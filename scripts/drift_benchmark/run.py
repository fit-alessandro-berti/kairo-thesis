"""Write one JSON artefact per event log, detector, and configuration."""

import argparse
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import traceback
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

# Set before numpy, scipy, sklearn or the worker processes import.
for env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[env_key] = "1"
os.environ.setdefault(
    "PM4PY_DEFAULT_ALIGNMENTS_VARIANT", "Variants.VERSION_DIJKSTRA_LESS_MEMORY"
)

ROOT = Path(__file__).resolve().parents[2]


def revision(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def json_value(value):
    """Serialize benchmark records and model metadata without address-based reprs."""
    import numpy as np
    import pandas as pd
    from pm4py.objects.petri_net.obj import Marking, PetriNet

    if value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return json_value(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (pd.Timedelta, np.timedelta64)):
        return pd.Timedelta(value).total_seconds()
    if isinstance(value, PetriNet):
        return {
            "places": len(value.places),
            "transitions": len(value.transitions),
            "silent_transitions": sum(item.label is None for item in value.transitions),
            "arcs": len(value.arcs),
        }
    if isinstance(value, Marking):
        return {"marked_places": len(value), "tokens": sum(value.values())}
    if isinstance(value, Mapping):
        return {
            key if isinstance(key, str) else json.dumps(json_value(key)): json_value(
                item
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((json_value(item) for item in value), key=repr)
    raise TypeError(f"Unsupported result type {type(value).__qualname__}")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    payload = json_value(payload)
    with temporary.open("w", encoding="utf-8") as stream:
        # Keep long evidence series to one row per observation.
        stream.write("{\n")
        for index, (key, value) in enumerate(payload.items()):
            stream.write(",\n" if index else "")
            stream.write(f"  {json.dumps(key)}: ")
            if key in {"scores", "versions"} and value:
                stream.write("[\n")
                for item_index, item in enumerate(value):
                    stream.write(",\n" if item_index else "")
                    stream.write(
                        "    " + json.dumps(item, ensure_ascii=False, allow_nan=False)
                    )
                stream.write("\n  ]")
            else:
                indent = None if key in {"scores", "versions"} else 2
                content = json.dumps(
                    value, ensure_ascii=False, allow_nan=False, indent=indent
                )
                stream.write(content.replace("\n", "\n  "))
        stream.write("\n}\n")
    temporary.replace(path)


def sample_scores(payload, full_scores):
    """Keep the regular series and alarm-linked scores when dense output is sampled."""
    scores = payload["scores"]
    stride = 1 if full_scores else max(1, math.ceil(len(scores) / 1200))
    if stride > 1:
        alarms = {item.get("index") for item in payload["changes"]}
        payload["scores"] = [
            item
            for i, item in enumerate(scores)
            if i % stride == 0 or i == len(scores) - 1 or item.get("index") in alarms
        ]
    payload["score_series"] = {
        "observation_count": len(scores),
        "published_count": len(payload["scores"]),
        "stride": stride,
        "keeps_alarm_indices_and_last_observation": True,
    }


def run_dataset(task):
    name, args, pm_revision, kairo_revision = task
    sys.path.insert(0, str(args.pm4py))
    import pandas as pd
    import pm4py

    if not Path(pm4py.__file__).resolve().is_relative_to(args.pm4py):
        raise RuntimeError(
            f"Imported PM4Py from {pm4py.__file__} instead of the selected checkout"
        )
    from . import datasets, evaluation, kairo_methods, pm_methods

    [(info, log)] = list(datasets.load_all(args.data_dir, {name}))
    lower = log[pm_methods.TIME].min()
    if pm_methods.START in log and log[pm_methods.START].notna().any():
        lower = min(lower, log[pm_methods.START].min())
    upper = log[pm_methods.TIME].max()
    span = upper - lower
    if span <= pd.Timedelta(0):
        raise ValueError(f"{name} has no positive time span")
    truth = info["truth"]
    assert len({item["id"] for item in truth}) == len(truth), name
    for item in truth:
        start, end = pd.Timestamp(item["start"]), pd.Timestamp(item["end"])
        if start.tzinfo is None or end.tzinfo is None or start > end:
            raise ValueError(f"{name} has malformed truth {item}")
    truth_episodes = evaluation.episodes(truth)
    here = args.output / name
    info.update(
        {
            "events": len(log),
            "cases": log[pm_methods.CASE].nunique(),
            "activities": log[pm_methods.ACTIVITY].nunique(),
            "resources": log[pm_methods.RESOURCE].nunique()
            if pm_methods.RESOURCE in log
            else 0,
            "first_timestamp": lower,
            "last_timestamp": upper,
            "observation_span_seconds": span.total_seconds(),
            "truth_episodes": truth_episodes,
        }
    )
    dataset_path = here / "dataset.json"
    reusable_dataset = (
        args.resume
        and dataset_path.is_file()
        and json.loads(dataset_path.read_text(encoding="utf-8")) == json_value(info)
    )
    if not reusable_dataset:
        write_json(dataset_path, info)
    counts = {"ok": 0, "unsupported": 0, "error": 0}

    def reusable(library, method, profile, parameters=None):
        if not reusable_dataset:
            return None
        path = here / library / method / f"{profile}.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["status"] not in {"ok", "unsupported"}:
            return None
        if payload["detector_source_revision"] != (
            pm_revision if library == "pm4py" else kairo_revision
        ):
            return None
        if parameters is not None and payload["configuration"] != json_value(
            parameters
        ):
            return None
        if method == "conformance-profile" and (
            payload["metadata"].get("precision_alignment_variant")
            != pm_methods.precision_alignment_variant()
        ):
            return None
        if payload["status"] == "ok":
            raw_count = payload["score_series"]["observation_count"]
            stride = 1 if args.full_scores else max(1, math.ceil(raw_count / 1200))
            if payload["score_series"]["stride"] != stride:
                return None
        return payload["status"]

    def record(library, method, profile, payload):
        path = here / library / method / f"{profile}.json"
        state = payload.get("status", "ok")
        if state == "ok":
            payload["evaluation"] = evaluation.evaluate(
                payload["changes"], truth_episodes, span
            )
            sample_scores(payload, args.full_scores)
        write_json(
            path,
            {
                "schema_version": 1,
                "dataset": name,
                "library": library,
                "method": method,
                "profile": profile,
                "status": state,
                "detector_source_revision": pm_revision
                if library == "pm4py"
                else kairo_revision,
                **payload,
            },
        )
        counts[state] += 1

    if args.library in {"both", "pm4py"}:
        trace_log = None
        for method, profile, parameters in pm_methods.configurations(log):
            if args.method and method not in args.method:
                continue
            existing = reusable("pm4py", method, profile, parameters)
            if existing is not None:
                counts[existing] += 1
                continue
            if trace_log is None:
                trace_log = pm_methods.prepare(log)
            begin = perf_counter()
            try:
                result = pm_methods.run_one(log, method, parameters, trace_log)
                result["configuration"] = parameters
                result["runtime_seconds"] = round(perf_counter() - begin, 6)
                record("pm4py", method, profile, result)
            except Exception as error:  # noqa: BLE001 - record failing runs without discarding other results
                traceback.print_exc()
                record(
                    "pm4py",
                    method,
                    profile,
                    {
                        "status": "error",
                        "configuration": parameters,
                        "error": f"{type(error).__name__}: {error}",
                    },
                )

    if args.library in {"both", "kairo"}:
        wanted = [
            (f"{perspective}-{method}", profile)
            for perspective in kairo_methods.PERSPECTIVES
            for method in kairo_methods.METHODS
            for profile in kairo_methods.PROFILES
            if not args.method or f"{perspective}-{method}" in args.method
        ]
        remaining = set()
        for method, profile in wanted:
            existing = reusable("kairo", method, profile)
            if existing is None:
                remaining.add((method, profile))
            else:
                counts[existing] += 1
        if remaining:
            for method, profile, result in kairo_methods.run(log):
                if (method, profile) in remaining:
                    record("kairo", method, profile, result)

    return name, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pm4py", type=Path, required=True, help="Selected PM4Py source checkout"
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Download/cache directory"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset", action="append", help="Repeat to run particular dataset IDs"
    )
    parser.add_argument("--library", choices=("both", "kairo", "pm4py"), default="both")
    parser.add_argument(
        "--method", action="append", help="Repeat to run particular detector IDs"
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--full-scores", action="store_true", help="Keep all score observations"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Reuse complete matching runs"
    )
    args = parser.parse_args()
    args.pm4py = args.pm4py.resolve()
    sys.path.insert(0, str(args.pm4py))
    from . import datasets

    names = args.dataset if args.dataset else datasets.default_ids()
    if len(set(names)) != len(names):
        parser.error("Dataset IDs must be unique")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    pm_revision = revision(args.pm4py)
    # The benchmark code has its own history. This is the assessed Kairo library commit.
    kairo_revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", "kairo/kairo"],
        text=True,
    ).strip()
    manifest_path = args.output / "manifest.json"
    full_run = not args.dataset and not args.method and args.library == "both"
    if full_run:
        manifest = {
            "schema_version": 1,
            "datasets": names,
            "pm4py_revision": pm_revision,
            "kairo_revision": kairo_revision,
            "full_scores": args.full_scores,
            "default_alignment_variant": os.environ["PM4PY_DEFAULT_ALIGNMENTS_VARIANT"],
            "runtime_versions": {
                "python": sys.version.split()[0],
                **{
                    key: version(key)
                    for key in (
                        "kairo",
                        "pandas",
                        "numpy",
                        "scikit-learn",
                        "scipy",
                        "minisom",
                        "river",
                        "hdbscan",
                    )
                },
            },
        }
        write_json(manifest_path, manifest)
    tasks = [(name, args, pm_revision, kairo_revision) for name in names]
    with ProcessPoolExecutor(
        max_workers=args.jobs, mp_context=mp.get_context("spawn")
    ) as pool:
        total_errors = 0
        totals = dict.fromkeys(("ok", "unsupported", "error"), 0)
        for future in as_completed(pool.submit(run_dataset, task) for task in tasks):
            name, counts = future.result()
            print(f"{name} {counts}", flush=True)
            total_errors += counts["error"]
            for state, count in counts.items():
                totals[state] += count
    if full_run:
        write_json(manifest_path, {**manifest, "counts": totals})
    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
