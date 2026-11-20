"""Repeat PM4Py control-flow EMD at wider windows on the Bose and Ceravolo logs."""

import argparse
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from math import ceil
from pathlib import Path
from time import perf_counter

from .run import revision, write_json

WINDOWS = (100, 200)
BOUNDARIES = 40
DEFAULT_DATASETS = (
    "cdrift_bose",
    "cdrift_ceravolo_rio",
    "cdrift_ceravolo_rio_noise15",
    "cdrift_ceravolo_rp",
)


def run_sensitivity(task):
    name, data_dir, output, pm4py_path, pm4py_revision = task
    sys.path.insert(0, str(pm4py_path))
    import pm4py

    if not Path(pm4py.__file__).resolve().is_relative_to(pm4py_path):
        raise RuntimeError(
            f"Imported PM4Py from {pm4py.__file__} instead of {pm4py_path}"
        )
    from . import datasets, evaluation, pm_methods

    ((info, log),) = datasets.load_all(data_dir, {name})
    cases = log[datasets.CASE].nunique()
    earliest = log[datasets.TIME].min()
    if datasets.START in log and log[datasets.START].notna().any():
        earliest = min(earliest, log[datasets.START].min())
    span = log[datasets.TIME].max() - earliest
    baseline = next(
        params
        for method, profile, params in pm_methods.configurations(log)
        if (method, profile) == ("earth-mover", "control-flow")
    )
    trace_log = pm_methods.prepare(log)
    for window in WINDOWS:
        # Bound the work independently of labeled drift positions. The new
        # width leaves (case count - 2 * width) eligible boundary increments.
        step = max(1, ceil((cases - 2 * window) / (BOUNDARIES - 1)))
        config = {**baseline, "window_size": window, "step": step}
        started = perf_counter()
        result = pm_methods.run_one(log, "earth-mover", config, trace_log)
        elapsed = perf_counter() - started
        write_json(
            output / name / f"window-{window}.json",
            {
                "schema_version": 1,
                "dataset": name,
                "library": "pm4py",
                "method": "earth-mover",
                "status": "ok",
                "detector_source_revision": pm4py_revision,
                "diagnostic": "post-hoc window sensitivity, distinct from the frozen benchmark",
                "configuration": config,
                "primary_configuration": baseline,
                "truth": info["truth"],
                "runtime_seconds": round(elapsed, 6),
                "evaluation": evaluation.evaluate(
                    result["changes"], info["truth"], span
                ),
                **result,
            },
        )
        print(
            name,
            window,
            len(result["scores"]),
            len(result["changes"]),
            round(elapsed, 2),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pm4py", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        help="Repeat to select archived cdrift dataset IDs (defaults to Bose and Ceravolo)",
    )
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    from .datasets import CDRIFT

    args.pm4py = args.pm4py.resolve()
    names = args.dataset or DEFAULT_DATASETS
    if len(set(names)) != len(names) or set(names).difference(CDRIFT):
        parser.error("choose unique IDs from " + ", ".join(CDRIFT))
    pm4py_revision = revision(args.pm4py)
    tasks = [
        (name, args.data_dir, args.output, args.pm4py, pm4py_revision) for name in names
    ]
    with ProcessPoolExecutor(
        max_workers=args.jobs, mp_context=mp.get_context("spawn")
    ) as pool:
        list(pool.map(run_sensitivity, tasks))


if __name__ == "__main__":
    main()
