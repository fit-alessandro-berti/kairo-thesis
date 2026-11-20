"""Verify a complete drift-benchmark result tree without loading the input logs."""

import argparse
import ast
import json
import math
import re
import statistics
import subprocess
from collections import Counter
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from . import evaluation

PM_PROFILES = {
    "bose": ("relation-count",),
    "dfg-frequency": ("permutation", "g-test-wide"),
    "event-relations": ("fixed", "adaptive"),
    "lcdd": ("local-completeness",),
    "relation-invariance": ("stable-relations",),
    "graph-metrics": ("pelt",),
    "run-distribution": ("fixed", "adaptive"),
    "trace-variant-frequency": ("permutation", "permutation-wide"),
    "earth-mover": ("control-flow", "time-aware"),
    "indicators": ("pelt-basic", "pelt-enriched", "adwin-basic"),
    "cv-indicators": ("pelt-enriched",),
    "conformance": ("page-hinkley-fixed", "adwin-relearn"),
    "conformance-profile": ("pelt",),
    "declare-profiles": ("pelt",),
    "version-clustering": ("similarity",),
    "cluster-frequency": ("trained-kmeans",),
    "multivariate": ("rbf-flow-timing",),
    "supervised-similarity": ("independent-prototype",),
}
KAIRO_METHODS = tuple(
    f"{perspective}-{cluster}"
    for perspective in ("intra", "resource", "inter")
    for cluster in ("kmeans", "som", "dbscan")
)
KAIRO_PROFILES = ("previous", "recent")
STATES = ("ok", "unsupported", "error")


class VerificationError(ValueError):
    """The result tree violates the benchmark format or scoring invariants."""


def require(condition, context, message):
    if not condition:
        raise VerificationError(f"{context}: {message}")


def _read(path):
    def reject_constant(value):
        raise VerificationError(f"{path}: invalid JSON number {value!r}")

    with path.open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject_constant)


def _timestamp(value, context):
    point = pd.Timestamp(value)
    require(
        not pd.isna(point) and point.utcoffset() == timedelta(0),
        context,
        f"expected a UTC timestamp, got {value!r}",
    )
    return point


def _count(value, context):
    require(
        type(value) is int and value >= 0,
        context,
        f"expected a nonnegative integer, got {value!r}",
    )
    return value


def _expected_files(has_resources):
    paths = {
        ("pm4py", method, profile)
        for method, profiles in PM_PROFILES.items()
        for profile in profiles
    }
    if has_resources:
        paths.add(("pm4py", "multivariate", "rbf-resource-assignment"))
    return paths | {
        ("kairo", method, profile)
        for method in KAIRO_METHODS
        for profile in KAIRO_PROFILES
    }


def _verify_checkout(checkout, revision):
    path = "pm4py/algo/concept_drift/algorithm.py"
    source = subprocess.check_output(
        ["git", "-C", str(checkout), "show", f"{revision}:{path}"], text=True
    )
    enum = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "Variants"
    )
    variants = {
        target.id.lower().replace("_", "-")
        for assignment in enum.body
        if isinstance(assignment, ast.Assign)
        for target in assignment.targets
        if isinstance(target, ast.Name)
    }
    require(
        variants == set(PM_PROFILES),
        checkout,
        f"unchecked or stale enum variants {variants ^ set(PM_PROFILES)}",
    )


def _verify_method_revision(checkout, base, revised, method):
    command = ["git", "-C", str(checkout)]
    subprocess.run([*command, "merge-base", "--is-ancestor", base, revised], check=True)
    changed = subprocess.check_output(
        [*command, "diff", "--name-only", base, revised, "--", "pm4py/"], text=True
    ).splitlines()
    expected = f"pm4py/algo/concept_drift/variants/{method.replace('-', '_')}.py"
    require(
        changed == [expected],
        checkout,
        f"revision override for {method} changes unexpected files {changed}",
    )


def _truth(dataset, context):
    truth = dataset["truth"]
    require(
        len({item["id"] for item in truth}) == len(truth),
        context,
        "duplicate truth IDs",
    )
    intervals = set()
    for item in truth:
        start = _timestamp(item["start"], context)
        end = _timestamp(item["end"], context)
        require(start <= end, context, f"reversed truth interval for {item['id']}")
        intervals.add((start, end))
    episodes = dataset["truth_episodes"]
    episode_bounds = {
        (_timestamp(item["start"], context), _timestamp(item["end"], context))
        for item in episodes
    }
    require(
        len({item["id"] for item in episodes}) == len(episodes),
        context,
        "duplicate truth episode IDs",
    )
    require(
        episode_bounds == intervals and len(episode_bounds) == len(episodes),
        context,
        "truth episodes do not preserve all distinct annotated intervals",
    )
    for actual, expected in zip(episodes, evaluation.episodes(truth)):
        require(
            actual["id"] == expected["id"]
            and actual["annotations"] == expected["annotations"],
            context,
            "truth episode IDs do not reconcile with raw annotations",
        )
    return episodes


def _verify_matches(result, truth, lower, upper, path):
    changes = result["changes"]
    span = upper - lower
    expected = evaluation.evaluate(changes, truth, span)
    observed = result["evaluation"]
    require(
        observed == expected,
        path,
        "stored evaluation differs from recomputed timestamp matching",
    )

    # Verify match cardinality separately from the evaluation's cost matrix.
    estimates = [_timestamp(change["timestamp"], path) for change in changes]
    bounds = [
        (_timestamp(item["start"], path), _timestamp(item["end"], path))
        for item in truth
    ]
    for fraction, metrics in observed["by_tolerance_fraction"].items():
        pairs = metrics["matches"]
        matched_truth = {pair["truth_id"] for pair in pairs}
        matched_changes = {pair["change_offset"] for pair in pairs}
        tp = metrics["tp"]
        require(
            tp == len(pairs) == len(matched_truth) == len(matched_changes),
            path,
            f"non one-to-one matches at tolerance {fraction}",
        )
        require(
            tp + metrics["fp"] == len(changes) and tp + metrics["fn"] == len(bounds),
            path,
            f"unreconciled detection and truth counts at tolerance {fraction}",
        )
        tolerance = pd.Timedelta(seconds=metrics["tolerance_seconds"])
        if bounds and estimates:
            adjacency = csr_matrix(
                np.array(
                    [
                        [
                            start - tolerance <= estimate <= end + tolerance
                            for estimate in estimates
                        ]
                        for start, end in bounds
                    ]
                )
            )
            optimum = int(np.count_nonzero(maximum_bipartite_matching(adjacency) >= 0))
            require(
                tp == optimum,
                path,
                f"matching does not maximize cardinality at tolerance {fraction}",
            )


def _verify_points(result, lower, upper, path, full_scores=False):
    is_kairo = result["library"] == "kairo"
    unit = result["observation_unit"]
    configuration = result["configuration"]
    window = pd.Timedelta(0)
    if is_kairo:
        require(unit == "signal_window", path, "unknown Kairo observation unit")
        window = pd.Timedelta(configuration["signal"]["window"])
    elif unit == "calendar_bin":
        window = pd.Timedelta(configuration["bin_size"])

    for key in ("changes", "scores", "versions"):
        recent = None
        for point in result.get(key, []):
            if "index" in point:
                _count(point["index"], path)
            if "score" in point and point["score"] is not None:
                require(math.isfinite(point["score"]), path, f"nonfinite {key} score")
            if "p_value" in point:
                require(0 <= point["p_value"] <= 1, path, f"out-of-range {key} p-value")
            moment = _timestamp(point["timestamp"], path)
            require(
                lower - max(window, pd.Timedelta(microseconds=1)) <= moment <= upper,
                path,
                f"{key} timestamp outside the observed time range {point['timestamp']!r}",
            )
            if key != "versions":
                available = _timestamp(point["available_at"], path)
                require(
                    moment <= available <= upper + window,
                    path,
                    f"invalid {key} reporting time",
                )
                if is_kairo:
                    require(
                        available == upper,
                        path,
                        "Kairo result was available before full-log fitting",
                    )
                    window_end = _timestamp(point["details"]["window_end"], path)
                    require(
                        window_end - moment == window
                        and moment.floor(window) == moment,
                        path,
                        "Kairo window does not match configuration",
                    )
                    if key == "changes":
                        require(
                            point["details"]["eligible"] and window_end <= upper,
                            path,
                            "Kairo alarm originates in an ineligible window",
                        )
            if is_kairo and key == "scores":
                require(
                    recent is None or point["index"] > recent,
                    path,
                    "Kairo calendar score indices are out of order",
                )
                recent = point["index"]

    for interval in result.get("intervals", []):
        start_index = _count(interval["start_index"], path)
        end_index = _count(interval["end_index"], path)
        start = _timestamp(interval["start_timestamp"], path)
        end = _timestamp(interval["end_timestamp"], path)
        available = _timestamp(interval["available_at"], path)
        require(
            start_index <= end_index
            and lower <= start <= end <= available <= upper + window,
            path,
            "invalid transition interval",
        )

    _verify_score_sampling(result, result["score_series"], full_scores, path)
    if is_kairo and result["score_series"]["stride"] == 1:
        _verify_kairo_threshold(result, path)


def _verify_kairo_threshold(result, path):
    scores = result["scores"]
    config = result["configuration"]["signal"]
    stored = result["metadata"]
    median_rows = statistics.median(
        _count(point["details"]["rows"], path) for point in scores
    )
    require(
        median_rows == stored["median_rows"],
        path,
        "Kairo median window size differs from evidence",
    )
    eligible = []
    for point in scores:
        details = point["details"]
        require(
            0 <= details["coverage"] <= 1,
            path,
            "Kairo coverage fraction outside [0, 1]",
        )
        passes = (
            point["score"] is not None
            and details["coverage"] == 1
            and details["rows"]
            >= config["min_rows_as_fraction_of_median"] * median_rows
        )
        require(
            details["eligible"] == passes,
            path,
            "Kairo score eligibility differs from window evidence",
        )
        if passes:
            eligible.append(point)

    require(
        stored["eligible_windows"] == len(eligible),
        path,
        "Kairo eligible-window count differs from evidence",
    )
    expected = []
    if eligible:
        threshold = max(
            1e-12,
            np.quantile(
                [point["score"] for point in eligible], config["threshold_quantile"]
            ),
        )
        require(
            math.isclose(threshold, stored["threshold"], rel_tol=1e-12),
            path,
            "invalid Kairo evaluation threshold",
        )
        active_until = None
        for point in eligible:
            if point["score"] >= stored["threshold"]:
                if point["timestamp"] != active_until:
                    expected.append(point)
                active_until = point["details"]["window_end"]
            else:
                active_until = None
    else:
        require(
            stored["threshold"] is None,
            path,
            "Kairo reports a threshold without eligible scores",
        )
    require(
        expected == result["changes"],
        path,
        "Kairo alarms differ from their threshold-crossing evidence",
    )


def _verify_score_sampling(result, series, full_scores, path):
    # Full validation of sampled evidence uses its declared size and preserved alarms.
    retained = len(result["scores"])
    original = series["observation_count"]
    stride = series["stride"]
    require(
        _count(original, path) >= retained,
        path,
        "stored score count exceeds source count",
    )
    require(
        series["published_count"] == retained,
        path,
        "published score count differs from score evidence",
    )
    expected_stride = 1 if full_scores else max(1, math.ceil(original / 1200))
    require(
        stride == expected_stride
        and series["keeps_alarm_indices_and_last_observation"],
        path,
        "incorrect score sampling declaration",
    )
    if stride == 1:
        require(retained == original, path, "full score series is incomplete")
    if retained < original:
        steps, remainder = divmod(original - 1, stride)
        require(
            retained >= steps + 1 + bool(remainder),
            path,
            "regular stride observations are missing",
        )
        score_indices = {point.get("index") for point in result["scores"]}
        changes = {point["index"] for point in result["changes"] if "index" in point}
        require(
            changes <= score_indices,
            path,
            "score sampling omitted evidence for at least one alarm",
        )


def _verify_unsupported(result, resources, path):
    require(
        result["library"] == "kairo",
        path,
        "expected a PM4Py result for this configuration",
    )
    require(
        not result["changes"] and not result["scores"] and "evaluation" not in result,
        path,
        "unsupported method has purported detector evidence",
    )
    reason, method = result["reason"], result["method"]
    missing_resource = (
        not resources
        and method.startswith("resource-")
        and reason == "The event log has no observed resources"
    )
    dbscan_limit = method.endswith("-dbscan") and re.fullmatch(
        r"\d+ distinct rows exceed the DBSCAN cap of \d+", reason
    )
    if dbscan_limit:
        count, cap = map(int, re.findall(r"\d+", reason))
        require(
            count > cap, path, "unsupported DBSCAN result is within its declared cap"
        )
    no_clusters = method.endswith("-dbscan") and reason.startswith(
        "DBSCAN found no clusters, every row is noise."
    )
    require(
        missing_resource or dbscan_limit or no_clusters,
        path,
        f"unexpected unsupported reason {reason!r}",
    )


def verify(output, expected_datasets, pm4py=None):
    output = Path(output)
    manifest = _read(output / "manifest.json")
    require(
        manifest["schema_version"] == 1, output, "unexpected manifest schema version"
    )
    names = manifest["datasets"]
    full_scores = manifest["full_scores"]
    require(type(full_scores) is bool, output, "invalid full-score setting")
    require(
        len(names) == len(set(names)) and set(names) == set(expected_datasets),
        output,
        "incomplete dataset manifest",
    )
    for library in ("kairo", "pm4py"):
        require(
            re.fullmatch(r"[0-9a-f]{40}", manifest[f"{library}_revision"]),
            output,
            f"invalid {library} source revision",
        )
    method_revisions = manifest.get("pm4py_method_revisions", {})
    require(
        isinstance(method_revisions, dict),
        output,
        "invalid PM4Py method revision overrides",
    )
    for method, revised in method_revisions.items():
        require(
            method in PM_PROFILES and re.fullmatch(r"[0-9a-f]{40}", revised),
            output,
            f"invalid PM4Py method revision override {method!r}",
        )
    if pm4py is not None:
        _verify_checkout(pm4py, manifest["pm4py_revision"])
        for method, revised in method_revisions.items():
            _verify_method_revision(pm4py, manifest["pm4py_revision"], revised, method)

    dataset_paths = {path.parent.name for path in output.glob("*/dataset.json")}
    require(
        dataset_paths == set(names),
        output,
        f"missing or stale dataset files {dataset_paths ^ set(names)}",
    )
    counts = {
        library: Counter({state: 0 for state in STATES})
        for library in ("kairo", "pm4py")
    }
    dataset_counts = {}
    for name in names:
        base = output / name
        dataset = _read(base / "dataset.json")
        require(
            dataset["id"] == name,
            base,
            "dataset metadata does not match its result folder",
        )
        truth = _truth(dataset, base)
        resources = _count(dataset["resources"], base)
        events = _count(dataset["events"], base)
        cases = _count(dataset["cases"], base)
        require(
            events >= cases > 0 and _count(dataset["activities"], base) > 0,
            base,
            "invalid log dimensions",
        )
        lower = _timestamp(dataset["first_timestamp"], base)
        upper = _timestamp(dataset["last_timestamp"], base)
        span = (upper - lower).total_seconds()
        require(
            span > 0 and span == dataset["observation_span_seconds"],
            base,
            "invalid observation span",
        )

        actual = {
            (*path.parent.relative_to(base).parts, path.stem)
            for path in base.glob("*/*/*.json")
        }
        expected = _expected_files(resources > 0)
        require(
            actual == expected,
            base,
            f"missing {sorted(expected - actual)} and stale {sorted(actual - expected)}",
        )
        dataset_counts[name] = Counter({state: 0 for state in STATES})
        for library, method, profile in sorted(actual):
            path = base / library / method / f"{profile}.json"
            result = _read(path)
            revision = manifest[f"{library}_revision"]
            if library == "pm4py":
                revision = method_revisions.get(method, revision)
            for key, value in {
                "schema_version": 1,
                "dataset": name,
                "library": library,
                "method": method,
                "profile": profile,
                "detector_source_revision": revision,
            }.items():
                require(result.get(key) == value, path, f"incorrect {key}")
            require(
                isinstance(result["configuration"], dict) and result["configuration"],
                path,
                "missing configuration",
            )
            status = result["status"]
            require(
                status in STATES and status != "error",
                path,
                f"detector error {result.get('error', status)!r}",
            )
            if status == "ok":
                if (
                    method == "conformance-profile"
                    and "default_alignment_variant" in manifest
                ):
                    expected_alignment = manifest["default_alignment_variant"].rsplit(
                        ".", 1
                    )[-1]
                    require(
                        result["metadata"].get("precision_alignment_variant")
                        == expected_alignment,
                        path,
                        "unexpected alignment solver for the precision profile",
                    )
                _verify_points(result, lower, upper, path, full_scores)
                _verify_matches(result, truth, lower, upper, path)
            else:
                _verify_unsupported(result, resources, path)
            counts[library][status] += 1
            dataset_counts[name][status] += 1

    totals = counts["kairo"].copy()
    totals.update(counts["pm4py"])
    require(
        manifest.get("counts") == totals,
        output,
        "manifest result counts differ from verified files",
    )
    return {
        "datasets": len(names),
        "techniques": {"kairo": len(KAIRO_METHODS), "pm4py": len(PM_PROFILES)},
        "results": counts,
        "results_by_dataset": dataset_counts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--pm4py",
        type=Path,
        help="Verify the recorded revision's dispatcher in this checkout",
    )
    args = parser.parse_args()
    from . import datasets

    try:
        report = verify(args.output, datasets.default_ids(), args.pm4py)
    except (KeyError, ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Verification failed: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
