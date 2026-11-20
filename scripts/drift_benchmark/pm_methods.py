"""Configurations and result adapter for the PM4Py concept-drift dispatcher."""

from functools import lru_cache
from itertools import combinations
from math import ceil

import numpy as np
import pandas as pd
import pm4py
from pm4py.algo.concept_drift import algorithm
from pm4py.algo.concept_drift.variants import supervised_similarity
from pm4py.algo.conformance.alignments.petri_net import algorithm as petri_alignments
from pm4py.objects.conversion.process_tree import converter as tree_converter
from pm4py.objects.log.obj import EventLog
from pm4py.objects.process_tree.obj import Operator, ProcessTree
from pm4py.util.string_distance import levenshtein_distance

CASE = "case:concept:name"
ACTIVITY = "concept:name"
TIME = "time:timestamp"
START = "start:timestamp"
RESOURCE = "org:resource"
SEED = 73
IMAGE_WINDOWS = 48


def precision_alignment_variant():
    return petri_alignments.DEFAULT_VARIANT.name


def calendar_bin(log):
    span = log[TIME].max() - log[TIME].min()
    hours = max(1, ceil(span / pd.Timedelta(hours=1) / 100))
    return f"{hours}h"


def has_service_times(log):
    return START in log and (log[START] < log[TIME]).any()


def indicator_features(log):
    basic = ["arrivals", "completions", "work_in_progress", "events", "case_duration"]
    frequent = log[ACTIVITY].value_counts().head(4).index
    features = [*basic, *(("activity_frequency", act) for act in frequent)]
    if has_service_times(log):
        features += [("service_time", act) for act in frequent]
    else:
        features += [("inter_event_gap", act) for act in frequent]
    if RESOURCE in log and log[RESOURCE].notna().any():
        features.append("active_resources")
        for act in frequent:
            resources = log.loc[log[ACTIVITY].eq(act), RESOURCE].value_counts().head(2)
            features += [("resource_assignment", act, res) for res in resources.index]
    return basic, features


def configurations(log):
    """Size windows from input volume, without reading labels or detector outcomes."""
    cases = log[CASE].nunique()
    rows = len(log)
    n = max(16, min(150, round(cases / 22)))
    wide = min(300, n * 2)
    step = max(2, n // 5, cases // 220)
    block = max(n, cases // 120)
    references = min(max(35, n), 120)
    # Exact search over the concurrent mined net was impractical on the
    # longest executions. A sequential medoid supplies a bounded reference.
    alignment = {"reference_size": references}
    recalibration = references
    if rows / cases > 35:
        alignment = {
            "reference_size": 30,
            "reference_model_strategy": "sequential_medoid",
        }
        recalibration = 1
    perms = {"permutations": 99, "random_state": SEED, "alpha": 0.03, "persistence": 2}
    narrow = dict(window_size=n, step=step, min_separation=n, **perms)
    smooth = dict(
        window_size=wide, step=max(step, wide // 5), min_separation=wide, **perms
    )
    events = max(100, min(1600, round(n * rows / cases)))
    event_step = max(2, events // 8, rows // 900)
    frequency = calendar_bin(log)
    basic, enriched = indicator_features(log)
    has_resources = RESOURCE in log and log[RESOURCE].notna().any()
    valid_service = (
        has_service_times(log)
        and log[START].notna().all()
        and (log[START] <= log[TIME]).all()
    )
    emd = {"window_size": min(n, 35), "step": max(4, cases // 105), "min_separation": n}

    yield (
        "bose",
        "relation-count",
        {
            "sub_log_size": max(5, min(50, cases // 80)),
            "window_size": 6,
            "num_permutations": 399,
            "thresh_p_value": 0.025,
            "max_no_change_points": 100,
            "random_state": SEED,
        },
    )
    yield "dfg-frequency", "permutation", {**narrow, "min_effect": 0.1}
    yield (
        "dfg-frequency",
        "g-test-wide",
        {**smooth, "test": "g_test", "min_effect": 0.1},
    )
    yield (
        "event-relations",
        "fixed",
        {
            "window_size": events,
            "adaptive": False,
            "step": event_step,
            "alpha": 0.01,
            "persistence": 2,
            "min_effect": 0.1,
            "min_separation": events,
        },
    )
    yield (
        "event-relations",
        "adaptive",
        {
            "window_size": events,
            "adaptive": True,
            "step": event_step,
            "alpha": 0.01,
            "persistence": 2,
            "min_effect": 0.1,
            "min_separation": events,
        },
    )
    yield (
        "lcdd",
        "local-completeness",
        {
            "complete_window_size": wide,
            "detection_window_size": n,
            "stable_period": max(2, n // 8),
        },
    )
    yield (
        "relation-invariance",
        "stable-relations",
        {
            "min_stable_traces": wide,
            "radius": max(3, n // 8),
            "min_separation": n,
        },
    )
    yield "graph-metrics", "pelt", {"window_size": block, "penalty": 12, "min_size": 3}
    yield (
        "run-distribution",
        "fixed",
        {
            "window_size": n,
            "step": step,
            "alpha": 0.025,
            "persistence": 3,
            "min_separation": n,
            "gradual": True,
        },
    )
    yield (
        "run-distribution",
        "adaptive",
        {
            "window_size": n,
            "step": step,
            "alpha": 0.025,
            "persistence": 3,
            "min_separation": n,
            "gradual": True,
            "adaptive": True,
        },
    )
    yield "trace-variant-frequency", "permutation", {**narrow, "min_effect": 0.12}
    yield "trace-variant-frequency", "permutation-wide", {**smooth, "min_effect": 0.12}
    yield "earth-mover", "control-flow", {**emd, "threshold": 0.20}
    timing = "service" if valid_service else "inter_event"
    yield (
        "earth-mover",
        "time-aware",
        {
            **emd,
            "threshold": 0.35,
            "time_feature": timing,
            "start_timestamp_key": START,
        },
    )
    yield (
        "indicators",
        "pelt-basic",
        {"bin_size": frequency, "features": basic, "penalty": 14},
    )
    yield (
        "indicators",
        "pelt-enriched",
        {
            "bin_size": frequency,
            "features": enriched,
            "penalty": 18,
            "start_timestamp_key": START,
        },
    )
    yield (
        "indicators",
        "adwin-basic",
        {
            "bin_size": frequency,
            "features": basic,
            "method": "adwin",
            "adwin_delta": 0.005,
            "adwin_clock": 1,
        },
    )
    yield (
        "cv-indicators",
        "pelt-enriched",
        {
            "bin_size": frequency,
            "features": enriched,
            "penalty_scale": 3,
            "start_timestamp_key": START,
        },
    )
    yield (
        "conformance",
        "page-hinkley-fixed",
        {
            **alignment,
            "page_hinkley_min_instances": 20,
            "page_hinkley_threshold": 5,
        },
    )
    yield (
        "conformance",
        "adwin-relearn",
        {
            **alignment,
            "method": "adwin",
            "adwin_delta": 0.005,
            "adwin_clock": 8,
            "model_policy": "on_alarm",
            "update_window": recalibration,
        },
    )
    yield (
        "conformance-profile",
        "pelt",
        {
            "window_size": block,
            "step": block,
            **alignment,
            "penalty": 0.1,
            "min_size": 3,
        },
    )
    # A fixed-size bank keeps pairwise rule extraction tractable on large alphabets.
    activities = log[ACTIVITY].value_counts().head(10).index.tolist()
    bank = [("existence", act) for act in activities] + [
        (relation, left, right)
        for left in activities
        for right in activities
        if left != right
        for relation in ("response", "precedence")
    ]
    yield (
        "declare-profiles",
        "pelt",
        {
            "window_size": block,
            "step": block,
            "penalty": 3,
            "constraints": bank,
        },
    )
    yield (
        "version-clustering",
        "similarity",
        {
            "num_windows": 64,
            "min_cluster_size": 4,
            "penalty": 3,
        },
    )
    yield (
        "cluster-frequency",
        "trained-kmeans",
        {
            **narrow,
            "training_size": references,
            "n_clusters": 5,
            "min_effect": 0.12,
            "cluster_random_state": SEED,
        },
    )
    yield (
        "multivariate",
        "rbf-flow-timing",
        {
            **narrow,
            "features": ["activity", "duration", "mean_gap"],
            "method": "mmd",
        },
    )
    if has_resources:
        yield (
            "multivariate",
            "rbf-resource-assignment",
            {
                **narrow,
                "features": ["activity_resource"],
                "method": "mmd",
            },
        )
    yield (
        "supervised-similarity",
        "independent-prototype",
        {
            "num_windows": IMAGE_WINDOWS,
            "min_confidence": 0.8,
        },
    )


@lru_cache(maxsize=1)
def train_prototype():
    """Fit the in-branch patch classifier on independent generated log sequences."""
    rng = np.random.default_rng(SEED)
    cases_per_window = 10
    case_count = IMAGE_WINDOWS * cases_per_window
    origin = pd.Timestamp("2010-01-01", tz="UTC")
    examples = []
    pairs = [
        (("A", "B", "C", "D"), ("A", "C", "B", "D")),
        (("A", "B", "D"), ("A", "C", "D")),
        (("A", "B", "C", "D"), ("A", "D")),
        (("A", "B", "B", "D"), ("A", "B", "C", "D")),
    ]
    for replicate in range(24):
        before, after = pairs[replicate % len(pairs)]
        if replicate % 2:
            before, after = after, before
        boundary = int(rng.integers(10, 36))
        start = boundary * cases_per_window
        width = 0 if replicate % 3 else int(rng.integers(3, 8))
        annotations = [
            supervised_similarity.Region(
                boundary, boundary + width, "gradual" if width else "sudden"
            )
        ]
        rows = []
        for index in range(case_count):
            if width:
                probability = np.clip(
                    (index - start) / (width * cases_per_window), 0, 1
                )
            else:
                probability = float(index >= start)
            noise = (replicate % 4) * 0.035
            use_after = rng.random() < noise + (1 - 2 * noise) * probability
            for position, act in enumerate(after if use_after else before):
                timestamp = origin + pd.Timedelta(hours=index, minutes=position)
                rows.append((str(index), act, timestamp))
        examples.append(
            (pd.DataFrame(rows, columns=[CASE, ACTIVITY, TIME]), annotations)
        )
    return supervised_similarity.fit(
        examples,
        {"num_windows": IMAGE_WINDOWS},
        radius=2,
        random_state=SEED,
    )


def point_record(item, include_details=False):
    record = {
        key: getattr(item, key)
        for key in ("index", "timestamp", "available_at", "score", "p_value")
        if getattr(item, key) is not None
    }
    if include_details and item.details:
        record["details"] = item.details
    return record


def prepare(log):
    """Reuse the supported legacy trace representation across detectors."""
    return pm4py.convert_to_event_log(log)


def sequential_reference(trace_log, size):
    """Build a sequential Petri net from a training-only trace medoid."""
    ordered = sorted(trace_log, key=lambda trace: max(event[TIME] for event in trace))
    variants = [
        tuple(event[ACTIVITY] for event in sorted(trace, key=lambda item: item[TIME]))
        for trace in ordered[:size]
    ]
    totals = [0.0] * len(variants)
    for first, second in combinations(range(len(variants)), 2):
        a, b = variants[first], variants[second]
        distance = levenshtein_distance(a, b) / max(len(a), len(b))
        totals[first] += distance
        totals[second] += distance
    representative = variants[min(range(len(variants)), key=totals.__getitem__)]
    tree = ProcessTree(operator=Operator.SEQUENCE)
    tree.children = [
        ProcessTree(label=activity, parent=tree) for activity in representative
    ]
    return tree_converter.apply(tree), EventLog(ordered[size:])


def run_one(log, method, configuration, trace_log=None):
    variant = algorithm.Variants[method.upper().replace("-", "_")]
    parameters = dict(configuration)
    if method == "supervised-similarity":
        parameters["predictor"] = train_prototype()
    use_traces = trace_log is not None and method != "event-relations"
    active_log = trace_log if use_traces else log
    reference_strategy = parameters.pop("reference_model_strategy", None)
    if reference_strategy == "sequential_medoid":
        size = parameters["reference_size"]
        parameters["reference_model"], active_log = sequential_reference(
            trace_log if use_traces else prepare(log), size
        )
        use_traces = True
    if method == "bose":
        # The historical revision only used NumPy's global permutation RNG.
        np.random.seed(SEED)
    result = algorithm.apply(active_log, variant=variant, parameters=parameters)
    if method == "bose":
        _, timestamps, p_values = result
        points = [
            {
                "timestamp": pd.to_datetime(time, unit="s", utc=True),
                "p_value": value,
                "available_at": log[TIME].max(),
            }
            for time, value in sorted(zip(timestamps, p_values))
        ]
        return {
            "observation_unit": "trace",
            "changes": points,
            "scores": [],
            "metadata": {
                "retrospective": True,
                "trace_order": "start",
                "input_format": ("EventLog" if use_traces else "DataFrame"),
                "score_output": "legacy API returns p-values only for alarms",
            },
        }
    metadata = dict(result.metadata)
    metadata["input_format"] = "EventLog" if use_traces else "DataFrame"
    if reference_strategy is not None:
        metadata["reference_model_strategy"] = reference_strategy
        metadata["source_training_traces_excluded"] = configuration["reference_size"]
    if method == "conformance-profile":
        metadata["precision_alignment_variant"] = precision_alignment_variant()
    if method == "supervised-similarity":
        metadata["training"] = {
            "source": "independently generated by train_prototype",
            "seed": SEED,
            "logs": 24,
            "cases_per_log": IMAGE_WINDOWS * 10,
            "radius": 2,
        }
    return {
        "observation_unit": result.observation_unit,
        "changes": [
            point_record(item, include_details=True) for item in result.changes
        ],
        "scores": [point_record(item) for item in result.scores],
        "intervals": [vars(item) for item in result.intervals],
        "versions": [vars(item) for item in result.versions],
        "metadata": metadata,
    }
