"""Run Kairo's public state and drift APIs with label-free benchmark settings."""

from collections.abc import Iterator
from math import log as logarithm

import numpy as np
import pandas as pd
from kairo.analysis import META
from sklearn.neighbors import NearestNeighbors

import kairo

CASE, TIME = META
ACTIVITY = "concept:name"
RESOURCE = "org:resource"
PERSPECTIVES = ("intra", "resource", "inter")
METHODS = ("kmeans", "som", "dbscan")
PROFILES = {
    "previous": {
        "divergence": "js",
        "reference": "previous",
        "quantile": 0.90,
        "window_factor": 1,
    },
    "recent": {
        "divergence": "tv",
        "reference": "recent",
        "lookback": 3,
        "quantile": 0.95,
        "window_factor": 2,
    },
}
WINDOWS = tuple(
    pd.Timedelta(frequency)
    for frequency in (
        "1s",
        "5s",
        "15s",
        "30s",
        "1min",
        "5min",
        "15min",
        "30min",
        "1h",
        "2h",
        "3h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1D",
        "2D",
        "3D",
        "7D",
        "14D",
        "28D",
        "56D",
        "112D",
    )
)
TIME_UNITS = ("1D", "1h", "1min", "1s")
CASE_ATTRIBUTES = {
    "case:amount": "numerical",
    "case:region": "categorical",
    "case:requestedamount": "numerical",
    "case:loanamount": "numerical",
    "case:applicationtype": "categorical",
    "case:loangoal": "categorical",
}
# The built-in feature extractors materialize dense matrices and DBSCAN may
# materialize a neighbor list for each distinct row.
MAX_FEATURE_CELLS = 15_000_000
MAX_DBSCAN_DISTINCT_ROWS = 10_000
POSITIVE_SCORE = 1e-12


def _frequency(window: pd.Timedelta) -> str:
    for unit in TIME_UNITS:
        length = pd.Timedelta(unit)
        if window % length == pd.Timedelta(0):
            return f"{window // length}{unit[1:]}"
    raise ValueError(f"Expected a whole-second window, got {window}")


def _closest(windows, target: pd.Timedelta) -> pd.Timedelta:
    target = max(target, WINDOWS[0])
    return min(windows, key=lambda width: abs(logarithm(width / target)))


def _windows(log: pd.DataFrame) -> tuple[pd.Timedelta, pd.Timedelta, pd.Timedelta]:
    span = log[TIME].max() - log[TIME].min()
    signal = _closest(WINDOWS, span / 50)
    divisors = [
        window
        for window in WINDOWS
        if window <= signal and signal % window == pd.Timedelta(0)
    ]
    resource_target = max(span / 500, (span / len(log)) * 20)
    inter_target = max(span / 500, (span / log[CASE].nunique()) * 5)
    return signal, _closest(divisors, resource_target), _closest(divisors, inter_target)


def _intra_features(log: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    activity_count = log[ACTIVITY].nunique()
    previous = log.groupby(CASE, sort=False)[ACTIVITY].shift()
    relations = pd.DataFrame({"previous": previous, "current": log[ACTIVITY]})
    relation_count = len(relations.dropna().drop_duplicates())
    widths = {
        "current_act": activity_count,
        "df_counts": relation_count,
        "past_acts": 3 * activity_count,
        "act_freqs": activity_count,
        "act_set": activity_count,
        "case_progress": 1,
    }
    # Reserve one column for case progress. When the log is wide, prioritize
    # features that represent the current step and the order of activities.
    remaining = max(0, MAX_FEATURE_CELLS // len(log) - 1)
    selected = {"case_progress"}
    for group, count in widths.items():
        if group != "case_progress" and count <= remaining:
            selected.add(group)
            remaining -= count
    groups = [group for group in kairo.INTRA_GLOSSARY if group in selected]
    spec = {
        "groups": groups,
        "sliding_window_size": 3,
        "omitted_feature_groups": sorted(set(widths) - selected),
    }
    return kairo.compute_features_intra(
        log, features=groups, sliding_window_size=3
    ), spec


def _duration(log: pd.DataFrame) -> tuple[pd.Series | None, str | None]:
    for column in ("event_duration", "event:duration_min"):
        if column in log:
            duration = pd.to_numeric(log[column], errors="coerce")
            duration = duration.where(np.isfinite(duration) & duration.ge(0))
            if duration.notna().any():
                return duration, column
    for column in ("start:timestamp", "start_timestamp"):
        if column in log:
            start = pd.to_datetime(
                log[column], utc=True, errors="coerce", format="mixed"
            )
            duration = (log[TIME] - start).dt.total_seconds() / 60
            duration = duration.where(duration.ge(0))
            if duration.notna().any():
                return duration, f"({TIME} - {column}) in minutes"
    return None, None


def _resource_features(
    log: pd.DataFrame, window: pd.Timedelta
) -> tuple[pd.DataFrame, dict]:
    feature_window = _frequency(window)
    n_windows = (
        int((log[TIME].max().floor(window) - log[TIME].min().floor(window)) // window)
        + 1
    )
    resource_count = log[RESOURCE].nunique()
    previous = log.groupby(CASE, sort=False)[RESOURCE].shift()
    handed_over = previous.notna() & log[RESOURCE].notna() & previous.ne(log[RESOURCE])
    handovers = pd.DataFrame(
        {"giver": previous[handed_over], "receiver": log.loc[handed_over, RESOURCE]}
    )
    widths = {
        group: resource_count for group in ("res_events", "res_cases", "res_waits")
    }
    duration, duration_source = _duration(log)
    if duration is not None:
        log = log.assign(event_duration=duration)
        widths["res_durations"] = resource_count
    widths["act_res_shares"] = len(log[[ACTIVITY, RESOURCE]].dropna().drop_duplicates())
    widths["handover_shares"] = len(handovers.drop_duplicates())
    remaining = MAX_FEATURE_CELLS // n_windows
    selected = set()
    for group, count in widths.items():
        if count <= remaining:
            selected.add(group)
            remaining -= count
    groups = [group for group in kairo.RESOURCE_GLOSSARY if group in selected]
    spec = {
        "groups": groups,
        "feature_window": feature_window,
        "resources": "all observed",
        "resource_count": int(resource_count),
        "missing_resource_rows": int(log[RESOURCE].isna().sum()),
        "event_duration_source": duration_source,
        "valid_duration_rows": int(duration.notna().sum())
        if duration is not None
        else 0,
        "omitted_feature_groups": sorted(set(kairo.RESOURCE_GLOSSARY) - selected),
    }
    if not groups:
        raise ValueError(
            "Resource feature matrices exceed the benchmark's dense feature budget"
        )
    return kairo.compute_features_resource(
        log, feature_window=feature_window, features=groups
    ), spec


def _inter_features(
    log: pd.DataFrame, window: pd.Timedelta
) -> tuple[pd.DataFrame, dict]:
    case_attributes = {
        column: CASE_ATTRIBUTES[column.lower()]
        for column in log
        if column.lower() in CASE_ATTRIBUTES and log[column].nunique() > 1
    }
    attribute_groups = {
        "attr_means": "numerical",
        "attr_stds": "numerical",
        "attr_shares": "categorical",
    }
    groups = [
        group
        for group in kairo.INTER_GLOSSARY
        if group not in attribute_groups
        or attribute_groups[group] in case_attributes.values()
    ]
    spec = {
        "groups": groups,
        "feature_window": _frequency(window),
        "case_attributes": case_attributes,
        "stall_threshold": "1D",
        "omitted_feature_groups": sorted(set(kairo.INTER_GLOSSARY) - set(groups)),
    }
    return kairo.compute_features_inter(
        log,
        feature_window=spec["feature_window"],
        features=groups,
        case_attributes=case_attributes,
        stall_threshold=spec["stall_threshold"],
    ), spec


def _reduce(features: pd.DataFrame, perspective: str) -> tuple[pd.DataFrame, dict]:
    scaling = "minmax" if perspective == "intra" else "zscore"
    scaled = kairo.standardize(features, scaling)
    numeric = scaled.drop(columns=META, errors="ignore")
    cut = min(5, *numeric.shape)
    if cut < 1:
        raise ValueError("Kairo did not return a nonempty feature matrix")
    spec = {
        "scaling": scaling,
        "feature_columns": numeric.shape[1],
        "feature_rows": len(numeric),
        "pca_components": None,
    }
    if len(numeric) == 1 or (numeric.nunique() <= 1).all():
        return scaled, spec
    # Kairo's sklearn PCA uses the NumPy global state for its randomized solver.
    state = np.random.get_state()
    try:
        np.random.seed(0)
        pca = kairo.compute_pca(scaled, cut)
    finally:
        np.random.set_state(state)
    spec.update(
        pca_components=cut,
        pca_seed=0,
        pca_retained_variance=float(pca.explained_variance_ratio_.sum()),
    )
    return kairo.apply_pca(scaled, pca, cut), spec


def _dbscan_settings(rows: np.ndarray, counts: np.ndarray) -> dict:
    min_samples = 5
    n_neighbors = min(min_samples, len(rows))
    # Match Kairo's treatment of identical rows by finding the distance to
    # min_samples total weight among the nearest distinct rows.
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=1).fit(rows)
    sample = rows[np.linspace(0, len(rows) - 1, min(len(rows), 2048), dtype=int)]
    distances, indices = neighbors.kneighbors(sample)
    cumulative_counts = counts[indices].cumsum(axis=1)
    positions = (cumulative_counts < min_samples).sum(axis=1).clip(max=n_neighbors - 1)
    kth_distances = distances[np.arange(len(sample)), positions]
    quantile = 0.80
    eps = max(1e-8, float(np.nextafter(np.quantile(kth_distances, quantile), np.inf)))
    return {
        "eps": eps,
        "min_samples": min_samples,
        "distance": "euclidean",
        "eps_k_distance_quantile": quantile,
        "eps_k_distance_sample_rows": len(sample),
    }


def _cluster_settings(
    perspective: str, method: str, rows: np.ndarray, counts: np.ndarray
) -> dict:
    if method == "kmeans":
        return {"k": min(5, len(rows)), "random_seed": 0}
    if method == "som":
        size = [5, 5] if perspective == "intra" else [3, 3]
        return {
            "size": size,
            "learning_rate": 0.5,
            "distance": "euclidean",
            "random_seed": 0,
        }
    return _dbscan_settings(rows, counts)


def _cluster(reduced: pd.DataFrame, method: str, spec: dict) -> pd.DataFrame:
    if method == "kmeans":
        model = kairo.compute_kmeans(reduced, k=spec["k"])
        labels = kairo.get_kmeans_states(reduced, model)
    elif method == "som":
        model = kairo.compute_som(
            reduced,
            size=tuple(spec["size"]),
            learning_rate=spec["learning_rate"],
            distance=spec["distance"],
        )
        labels = kairo.get_som_states(reduced, model)
    else:
        model = kairo.compute_dbscan(
            reduced,
            eps=spec["eps"],
            min_samples=spec["min_samples"],
            distance=spec["distance"],
        )
        labels = kairo.get_dbscan_states(reduced, model)
    return labels


def _signal_spec(base_window: pd.Timedelta, profile: dict) -> dict:
    return {
        "window": _frequency(base_window * profile["window_factor"]),
        "divergence": profile["divergence"],
        "reference": profile["reference"],
        "lookback": profile.get("lookback"),
        "threshold_quantile": profile["quantile"],
        "min_rows_as_fraction_of_median": 0.25,
    }


def _detect(
    states: pd.DataFrame,
    signal_spec: dict,
    beginning: pd.Timestamp,
    ending: pd.Timestamp,
) -> dict:
    window = pd.Timedelta(signal_spec["window"])
    distributions = kairo.compute_state_distributions(states, signal_spec["window"])
    signal = kairo.compute_drift_signal(
        distributions,
        divergence=signal_spec["divergence"],
        reference=signal_spec["reference"],
        lookback=signal_spec["lookback"] or 5,
    )
    rows = kairo.compute_signal_window_rows(states, signal_spec["window"])["rows"]
    median_rows = float(rows.median())
    available = ending.isoformat()
    observations, changes = [], []
    eligible_scores = []
    for i, ((start, score), count) in enumerate(
        zip(signal.itertuples(index=False, name=None), rows)
    ):
        end = start + window
        covered = (
            max(pd.Timedelta(0), min(end, ending) - max(start, beginning)) / window
        )
        eligible = (
            covered == 1
            and count >= signal_spec["min_rows_as_fraction_of_median"] * median_rows
        )
        eligible = bool(eligible and np.isfinite(score))
        observation = {
            "index": i,
            "timestamp": start.isoformat(),
            "available_at": available,
            "score": float(score) if np.isfinite(score) else None,
            "details": {
                "window_end": end.isoformat(),
                "rows": int(count),
                "coverage": float(covered),
                "eligible": eligible,
            },
        }
        observations.append(observation)
        if eligible:
            eligible_scores.append(score)

    threshold = None
    if eligible_scores:
        threshold = max(
            POSITIVE_SCORE,
            kairo.compute_drift_signal_quantile(
                pd.DataFrame({"drift_score": eligible_scores}),
                signal_spec["threshold_quantile"],
            ),
        )
        active_end = None
        for observation in observations:
            details = observation["details"]
            if details["eligible"] and observation["score"] >= threshold:
                if observation["timestamp"] != active_end:
                    changes.append(observation)
                active_end = details["window_end"]
            else:
                active_end = None

    return {
        "changes": changes,
        "scores": observations,
        "metadata": {
            "threshold": threshold,
            "median_rows": median_rows,
            "eligible_windows": len(eligible_scores),
        },
    }


def run(log: pd.DataFrame) -> Iterator[tuple[str, str, dict]]:
    """Emit one run per perspective, clustering algorithm and detection profile."""
    log = log.sort_values(TIME, kind="stable").reset_index(drop=True)
    base, resource_window, inter_window = _windows(log)
    beginning, ending = log[TIME].iloc[[0, -1]]
    for perspective in PERSPECTIVES:
        feature_error = None
        feature_spec, reduction_spec = {}, {}
        try:
            if perspective == "intra":
                features, feature_spec = _intra_features(log)
            elif perspective == "resource":
                if RESOURCE not in log or not log[RESOURCE].notna().any():
                    raise ValueError("The event log has no observed resources")
                features, feature_spec = _resource_features(log, resource_window)
            else:
                features, feature_spec = _inter_features(log, inter_window)
            reduced, reduction_spec = _reduce(features, perspective)
            del features
            rows, counts = np.unique(
                reduced.drop(columns=META, errors="ignore").to_numpy(),
                axis=0,
                return_counts=True,
            )
        except ValueError as error:
            feature_error = str(error)

        for method in METHODS:
            reason = feature_error
            clustering_spec, info = {}, {}
            if (
                reason is None
                and method == "dbscan"
                and len(rows) > MAX_DBSCAN_DISTINCT_ROWS
            ):
                reason = f"{len(rows)} distinct rows exceed the DBSCAN cap of {MAX_DBSCAN_DISTINCT_ROWS}"
            if reason is None:
                try:
                    clustering_spec = _cluster_settings(
                        perspective, method, rows, counts
                    )
                    labels = _cluster(reduced, method, clustering_spec)
                    meta = [column for column in META if column in reduced]
                    states = reduced[meta].join(labels)
                    info = {
                        "occupied_states": len(labels.drop_duplicates()),
                        "distinct_feature_rows": len(rows),
                    }
                    if method == "dbscan":
                        info["noise_rows"] = int(labels["cluster"].eq(-1).sum())
                except ValueError as error:
                    reason = str(error)
            for profile_name, profile in PROFILES.items():
                signal_spec = _signal_spec(base, profile)
                configuration = {
                    "perspective": perspective,
                    "features": feature_spec,
                    "reduction": reduction_spec,
                    "clustering": {"method": method, **clustering_spec},
                    "signal": signal_spec,
                    "fit_scope": "full event log",
                }
                payload = {
                    "configuration": configuration,
                    "observation_unit": "signal_window",
                }
                if reason is None:
                    detected = _detect(states, signal_spec, beginning, ending)
                    detected["metadata"].update(info)
                    payload.update(detected)
                else:
                    payload.update(
                        status="unsupported", reason=reason, changes=[], scores=[]
                    )
                yield f"{perspective}-{method}", profile_name, payload
