"""Common temporal matching for drift alarms from different observation axes."""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

TOLERANCES = (0.01, 0.025, 0.05)
PRIMARY_TOLERANCE = 0.025


def episodes(truth):
    """Group simultaneous annotations with identical bounds for untyped scoring."""
    by_interval = {}
    for item in truth:
        bounds = (pd.Timestamp(item["start"]), pd.Timestamp(item["end"]))
        if bounds not in by_interval:
            by_interval[bounds] = {
                "id": item["id"],
                "start": bounds[0],
                "end": bounds[1],
                "annotations": [],
            }
        by_interval[bounds]["annotations"].append(item["id"])
    return list(by_interval.values())


def match(changes, truth, tolerance):
    """Maximize one-to-one matches, then minimize total distance to truth intervals."""
    tolerance = pd.Timedelta(tolerance).total_seconds()
    matched = []
    if changes and truth:
        estimate = [pd.Timestamp(change["timestamp"]) for change in changes]
        bounds = [
            (pd.Timestamp(item["start"]), pd.Timestamp(item["end"])) for item in truth
        ]
        distances = np.array(
            [
                [
                    max(0, (start - time).total_seconds(), (time - end).total_seconds())
                    for time in estimate
                ]
                for start, end in bounds
            ]
        )

        # A dummy column represents an unmatched truth. Each feasible distance
        # costs at most one. The dummy penalty makes cardinality dominate distance.
        dummy = len(truth) + 1
        real_cost = np.where(
            distances <= tolerance, distances / max(tolerance, 1), dummy * 2
        )
        costs = np.column_stack([real_cost, np.full((len(truth), len(truth)), dummy)])
        rows, columns = linear_sum_assignment(costs)
        for row, column in zip(rows, columns):
            if column >= len(changes) or distances[row, column] > tolerance:
                continue
            available = changes[column].get("available_at")
            matched.append(
                {
                    "truth_id": truth[row]["id"],
                    "change_offset": int(column),
                    "estimate_minus_onset_seconds": (
                        estimate[column] - bounds[row][0]
                    ).total_seconds(),
                    "distance_to_truth_interval_seconds": float(distances[row, column]),
                    "availability_minus_onset_seconds": (
                        (pd.Timestamp(available) - bounds[row][0]).total_seconds()
                        if available is not None
                        else None
                    ),
                }
            )

    matched_truth = {pair["truth_id"] for pair in matched}
    matched_changes = {pair["change_offset"] for pair in matched}
    tp = len(matched)
    fp = len(changes) - tp
    fn = len(truth) - tp
    return {
        "tolerance_seconds": tolerance,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if truth else None,
        "matches": matched,
        "unmatched_truth": [
            item["id"] for item in truth if item["id"] not in matched_truth
        ],
        "unmatched_changes": [
            i for i in range(len(changes)) if i not in matched_changes
        ],
    }


def evaluate(changes, truth, span):
    """Apply the same timestamp tolerances to every method within a log."""
    for change in changes:
        if pd.isna(pd.Timestamp(change.get("timestamp"))):
            raise ValueError("Every detected change needs an estimated timestamp")
    return {
        "primary_tolerance_fraction": PRIMARY_TOLERANCE,
        "by_tolerance_fraction": {
            str(fraction): match(changes, truth, span * fraction)
            for fraction in TOLERANCES
        },
    }
