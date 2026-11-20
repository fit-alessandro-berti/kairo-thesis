"""Evaluate PM4Py's case attributes on the assessment's nine Rheon logs."""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .run import revision, sample_scores, write_json


def run_dataset(task):
    name, data_dir, output, checkout, pm_revision = task
    sys.path.insert(0, str(checkout))

    import pandas as pd
    import pm4py

    from . import datasets, evaluation, pm_methods

    if not Path(pm4py.__file__).resolve().is_relative_to(checkout):
        raise RuntimeError(
            f"PM4Py imported outside the selected checkout: {pm4py.__file__}"
        )
    [(info, log)] = list(datasets.load_all(data_dir, {name}))
    region = log.get("case:region", pd.Series(dtype=str)).dropna()
    amount = pd.to_numeric(
        log.get("case:amount", pd.Series(dtype=float)), errors="coerce"
    )
    region_values = sorted(region.unique().tolist())
    indicator_features = [("case_type", "region", value) for value in region_values]
    joint_features = [("case_category", "region")] if region_values else []
    if amount.notna().any():
        indicator_features.append(("case_value", "amount"))
        joint_features.append(("case_numeric", "amount"))
    if not joint_features:
        raise ValueError(f"No case-composition fields in {name}")

    baseline = {
        (method, profile): config
        for method, profile, config in pm_methods.configurations(log)
    }
    configurations = {
        "indicators": {
            **baseline["indicators", "pelt-basic"],
            "features": indicator_features,
        },
        "multivariate": {
            **baseline["multivariate", "rbf-flow-timing"],
            "features": joint_features,
        },
    }
    lower = log[pm_methods.TIME].min()
    if pm_methods.START in log and log[pm_methods.START].notna().any():
        lower = min(lower, log[pm_methods.START].min())
    upper = log[pm_methods.TIME].max()
    truth = evaluation.episodes(info["truth"])
    trace_log = pm_methods.prepare(log)
    findings = {}
    for method, config in configurations.items():
        result = pm_methods.run_one(log, method, config, trace_log)
        metrics = evaluation.evaluate(result["changes"], truth, upper - lower)
        sample_scores(result, full_scores=True)
        write_json(
            output / name / method / "case-composition.json",
            {
                "schema_version": 1,
                "dataset": name,
                "input": info["source"],
                "first_timestamp": lower,
                "last_timestamp": upper,
                "truth_annotations": info["truth"],
                "truth_episodes": truth,
                "library": "pm4py",
                "method": method,
                "profile": "case-composition",
                "detector_source_revision": pm_revision,
                "status": "ok",
                "configuration": config,
                "evaluation": metrics,
                **result,
            },
        )
        primary = metrics["by_tolerance_fraction"][str(evaluation.PRIMARY_TOLERANCE)]
        findings[method] = {key: primary[key] for key in ("tp", "fp", "fn")}
    return name, findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pm4py", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()

    from . import datasets

    checkout = args.pm4py.resolve()
    pm_revision = revision(checkout)
    tasks = [
        (name, args.data_dir, args.output, checkout, pm_revision)
        for name in datasets.RHEON
    ]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for name, findings in pool.map(run_dataset, tasks):
            print(name, findings, flush=True)


if __name__ == "__main__":
    main()
