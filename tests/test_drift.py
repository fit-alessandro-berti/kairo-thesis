from collections import Counter, defaultdict
import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st
from numpy.testing import assert_allclose

import kairo
from oracles import CASE, TIME, window_start


def divergence_oracle(p, q, method):
    if method == "kl":
        # Kairo documents an epsilon for absent states; compare positive entries
        # to ordinary KL and allow only the rounding introduced by 1e-9 smoothing.
        return sum(a * math.log(a / b) for a, b in zip(p, q))
    if method == "js":
        middle = [(a + b) / 2 for a, b in zip(p, q)]
        return (divergence_oracle(p, middle, "kl") + divergence_oracle(q, middle, "kl")) / 2
    if method == "tv":
        return sum(abs(a - b) for a, b in zip(p, q)) / 2
    return math.sqrt(sum((math.sqrt(a) - math.sqrt(b)) ** 2 for a, b in zip(p, q)) / 2)


@pytest.mark.parametrize("method", ["kl", "js", "tv", "hellinger"])
@pytest.mark.parametrize("reference", ["previous", "recent", "baseline"])
def test_drift_all_divergences_and_references_against_scalar_math(method, reference):
    rows = [[.7, .2, .1], [.1, .3, .6], [.2, .4, .4], [.8, .1, .1]]
    dates = pd.date_range("2025-01-01", periods=len(rows), freq="7D", tz="UTC")
    distributions = pd.DataFrame(rows, index=dates)
    actual = kairo.compute_drift_signal(distributions, method, reference, lookback=2)
    assert actual.signal_window.tolist() == list(dates)
    expected = []
    for i, row in enumerate(rows):
        history = rows if reference == "baseline" else rows[max(0, i - (2 if reference == "recent" else 1)):i]
        required = 2 if reference == "recent" else 1
        if len(history) < required:
            expected.append(float("nan"))
        else:
            against = [sum(values) / len(history) for values in zip(*history)]
            expected.append(divergence_oracle(row, against, method))
    assert_allclose(actual.drift_score, expected, rtol=1e-7, atol=1e-8, equal_nan=True)
    assert kairo.compute_drift_signal_mean(actual) == pytest.approx(np.nanmean(expected), abs=1e-8)
    for quantile in [0, .25, .5, .9, 1]:
        assert kairo.compute_drift_signal_quantile(actual, quantile) == pytest.approx(np.nanquantile(expected, quantile), abs=1e-8)


@pytest.mark.parametrize("method,expected", [("js", math.log(2)), ("tv", 1.), ("hellinger", 1.)])
def test_disjoint_and_identical_distributions(method, expected):
    frame = pd.DataFrame([[1., 0.], [1., 0.], [0., 1.]])
    actual = kairo.compute_drift_signal(frame, method)
    assert np.isnan(actual.drift_score.iloc[0])
    assert actual.drift_score.iloc[1] == pytest.approx(0)
    assert actual.drift_score.iloc[2] == pytest.approx(expected, abs=5e-8)
    kl = kairo.compute_drift_signal(frame, "kl")
    assert np.isfinite(kl.drift_score.iloc[2]) and kl.drift_score.iloc[2] > 20


@pytest.mark.parametrize("kind", ["som", "cluster"])
@pytest.mark.parametrize("frequency", ["1D", "3D", "7D"])
def test_state_distributions_counts_and_empty_window_omission(kind, frequency):
    times = pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-12", "2025-01-12"], utc=True)
    frame = pd.DataFrame({CASE: ["a"] * 5, TIME: times})
    if kind == "som":
        frame["i"], frame["j"] = [0, 1, 0, 1, 1], [0, 2, 0, 2, 2]
        labels = ["(0, 0)", "(1, 2)", "(0, 0)", "(1, 2)", "(1, 2)"]
    else:
        frame["cluster"] = [0, 1, -1, 1, 1]
        labels = ["0", "1", "noise", "1", "1"]
    counts = defaultdict(Counter)
    for time, label in zip(times, labels):
        counts[window_start(time, pd.Timedelta(frequency))][label] += 1
    actual = kairo.compute_state_distributions(frame, frequency)
    rows = kairo.compute_signal_window_rows(frame, frequency)
    assert list(actual.index) == sorted(counts)
    assert rows.signal_window.tolist() == list(actual.index)
    for i, (start, row) in enumerate(actual.iterrows()):
        total = sum(counts[start].values())
        assert rows.rows.iloc[i] == total
        for label, share in row.items():
            assert share == pytest.approx(counts[start][label] / total)
    assert_allclose(actual.sum(axis=1), 1)
    assert kairo.compute_signal_window_rows_mean(rows) == pytest.approx(len(frame) / len(rows))
    assert kairo.compute_signal_window_rows_quantile(rows, .5) == np.median(rows.rows)


def test_rejected_windows_quantiles_and_modes():
    frame = pd.DataFrame({TIME: pd.date_range("2025-01-01", periods=3, freq="3D"), "cluster": [0, 1, 0]})
    for function in [kairo.compute_state_distributions, kairo.compute_signal_window_rows]:
        with pytest.raises(ValueError, match="at least as long"):
            function(frame, "1D")
    for value in [-.1, 1.1, float("nan")]:
        with pytest.raises(ValueError, match="quantile"):
            kairo.compute_drift_signal_quantile(pd.DataFrame({"drift_score": [0, 1]}), value)
        with pytest.raises(ValueError, match="quantile"):
            kairo.compute_signal_window_rows_quantile(pd.DataFrame({"rows": [1, 2]}), value)
    with pytest.raises(ValueError, match="Unknown divergence"):
        kairo.compute_drift_signal(pd.DataFrame([[.5, .5]]), "wrong")
    with pytest.raises(ValueError, match="Unknown reference"):
        kairo.compute_drift_signal(pd.DataFrame([[.5, .5]]), reference="wrong")


@given(st.lists(st.integers(1, 100), min_size=6, max_size=6))
def test_bounded_divergences_and_symmetry(values):
    p, q = np.array(values[:3], dtype=float), np.array(values[3:], dtype=float)
    p, q = p / p.sum(), q / q.sum()
    for method, bound in [("js", math.log(2)), ("tv", 1), ("hellinger", 1)]:
        forward = kairo.compute_drift_signal(pd.DataFrame([p, q]), method).drift_score.iloc[1]
        backward = kairo.compute_drift_signal(pd.DataFrame([q, p]), method).drift_score.iloc[1]
        assert 0 <= forward <= bound + 1e-12
        assert forward == pytest.approx(backward)
        assert forward == pytest.approx(divergence_oracle(p, q, method), abs=1e-8)
