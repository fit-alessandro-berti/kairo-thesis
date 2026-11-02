"""Desired-behavior reproductions. Findings are explained in COMPLETE_REPORT.md."""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.cluster import DBSCAN

import kairo
from oracles import ACT, CASE, RES, TIME


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="KAIRO-001: missing resources inflate activity shares")
def test_missing_resource_stays_in_activity_share_denominator():
    log = pd.DataFrame({CASE: ["a", "b"], ACT: ["A", "A"],
                        TIME: pd.to_datetime(["2025-01-01", "2025-01-01"], utc=True), RES: ["r1", None]})
    result = kairo.compute_features_resource(log, features=["act_res_shares"])
    assert result["act_res_shares:A by r1"].iloc[0] == pytest.approx(.5)


@pytest.mark.xfail(strict=True, raises=ValueError, reason="KAIRO-002: shuffled duplicate indices cannot be restored")
def test_duplicate_indices_preserve_each_event(event_log):
    log = event_log.sample(frac=1, random_state=13)
    log.index = [0, 0, 1, 1, 2, 2, 3, 3]
    expected = kairo.compute_features_intra(log.reset_index(drop=True), sliding_window_size=1)
    actual = kairo.compute_features_intra(log, sliding_window_size=1)
    assert len(actual) == len(log)
    assert_array_equal(actual.drop(columns=[CASE, TIME]), expected.drop(columns=[CASE, TIME]))
    assert actual.index.equals(log.index)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="KAIRO-003: delimiter collisions merge distinct transitions")
def test_directly_follows_names_do_not_merge_distinct_activity_pairs():
    log = pd.DataFrame({CASE: ["a", "a", "b", "b"], ACT: ["A -> B", "C", "A", "B -> C"],
                        TIME: pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")})
    result = kairo.compute_features_intra(log, features=["df_counts"])
    assert result.drop(columns=[CASE, TIME]).shape[1] == 2


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="KAIRO-004: cosine DBSCAN merges duplicate zero vectors")
def test_cosine_dbscan_deduplication_preserves_zero_vector_noise():
    features = pd.DataFrame([[0., 0.], [0., 0.], [0., 0.], [1., 0.], [1., 0.], [1., 0.]], columns=["x", "y"])
    reference = DBSCAN(eps=.1, min_samples=3, metric="cosine").fit_predict(features)
    actual = kairo.get_dbscan_states(features, kairo.compute_dbscan(features, .1, 3, "cosine")).cluster.to_numpy()
    assert_array_equal(actual == -1, reference == -1)


@pytest.mark.xfail(strict=True, raises=IndexError, reason="KAIRO-005: single-cluster abstraction has no other state")
def test_single_cluster_state_abstraction():
    frequencies = pd.Series([3], index=["0"], name="events")
    distances = pd.DataFrame([[0.]], index=["0"], columns=["0"])
    text = kairo.abstract_states(frequencies, distances, {0: "#888888"})
    assert "3 events" in text


@pytest.mark.parametrize("method", ["som", "kmeans", "dbscan"])
@pytest.mark.xfail(strict=True, raises=pytest.fail.Exception, reason="KAIRO-006: one feature window yields NaT duration")
def test_single_window_trajectory_rejects_uninferable_duration(method):
    frame = pd.DataFrame({TIME: [pd.Timestamp("2025-01-01", tz="UTC")]})
    if method == "som":
        frame["i"], frame["j"] = 0, 0
    else:
        frame["cluster"] = 0
    # With no window-width metadata, silently returning NaT is misleading.
    # A future API accepting an explicit width could return a finite duration.
    with pytest.raises(ValueError, match="[Ww]indow|[Dd]uration|[Tt]wo"):
        getattr(kairo, f"get_{method}_log_trajectory")(frame)


@pytest.mark.parametrize("cut", [0, -1, 4])
@pytest.mark.xfail(strict=True, raises=pytest.fail.Exception, reason="KAIRO-007: invalid PCA cuts are silently accepted")
def test_invalid_pca_cut_rejected(numeric_features, cut):
    model = kairo.compute_pca(numeric_features, 3)
    with pytest.raises(ValueError):
        kairo.apply_pca(numeric_features, model, cut)


@pytest.mark.xfail(strict=True, raises=pytest.fail.Exception, reason="KAIRO-008: zero lookback silently returns NaN scores")
def test_recent_reference_requires_positive_lookback():
    with pytest.raises(ValueError):
        kairo.compute_drift_signal(pd.DataFrame([[.2, .8], [.6, .4]]), reference="recent", lookback=0)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="KAIRO-009: initial distribution described with nan% changes")
def test_distribution_abstraction_does_not_invent_initial_change():
    distributions = pd.DataFrame([[1., 0.], [0., 1.]], columns=["0", "1"],
                                 index=pd.date_range("2025-01-01", periods=2, freq="D"))
    text = kairo.abstract_state_distributions(distributions)
    assert "nan%" not in text


@pytest.mark.parametrize("method", ["som", "kmeans", "dbscan"])
@pytest.mark.xfail(strict=True, raises=TypeError, reason="KAIRO-010: no matching cases crashes trajectory construction")
def test_case_range_without_matching_cases_returns_empty(method):
    frame = pd.DataFrame({CASE: ["a", "a"], TIME: pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")})
    if method == "som":
        frame["i"], frame["j"] = 0, 0
    else:
        frame["cluster"] = 0
    assert getattr(kairo, f"get_{method}_case_trajectories")(frame, start_date="2026-01-01").empty
