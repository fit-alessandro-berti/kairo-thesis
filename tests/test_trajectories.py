import pandas as pd
import pytest
from numpy.testing import assert_array_equal

import kairo
from oracles import CASE, TIME


@pytest.fixture(params=["som", "kmeans", "dbscan"])
def state_events(request):
    method = request.param
    frame = pd.DataFrame({CASE: ["b", "a", "a", "a", "b", "a", "a"],
                          TIME: pd.to_datetime(["2025-01-01 01:00", "2025-01-01 00:00", "2025-01-01 01:00",
                                                "2025-01-01 02:00", "2025-01-01 04:00", "2025-01-01 03:00",
                                                "2025-01-01 05:00"], utc=True)})
    labels = [0, 0, 0, 1, 1, 1, 0]
    if method == "som":
        frame["i"], frame["j"] = labels, 0
        names = ["(0, 0)", "(1, 0)"]
    else:
        frame["cluster"] = labels
        names = ["0", "1"]
    return method, frame, names


def test_case_visits_and_frequency_conservation(state_events):
    method, frame, names = state_events
    visits = getattr(kairo, f"get_{method}_case_trajectory")(frame, "a")
    assert visits.state.tolist() == [names[0], names[1], names[0]]
    assert visits.events.tolist() == [2, 2, 1]
    assert visits.duration.tolist() == [pd.Timedelta(hours=2), pd.Timedelta(hours=3), pd.Timedelta(0)]
    assert visits.start.tolist() == list(pd.to_datetime(["2025-01-01 00:00", "2025-01-01 02:00", "2025-01-01 05:00"], utc=True))
    assert visits.end.tolist() == list(pd.to_datetime(["2025-01-01 02:00", "2025-01-01 05:00", "2025-01-01 05:00"], utc=True))
    all_visits = getattr(kairo, f"get_{method}_case_trajectories")(frame)
    assert all_visits.events.sum() == len(frame)
    frequencies = getattr(kairo, f"get_{method}_state_frequencies")(frame)
    assert frequencies.to_dict() == {names[0]: 4, names[1]: 3}
    assert frequencies.name == "events" and frequencies.index.name == "state"
    limited = getattr(kairo, f"get_{method}_state_frequencies")(frame, "2025-01-01 02:00", "2025-01-01 04:00")
    assert limited.sum() == 3 and limited[names[1]] == 3
    with pytest.raises(ValueError, match="No events found"):
        getattr(kairo, f"get_{method}_case_trajectory")(frame, "missing")


def test_whole_case_range_filter_limits_and_numeric_case_ids(state_events):
    method, frame, _ = state_events
    many = getattr(kairo, f"get_{method}_case_trajectories")
    assert set(many(frame, "2025-01-01 01:00", "2025-01-01 04:00").case) == {"b"}
    assert set(many(frame, max_cases=1).case) == {"a"}
    numeric = frame.assign(**{CASE: frame[CASE].map({"a": 1, "b": 2})})
    assert getattr(kairo, f"get_{method}_case_trajectory")(numeric, "1").events.sum() == 5


def test_log_visits_include_last_window_and_filter_range(state_events):
    method, _, names = state_events
    frame = pd.DataFrame({TIME: pd.date_range("2025-01-01", periods=5, freq="1D", tz="UTC")})
    labels = [0, 0, 1, 1, 0]
    if method == "som":
        frame["i"], frame["j"] = labels, 0
    else:
        frame["cluster"] = labels
    visits = getattr(kairo, f"get_{method}_log_trajectory")(frame.sample(frac=1, random_state=3))
    assert visits.state.tolist() == [names[0], names[1], names[0]]
    assert visits.feature_windows.tolist() == [2, 2, 1]
    assert_array_equal(visits.duration, pd.to_timedelta([2, 2, 1], unit="D"))
    assert visits.end.iloc[-1] == pd.Timestamp("2025-01-06", tz="UTC")
    limited = getattr(kairo, f"get_{method}_log_trajectory")(frame, "2025-01-02", "2025-01-03")
    assert limited.feature_windows.tolist() == [1, 1]
    assert limited.end.iloc[-1] == pd.Timestamp("2025-01-04", tz="UTC")
    assert getattr(kairo, f"get_{method}_state_frequencies")(frame).name == "feature windows"
    assert getattr(kairo, f"get_{method}_log_trajectory")(frame, "2026-01-01").empty


def test_noise_is_retained_in_dbscan_visits_and_distributions():
    frame = pd.DataFrame({CASE: ["a"] * 3, TIME: pd.date_range("2025-01-01", periods=3, freq="h"), "cluster": [0, -1, 0]})
    assert kairo.get_dbscan_case_trajectory(frame, "a").state.tolist() == ["0", "noise", "0"]
    assert kairo.get_dbscan_state_frequencies(frame).to_dict() == {"noise": 1, "0": 2}
    assert kairo.compute_state_distributions(frame, "1D").iloc[0]["noise"] == pytest.approx(1 / 3)
