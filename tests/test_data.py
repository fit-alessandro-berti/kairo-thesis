import pandas as pd
import pm4py
import pytest
from pandas.testing import assert_frame_equal

import kairo
from oracles import ACT, CASE, RES, TIME


def test_csv_all_column_mappings_and_documented_text_timestamps(event_log, tmp_path):
    source = event_log.assign(event_id=range(len(event_log)), start_timestamp=event_log[TIME] - pd.Timedelta(minutes=2))
    names = {CASE: "case", ACT: "activity", TIME: "when", RES: "worker", "event_duration": "minutes",
             "event_id": "id", "start_timestamp": "start"}
    path = tmp_path / "events.CSV"
    source.rename(columns=names).to_csv(path, index=False)
    actual = kairo.read_log(path, case_id="case", activity="activity", timestamp="when", event_id="id",
                            start_timestamp="start", resource="worker", event_duration="minutes")
    expected = pd.read_csv(path).rename(columns={v: k for k, v in names.items()})
    assert_frame_equal(actual, expected)
    assert isinstance(actual[TIME].iloc[0], str)
    actual[TIME] = pd.to_datetime(actual[TIME], utc=True)
    assert kairo.compute_log_stats(actual)["events"] == len(source)


def test_xes_roundtrip(event_log, tmp_path):
    path = tmp_path / "events.xes"
    pm4py.write_xes(event_log, str(path))
    actual = kairo.read_log(str(path), resource=RES, event_duration="event_duration")
    for case in event_log[CASE].unique():
        expected_rows = event_log[event_log[CASE] == case].sort_values(TIME)
        actual_rows = actual[actual[CASE] == case].sort_values(TIME)
        assert actual_rows[ACT].tolist() == expected_rows[ACT].tolist()
        assert actual_rows[RES].tolist() == expected_rows[RES].tolist()
        assert actual_rows[TIME].tolist() == expected_rows[TIME].tolist()
        assert actual_rows.event_duration.tolist() == expected_rows.event_duration.tolist()


@pytest.mark.parametrize("argument", ["case_id", "activity", "timestamp", "event_id", "start_timestamp", "resource", "event_duration"])
def test_missing_mapped_column_rejected(event_log, tmp_path, argument):
    path = tmp_path / "events.csv"
    event_log.to_csv(path, index=False)
    with pytest.raises(ValueError, match="not found"):
        kairo.read_log(path, **{argument: "absent"})


def test_unsupported_extension_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported file type"):
        kairo.read_log(tmp_path / "not-a-log.json")


def test_log_statistics_hand_calculated(event_log):
    original = event_log.copy(deep=True)
    actual = kairo.compute_log_stats(event_log.sample(frac=1, random_state=7))
    assert {key: actual[key] for key in ["cases", "events", "activities", "resources"]} == {
        "cases": 3, "events": 8, "activities": 3, "resources": 3}
    assert actual["span_minutes"] == 95 * 60
    assert actual["start"] == pd.Timestamp("2025-01-01 01:00", tz="UTC")
    assert actual["end"] == pd.Timestamp("2025-01-05", tz="UTC")
    for metric, expected in {"min": 1 / 24, "mean": 98 / 72, "median": 2 / 24, "max": 95 / 24}.items():
        assert actual[f"tpt_days_{metric}"] == pytest.approx(expected)
    for metric, expected in {"min": 2, "mean": 8 / 3, "median": 2, "max": 4}.items():
        assert actual[f"length_{metric}"] == pytest.approx(expected)
    assert actual["activity_counts"].to_dict() == {"A": 4, "B": 2, "C": 2}
    assert actual["resource_counts"].to_dict() == {"r1": 3, "r2": 3, "r3": 2}
    assert_frame_equal(event_log, original)
    minimal = kairo.compute_log_stats(event_log.drop(columns=RES))
    assert minimal["resources"] == 0 and minimal["resource_counts"].empty


def test_single_event_statistics(event_log):
    actual = kairo.compute_log_stats(event_log.iloc[:1])
    assert actual["cases"] == actual["events"] == actual["length_min"] == 1
    assert actual["span_minutes"] == actual["tpt_days_max"] == 0


def test_default_csv_mapping_without_optional_columns(event_log, tmp_path):
    source = event_log[[CASE, ACT, TIME]]
    path = tmp_path / "minimal.csv"
    source.to_csv(path, index=False)
    actual = kairo.read_log(path)
    assert actual.columns.tolist() == [CASE, ACT, TIME]
    assert len(actual) == len(source)
    actual[TIME] = pd.to_datetime(actual[TIME], utc=True)
    assert kairo.compute_log_stats(actual)["resources"] == 0
