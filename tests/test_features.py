import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st
from pandas.testing import assert_frame_equal

import kairo
from oracles import ACT, CASE, RES, TIME, intra_oracle, window_oracles


def same_values(actual, expected):
    assert set(actual.columns) == set(expected.columns)
    assert_frame_equal(actual.sort_index(axis=1), expected.sort_index(axis=1),
                       check_dtype=False, check_names=False, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("lag", [0, 1, 3, 10])
def test_intra_every_feature_against_prefix_enumeration(event_log, lag):
    shuffled = event_log.sample(frac=1, random_state=7)
    original = shuffled.copy(deep=True)
    same_values(kairo.compute_features_intra(shuffled, sliding_window_size=lag), intra_oracle(shuffled, lag))
    assert_frame_equal(shuffled, original)


@pytest.mark.parametrize("frequency", ["1h", "12h", "1D", "3D", "7D"])
@pytest.mark.parametrize("threshold", ["1h", "1D", "3D"])
def test_resource_and_inter_all_features_against_window_enumeration(event_log, frequency, threshold):
    shuffled = event_log.sample(frac=1, random_state=13)
    original = shuffled.copy(deep=True)
    resource, inter = window_oracles(shuffled, frequency, threshold)
    same_values(kairo.compute_features_resource(shuffled, frequency), resource)
    same_values(kairo.compute_features_inter(shuffled, frequency,
                case_attributes={"amount": "numerical", "tier": "categorical"}, stall_threshold=threshold), inter)
    assert_frame_equal(shuffled, original)


@pytest.mark.parametrize("resources", [["r1"], ["r2", "r3"], ["r3", "r1", "r2"]])
def test_resource_selection_preserves_full_population_denominators(event_log, resources):
    expected, _ = window_oracles(event_log, resources=resources)
    same_values(kairo.compute_features_resource(event_log, resources=resources), expected)


@pytest.mark.parametrize("perspective", ["intra", "resource", "inter"])
def test_each_feature_group_individually_matches_combined_result(event_log, perspective):
    function = getattr(kairo, f"compute_features_{perspective}")
    glossary = getattr(kairo, f"{perspective.upper()}_GLOSSARY")
    kwargs = {"sliding_window_size": 2} if perspective == "intra" else (
        {"case_attributes": {"amount": "numerical", "tier": "categorical"}} if perspective == "inter" else {})
    all_features = function(event_log, **kwargs)
    for group in glossary:
        one = function(event_log, features=[group], **kwargs)
        assert len(one.columns) > (2 if perspective == "intra" else 1)
        assert_frame_equal(one, all_features[one.columns], check_names=False)
    with pytest.raises(ValueError, match="[Nn]o (known feature|feature)"):
        function(event_log, features=["unknown"], **kwargs)


def test_declared_resource_and_attribute_errors(event_log):
    with pytest.raises(ValueError, match="resource column"):
        kairo.compute_features_resource(event_log.drop(columns=RES))
    with pytest.raises(ValueError, match="duration column"):
        kairo.compute_features_resource(event_log.drop(columns="event_duration"))
    with pytest.raises(ValueError, match="not found"):
        kairo.compute_features_resource(event_log, resources=["absent"])
    for attributes, message in [({"absent": "numerical"}, "not found"),
                                ({"amount": "invalid"}, "pick numerical or categorical"),
                                ({"tier": "numerical"}, "holds no numbers")]:
        with pytest.raises(ValueError, match=message):
            kairo.compute_features_inter(event_log, case_attributes=attributes)
    with pytest.raises(ValueError, match="No feature computed"):
        kairo.compute_features_inter(event_log, features=["attr_means"])


def test_numeric_strings_and_missing_optional_columns(event_log):
    converted = event_log.assign(amount=event_log.amount.astype(str), event_duration=event_log.event_duration.astype(str))
    assert_frame_equal(kairo.compute_features_resource(converted), kairo.compute_features_resource(event_log))
    assert_frame_equal(kairo.compute_features_inter(converted, case_attributes={"amount": "numerical"}),
                       kairo.compute_features_inter(event_log, case_attributes={"amount": "numerical"}))
    result = kairo.compute_features_resource(event_log.drop(columns="event_duration"), features=["res_events"])
    assert result.filter(like="res_events:").to_numpy().sum() == len(event_log)


def test_stalls_strict_threshold_and_resume_boundaries():
    log = pd.DataFrame({CASE: ["a"] * 3, ACT: ["A"] * 3,
                        TIME: pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-06"], utc=True)})
    result = kairo.compute_features_inter(log, features=["stalled_cases"], stall_threshold="1D")
    # Exactly one day is not a stall; an event at the window end resumes the case.
    assert result.stalled_cases.tolist() == [0, 0, 0, 1, 0, 0]


@given(st.lists(st.tuples(st.integers(0, 4), st.integers(0, 3), st.integers(0, 200)), min_size=1, max_size=35))
def test_prefix_properties_on_generated_interleaved_logs(events):
    log = pd.DataFrame([{CASE: f"case-{case}", ACT: f"activity-{activity}",
                         TIME: pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(minutes=minute)}
                        for case, activity, minute in events])
    log.index = np.arange(len(log)) * 3 + 10
    same_values(kairo.compute_features_intra(log, sliding_window_size=2), intra_oracle(log, 2))


@given(st.lists(st.tuples(st.integers(0, 3), st.integers(0, 2), st.integers(0, 80), st.integers(0, 2)),
                min_size=2, max_size=30))
def test_window_properties_on_generated_logs(events):
    log = pd.DataFrame([{CASE: f"c{case}", ACT: f"a{activity}", RES: f"r{resource}",
                         TIME: pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(hours=hour),
                         "event_duration": hour + 1, "amount": case * 10, "tier": f"t{case % 2}"}
                        for case, activity, hour, resource in events])
    resource, inter = window_oracles(log)
    same_values(kairo.compute_features_resource(log), resource)
    same_values(kairo.compute_features_inter(log, case_attributes={"amount": "numerical", "tier": "categorical"}), inter)
