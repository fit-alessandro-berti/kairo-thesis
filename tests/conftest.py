"""Small, deterministic logs whose expected values can be calculated independently."""
import numpy as np
import pandas as pd
import pytest
from hypothesis import settings

settings.register_profile("audit", max_examples=60, derandomize=True, deadline=None)
settings.load_profile("audit")


@pytest.fixture
def event_log():
    rows = [
        ("a", "A", "2025-01-01 01:00", "r1", 10, 10, "gold"),
        ("b", "B", "2025-01-01 02:00", "r2", 5, 30, "silver"),
        ("a", "B", "2025-01-01 03:00", "r2", 20, 10, "gold"),
        ("b", "A", "2025-01-01 04:00", "r3", 15, 30, "silver"),
        ("a", "A", "2025-01-03 01:00", "r2", 30, 10, "gold"),
        ("c", "A", "2025-01-03 02:00", "r1", 25, 50, "gold"),
        ("c", "C", "2025-01-03 03:00", "r3", 35, 50, "gold"),
        ("a", "C", "2025-01-05 00:00", "r1", 40, 10, "gold"),
    ]
    log = pd.DataFrame(rows, columns=["case:concept:name", "concept:name", "time:timestamp",
                                      "org:resource", "event_duration", "amount", "tier"])
    log["time:timestamp"] = pd.to_datetime(log["time:timestamp"], utc=True)
    log.index = pd.Index([11, 23, 35, 47, 59, 61, 73, 89], name="event")
    return log


@pytest.fixture
def numeric_features():
    rng = np.random.default_rng(2026)
    values = rng.normal(size=(36, 5))
    values[18:] += 5
    frame = pd.DataFrame(values, columns=list("abcde"), index=np.arange(100, 136))
    frame.insert(0, "time:timestamp", pd.date_range("2025-01-01", periods=len(frame), freq="h", tz="UTC"))
    frame.insert(0, "case:concept:name", [str(i // 4) for i in range(len(frame))])
    return frame
