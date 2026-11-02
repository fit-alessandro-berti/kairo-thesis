"""Deliberately simple Python-loop oracles, independent of pandas groupby/crosstab code."""
from collections import Counter, defaultdict
from statistics import mean, stdev

import pandas as pd

CASE, ACT, TIME, RES = "case:concept:name", "concept:name", "time:timestamp", "org:resource"


def ordered_records(log):
    # Python's sort is stable, including equal timestamps.
    return sorted(log.to_dict("records"), key=lambda row: row[TIME])


def window_start(time, width):
    epoch = pd.Timestamp("1970-01-01", tz=time.tz)
    return epoch + ((time - epoch) // width) * width


def intra_oracle(log, lag):
    cases = defaultdict(list)
    by_position = {}
    activities = sorted(log[ACT].unique())
    pairs = set()
    for position, row in sorted(enumerate(log.to_dict("records")), key=lambda item: item[1][TIME]):
        trace = cases[row[CASE]]
        if trace:
            pairs.add((trace[-1][ACT], row[ACT]))
        trace.append(row)
        by_position[position] = list(trace)
    result = []
    for position, row in enumerate(log.to_dict("records")):
        prefix = by_position[position]
        counts = Counter(event[ACT] for event in prefix)
        follows = Counter((a[ACT], b[ACT]) for a, b in zip(prefix, prefix[1:]))
        values = {CASE: row[CASE], TIME: row[TIME]}
        for activity in activities:
            values[f"act_freqs:{activity}"] = counts[activity] / len(prefix)
            values[f"act_set:{activity}"] = float(counts[activity] > 0)
            values[f"current_act:{activity}"] = float(row[ACT] == activity)
            for back in range(1, lag + 1):
                values[f"past_acts_{back}:{activity}"] = float(
                    len(prefix) > back and prefix[-back - 1][ACT] == activity)
        for a, b in sorted(pairs):
            values[f"df_counts:{a} -> {b}"] = float(follows[a, b])
        values["case_progress"] = len(prefix) / len(cases[row[CASE]])
        result.append(values)
    return pd.DataFrame(result, index=log.index)


def window_oracles(log, frequency="1D", threshold="1D", resources=None):
    width = pd.Timedelta(frequency)
    threshold = pd.Timedelta(threshold)
    rows = ordered_records(log)
    cases = defaultdict(list)
    predecessors = {}
    for i, row in enumerate(rows):
        predecessors[i] = cases[row[CASE]][-1] if cases[row[CASE]] else None
        cases[row[CASE]].append(row)
    resources = resources or sorted({row[RES] for row in rows})
    activity_pairs = sorted({(row[ACT], row[RES]) for row in rows})
    handover_pairs = sorted({(predecessors[i][RES], row[RES]) for i, row in enumerate(rows)
                             if predecessors[i] and predecessors[i][RES] != row[RES]})
    starts = []
    start = window_start(rows[0][TIME], width)
    while start <= rows[-1][TIME]:
        starts.append(start)
        start += width
    resource_result, inter_result = [], []
    for start in starts:
        end = start + width
        in_window = [(i, row) for i, row in enumerate(rows) if start <= row[TIME] < end]
        events = [row for _, row in in_window]
        handovers = [(predecessors[i], row) for i, row in in_window
                     if predecessors[i] and predecessors[i][RES] != row[RES]]
        gaps = [(row[TIME] - predecessors[i][TIME]).total_seconds() / 60
                for i, row in in_window if predecessors[i]]
        resource = {TIME: start}
        for name in resources:
            own = [row for row in events if row[RES] == name]
            waits = [(after[TIME] - before[TIME]).total_seconds() / 60
                     for before, after in handovers if after[RES] == name]
            resource[f"res_events:{name}"] = len(own)
            resource[f"res_cases:{name}"] = len({row[CASE] for row in own})
            resource[f"res_durations:{name}"] = mean(row["event_duration"] for row in own) if own else 0
            resource[f"res_waits:{name}"] = mean(waits) if waits else 0
        for activity, name in activity_pairs:
            if name in resources:
                denominator = sum(row[ACT] == activity for row in events)
                numerator = sum(row[ACT] == activity and row[RES] == name for row in events)
                resource[f"act_res_shares:{activity} by {name}"] = numerator / denominator if denominator else 0
        for giver, receiver in handover_pairs:
            if giver in resources and receiver in resources:
                denominator = sum(before[RES] == giver for before, _ in handovers)
                numerator = sum(before[RES] == giver and after[RES] == receiver for before, after in handovers)
                resource[f"handover_shares:{giver} -> {receiver}"] = numerator / denominator if denominator else 0
        resource_result.append(resource)
        active = len({row[CASE] for row in events})
        stalled = 0
        for trace in cases.values():
            before = [row for row in trace if row[TIME] <= end]
            after = [row for row in trace if row[TIME] > end]
            if before and after and end - before[-1][TIME] > threshold:
                stalled += 1
        amounts = [float(row["amount"]) for row in events]
        inter = {
            TIME: start,
            "active_cases": active,
            "new_arrivals": sum(start <= trace[0][TIME] < end for trace in cases.values()),
            "completions": sum(start <= trace[-1][TIME] < end for trace in cases.values()),
            "events_per_case": len(events) / active if active else 0,
            "mean_delta_t": mean(gaps) if gaps else 0,
            "std_delta_t": stdev(gaps) if len(gaps) > 1 else 0,
            "stalled_cases": stalled,
            "attr_means:amount": mean(amounts) if amounts else 0,
            "attr_stds:amount": stdev(amounts) if len(amounts) > 1 else 0,
        }
        for tier in sorted({row["tier"] for row in rows}):
            inter[f"attr_shares:tier={tier}"] = sum(row["tier"] == tier for row in events) / len(events) if events else 0
        inter_result.append(inter)
    return pd.DataFrame(resource_result), pd.DataFrame(inter_result)
