# Kairo library: functioning and correctness report

**Audit date:** 2026-11-02 (UTC)

**Library:** `kairo` 0.1.0

**Baseline commit:** `a897420b9f35afae09d71eeaf5fd4dfbe29b7b39`

**Audited library source tree:** `cbd6ebb74280dadac6899946f39bd2a2c9c05cc9`

## 1. Assessment

The ordinary analysis workflows work in the tested environment. Independent
checks agree with the documented feature definitions and numerical calculations,
and all nine combinations of three perspectives and three clustering methods
complete through plots and drift signals. A separate synthetic experiment locates
an abrupt process change on the correct day in all three perspectives.

**The library is not fully correct on all tested inputs. Ten confirmed findings
remain**, including incorrect results, crashes on valid boundary cases, and input
validation gaps. They are documented in section 5 and preserved as executable
regression cases. Library implementation files were not modified during this audit.

| Check | Observed result |
| --- | --- |
| Full source test suite | **128 passed, 16 expected failures**, no unexpected failures or skips |
| Tests against the built wheel | **128 passed, 16 expected failures**, same findings |
| Known-defect tests with expected-failure handling disabled | **16 failed**, confirming the reproductions still fail |
| Property-based checks | **300 passing generated examples** across five parametrized test instances |
| Public API exercised | **81 exported functions and `LLMConnector`**, including its call and streaming methods |
| Statement coverage | **953 / 953 = 100%** |
| Branch coverage | **239 / 242 = 98.76%** |
| Combined statement/branch coverage | **99.75%** |
| Wheel and source-distribution builds | Both successful |
| Installed dependency consistency | All 185 installed packages compatible |
| Actual image rendering | Chromium/Kaleido generated valid PNGs; both supported message image formats decoded successfully |

The 16 expected failures represent 10 findings because some affect all three
clustering families or several invalid parameter values. They are **failures of
desired behavior**, not successful correctness checks. They use strict `xfail`
markers with expected exception types; an unexpected pass fails the suite and
signals that the finding should be reviewed. Coverage includes these reproductions
and measures execution, not proof of correctness.

## 2. Scope, evidence, and environment

The audit covers the implementation in `kairo/kairo`, the public exports in its
`__init__.py`, and the contracts in [the API documentation](kairo/index.html) and
feature glossaries. It includes loading, statistics, every feature group, scaling,
PCA, SOM, k-means, DBSCAN, state frequencies/distances/trajectories, quality scores,
distributions, drift signals, plotting, text abstractions, and LLM request adapters.

The repository initially had a Docker/dashboard smoke check but no dedicated
library test suite. This change adds the tests under `tests/`, this report,
`pytest-cov` as a development dependency, pytest configuration, and ignores for
local test artifacts. The lockfile adds only `coverage` and `pytest-cov`; **no
previously locked package version changed**. Neither library source nor its public
API was changed to make tests pass.

### Environment used

| Component | Version / environment |
| --- | --- |
| Operating system | Linux x86_64, kernel `6.12.113+deb13-amd64`, glibc 2.41 |
| Python | CPython 3.12.14, project `.venv` |
| NumPy / pandas / SciPy | 2.4.4 / 3.0.2 / 1.17.1 |
| scikit-learn / MiniSom | 1.8.0 / 2.3.6 |
| PM4Py / r4pm | 2.7.22.3 / 0.6.2 |
| Plotly / Kaleido | 6.7.0 / 1.3.0 |
| Chromium | 156.0.8078.73, Debian 13 build |
| OpenAI / Anthropic SDKs | 3.16.2 / 1.7.0 |
| pytest / Hypothesis | 9.0.3 / 6.152.4 |
| pytest-cov / coverage.py | 7.1.0 / 7.17.0 |

The full source run took 29.61 seconds and the wheel run 18.41 seconds on this
machine. They overlapped, so these are observations, not comparative benchmarks.
The source coverage run emitted 38 warnings: one PM4Py notice about its r4pm I/O
backend and 37 pandas warnings about future DatetimeIndex concatenation sorting.
No warning was suppressed to obtain the results.

### Test inventory

| Test file | Passing cases | Expected failures | Main evidence |
| --- | ---: | ---: | --- |
| [test_data.py](tests/test_data.py) | 13 | 0 | CSV mappings, timestamp contract, XES round trip, hand-calculated statistics, optional columns, rejected mappings |
| [test_features.py](tests/test_features.py) | 30 | 0 | Independent prefix/window enumeration; every feature group; resource subsets; validation; generated logs |
| [test_models.py](tests/test_models.py) | 13 | 0 | Manual scaling, SVD/reconstruction, deterministic SOM/k-means, nearest-neuron checks, DBSCAN equivalence, clustering scores |
| [test_trajectories.py](tests/test_trajectories.py) | 10 | 0 | Exact visit boundaries, durations, conservation, whole-case selection, numeric IDs, noise, log-window endpoints |
| [test_drift.py](tests/test_drift.py) | 23 | 0 | Scalar divergence/reference calculations, normalization, count conservation, bounds, symmetry, quantiles, invalid modes |
| [test_llm.py](tests/test_llm.py) | 21 | 0 | Mocked provider adapters, streaming, fallback, request content, image encoding, token estimates, summary content |
| [test_plots_and_pipeline.py](tests/test_plots_and_pipeline.py) | 18 | 0 | Nine complete pipelines, plotted values, color invariants, real PNGs, three planted-change experiments |
| [test_known_defects.py](tests/test_known_defects.py) | 0 | 16 | Desired-behavior reproductions of the ten findings |
| **Total** | **128** | **16** | **144 collected cases** |

The 300 Hypothesis examples execute inside five of those cases; they are not 300
additional pytest cases. The profile is deterministic, uses 60 examples per
instance, and disables timing deadlines. Hypothesis also discarded 24 generated
examples during generation; none of the 300 executed examples failed. See
[conftest.py](tests/conftest.py) and the independent [oracles](tests/oracles.py).

## 3. What the passing tests establish

### Loading and statistics

- CSV and XES inputs load; all seven configurable role mappings are exercised.
- Missing mapped columns and unsupported extensions raise the documented errors.
- CSV timestamps deliberately remain text, as documented. The tests explicitly
  convert them before downstream analysis; that behavior is not reported as a bug.
- An eight-event, three-case fixture has independently checked counts, activity and
  resource frequencies, start/end, throughput-time statistics, and trace lengths.
  Its total span is 95 hours, case lengths are 4/2/2, and case throughput times are
  95/2/1 hours. A one-event log and absent optional resource columns also work.
- XES round trips preserve the tested per-case activities, resources, timestamps,
  and event durations.

### Features and time windows

The oracles use Python loops, counters, and scalar statistics rather than copying
the library's pandas groupby/crosstab implementation. Every output feature value
is compared, not merely its shape or absence of exceptions.

- Intra-case activity frequencies, directly-follows counts, activity sets, progress,
  current activity, and past-activity blocks agree with explicit prefix enumeration.
- Lag sizes 0/1/3/10, interleaved cases, shuffled input, stable equal-timestamp
  ordering, nonconsecutive unique indices, and input immutability are checked.
- Resource event/case counts, duration means, incoming waits, activity shares, and
  handover shares agree with enumerated receiving events and denominators.
- Selecting resources preserves denominators over all resources in the complete
  fixture; this does not cover the missing-resource defect in KAIRO-001.
- Inter-case active cases, arrivals, completions, events per case, gaps, sample
  standard deviations, stalls, and numerical/categorical attributes match explicit
  calculations. Numerical attributes are event-weighted, as the docs specify.
- Window lengths 1h/12h/1D/3D/7D and stall thresholds 1h/1D/3D are exercised,
  including empty windows and epoch-aligned multi-day boundaries.
- Stalls use a strict threshold and exclude cases resumed or completed at the
  window end. A dedicated boundary example expects `[0, 0, 0, 1, 0, 0]` stalls.
- Each feature group works separately and agrees with the corresponding combined
  values. Numeric strings and documented missing-column/attribute errors are checked.

### Scaling, PCA, clustering, and quality scores

- z-score and min-max values match manual formulas; constants become zero, excluded
  binary groups and metadata remain unchanged, and source frames are not mutated.
- PCA variances and reconstruction errors agree with an independent NumPy SVD.
  Reconstructing rows from each retained prefix gives the reported mean squared
  Euclidean reconstruction error; errors decrease as components are retained.
- SOM and k-means are repeatable for the tested data. SOM winners are nearest
  neurons for both supported distances, and quantization errors match those
  distances. k-means centers and inertia match member means and squared residuals.
- Optimized DBSCAN matches full-data sklearn DBSCAN on 120 generated nonzero-data
  cases, comparing noise and partitions independently of cluster label numbering.
  This equivalence fails for the zero-vector case described in KAIRO-004.
- DBSCAN distances use means of distinct core vectors, matching the documented
  convention; they are not occurrence-weighted cluster centroids.
- Noise is omitted from silhouette/Calinski–Harabasz scoring but retained in state
  frequencies, trajectories, and distributions. Hand calculations give silhouette
  `(19/21 + 17/19)/2` and Calinski–Harabasz `200` for a two-cluster example.
- Occupancy entropy has checked uniform, partially occupied, single-occupied, and
  one-neuron cases. Undefined quality-score cases return NaN as documented.

### Trajectories and drift

- Consecutive visits, event counts, state names, start/end times, and durations are
  checked exactly for all three clustering families. The final case visit ends at
  the final event; the final log visit includes its final feature window.
- Whole-case date filtering, earliest-case limits, numeric case IDs, inclusive
  frequency filtering, unknown-case errors, and empty log-window selections work.
  Empty *case* selections and a single available feature window have separate findings.
- State shares sum to one, counts are conserved, noise is retained, and windows
  without rows are omitted. Row-count and distribution windows agree.
- All four divergences and all three references are compared with scalar math.
  Previous/recent references correctly leave initial scores undefined; recent
  references use only preceding windows; baseline uses all supplied windows.
- Positive-probability KL/JS agree within the tested epsilon tolerance. The
  implementation adds `1e-9` to entries: disjoint-support KL is therefore finite
  and large, rather than the mathematical infinity of unsmoothed KL. This is an
  intentional approximation, not an audit finding.
- JS/TV/Hellinger symmetry and bounds, identical/disjoint distributions, means,
  quantiles, invalid quantiles, and too-short signal windows are tested.
- A 40-day, 320-event synthetic log changes activities, resources, durations, and
  event volume on **2025-01-21**. The feature → scaling → PCA → k-means → daily TV
  pipeline has its only positive peak on that date in each perspective. This
  establishes functioning on the planted abrupt example, not general detection
  accuracy or statistical significance on real logs.

### Plots, abstractions, and LLM adapters

- Every exported plotting function is exercised, its Plotly representation
  validates/serializes, and selected bar, heatmap, k-distance, signal, and threshold
  values are checked against their inputs.
- State colors are deterministic valid hexadecimal RGB strings in the tested
  cases, including one cluster, 12 clusters, and coincident SOM neurons.
- Real Chromium/Kaleido rendering produces nonblank PNGs of the requested size.
  Actual PNG attachments also decode from OpenAI-compatible and Anthropic messages.
- Text abstractions preserve checked counts, percentages, PCA cut information,
  nearest/farthest state distances, trajectories, and drift statistics. Long log
  trajectories exercise the 200-visit summary limit.
- Mocked SDK clients verify OpenAI, Google, custom-host, and Anthropic routing;
  request arguments; system-message handling; call/stream results; partial chunks;
  reported usage; usage-free fallback; and the custom-host retry path.
- Token tests validate the library's documented estimates for text and three image
  sizes. They do not establish exact provider/model-specific token billing.

## 4. Coverage and packaging

| Library module | Statements executed / total | Branches executed / total |
| --- | ---: | ---: |
| `kairo/__init__.py` | 8 / 8 | 0 / 0 |
| `analysis/__init__.py` | 260 / 260 | 50 / 50 |
| `analysis/intra.py` | 57 / 57 | 18 / 18 |
| `analysis/resource.py` | 71 / 71 | 20 / 20 |
| `analysis/inter.py` | 84 / 84 | 34 / 34 |
| `analysis/plots.py` | 184 / 184 | 25 / 26 |
| `data/__init__.py` | 41 / 41 | 14 / 14 |
| `llm/__init__.py` | 248 / 248 | 78 / 80 |
| **Total** | **953 / 953** | **239 / 242** |

The three unexecuted branches are: omitting cell labels in the large-grid SOM color
plot (`plots.py:129`); ignoring an unrecognized multimodal content block
(`llm/__init__.py:284`); and completing a state abstraction without supplied quality
scores (`llm/__init__.py:439`). No source lines were excluded from measurement.

The exported surface comprises 2 data functions, 36 general analysis functions,
3 perspective feature functions, 26 plotting/color functions, and 14 LLM utility/
abstraction functions plus `LLMConnector`. All are exercised. This does not imply
that every argument combination, dependency behavior, or numerical condition is covered.

Both `kairo-0.1.0-py3-none-any.whl` and `kairo-0.1.0.tar.gz` built successfully.
The wheel was installed into a separate temporary import directory without editable
source resolution; `kairo.__file__` was checked to point into that directory. The
entire suite then ran against the wheel with the same locked dependency environment
and produced the same 128 passes / 16 expected failures. This checks distribution
contents and imports; it is not a fresh independent resolution of all dependencies.

## 5. Confirmed findings

Severity describes observed impact within this library, not a security rating.
All findings remain open. The IDs appear directly in the strict expected-failure
tests and can be reproduced using section 6.

### KAIRO-001 — Missing resources inflate activity-resource shares

**Priority: high; numerical correctness.**

**Location:** [resource.py:124](kairo/kairo/analysis/resource.py#L124)

**Test:** `test_missing_resource_stays_in_activity_share_denominator`

For two same-window events of activity A, one assigned to r1 and one with a missing
resource, the documented denominator is both A events. Expected r1 share: **0.5**;
observed: **1.0**. The crosstab drops the missing-resource event before the totals
are calculated. This can distort resource features and downstream states.

**Suggested correction:** calculate activity totals from all events independently
of the resource crosstab, or explicitly validate/document a different missing-value
policy. Add cases with partially and entirely missing resources.

### KAIRO-002 — Shuffled duplicate indices break intra-case features

**Priority: medium; input/index support gap.**

**Location:** [intra.py:64](kairo/kairo/analysis/intra.py#L64)

**Test:** `test_duplicate_indices_preserve_each_event`

A shuffled valid event log with repeated DataFrame index labels raises
`ValueError: cannot reindex on an axis with duplicate labels`. Chronological sorting
is followed by label-based restoration of the original order. The public docs do
not state that index labels must be unique; concatenated logs commonly retain
repeated labels.

**Suggested correction:** restore order by a temporary unique positional identifier,
or validate and document uniqueness. A caller workaround is to reset the index
before computing features, preserving any external event identifier separately.

### KAIRO-003 — Activity-pair label collisions merge distinct transitions

**Priority: high; silent information loss.**

**Location:** [intra.py:79](kairo/kairo/analysis/intra.py#L79)

**Test:** `test_directly_follows_names_do_not_merge_distinct_activity_pairs`

The transitions `("A -> B", "C")` and `("A", "B -> C")` both become the string
`A -> B -> C`. Two distinct directly-follows pairs therefore produce **one** feature
column instead of two. These are valid distinct activity names, and the merge
silently changes the feature representation.

**Suggested correction:** use structured pair keys internally and collision-safe
encoding for display/column names. This reproduction covers intra-case pairs;
other string-composed feature names were not exhaustively collision-tested.

### KAIRO-004 — Cosine DBSCAN deduplication changes zero-vector clustering

**Priority: high; clustering equivalence.**

**Location:** [analysis/__init__.py:204](kairo/kairo/analysis/__init__.py#L204)

**Test:** `test_cosine_dbscan_deduplication_preserves_zero_vector_noise`

With three `[0, 0]` rows and three `[1, 0]` rows, `eps=0.1`, `min_samples=3`, and
cosine distance, full-data sklearn DBSCAN marks the zero rows as noise. Kairo marks
all six rows as clustered. Its deduplication combines the zero rows into one sample
with weight three, changing the neighborhood count under sklearn's zero-vector
cosine-distance convention.

Zero vectors can arise after scaling constant features or from idle windows. The
optimization's claimed equivalence does not hold for this supported metric/input
combination. This is specifically an equivalence finding; cosine similarity of a
zero vector has no unique mathematical definition independent of the convention.

**Suggested correction:** handle zero vectors without this coalescing optimization,
or reject/document them explicitly for cosine DBSCAN. Retain the nonzero-data
reference comparisons as well as this regression case.

### KAIRO-005 — A single-cluster state summary crashes

**Priority: medium; valid degenerate-model failure.**

**Location:** [llm/__init__.py:425](kairo/kairo/llm/__init__.py#L425)

**Test:** `test_single_cluster_state_abstraction`

Summarizing one state, its frequency, and a 1×1 zero-distance matrix raises
`IndexError`. The code removes the current state and then indexes the nonexistent
nearest/farthest other state. One cluster is a valid DBSCAN result and one neuron
is a valid SOM configuration.

**Suggested correction:** state that no other cluster exists and omit nearest/
farthest comparisons. The frequency and color information can still be reported.

### KAIRO-006 — Single-window log trajectories silently have NaT durations

**Priority: medium; duration information/validation gap.**

**Location:** [analysis/__init__.py:521](kairo/kairo/analysis/__init__.py#L521)

**Tests:** `test_single_window_trajectory_rejects_uninferable_duration[som|kmeans|dbscan]`

If the complete input contains only one feature window, its width cannot be inferred
from timestamp differences. All three log-trajectory getters return a visit whose
`end` and `duration` are `NaT`. The regression cases require an informative rejection
rather than this silent result with the current signature.

**Suggested correction:** retain/accept explicit feature-window width, or reject an
uninferable width clearly. Do not guess a duration. A filtered single window from a
larger regular input is different: the width can be inferred before filtering.

### KAIRO-007 — Invalid PCA cuts are silently accepted

**Priority: medium; invalid-input validation gap.**

**Location:** [analysis/__init__.py:54](kairo/kairo/analysis/__init__.py#L54)

**Tests:** `test_invalid_pca_cut_rejected[0|-1|4]`

For a PCA fitted with three components, the documented cut domain is 1–3. Cut 0
returns no principal components, cut -1 returns two, and cut 4 returns three instead
of rejecting the request. Python slicing determines these results silently.

**Suggested correction:** require an integer in `1..pca.n_components_`. This is a
validation finding for out-of-domain inputs, not an error in valid PCA arithmetic.

### KAIRO-008 — A zero recent-reference lookback silently produces NaNs

**Priority: medium; invalid-input validation gap.**

**Location:** [analysis/__init__.py:330](kairo/kairo/analysis/__init__.py#L330)

**Test:** `test_recent_reference_requires_positive_lookback`

The docs require lookback to be an integer from one. `reference="recent", lookback=0`
is accepted and returns all-NaN scores, indistinguishable to a caller from simply
having insufficient history.

**Suggested correction:** validate a positive integer when the recent reference is
selected. Valid positive lookbacks passed the independent numerical checks.

### KAIRO-009 — Distribution summaries describe an undefined initial change

**Priority: low; misleading text supplied to LLMs.**

**Location:** [llm/__init__.py:487](kairo/kairo/llm/__init__.py#L487)

**Test:** `test_distribution_abstraction_does_not_invent_initial_change`

For two distributions, the "largest changes" section includes the first window as
`0.0% of the mix moved, 0 +nan%, 1 +nan%`. There is no previous distribution for
that window. Summing the all-NaN difference produces zero, and it enters the top-five
list before its per-state NaNs are formatted.

**Suggested correction:** exclude rows with no preceding distribution from change
ranking, and describe a single-window input as having no measured transition.

### KAIRO-010 — Empty case selections crash trajectory construction

**Priority: medium; valid selection failure.**

**Location:** [analysis/__init__.py:501](kairo/kairo/analysis/__init__.py#L501)

**Tests:** `test_case_range_without_matching_cases_returns_empty[som|kmeans|dbscan]`

Selecting a date range with no complete cases raises
`TypeError: Cannot cast DatetimeArray to dtype float64` under the locked pandas 3.0.2
environment. The empty grouped result reaches datetime filling with incompatible
empty dtypes. All three public case-trajectory families share this path.

**Suggested correction:** return a correctly typed empty trajectory frame before
the fill operation. This is distinct from an unknown explicit case ID, for which
an informative `ValueError` is already documented and tested.

## 6. Reproduction

Run from the repository root. Python 3.12 is selected by `.python-version`; `uv sync`
installs the lockfile's development dependencies. The full rendering case requires
Chromium/Chrome and its system libraries; the audit used `/usr/bin/chromium`.

```bash
uv sync --locked
uv pip check --python .venv/bin/python
BROWSER_PATH=/usr/bin/chromium uv run --locked pytest -q -ra \
  --cov=kairo --cov-branch --cov-report=term-missing \
  --cov-report=json:/tmp/kairo-audit-coverage.json \
  --junitxml=/tmp/kairo-audit-results.xml --hypothesis-show-statistics
```

Expected audited summary: **128 passed, 16 xfailed**. On a machine without a browser,
`uv run --locked pytest -m 'not rendering'` runs the other checks, but does not
validate real image rendering and is not equivalent to the full run.

To expose the known issues as ordinary failures:

```bash
uv run --locked pytest tests/test_known_defects.py --runxfail -q --tb=short
```

Expected audited summary: **16 failed**, process exit status **1**. To reproduce one
finding, append `::test_name` to the test-file path (names are in section 5).
Do not remove `xfail` markers merely to hide results or convert the incorrect
behavior into a passing expectation.

Packaging verification performed:

```bash
uv build ./kairo --wheel --sdist --out-dir /tmp/kairo-library-audit-dist
uv pip install --python .venv/bin/python --no-deps --link-mode=copy \
  --target /tmp/kairo-wheel-audit-import \
  /tmp/kairo-library-audit-dist/kairo-0.1.0-py3-none-any.whl
PYTHONPATH=/tmp/kairo-wheel-audit-import uv run --locked python -c \
  'import kairo; print(kairo.__file__)'
PYTHONPATH=/tmp/kairo-wheel-audit-import BROWSER_PATH=/usr/bin/chromium \
  uv run --locked pytest -q --tb=short
```

Use a fresh target directory when repeating a wheel install. Verify that the printed
import path points into that target before relying on the wheel test result.

Audit execution artifacts were written to `/tmp/kairo-audit-full.log`,
`/tmp/kairo-audit-results.xml`, `/tmp/kairo-audit-coverage.json`,
`/tmp/kairo-audit-wheel.log`, `/tmp/kairo-audit-wheel-results.xml`, and
`/tmp/kairo-audit-defects.log`. They are local, regenerable artifacts; the durable
evidence committed here is the test suite and this report.

## 7. Limits and follow-up priorities

- Only the listed Linux/Python/dependency combination was executed. The package's
  full Python-version range, other operating systems, architectures, and alternate
  dependency versions were not certified.
- Naive and UTC timestamps are exercised. Daylight-saving transitions and arbitrary
  mixed/non-UTC time zones were not systematically tested.
- Provider calls use mocked SDK clients and synthetic responses. No paid API calls,
  authentication, live endpoint compatibility, provider token billing, or generated
  language-model answer quality was tested.
- The real rendering test checks encoding, dimensions, nonblank pixels, and message
  payloads. Plot trace values are tested separately; there was no exhaustive visual
  review of every chart, legend, or color-accessibility property.
- Generated logs and the planted-change experiment are small deterministic tests.
  Large real datasets, memory limits, long-running sessions, throughput, gradual/
  recurring drift detection accuracy, false-positive rates, and statistical
  calibration require separate evaluation.
- Tests enforce many documented errors and preserve the discovered validation
  gaps. They do not establish a universal contract for every malformed DataFrame,
  missing required value, empty log, invalid frequency, or numerical degeneracy.
- Passing model tests do not prove global optimization of clustering objectives or
  that a selected number of states is scientifically appropriate for a process.

Prioritize the silent feature/clustering errors **KAIRO-001, KAIRO-003, KAIRO-004**,
then the valid-input crashes **KAIRO-005 and KAIRO-010**. Address duration/index
handling and parameter validation next. After each fix, run its reproduction with
`--runxfail`, remove its expected-failure marker only when justified, and rerun the
full source and wheel suites. The current results support ordinary tested workflows
but do not justify treating the library as free of correctness defects.
