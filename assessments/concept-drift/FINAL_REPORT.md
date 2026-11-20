# Final report on concept drift detection with Kairo and PM4Py

## Conclusion

Kairo provides a useful signal for investigating changes in process behavior. It identified resource and timing changes cleanly on selected Rheon logs and matched PM4Py version clustering on the three Padella control-flow logs. Across the full benchmark, its best profile matched more annotated shifts than the best PM4Py profile, and its frequent unmatched alarms made it less reliable as a general-purpose automatic detector under the settings tested.

## Scope and evaluation

The assessment covered 23 complete event logs, 1,086,668 events, and 37 annotated drift episodes. It included eight Padella logs, nine Rheon logs, and six historical Bose, Ceravolo, and Ostovar logs. Two Padella logs served as controls with no injected drift. All 18 PM4Py variants were evaluated alongside nine Kairo combinations of process perspective and clustering method. Of 1,030 result files, 992 record completed runs. The other 38 document unavailable Kairo configurations due to missing resource data or the benchmark's DBSCAN size limit. [Results](results/) · [Run manifest](results/manifest.json)

The benchmark chose settings without consulting the drift labels. For Kairo, it applied a quantile threshold to changes in state distributions and recorded one alarm at the start of each consecutive run of qualifying windows. PM4Py supplied its native alarms. The evaluation matched estimated alarm timestamps one-to-one with annotated points or intervals. It used a margin of 2.5% of each log's time span, with 1% and 5% as sensitivity checks. Each unmatched alarm counted as a false positive for the benchmark, including repeat alarms for the same episode and possible natural changes outside the annotations. [Method and reproduction instructions](README.md#method)

## Results

The two profiles below had the highest micro F1 within their respective libraries at the primary tolerance. Each ran on all 23 logs.

- **Kairo intra-case K-means with a previous-window reference** matched 21 of 37 episodes and raised 72 unmatched alarms. Precision was 22.6%, recall was 56.8%, and micro F1 was **0.323**.
- **PM4Py version clustering** matched 15 of 37 episodes and raised 10 unmatched alarms. Precision was 60.0%, recall was 40.5%, and micro F1 was **0.484**.

Micro F1 pools the per-log counts, including unmatched alarms from the two controls. Those controls accounted for nine Kairo and two PM4Py alarms from the two profiles above. PM4Py version clustering scored 0.419 at the 1% tolerance and 0.516 at 5%. The same Kairo K-means profile scored 0.231 and 0.400, respectively. Each profile's total number of alarms stayed fixed as the matching tolerance changed.

- **Control flow.** Kairo's recent-window intra-case SOM and PM4Py version clustering each matched all three Padella control-flow injections and raised two extra alarms. On the Rheon log with four sudden control-flow changes, Kairo's previous-window intra-case SOM matched all four and raised one extra alarm. PM4Py's wide trace-variant profile matched all four without an extra alarm. [Padella results](results/) · [Rheon control-flow results](results/rheon_multi_02/)
- **Resources and timing.** Kairo's recent-window resource profiles matched the Rheon sudden reassignment without extra alarms. Its recent-window inter-case SOM did the same for the Rheon gradual waiting-time shift. On the three Padella timing logs, Kairo's previous-window resource SOM and PM4Py's fixed Page–Hinkley conformance each matched two injections and raised nine extra alarms. [Rheon reassignment](results/rheon_single_11/) · [Rheon waiting time](results/rheon_single_15/)
- **Case composition.** Kairo's inter-case SOM matched all three Rheon region changes with two extra alarms using the previous-window profile. PM4Py PELT and multivariate MMD each matched all three without extra alarms when configured with explicit case attributes in the follow-up diagnostic. [Rheon region results](results/rheon_multi_03/kairo/inter-som/) · [PM4Py case-attribute diagnostic](diagnostics/case-composition/rheon_multi_03/)

## Implementation checks and limits

The assessment uncovered two errors in PM4Py's Bose detector. The fixes order cases chronologically and count outgoing eventually-follows relations as specified by Relation Type Count. On the historical Bose log, the relation-count fix kept three of four matches and reduced unmatched alarms from 17 to 12. The Kairo repository changes consist of the benchmark and its artifacts. [Bose case-order diagnostic](diagnostics/bose-case-id-order.json) · [Archived-log comparison](diagnostics/bose-archive-effect.json)

The results describe retrospective drift localization on a purposive sample. Kairo's state models and alarm thresholds were fitted on the full log, and the best profiles were identified on that same benchmark. The quantile rule flags peaks relative to each log's own score distribution, so the unmatched alarms reflect the tested combination of state representation and alarm rule. The clearest evidence for Kairo is its ability to surface particular resource, timing, and control-flow changes. Establishing reliability for unattended alerts would require evaluating a fixed configuration prospectively on separate logs.
