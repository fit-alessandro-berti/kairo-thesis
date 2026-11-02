import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from numpy.testing import assert_allclose

import kairo
from oracles import CASE, TIME


def validate_figure(figure):
    assert isinstance(figure, go.Figure)
    payload = json.loads(figure.to_json())
    assert payload["data"] and payload["layout"]


@pytest.mark.parametrize("method", ["som", "kmeans", "dbscan"])
@pytest.mark.parametrize("perspective", ["intra", "resource", "inter"])
def test_complete_pipeline_every_perspective_and_clustering_method(event_log, method, perspective):
    # This intentionally uses the public API, with no internal helpers or mocks.
    kwargs = {"sliding_window_size": 2} if perspective == "intra" else {}
    features = getattr(kairo, f"compute_features_{perspective}")(event_log, **kwargs)
    scaled = kairo.standardize(features, "minmax")
    pca = kairo.compute_pca(scaled, min(3, len(features)))
    compressed = kairo.apply_pca(scaled, pca, min(2, pca.n_components_))
    if method == "som":
        model = kairo.compute_som(compressed, size=(2, 2))
    elif method == "kmeans":
        model = kairo.compute_kmeans(compressed, k=2)
    else:
        # Singleton clusters are valid DBSCAN clusters and avoid all-noise in this tiny fixture.
        model = kairo.compute_dbscan(compressed, eps=.25, min_samples=1)
    labels = getattr(kairo, f"get_{method}_states")(compressed, model)
    states = compressed[[column for column in [CASE, TIME] if column in compressed]].join(labels)
    assert len(states) == len(features)
    colors = getattr(kairo, f"get_{method}_state_colors")(model)
    assert colors == getattr(kairo, f"get_{method}_state_colors")(model)
    assert all(re.fullmatch(r"#[0-9a-f]{6}", color) for color in colors.values())
    distances = getattr(kairo, f"get_{method}_state_distances")(model)
    assert_allclose(distances, distances.T, atol=1e-10)
    assert_allclose(np.diag(distances), 0, atol=1e-10)
    frequencies = getattr(kairo, f"get_{method}_state_frequencies")(states)
    assert frequencies.sum() == len(features)
    distributions = kairo.compute_state_distributions(states, "1D")
    assert_allclose(distributions.sum(axis=1), 1)
    drift = kairo.compute_drift_signal(distributions, "js", "previous")
    assert np.isfinite(drift.drift_score.iloc[1:]).all()
    figures = [kairo.plot_activity_counts(kairo.compute_log_stats(event_log)),
               kairo.plot_pca_variances(pca), kairo.plot_pca_variances(pca, 1),
               kairo.plot_pca_reconstruction_errors(kairo.compute_pca_reconstruction_errors(scaled, pca)),
               kairo.plot_pca_reconstruction_errors(kairo.compute_pca_reconstruction_errors(scaled, pca), 1),
               kairo.plot_state_distributions(distributions, colors), kairo.plot_drift_signal(drift, "js"),
               kairo.plot_drift_signal(drift, "js", threshold=.3),
               kairo.plot_signal_window_rows(kairo.compute_signal_window_rows(states, "1D"))]
    if method == "som":
        figures.extend([kairo.plot_som_state_colors(states, colors), kairo.plot_som_u_matrix(model),
                        kairo.plot_som_heatmap(states, (2, 2)),
                        kairo.plot_som_heatmap(states, (2, 2), "2025-01-02", "2025-01-04")])
    else:
        figures.extend([getattr(kairo, f"plot_{method}_state_frequencies")(frequencies, colors),
                        getattr(kairo, f"plot_{method}_state_distances")(distances)])
    if method == "dbscan":
        figures.append(kairo.plot_dbscan_k_distance(compressed, min_samples=2))
    if perspective == "intra":
        single = getattr(kairo, f"get_{method}_case_trajectory")(states, "a")
        many = getattr(kairo, f"get_{method}_case_trajectories")(states)
        assert many.events.sum() == len(states)
        figures.extend([getattr(kairo, f"plot_{method}_case_trajectory")(single, colors, "a"),
                        getattr(kairo, f"plot_{method}_case_trajectories")(many, colors)])
    else:
        visits = getattr(kairo, f"get_{method}_log_trajectory")(states)
        assert visits.feature_windows.sum() == len(states)
        figures.append(getattr(kairo, f"plot_{method}_log_trajectory")(visits, colors))
    for figure in figures:
        validate_figure(figure)


def test_plot_data_matches_input_quantities(event_log):
    stats = kairo.compute_log_stats(event_log)
    figure = kairo.plot_activity_counts(stats)
    assert dict(zip(figure.data[0].y, figure.data[0].x)) == stats["activity_counts"].to_dict()
    features = pd.DataFrame({"x": [0., 1., 4., 10.], "y": [0., 0., 0., 0.]})
    curve = kairo.plot_dbscan_k_distance(features, min_samples=2)
    assert_allclose(curve.data[0].y, [1., 1., 3., 6.])
    states = pd.DataFrame({TIME: pd.date_range("2025-01-01", periods=3, freq="D"), "i": [0, 1, 0], "j": [0, 0, 0]})
    heatmap = kairo.plot_som_heatmap(states, (2, 2))
    assert_allclose(heatmap.data[0].z, [[2, 0], [1, 0]])
    empty_range = kairo.plot_som_heatmap(states, (2, 2), "2026-01-01")
    assert_allclose(empty_range.data[0].z, np.zeros((2, 2)))
    colors = {0: "#123456", 1: "#abcdef", -1: "#e4e4e4"}
    frequencies = pd.Series([2], index=["0"], name="events")
    bars = kairo.plot_dbscan_state_frequencies(frequencies, colors)
    assert {trace.name: float(trace.y[0]) for trace in bars.data} == {"0": 2., "1": 0., "noise": 0.}
    signal = pd.DataFrame({"signal_window": pd.date_range("2025-01-01", periods=3), "drift_score": [np.nan, .2, .8]})
    line = kairo.plot_drift_signal(signal, "tv", threshold=.5)
    assert_allclose(line.data[0].y, signal.drift_score, equal_nan=True)
    assert line.layout.shapes[0].y0 == line.layout.shapes[0].y1 == .5
    assert line.layout.annotations[0].text == "Threshold 0.5"


@pytest.mark.parametrize("count", [1, 2, 12])
def test_state_colors_for_degenerate_and_large_models(count):
    data = pd.DataFrame({"x": np.arange(max(count, 2), dtype=float), "y": np.zeros(max(count, 2))})
    model = kairo.compute_kmeans(data, count)
    colors = kairo.get_kmeans_state_colors(model)
    assert len(colors) == count
    assert all(re.fullmatch(r"#[0-9a-f]{6}", color) for color in colors.values())
    assert colors == kairo.get_kmeans_state_colors(model)
    if count > 1:
        assert len(set(colors.values())) > 1
    frequencies = pd.Series(1, index=[str(i) for i in range(count)], name="events")
    validate_figure(kairo.plot_kmeans_state_frequencies(frequencies, colors))
    validate_figure(kairo.plot_kmeans_state_distances(kairo.get_kmeans_state_distances(model)))


def test_identical_neurons_get_identical_colors():
    model = kairo.compute_som(pd.DataFrame({"x": [1., 1., 1.], "y": [1., 1., 1.]}), (2, 2))
    assert len(set(kairo.get_som_state_colors(model).values())) == 1


@pytest.mark.rendering
def test_real_png_export_and_copilot_image_payload(event_log):
    figure = kairo.plot_activity_counts(kairo.compute_log_stats(event_log))
    png = figure.to_image(format="png", width=640, height=480)
    assert png.startswith(b"\x89PNG\r\n\x1a\n") and len(png) > 1000
    import base64
    from io import BytesIO
    from PIL import Image
    with Image.open(BytesIO(png)) as image:
        assert image.size == (640, 480)
        assert image.convert("RGB").getextrema()[0][0] < image.convert("RGB").getextrema()[0][1]
    for provider in ["openai", "anthropic"]:
        messages = kairo.create_messages(provider, None, ["Explain these counts."], [figure])
        block = messages[0]["content"][1]
        encoded = block["source"]["data"] if provider == "anthropic" else block["image_url"]["url"].split(",", 1)[1]
        with Image.open(BytesIO(base64.b64decode(encoded))) as image:
            assert image.format == "PNG" and image.width > 0


@pytest.mark.parametrize("perspective", ["intra", "resource", "inter"])
def test_planted_abrupt_change_has_peak_at_known_day(perspective):
    rows = []
    start = pd.Timestamp("2025-01-01", tz="UTC")
    for day in range(40):
        changed = day >= 20
        for case in range(6 if changed else 2):
            for event in range(2):
                rows.append({CASE: f"{day}-{case}", "concept:name": f"{'new' if changed else 'old'}-{event}",
                             TIME: start + pd.Timedelta(days=day, hours=case * 3 + event),
                             "org:resource": "new-team" if changed else "old-team",
                             "event_duration": 30 if changed else 10})
    log = pd.DataFrame(rows)
    features = getattr(kairo, f"compute_features_{perspective}")(log)
    scaled = kairo.standardize(features, "minmax")
    reduced = kairo.apply_pca(scaled, kairo.compute_pca(scaled, 2), 2)
    model = kairo.compute_kmeans(reduced, 2)
    states = reduced[[column for column in [CASE, TIME] if column in reduced]].join(kairo.get_kmeans_states(reduced, model))
    signal = kairo.compute_drift_signal(kairo.compute_state_distributions(states, "1D"), "tv")
    peak = signal.loc[signal.drift_score.idxmax()]
    assert peak.signal_window == start + pd.Timedelta(days=20)
    # Clusters need not coincide exactly with regimes (intra-case prefixes can
    # share a state), so the contract here is a unique positive peak at the change.
    assert 0 < peak.drift_score <= 1
    assert (signal.drift_score.dropna().drop(peak.name).abs() < 1e-12).all()
