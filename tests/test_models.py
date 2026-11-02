import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from sklearn.cluster import DBSCAN
from sklearn.metrics import calinski_harabasz_score, pairwise_distances, silhouette_score

import kairo
from oracles import CASE, TIME


@pytest.mark.parametrize("method", ["zscore", "minmax"])
def test_scaling_matches_manual_formula_and_preserves_metadata(method):
    data = pd.DataFrame({CASE: ["a", "b", "c"], TIME: pd.date_range("2025-01-01", periods=3),
                         "x": [1., 2., 3.], "constant": [7., 7., 7.], "current_act:A": [0., 1., 0.],
                         "past_acts_1:A": [0., 0., 1.], "act_set:A": [1., 1., 1.]}, index=[8, 4, 2])
    original = data.copy(deep=True)
    scaled = kairo.standardize(data, method)
    assert_allclose(scaled.x, [-1, 0, 1] if method == "zscore" else [0, .5, 1])
    assert_allclose(scaled.constant, 0)
    assert_frame_equal(scaled[[CASE, TIME, "current_act:A", "past_acts_1:A", "act_set:A"]],
                       data[[CASE, TIME, "current_act:A", "past_acts_1:A", "act_set:A"]])
    all_scaled = kairo.standardize(data, method, exclude=[])
    assert_allclose(all_scaled["act_set:A"], 0)
    assert_frame_equal(data, original)
    with pytest.raises(ValueError, match="Unknown method"):
        kairo.standardize(data, "other")


@pytest.mark.parametrize("components", [None, 1, 3, 5])
def test_pca_variance_and_reconstruction_against_svd(numeric_features, components):
    original = numeric_features.copy(deep=True)
    values = numeric_features.drop(columns=[CASE, TIME]).to_numpy()
    centered = values - values.mean(axis=0)
    _, singular, _ = np.linalg.svd(centered, full_matrices=False)
    model = kairo.compute_pca(numeric_features, components)
    count = components or values.shape[1]
    assert_allclose(model.explained_variance_, singular[:count] ** 2 / (len(values) - 1))
    assert_allclose(model.explained_variance_ratio_, singular[:count] ** 2 / (singular ** 2).sum())
    errors = kairo.compute_pca_reconstruction_errors(numeric_features, model)
    assert_allclose(errors, [(singular[k:] ** 2).sum() / len(values) for k in range(1, count + 1)], atol=1e-12)
    assert (np.diff(errors) <= 1e-12).all()
    for cut in range(1, count + 1):
        reduced = kairo.apply_pca(numeric_features, model, cut)
        assert reduced.columns.tolist() == [CASE, TIME] + [f"pc_{i}" for i in range(1, cut + 1)]
        assert_frame_equal(reduced[[CASE, TIME]], numeric_features[[CASE, TIME]])
        reconstructed = reduced.drop(columns=[CASE, TIME]).to_numpy() @ model.components_[:cut] + model.mean_
        assert np.square(values - reconstructed).sum(axis=1).mean() == pytest.approx(errors.loc[cut], abs=1e-12)
    assert_frame_equal(numeric_features, original)


@pytest.mark.parametrize("distance", ["euclidean", "cosine"])
def test_som_determinism_winners_distances_quantization(numeric_features, distance):
    model = kairo.compute_som(numeric_features, (2, 3), distance=distance)
    again = kairo.compute_som(numeric_features, (2, 3), distance=distance)
    assert_allclose(model.get_weights(), again.get_weights())
    weights = model.get_weights().reshape(6, -1)
    values = numeric_features.drop(columns=[CASE, TIME]).to_numpy()
    distances = pairwise_distances(values, weights, metric=distance)
    expected = np.array([divmod(index, 3) for index in distances.argmin(axis=1)])
    states = kairo.get_som_states(numeric_features, model)
    assert states.index.equals(numeric_features.index)
    assert_array_equal(states.to_numpy(), expected)
    assert kairo.compute_som_quantization_error(numeric_features, states, model, distance) == pytest.approx(distances.min(axis=1).mean())
    all_distances = kairo.get_som_state_distances(model, distance)
    assert_allclose(all_distances, pairwise_distances(weights, metric=distance), atol=1e-12)
    assert all_distances.index.tolist() == [f"({i}, {j})" for i in range(2) for j in range(3)]
    with pytest.raises(ValueError, match="Unknown distance"):
        kairo.compute_som_quantization_error(numeric_features, states, model, "invalid")


def test_kmeans_centers_inertia_and_determinism(numeric_features):
    model = kairo.compute_kmeans(numeric_features, 2)
    states = kairo.get_kmeans_states(numeric_features, model)
    values = numeric_features.drop(columns=[CASE, TIME]).to_numpy()
    expected_sse = 0
    for cluster in range(2):
        members = values[states.cluster == cluster]
        center = members.mean(axis=0)
        assert_allclose(model.cluster_centers_[cluster], center)
        expected_sse += np.square(members - center).sum()
    assert model.inertia_ == pytest.approx(expected_sse)
    assert_array_equal(states.cluster, kairo.compute_kmeans(numeric_features, 2).labels_)
    assert states.index.equals(numeric_features.index)
    assert_allclose(kairo.get_kmeans_state_distances(model), pairwise_distances(model.cluster_centers_), atol=1e-12)


@pytest.mark.parametrize("distance,eps", [("euclidean", .3), ("cosine", .03)])
@given(st.integers(min_value=0, max_value=1000))
def test_weighted_dbscan_matches_full_data_on_nonzero_clusters(distance, eps, seed):
    rng = np.random.default_rng(seed)
    prototypes = np.vstack([rng.normal([2, 0], .03, (4, 2)), rng.normal([0, 2], .03, (4, 2)), [[-10, -10]]])
    values = np.repeat(prototypes, [3, 2, 4, 2, 2, 3, 2, 4, 1], axis=0)
    rng.shuffle(values)
    features = pd.DataFrame(values, columns=["x", "y"])
    reference = DBSCAN(eps=eps, min_samples=3, metric=distance).fit_predict(values)
    model = kairo.compute_dbscan(features, eps, 3, distance)
    actual = kairo.get_dbscan_states(features, model).cluster.to_numpy()
    assert_array_equal(actual == -1, reference == -1)
    assert_array_equal(actual[:, None] == actual, reference[:, None] == reference)
    assert len(model.labels_) == len(prototypes)
    cores = model.components_
    labels = model.labels_[model.core_sample_indices_]
    centers = np.array([cores[labels == label].mean(axis=0) for label in range(actual.max() + 1)])
    assert_allclose(kairo.get_dbscan_state_distances(model), pairwise_distances(centers, metric=distance), atol=1e-10)


def test_all_noise_rejected():
    with pytest.raises(ValueError, match="every row is noise"):
        kairo.compute_dbscan(pd.DataFrame({"x": [0., 5., 10.]}), eps=.1, min_samples=2)


def test_quality_scores_and_noise_exclusion():
    features = pd.DataFrame({"x": [0., 1., 10., 11., 999.], "y": [0., 0., 0., 0., 999.]})
    states = pd.DataFrame({"cluster": [0, 0, 1, 1, -1]})
    # Independent hand calculations: within-cluster mean distance is 1, the
    # other-cluster means are 10.5 and 9.5; between/within sums are 100 and 1.
    assert kairo.compute_silhouette(features, states) == pytest.approx((19 / 21 + 17 / 19) / 2)
    assert kairo.compute_calinski_harabasz(features, states) == pytest.approx(200)
    assert kairo.compute_silhouette(features, states) == pytest.approx(silhouette_score(features.iloc[:4], [0, 0, 1, 1]))
    assert kairo.compute_calinski_harabasz(features, states) == pytest.approx(calinski_harabasz_score(features.iloc[:4], [0, 0, 1, 1]))
    som_states = pd.DataFrame({"i": [0, 0, 1, 1], "j": [0, 0, 0, 0]})
    assert kairo.compute_silhouette(features.iloc[:4], som_states) == pytest.approx(kairo.compute_silhouette(features, states))
    for labels in [[0] * 5, list(range(5)), [-1] * 5]:
        for score in (kairo.compute_silhouette, kairo.compute_calinski_harabasz):
            assert np.isnan(score(features, pd.DataFrame({"cluster": labels})))
    assert kairo.compute_som_occupancy_entropy(som_states, (2, 1)) == pytest.approx(1)
    assert kairo.compute_som_occupancy_entropy(som_states, (2, 2)) == pytest.approx(.5)
    assert kairo.compute_som_occupancy_entropy(pd.DataFrame({"i": [0, 0], "j": [0, 0]}), (2, 2)) == 0
    assert kairo.compute_som_occupancy_entropy(som_states.iloc[:1], (1, 1)) == 0
