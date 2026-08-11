from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier, SGDRegressor

from mlektic import explain_logistic_prediction, export_figure, visualize_logistic, visualize_lr
from mlektic.api.linear import explain_lr_prediction
from mlektic.domain.config import LinearHistoryConfig, LogisticHistoryConfig
from mlektic.services.linear_history import fit_history, fit_history_logistic


def _linear_data():
    X = np.linspace(-2.0, 2.0, 40).reshape(-1, 1)
    return X, 1.5 + 2.25 * X[:, 0]


def _binary_data(*, string_labels=False):
    X = np.linspace(-2.0, 2.0, 40).reshape(-1, 1)
    numeric = (X[:, 0] >= 0).astype(int)
    y = np.where(numeric == 1, "accepted", "rejected") if string_labels else numeric
    return X, y


def test_interpolation_contract_is_explicit_and_preserves_alpha():
    X, y = _linear_data()
    model = LinearRegression().fit(X, y)

    history = fit_history(model, X, y, steps=11, max_frames=4)

    assert history["history_source"] == "interpolated"
    assert history["metadata"]["source"] == "interpolated"
    assert history["metadata"]["captured_steps"] == 11
    assert history["metadata"]["displayed_steps"] == 4
    assert history["metadata"]["final_state_matches_estimator"] is True
    assert history["metadata"]["source_detail"]["baseline"] == "mean"
    np.testing.assert_allclose(history["alpha_values"][[0, -1]], [0.0, 1.0])
    assert history["metadata"]["warnings"][0]["code"] == "synthetic_interpolation"


def test_replay_contract_separates_raw_and_smoothed_loss():
    X, y = _linear_data()
    model = SGDRegressor(max_iter=12, random_state=7).fit(X, y)

    history = fit_history(model, X, y, steps=10, max_frames=4, smooth="ema", smooth_beta=0.8)

    assert history["metadata"]["source"] == "replayed"
    assert history["metadata"]["displayed_step_indices"].tolist() == [1, 4, 7, 10]
    assert history["loss_raw"] is not history["loss_display"]
    assert not np.allclose(history["loss_raw"], history["loss_display"])
    np.testing.assert_allclose(history["loss_hist"], history["loss_display"])
    np.testing.assert_allclose(history["metrics_hist"]["Loss"], history["loss_display"])
    assert history["metadata"]["warnings"][0]["code"] == "replay_not_original_training"
    effective = history["metadata"]["source_detail"]["effective_replay_parameters"]
    assert effective["max_iter"] == 1
    assert effective["shuffle"] is False
    assert history["metadata"]["source_detail"]["endpoint_policy"] == "supplied_fitted_estimator"
    assert history["metadata"]["final_state_matches_estimator"] is True
    assert history["metadata"]["displayed_state_origins"][-1] == "fitted_estimator"
    np.testing.assert_allclose(history["w_hist"][-1], model.coef_)
    np.testing.assert_allclose(history["b_hist"][-1], model.intercept_[0])


@pytest.mark.parametrize(
    ("config_type", "kwargs", "error_type"),
    [
        (LinearHistoryConfig, {"mode": "mystery"}, ValueError),
        (LinearHistoryConfig, {"baseline": "prior"}, ValueError),
        (LinearHistoryConfig, {"smooth": "median"}, ValueError),
        (LinearHistoryConfig, {"steps": 0}, ValueError),
        (LinearHistoryConfig, {"metrics": ["invisible_metric"]}, ValueError),
        (LogisticHistoryConfig, {"multiclass_link": "sigmoid"}, ValueError),
    ],
)
def test_invalid_history_configuration_fails_early(config_type, kwargs, error_type):
    with pytest.raises(error_type):
        config_type(**kwargs)


def test_iterative_mode_rejects_non_incremental_estimator():
    X, y = _linear_data()
    model = LinearRegression().fit(X, y)
    with pytest.raises(ValueError, match="partial_fit"):
        fit_history(model, X, y, mode="iterative")


def test_visual_timeline_reports_source_and_retained_indices():
    X, y = _linear_data()
    model = SGDRegressor(max_iter=8, random_state=7).fit(X, y)

    figure = visualize_lr(model, X, y, steps=8, max_frames=3, animation_mode="native")

    assert "Reconstructed replay" in figure.layout.title.text
    assert "fitted endpoint" in figure.layout.title.text
    assert "3/8 states" in figure.layout.title.text
    assert "n_iter_=" in figure.layout.title.text
    assert figure.layout.sliders[0].currentvalue.prefix == "Replay + fitted endpoint (3/8) · state: "
    assert [step.label for step in figure.layout.sliders[0].steps] == ["1", "4", "fitted"]
    assert figure.layout.meta["mlektic_history"]["captured_steps"] == 8


def test_history_subtitle_can_be_hidden_without_removing_context():
    X, y = _linear_data()
    model = SGDRegressor(max_iter=8, random_state=7).fit(X, y)

    figure = visualize_lr(
        model,
        X,
        y,
        steps=8,
        max_frames=3,
        animation_mode="native",
        show_history_context=False,
    )

    assert "Reconstructed replay" not in figure.layout.title.text
    assert "Replay + fitted endpoint (3/8)" in figure.layout.sliders[0].currentvalue.prefix
    assert figure.layout.meta["mlektic_history"]["source"] == "replayed"
    with pytest.raises(TypeError, match="show_history_context"):
        visualize_lr(model, X, y, show_history_context="no")


def test_linear_prediction_verifies_values_and_marks_extrapolation():
    X, y = _linear_data()
    model = LinearRegression().fit(X, y)

    with pytest.raises(ValueError, match="does not match"):
        explain_lr_prediction(model, X, y, x_query=[[4.0]], yhat=-999.0)
    with pytest.raises(ValueError, match="exactly one sample"):
        explain_lr_prediction(model, X, y, x_query=[[0.0], [1.0]])

    figure = explain_lr_prediction(
        model,
        X,
        y,
        x_query=[[4.0]],
        yhat=-999.0,
        prediction_source="provided",
    )
    assert "Extrapolation" in figure.layout.title.text
    assert figure.layout.meta["mlektic_prediction"]["source"] == "provided"
    assert figure.layout.xaxis2.range[1] > 4.0


def test_logistic_prediction_supports_string_labels_and_verifies_probability():
    X, y = _binary_data(string_labels=True)
    model = LogisticRegression().fit(X, y)
    probability = model.predict_proba([[0.5]])[0, 1]

    with pytest.raises(ValueError, match="does not match"):
        explain_logistic_prediction(model, X, y, x_query=[[0.5]], p_hat=1.0 - probability)

    figure = explain_logistic_prediction(model, X, y, x_query=[[0.5]])
    assert figure.layout.meta["mlektic_prediction"]["source"] == "model"
    assert "model-verified" in figure.layout.title.text


def test_binary_2d_prediction_maps_string_classes_and_shows_model_geometry():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 2))
    y = np.where(X[:, 0] + 0.5 * X[:, 1] > 0, "accepted", "rejected")
    model = LogisticRegression().fit(X, y)

    figure = explain_logistic_prediction(model, X, y, x_query=[[0.4, 0.1]])

    data_trace, model_surface, boundary, _prediction = figure.data
    assert set(np.asarray(data_trace.z, dtype=float)) == {0.0, 1.0}
    assert model_surface.type == "surface"
    assert np.ptp(np.asarray(model_surface.z, dtype=float)) > 0.5
    assert model_surface.name == "Model: p1"
    assert boundary.type == "scatter3d"
    assert len(boundary.z) >= 2
    np.testing.assert_allclose(boundary.z, 0.5)

    output_annotations = figure.layout.updatemenus[0].buttons[2].args[1]["annotations"]
    result_text = output_annotations[-1]["text"]
    assert r"\hat{\mathbf{p}}" in result_text
    assert r"\hat{y} &= \arg\max" in result_text
    assert r"=0" in result_text
    assert "accepted" not in result_text
    assert "rejected" not in result_text

    metadata = figure.layout.meta["mlektic_prediction"]
    assert metadata["classes"] == ["accepted", "rejected"]
    assert metadata["probability_target_class"] == "rejected"
    assert metadata["decision_threshold"] == 0.5
    assert metadata["show_class_labels"] is False
    np.testing.assert_allclose(metadata["model_class_probabilities"], model.predict_proba([[0.4, 0.1]])[0])

    labeled_figure = explain_logistic_prediction(
        model, X, y, x_query=[[0.4, 0.1]], show_class_labels=True
    )
    assert labeled_figure.data[1].name == "Model: P(rejected | x)"
    labeled_output = labeled_figure.layout.updatemenus[0].buttons[2].args[1]["annotations"][-1]["text"]
    assert r"=0\;(\mathrm{accepted})" in labeled_output
    assert "accepted" in labeled_figure.layout.scene.zaxis.ticktext[0]
    assert "rejected" in labeled_figure.layout.scene.zaxis.ticktext[-1]


def test_class_label_visibility_is_consistent_in_training_figures():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(70, 2))
    y = np.where(X[:, 0] > X[:, 1], "accepted", "rejected")
    model = LogisticRegression().fit(X, y)

    indexed = visualize_logistic(model, X, y, steps=5)
    labeled = visualize_logistic(model, X, y, steps=5, show_class_labels=True)

    assert indexed.layout.scene.zaxis.ticktext == ("0", "0.5", "1")
    assert all(label not in " ".join(indexed.layout.scene.zaxis.ticktext) for label in model.classes_)
    assert "accepted" in labeled.layout.scene.zaxis.ticktext[0]
    assert "rejected" in labeled.layout.scene.zaxis.ticktext[-1]
    assert indexed.layout.meta["mlektic_classes"]["show_class_labels"] is False
    assert labeled.layout.meta["mlektic_classes"]["classes"] == model.classes_.tolist()

    with pytest.raises(TypeError, match="show_class_labels"):
        visualize_logistic(model, X, y, show_class_labels="yes")
    with pytest.raises(TypeError, match="show_class_labels"):
        explain_logistic_prediction(model, X, y, x_query=[[0.0, 0.0]], show_class_labels="yes")


def test_multiclass_figures_hide_semantic_labels_by_default_and_can_reveal_them():
    X = np.linspace(-3.0, 3.0, 90).reshape(-1, 1)
    y = np.where(X[:, 0] < -0.7, "left", np.where(X[:, 0] > 0.7, "right", "center"))
    model = LogisticRegression(max_iter=1000).fit(X, y)

    indexed_training = visualize_logistic(model, X, y, steps=5)
    labeled_training = visualize_logistic(model, X, y, steps=5, show_class_labels=True)
    indexed_names = [trace.name for trace in indexed_training.data if str(trace.name).startswith("p(class")]
    labeled_names = [trace.name for trace in labeled_training.data if str(trace.name).startswith("p(class")]
    assert all(label not in " ".join(indexed_names) for label in model.classes_)
    assert all(label in " ".join(labeled_names) for label in model.classes_)

    indexed_prediction = explain_logistic_prediction(model, X, y, x_query=[[1.5]])
    labeled_prediction = explain_logistic_prediction(
        model, X, y, x_query=[[1.5]], show_class_labels=True
    )
    indexed_result = indexed_prediction.layout.updatemenus[0].buttons[2].args[1]["annotations"][5]["text"]
    labeled_result = labeled_prediction.layout.updatemenus[0].buttons[2].args[1]["annotations"][5]["text"]
    assert all(label not in indexed_result for label in model.classes_)
    assert str(model.predict([[1.5]])[0]) in labeled_result


def test_logistic_history_uses_fitted_positive_class_for_string_label_f1():
    X, y = _binary_data(string_labels=True)
    model = LogisticRegression().fit(X, y)

    history = fit_history_logistic(model, X, y, steps=5)

    assert "F1 Score" in history["metrics_hist"]
    assert np.all(np.isfinite(history["metrics_hist"]["F1 Score"]))
    figure = visualize_logistic(model, X, y, steps=5)
    assert figure.layout.meta["mlektic_history"]["source"] == "interpolated"


def test_logistic_raw_loss_is_preserved_when_display_is_smoothed():
    X, y = _binary_data()
    model = SGDClassifier(loss="log_loss", max_iter=10, random_state=4).fit(X, y)
    history = fit_history_logistic(model, X, y, steps=8, smooth="ema", max_frames=4)

    assert not np.allclose(history["loss_raw"], history["loss_display"])
    np.testing.assert_allclose(history["metrics_hist"]["Log-loss"], history["loss_display"])


def test_binary_logistic_2d_equations_have_vertical_separation():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(60, 2))
    y = (X[:, 0] - 0.4 * X[:, 1] > 0).astype(int)
    model = LogisticRegression().fit(X, y)

    figure = visualize_logistic(model, X, y, steps=5)

    formula, substitution = figure.layout.annotations[:2]
    assert formula.y - substitution.y >= 0.15
    assert figure.layout.margin.t == 180


def test_multiclass_2d_probability_examples_form_a_compact_vertical_block():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(70, 2))
    scores = np.column_stack([X[:, 0], X[:, 1], -X[:, 0] - X[:, 1]])
    y = np.array(["red", "green", "blue"])[np.argmax(scores, axis=1)]
    model = LogisticRegression().fit(X, y)

    figure = visualize_logistic(model, X, y, steps=5)

    theta_matrix, bias_vector, score_equation = figure.layout.annotations[1:4]
    first_probability = figure.layout.annotations[4]
    vertical_dots = figure.layout.annotations[5]
    last_probability = figure.layout.annotations[6]
    assert theta_matrix.y > bias_vector.y > score_equation.y
    assert bias_vector.y == pytest.approx((theta_matrix.y + score_equation.y) / 2, abs=0.01)
    assert first_probability.y == pytest.approx(0.35)
    assert first_probability.x == vertical_dots.x == last_probability.x
    assert first_probability.y > vertical_dots.y > last_probability.y
    assert r"\\[10pt]" in first_probability.text


@pytest.mark.parametrize(("feature_count", "stack_start"), [(1, 3), (2, 4), (4, 4)])
def test_multiclass_probability_stack_style_and_position_are_stable_across_frames(feature_count, stack_start):
    rng = np.random.default_rng(23 + feature_count)
    X = rng.normal(size=(72, feature_count))
    if feature_count == 1:
        y = np.where(X[:, 0] < -0.4, "left", np.where(X[:, 0] > 0.4, "right", "center"))
    else:
        scores = np.column_stack([X[:, 0], X[:, 1], -X[:, 0] - X[:, 1]])
        y = np.array(["red", "green", "blue"])[np.argmax(scores, axis=1)]
    model = LogisticRegression(max_iter=1000).fit(X, y)

    figure = visualize_logistic(model, X, y, steps=5)

    def stack_signature(annotations):
        return tuple(
            (annotation.x, annotation.y, annotation.xanchor, annotation.yanchor, annotation.font.size)
            for annotation in annotations[stack_start : stack_start + 3]
        )

    probability_stack = figure.layout.annotations[stack_start : stack_start + 3]
    first_probability, vertical_dots, last_probability = probability_stack
    assert r"\\[10pt]" in first_probability.text
    assert first_probability.font.size == last_probability.font.size == 13
    assert vertical_dots.font.size == 22

    expected_signature = stack_signature(figure.layout.annotations)
    for frame in figure.frames:
        assert stack_signature(frame.layout.annotations) == expected_signature


def test_multiclass_1d_and_nd_ellipses_are_centered_for_their_layouts():
    rng = np.random.default_rng(37)

    X_1d = rng.normal(size=(72, 1))
    y_1d = np.where(X_1d[:, 0] < -0.4, "left", np.where(X_1d[:, 0] > 0.4, "right", "center"))
    figure_1d = visualize_logistic(LogisticRegression(max_iter=1000).fit(X_1d, y_1d), X_1d, y_1d, steps=5)
    first_1d, dots_1d, last_1d = figure_1d.layout.annotations[3:6]
    assert first_1d.x == dots_1d.x == last_1d.x
    assert dots_1d.y == pytest.approx(0.19)

    X_nd = rng.normal(size=(72, 4))
    scores = np.column_stack([X_nd[:, 0], X_nd[:, 1], -X_nd[:, 0] - X_nd[:, 1]])
    y_nd = np.array(["red", "green", "blue"])[np.argmax(scores, axis=1)]
    figure_nd = visualize_logistic(LogisticRegression(max_iter=1000).fit(X_nd, y_nd), X_nd, y_nd, steps=5)
    input_vector = figure_nd.layout.annotations[1]
    dots_nd = figure_nd.layout.annotations[5]
    assert r" \\ " not in input_vector.text
    assert dots_nd.x == pytest.approx(0.31)
    assert dots_nd.y == pytest.approx(0.08)


def test_multiclass_replay_uses_exact_compact_probability_fractions_when_loss_is_visible():
    rng = np.random.default_rng(41)
    X = rng.normal(size=(90, 2))
    scores = np.column_stack([X[:, 0], X[:, 1], -X[:, 0] - X[:, 1]])
    y = np.argmax(scores, axis=1)
    model = SGDClassifier(loss="log_loss", max_iter=8, random_state=5).fit(X, y)

    figure = visualize_logistic(model, X, y, steps=5, show_loss=True, smooth=None)

    first_probability = figure.layout.annotations[4].text
    last_probability = figure.layout.annotations[6].text
    assert r"\sum_{j=1}^{3}q_j" in first_probability
    assert r"q_j=\sigma(z_j)" in first_probability
    assert r"\sum_{j=1}^{3}q_j" in last_probability
    assert len(first_probability) < 500
    assert len(last_probability) < 250

    def geometry_signature(annotations):
        return tuple(
            (item.x, item.y, item.xanchor, item.yanchor, item.font.size)
            for item in annotations
        )

    expected_geometry = geometry_signature(figure.layout.annotations)
    for frame in figure.frames:
        assert geometry_signature(frame.layout.annotations) == expected_geometry


def test_dense_multiclass_replay_honors_theta_column_cap_and_uses_taller_layout():
    rng = np.random.default_rng(47)
    X = rng.normal(size=(112, 20))
    weights = rng.normal(size=(8, 20))
    y = np.argmax(X @ weights.T, axis=1)
    y[:8] = np.arange(8)
    model = SGDClassifier(loss="log_loss", max_iter=8, random_state=7).fit(X, y)

    figure = visualize_logistic(
        model,
        X,
        y,
        steps=4,
        show_loss=True,
        smooth=None,
        max_theta_cols=5,
    )

    theta_matrix = figure.layout.annotations[2].text
    assert r"\begin{array}{ccccc}" in theta_matrix
    assert figure.layout.height == 720
    assert r"\sum_{j=1}^{8}q_j" in figure.layout.annotations[4].text


def test_nd_figures_do_not_keep_an_empty_loss_subplot_when_loss_is_hidden():
    rng = np.random.default_rng(43)
    X = rng.normal(size=(72, 4))

    y_linear = X @ np.array([0.5, -0.4, 0.8, 0.2]) + 0.3
    linear_figure = visualize_lr(LinearRegression().fit(X, y_linear), X, y_linear, steps=5, show_loss=False)
    assert len(linear_figure.data) == 0
    assert linear_figure.layout.xaxis.visible is False
    assert not hasattr(linear_figure.layout, "xaxis2")

    y_binary = (X[:, 0] - 0.5 * X[:, 1] > 0).astype(int)
    logistic_figure = visualize_logistic(LogisticRegression(max_iter=1000).fit(X, y_binary), X, y_binary, steps=5)
    assert len(logistic_figure.data) == 0
    assert logistic_figure.layout.xaxis.visible is False
    assert not hasattr(logistic_figure.layout, "xaxis2")


def test_export_figure_declares_plotly_and_mathjax_dependencies(tmp_path):
    figure = go.Figure(go.Scatter(x=[0, 1], y=[0, 1])).update_layout(title=r"$y=x$")
    destination = export_figure(figure, tmp_path / "mathematical-figure")

    assert destination == (tmp_path / "mathematical-figure.html").resolve()
    assert isinstance(destination, Path)
    html = destination.read_text(encoding="utf-8")
    assert "MathJax.js" in html
    assert "plotly.js" in html
    assert r"$y=x$" in html
