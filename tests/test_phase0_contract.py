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
    assert labeled_figure.data[1].name == "Model: p1"
    labeled_output = labeled_figure.layout.updatemenus[0].buttons[2].args[1]["annotations"]
    assert r"\hat{y} &= \arg\max" in labeled_output[-1]["text"]
    assert "accepted" in next(item["text"] for item in labeled_output if "Predicted class:" in item["text"])
    assert tuple(labeled_figure.layout.scene.zaxis.ticktext) == ("0", "0.5", "1")


@pytest.mark.parametrize("dimensions", [1, 2])
def test_binary_prediction_keeps_long_class_labels_below_inputs_and_results(dimensions):
    rng = np.random.default_rng(202 + dimensions)
    X = rng.normal(size=(80, dimensions))
    labels = (
        "accepted_application_with_additional_review",
        "rejected_application_after_initial_screening",
    )
    y = np.where(X[:, 0] > 0, labels[0], labels[1])
    model = LogisticRegression(max_iter=1000).fit(X, y)
    figure = explain_logistic_prediction(model, X, y, x_query=X[0], show_class_labels=True)

    output = figure.layout.updatemenus[0].buttons[2].args[1]["annotations"]
    class_rows = [item for item in output if str(item["text"]).startswith(("c₀ =", "c₁ ="))]
    assert len(class_rows) == 2
    assert class_rows[0]["y"] > class_rows[1]["y"]
    assert all("<br>" in item["text"] for item in class_rows)
    assert any("Predicted class:" in item["text"] for item in output)
    assert all(label not in str(trace.name) for trace in figure.data for label in labels)
    if dimensions == 1:
        assert figure.layout.yaxis2.title.text == "p₁"
        assert tuple(figure.layout.yaxis2.ticktext) == ("0", "0.5", "1")
    else:
        assert figure.layout.scene.zaxis.title.text == "p₁"
        assert tuple(figure.layout.scene.zaxis.ticktext) == ("0", "0.5", "1")


@pytest.mark.parametrize("dimensions", [1, 5])
def test_multiclass_prediction_wraps_long_semantic_labels(dimensions):
    rng = np.random.default_rng(302 + dimensions)
    X = rng.normal(size=(120, dimensions))
    labels = np.array([
        "class_zero_with_a_very_long_description",
        "class_one_with_a_very_long_description",
        "class_two_with_a_very_long_description",
    ])
    y = labels[np.digitize(X[:, 0], [-0.5, 0.5])]
    model = LogisticRegression(max_iter=1000).fit(X, y)
    figure = explain_logistic_prediction(model, X, y, x_query=X[0], show_class_labels=True)
    output = figure.layout.updatemenus[0].buttons[2].args[1]["annotations"]

    result = next(item["text"] for item in output if r"\hat{y} = \arg\max" in str(item["text"]))
    assert all(label not in result for label in labels)
    predicted = next(item["text"] for item in output if "Predicted class:" in str(item["text"]))
    assert str(model.predict(X[:1])[0]) == predicted.replace("<br>", "").replace("Predicted class:", "").strip()
    if dimensions == 1:
        assert figure.layout.yaxis2.title.text == "pₖ"
        assert all("<br>" in trace.name for trace in figure.data if str(trace.name).startswith("Class "))
    else:
        assert sum(" = " in str(item["text"]) and str(item["text"]).startswith("c") for item in output) == 3


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
    labeled_annotations = labeled_prediction.layout.updatemenus[0].buttons[2].args[1]["annotations"]
    assert all(label not in indexed_result for label in model.classes_)
    assert str(model.predict([[1.5]])[0]) in next(
        item["text"] for item in labeled_annotations if "Predicted class:" in item["text"]
    )


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


def test_multiclass_2d_probability_examples_match_1d_mathematical_order():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(70, 2))
    scores = np.column_stack([X[:, 0], X[:, 1], -X[:, 0] - X[:, 1]])
    y = np.array(["red", "green", "blue"])[np.argmax(scores, axis=1)]
    model = LogisticRegression().fit(X, y)

    figure = visualize_logistic(model, X, y, steps=5)

    parameters, score_equation = figure.layout.annotations[1:3]
    score_examples, first_probability, vertical_dots, last_probability = figure.layout.annotations[3:7]
    assert r"\Theta&=" in parameters.text
    assert r"\boldsymbol{\theta}_0&=" in parameters.text
    assert parameters.y > score_equation.y
    ordered_rows = (
        score_equation,
        score_examples,
        first_probability,
        vertical_dots,
        last_probability,
    )
    assert [item.y for item in ordered_rows] == pytest.approx([0.62, 0.44, 0.35, 0.22, 0.14])
    aligned_rows = (score_equation, score_examples, first_probability, last_probability)
    assert len({item.x for item in aligned_rows}) == 1
    assert {item.xanchor for item in aligned_rows} == {"center"}
    assert r"z_0(\mathbf{x})=" in score_examples.text
    assert r"\;\cdots\; z_{2}(\mathbf{x})=" in score_examples.text
    assert not any(r"\hat p_{0}=" in str(item.text) for item in figure.layout.annotations)


@pytest.mark.parametrize("feature_count", [1, 2, 4])
def test_multiclass_probability_stack_style_and_position_are_stable_across_frames(feature_count):
    rng = np.random.default_rng(23 + feature_count)
    X = rng.normal(size=(72, feature_count))
    if feature_count == 1:
        y = np.where(X[:, 0] < -0.4, "left", np.where(X[:, 0] > 0.4, "right", "center"))
    else:
        scores = np.column_stack([X[:, 0], X[:, 1], -X[:, 0] - X[:, 1]])
        y = np.array(["red", "green", "blue"])[np.argmax(scores, axis=1)]
    model = LogisticRegression(max_iter=1000).fit(X, y)

    figure = visualize_logistic(model, X, y, steps=5)

    stack_start = 3 if feature_count in (1, 2) else 4
    stack_length = 4 if feature_count in (1, 2) else 2

    def stack_signature(annotations):
        return tuple(
            (annotation.x, annotation.y, annotation.xanchor, annotation.yanchor, annotation.font.size)
            for annotation in annotations[stack_start : stack_start + stack_length]
        )

    probability_stack = figure.layout.annotations[stack_start : stack_start + stack_length]
    if feature_count == 1:
        score_examples, first_probability, vertical_dots, last_probability = probability_stack
        assert r"\quad\cdots\quad " in score_examples.text
        assert first_probability.font.size == last_probability.font.size == 13
        assert vertical_dots.font.size == 22
    elif feature_count == 2:
        score_examples, first_probability, omitted, last_probability = probability_stack
        assert score_examples.font.size == 13
        assert first_probability.font.size == last_probability.font.size == 13
        assert omitted.font.size == 22
    else:
        score_1, probability_stack_row = probability_stack
        assert score_1.font.size == 14
        assert probability_stack_row.font.size == 16
        assert r"\begin{array}{r@{\,}c@{\,}l}" in probability_stack_row.text
        assert r"&&\smash{\overset{\hphantom{" in probability_stack_row.text
        assert r"}{\vdots}" in probability_stack_row.text

    expected_signature = stack_signature(figure.layout.annotations)
    for annotations in [figure.layout.annotations, *(frame.layout.annotations for frame in figure.frames)]:
        if feature_count in (1, 2):
            assert r"\\[5pt]\boldsymbol{\theta}_0&=" in annotations[1].text
    for frame in figure.frames:
        assert stack_signature(frame.layout.annotations) == expected_signature


def test_multiclass_1d_and_nd_ellipses_are_centered_for_their_layouts():
    rng = np.random.default_rng(37)

    X_1d = rng.normal(size=(72, 1))
    y_1d = np.where(X_1d[:, 0] < -0.4, "left", np.where(X_1d[:, 0] > 0.4, "right", "center"))
    figure_1d = visualize_logistic(LogisticRegression(max_iter=1000).fit(X_1d, y_1d), X_1d, y_1d, steps=5)
    score_1d, first_1d, dots_1d, last_1d = figure_1d.layout.annotations[3:7]
    assert score_1d.x == first_1d.x == last_1d.x == pytest.approx(0.27)
    assert dots_1d.x == pytest.approx(0.328)
    assert dots_1d.y == pytest.approx(0.22)

    X_nd = rng.normal(size=(72, 4))
    scores = np.column_stack([X_nd[:, 0], X_nd[:, 1], -X_nd[:, 0] - X_nd[:, 1]])
    y_nd = np.array(["red", "green", "blue"])[np.argmax(scores, axis=1)]
    figure_nd = visualize_logistic(LogisticRegression(max_iter=1000).fit(X_nd, y_nd), X_nd, y_nd, steps=5)
    input_vector = figure_nd.layout.annotations[1]
    probability_stack_nd = figure_nd.layout.annotations[5]
    assert input_vector.text.count(r" \\ ") == 3
    assert " & " not in input_vector.text
    assert probability_stack_nd.x == pytest.approx(0.025)
    assert probability_stack_nd.y == pytest.approx(0.22)
    assert r"&&\smash{\overset{\hphantom{" in probability_stack_nd.text
    assert r"}{\vdots}" in probability_stack_nd.text


def test_three_class_2d_ovr_replay_uses_exact_compact_probability_fractions():
    rng = np.random.default_rng(41)
    X = rng.normal(size=(90, 2))
    scores = np.column_stack([X[:, 0], X[:, 1], -X[:, 0] - X[:, 1]])
    y = np.argmax(scores, axis=1)
    model = SGDClassifier(loss="log_loss", max_iter=8, random_state=5).fit(X, y)

    figure = visualize_logistic(model, X, y, steps=5, show_loss=True, smooth=None)

    first_probability = figure.layout.annotations[4].text
    last_probability = figure.layout.annotations[6].text
    assert r"\frac{q_{0}}{\sum_{\ell=0}^{K-1}q_\ell}" in first_probability
    assert r"\frac{q_{2}}{\sum_{\ell=0}^{K-1}q_\ell}" in last_probability
    assert r"q_\ell=\sigma(z_\ell)" in first_probability
    assert r"q_\ell=\sigma(z_\ell)" in last_probability
    assert figure.layout.annotations[4].font.size == 13
    assert figure.layout.annotations[6].font.size == 13
    assert figure.layout.annotations[2].x == pytest.approx(0.275)
    assert figure.layout.annotations[3].x == pytest.approx(0.30)
    assert figure.layout.annotations[5].x == pytest.approx(0.308)
    assert r"\;\cdots\; " in figure.layout.annotations[3].text
    assert r"\quad\cdots\quad " not in figure.layout.annotations[3].text

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
    assert r"\sum_{\ell=0}^{K-1}q_\ell" in figure.layout.annotations[5].text


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
