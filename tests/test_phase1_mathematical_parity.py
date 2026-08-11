import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from mlektic import visualize_logistic, visualize_lr
from mlektic.services.linear_history import fit_history_logistic
from mlektic.utils.math import _sigmoid
from mlektic.utils.probability import multiclass_probabilities


def _linear_data(dimensions=3):
    rng = np.random.default_rng(101 + dimensions)
    X = rng.normal(size=(90, dimensions))
    coefficients = np.linspace(0.5, 1.5, dimensions)
    y = 0.75 + X @ coefficients + rng.normal(scale=0.03, size=X.shape[0])
    return X, y


def _binary_data(dimensions=2, *, string_labels=False):
    rng = np.random.default_rng(211 + dimensions)
    X = rng.normal(size=(100, dimensions))
    score = 0.4 + X[:, 0] - 0.65 * X[:, min(1, dimensions - 1)]
    binary = (score > 0).astype(int)
    y = np.where(binary, "admitted", "not_admitted") if string_labels else binary
    return X, y


def _multiclass_data(dimensions=2):
    rng = np.random.default_rng(307 + dimensions)
    X = rng.normal(size=(120, dimensions))
    second = X[:, 1] if dimensions > 1 else -X[:, 0]
    scores = np.column_stack([X[:, 0], second, -X[:, 0] - second])
    y = np.array(["blue", "green", "red"])[np.argmax(scores, axis=1)]
    y[:3] = np.array(["blue", "green", "red"])
    return X, y


def test_linear_academic_contract_reconstructs_prediction_and_metrics():
    X, y = _linear_data(3)
    model = LinearRegression().fit(X, y)

    figure = visualize_lr(
        model,
        X,
        y,
        steps=7,
        detail="complete",
        feature_names=["length", "width", "density"],
        sample_index=5,
    )

    contract = figure.layout.meta["mlektic_math"]
    sample = contract["sample"]
    assert contract["detail"] == "complete"
    assert contract["feature_names"] == ["length", "width", "density"]
    assert sample["matches_model"] is True
    np.testing.assert_allclose(
        sample["contribution_sum"] + contract["parameters"]["intercept"],
        model.predict(X[[5]])[0],
    )
    np.testing.assert_allclose(contract["objective"]["value"], np.mean((y - model.predict(X)) ** 2))
    assert contract["objective"]["role"] == "empirical evaluation metric"
    assert contract["update_rule"]["shown_as_exact_estimator_rule"] is False
    assert figure.layout.height > 600
    assert any("Fitted-model derivation" in annotation.text for annotation in figure.layout.annotations)


def test_standard_scaler_original_space_conversion_preserves_linear_prediction():
    X, y = _linear_data(4)
    model = make_pipeline(StandardScaler(), LinearRegression()).fit(X, y)

    figure = visualize_lr(model, X, y, steps=4, detail="academic", sample_index=11)
    contract = figure.layout.meta["mlektic_math"]

    assert contract["equation_space"] == "original"
    assert contract["feature_space"]["is_affine"] is True
    assert contract["feature_space"]["raw_space_coefficients_available"] is True
    assert contract["sample"]["matches_model"] is True
    np.testing.assert_allclose(
        contract["sample"]["reconstructed_prediction"],
        model.predict(X[[11]])[0],
        rtol=1e-10,
        atol=1e-12,
    )


def test_non_affine_pipeline_uses_transformed_feature_mathematics_honestly():
    X = np.linspace(-2.0, 2.0, 70).reshape(-1, 1)
    y = 1.0 - 0.5 * X[:, 0] + 2.0 * X[:, 0] ** 2
    model = make_pipeline(PolynomialFeatures(2, include_bias=False), LinearRegression()).fit(X, y)

    figure = visualize_lr(model, X, y, steps=5, detail="complete", sample_index=9)
    contract = figure.layout.meta["mlektic_math"]

    assert contract["equation_space"] == "transformed"
    assert contract["feature_space"]["is_affine"] is False
    assert contract["feature_space"]["raw_space_coefficients_available"] is False
    assert contract["feature_names"] == ["x_1", "x_1^2"]
    assert contract["sample"]["matches_model"] is True
    assert "Raw-space coefficients are not claimed" in contract["feature_space"]["statement"]


def test_non_affine_logistic_pipeline_reconstructs_transformed_probability():
    X = np.linspace(-3.0, 3.0, 100).reshape(-1, 1)
    y = ((X[:, 0] < -1.0) | (X[:, 0] > 1.0)).astype(int)
    model = make_pipeline(
        PolynomialFeatures(2, include_bias=False),
        LogisticRegression(max_iter=1000),
    ).fit(X, y)

    figure = visualize_logistic(model, X, y, steps=5, detail="academic", sample_index=12)
    contract = figure.layout.meta["mlektic_math"]

    assert contract["equation_space"] == "transformed"
    assert contract["feature_names"] == ["x_1", "x_1^2"]
    assert contract["sample"]["matches_model"] is True
    np.testing.assert_allclose(
        contract["sample"]["model_probabilities"],
        model.predict_proba(X[[12]])[0],
        atol=1e-10,
    )


def test_binary_contract_uses_fitted_class_order_and_custom_threshold_without_label_noise():
    X, y = _binary_data(2, string_labels=True)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    threshold = 0.7

    figure = visualize_logistic(
        model,
        X,
        y,
        steps=5,
        detail="academic",
        threshold=threshold,
        sample_index=6,
    )
    contract = figure.layout.meta["mlektic_math"]
    probabilities = model.predict_proba(X[[6]])[0]
    expected_winner = 1 if probabilities[1] >= threshold else 0

    assert contract["classes"] == model.classes_.tolist()
    assert contract["decision"]["positive_class_label"] == model.classes_[1]
    assert contract["sample"]["winning_class_index"] == expected_winner
    np.testing.assert_allclose(contract["sample"]["model_probabilities"], probabilities)
    np.testing.assert_allclose(contract["sample"]["reconstructed_probabilities"], probabilities)
    assert tuple(figure.layout.scene.zaxis.tickvals) == (0.0, threshold, 1.0)
    assert "admitted" not in figure.layout.annotations[-1].text
    assert "not_admitted" not in figure.layout.annotations[-1].text


@pytest.mark.parametrize(
    ("estimator", "expected_link"),
    [
        (LogisticRegression(max_iter=1000), "softmax"),
        (SGDClassifier(loss="log_loss", max_iter=1000, random_state=19), "ovr"),
    ],
)
def test_multiclass_contract_matches_estimator_link_and_selected_surface(estimator, expected_link):
    X, y = _multiclass_data(2)
    model = estimator.fit(X, y)

    figure = visualize_logistic(
        model,
        X,
        y,
        steps=4,
        detail="academic",
        class_focus="green",
        sample_index=8,
    )
    contract = figure.layout.meta["mlektic_math"]
    selected = int(np.flatnonzero(model.classes_ == "green")[0])

    assert contract["probability_link"] == expected_link
    assert contract["class_focus_index"] == selected
    assert contract["class_focus_label"] == "green"
    np.testing.assert_allclose(
        contract["sample"]["reconstructed_probabilities"],
        model.predict_proba(X[[8]])[0],
        atol=1e-10,
    )
    surface_visibility = [trace.visible for trace in figure.data[1:4]]
    assert surface_visibility == [index == selected for index in range(3)]
    assert f"class focus c_{selected} (1/3)" in figure.layout.title.text


def test_logistic_parameter_interpolation_is_mathematically_synchronized():
    X, y = _binary_data(1)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    history = fit_history_logistic(model, X, y, steps=9, mode="final_interp")

    scores = history["grid"]["x1_grid"][None, :] * history["w_hist"][:, 0, None]
    scores = scores + history["b_hist"][:, None]
    np.testing.assert_allclose(history["p_line_hist"], _sigmoid(scores), atol=1e-12)
    assert history["metadata"]["source_detail"]["interpolation_target"] == "parameters"


def test_incremental_logistic_replay_ends_at_the_exact_fitted_estimator():
    X, y = _binary_data(1)
    model = SGDClassifier(loss="log_loss", max_iter=12, random_state=17).fit(X, y)
    history = fit_history_logistic(model, X, y, steps=7, max_frames=4, smooth=None)

    np.testing.assert_allclose(history["w_hist"][-1], model.coef_[0])
    np.testing.assert_allclose(history["b_hist"][-1], model.intercept_[0])
    np.testing.assert_allclose(
        history["p_line_hist"][-1],
        model.predict_proba(history["grid"]["x1_grid"].reshape(-1, 1))[:, 1],
    )
    assert history["metadata"]["final_state_matches_estimator"] is True
    assert history["metadata"]["displayed_state_origins"][-1] == "fitted_estimator"
    assert history["metadata"]["source_detail"]["replayed_states"] == 6


def test_scaled_logistic_interpolation_matches_original_space_coefficients():
    X, y = _binary_data(1)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y)
    history = fit_history_logistic(
        model,
        X,
        y,
        steps=7,
        mode="final_interp",
        display_space="original",
    )

    grid = history["grid"]["x1_grid"]
    scores = grid[None, :] * history["w_hist"][:, 0, None] + history["b_hist"][:, None]
    np.testing.assert_allclose(history["p_line_hist"], _sigmoid(scores), atol=1e-10)
    assert history["metadata"]["coefficient_space"] == "original"


def test_multiclass_parameter_interpolation_matches_resolved_probability_link():
    X, y = _multiclass_data(1)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    history = fit_history_logistic(model, X, y, steps=6, mode="final_interp")

    grid = history["grid"]["x1_grid"].reshape(-1, 1)
    for frame in range(history["w_hist"].shape[0]):
        scores = grid @ history["w_hist"][frame] + history["b_hist"][frame]
        expected = multiclass_probabilities(scores, history["probability_link"])
        np.testing.assert_allclose(history["p_curves_hist"][frame], expected, atol=1e-12)


def test_complete_regularization_contract_is_conservative_and_estimator_backed():
    X, y = _linear_data(2)
    model = SGDRegressor(penalty="elasticnet", alpha=0.02, l1_ratio=0.3, random_state=3).fit(X, y)

    figure = visualize_lr(model, X, y, steps=4, detail="complete")
    regularization = figure.layout.meta["mlektic_math"]["regularization"]

    assert regularization["type"] == "elasticnet"
    assert regularization["strength_parameter"] == {"name": "alpha", "value": 0.02}
    assert regularization["l1_ratio"] == 0.3
    assert regularization["intercept_penalty"] == "not introspected"
    assert "exact internal normalization is not claimed" in regularization["claim"]


def test_closed_form_nd_shows_default_empirical_path_curve_without_calling_it_optimizer_loss():
    X, y = _linear_data(6)
    model = LinearRegression().fit(X, y)

    figure = visualize_lr(model, X, y, steps=8, max_frames=5)
    semantics = figure.layout.meta["mlektic_history"]["loss_display_semantics"]

    assert figure.data[0].uid == "LOSS_LINE"
    assert figure.data[0].name == "Interpolation MSE"
    assert figure.layout.xaxis2.title.text == "Interpolation progress"
    assert figure.layout.yaxis2.title.text == "Empirical MSE along interpolation"
    assert figure.layout.yaxis2.domain == (0.12, 0.68)
    assert figure.layout.height == 640
    assert semantics["optimizer_loss"] is False
    assert semantics["role"] == "empirical evaluation along a synthetic interpolation"
    assert semantics["smoothing"] is None
    assert "already a smooth mathematical path" in figure.layout.meta["mlektic_history"]["smoothing"]["reason"]
    assert any("INTERP. MSE" in annotation.text for annotation in figure.layout.annotations)
    np.testing.assert_allclose(figure.frames[-1].data[0].y[-1], np.mean((y - model.predict(X)) ** 2))


def test_nd_fitted_contributions_wrap_without_horizontal_truncation():
    X, y = _linear_data(6)
    names = ["attendance", "assignments", "projects", "sleep", "practice", "prior_score"]
    model = LinearRegression().fit(X, y)

    figure = visualize_lr(
        model,
        X,
        y,
        steps=4,
        detail="complete",
        feature_names=names,
        sample_index=7,
    )
    calculation = next(
        annotation.text
        for annotation in figure.layout.annotations
        if annotation.y < 0 and r"\begin{aligned}" in annotation.text
    )

    assert calculation.count(r"\\[4pt]") == 3
    latex_names = [name.replace("_", r"\_") for name in names]
    assert all(rf"\mathrm{{{name}}}" in calculation for name in latex_names)
    assert "display truncated" not in calculation


def test_high_dimensional_panel_wraps_a_bounded_selection_and_discloses_omissions():
    X, y = _linear_data(20)
    model = LinearRegression().fit(X, y)

    figure = visualize_lr(
        model,
        X,
        y,
        steps=3,
        detail="academic",
        feature_names=[f"feature_{index + 1}" for index in range(20)],
    )
    calculation = next(
        annotation.text
        for annotation in figure.layout.annotations
        if annotation.y < 0 and r"\begin{aligned}" in annotation.text
    )

    assert calculation.count(r"\underbrace") == 9
    assert calculation.count(r"\\[4pt]") == 4
    assert any("Showing 9 of 20 contributions" in annotation.text for annotation in figure.layout.annotations)
    assert "every value remains available" not in calculation


def test_logistic_interpolation_can_show_empirical_path_without_optimizer_claims():
    X, y = _binary_data(2)
    model = LogisticRegression(max_iter=1000).fit(X, y)

    figure = visualize_logistic(model, X, y, steps=6, show_loss=True)
    loss_trace = next(trace for trace in figure.data if trace.uid == "LOSS_LINE")
    semantics = figure.layout.meta["mlektic_history"]["loss_display_semantics"]

    assert loss_trace.name == "Interpolation log-loss"
    assert figure.layout.yaxis.title.text == "Empirical log-loss along interpolation"
    assert semantics["optimizer_loss"] is False
    assert semantics["smoothing"] is None


def test_logistic_default_regularization_reports_effective_l2_not_deprecation_sentinel():
    X, y = _binary_data(2)
    model = LogisticRegression(max_iter=1000).fit(X, y)

    figure = visualize_logistic(model, X, y, steps=3, detail="complete")
    regularization = figure.layout.meta["mlektic_math"]["regularization"]

    assert regularization["type"] == "l2"
    assert regularization["strength_parameter"]["name"] == "C"
    assert regularization["strength_parameter"]["meaning"] == "inverse strength"


def test_all_detail_levels_share_the_hybrid_latex_math_band_and_smooth_motion():
    X, y = _linear_data(1)
    model = SGDRegressor(max_iter=12, random_state=5).fit(X, y)

    classic = visualize_lr(model, X, y, steps=5)
    academic = visualize_lr(model, X, y, steps=5, detail="academic")
    complete = visualize_lr(model, X, y, steps=5, detail="complete")

    assert len(classic.frames) == len(academic.frames)
    assert len(classic.data) == len(academic.data)
    for figure in (classic, academic, complete):
        equation = figure.data[3]
        assert equation.uid == "NUMERIC_EQUATION"
        assert equation.xaxis == "x4"
        assert equation.yaxis == "y4"
        assert equation.text[0].startswith("$\\hat{y}=")
        assert figure.frames[0].data[2].text != figure.frames[-1].data[2].text
        assert figure.layout.yaxis.domain[1] < figure.layout.yaxis4.domain[0]
        assert figure.layout.updatemenus[0].buttons[0].args[1]["frame"]["redraw"] is False
        assert all(not frame.layout.to_plotly_json() for frame in figure.frames)

        fitted_equation = figure.frames[-1].data[2].text[0]
        assert f"({model.coef_[0]:.4f})" in fitted_equation
        assert f"({model.intercept_[0]:.4f})" in fitted_equation
        assert figure.layout.meta["mlektic_history"]["final_state_matches_estimator"] is True
        assert figure.layout.sliders[0].steps[-1].label == "fitted"

    panel_annotations = [
        annotation
        for annotation in academic.layout.annotations
        if "Fitted-model derivation" in annotation.text
    ]
    assert len(panel_annotations) == 1
    assert panel_annotations[0].y < 0


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"detail": "dense"}, ValueError, "detail"),
        ({"show_objective": "sometimes"}, TypeError, "show_objective"),
        ({"show_regularization": 1}, TypeError, "show_regularization"),
        ({"feature_names": ["only_one"]}, ValueError, "feature_names"),
        ({"sample_index": 1000}, ValueError, "sample_index"),
    ],
)
def test_linear_phase1_options_fail_early(kwargs, error, match):
    X, y = _linear_data(2)
    model = LinearRegression().fit(X, y)
    with pytest.raises(error, match=match):
        visualize_lr(model, X, y, steps=3, **kwargs)


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.1, 1.1, "0.5"])
def test_invalid_binary_threshold_is_rejected(threshold):
    X, y = _binary_data(1)
    model = LogisticRegression().fit(X, y)
    with pytest.raises((TypeError, ValueError), match="threshold"):
        visualize_logistic(model, X, y, steps=3, threshold=threshold)
