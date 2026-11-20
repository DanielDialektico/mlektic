"""Regression tests for the mathematical review of figures and contracts."""

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import log_loss, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

from mlektic import explain_logistic_prediction, explain_lr_prediction, visualize_logistic
from mlektic.history.metrics import build_linear_metrics, build_logistic_metrics
from mlektic.mathematics import _regularization_contract
from mlektic.neural.hyperparameters import _optimizer_operation, _scheduler_operation
from mlektic.neural.metrics import infer_performance_metrics, output_kind
from mlektic.utils.math import _binary_log_loss_from_p, _multiclass_cross_entropy, r2_finite
from mlektic.utils.preprocessing import affine_preprocessing
from mlektic.visualization.design import _strip_negative_zero


def test_r2_follows_sklearn_finite_convention():
    y = np.array([2.0, 2.0, 2.0])
    assert r2_finite(0.0, 0.0) == r2_score(y, y, force_finite=True) == 1.0
    assert r2_finite(1.0, 0.0) == r2_score(y, y + [0, 0, 1], force_finite=True) == 0.0
    tiny = np.array([0.0, 1e-8, 2e-8])
    hist = build_linear_metrics(tiny, tiny.reshape(-1, 1) * 0.9, np.zeros(1))
    assert hist["R²"][0] == pytest.approx(r2_score(tiny, tiny * 0.9))


def test_log_loss_clipping_matches_sklearn_everywhere():
    p = np.array([1.0, 0.0, 0.7])
    y = np.array([0, 1, 1])
    expected = log_loss(y, np.column_stack([1 - p, p]))
    assert _binary_log_loss_from_p(p, y) == pytest.approx(expected)
    probabilities = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    labels = np.array([1, 2])
    assert _multiclass_cross_entropy(probabilities, labels, np.array([0, 1, 2])) == pytest.approx(
        log_loss(labels, probabilities, labels=[0, 1, 2])
    )


def test_binary_metrics_use_the_displayed_threshold():
    X = np.linspace(-3, 3, 90).reshape(-1, 1)
    y = (X[:, 0] > 0).astype(int)
    w_hist = np.array([[2.0]])
    b_hist = np.array([0.0])
    at_half = build_logistic_metrics(y, X, w_hist, b_hist, np.zeros(1), np.array([0, 1]), is_multiclass=False)
    strict = build_logistic_metrics(
        y, X, w_hist, b_hist, np.zeros(1), np.array([0, 1]), is_multiclass=False, threshold=0.9
    )
    assert at_half["Accuracy"][0] == 1.0
    assert strict["Accuracy"][0] < 1.0


def test_visualize_logistic_metric_cards_follow_threshold():
    X = np.linspace(-3, 3, 90).reshape(-1, 1)
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    default = visualize_logistic(model, X, y, steps=4, max_frames=4, show_loss=True)
    strict = visualize_logistic(model, X, y, steps=4, max_frames=4, threshold=0.9, show_loss=True)

    def final_accuracy(figure):
        texts = [str(a.text) for a in figure.frames[-1].layout.annotations or ()]
        texts += [str(a.text) for a in figure.layout.annotations]
        return [t for t in texts if "Accuracy" in t]

    assert final_accuracy(default) != final_accuracy(strict)


def test_multiclass_metric_is_labelled_macro_f1():
    X = np.random.default_rng(0).normal(size=(60, 1))
    y = np.digitize(X[:, 0], [-0.5, 0.5])
    metrics = build_logistic_metrics(
        y, X, np.zeros((1, 1, 3)), np.zeros((1, 3)), np.zeros(1), np.array([0, 1, 2]), is_multiclass=True
    )
    assert "Macro F1" in metrics and "F1 Score" not in metrics


@pytest.mark.parametrize(
    ("estimator", "kind"),
    [(Ridge(alpha=2.0), "l2"), (Lasso(alpha=0.5), "l1"), (ElasticNet(alpha=0.1, l1_ratio=0.3), "elasticnet")],
)
def test_implied_penalties_are_reported(estimator, kind):
    contract = _regularization_contract(estimator, family="linear")
    assert contract["active"] and contract["type"] == kind
    assert contract["strength_parameter"]["name"] == "alpha"


def test_unpenalized_linear_regression_and_elastic_net_half():
    assert not _regularization_contract(LinearRegression(), family="linear")["active"]
    formula = _regularization_contract(ElasticNet(), family="linear")["formula"]
    assert "(1-rho)/2" in formula


@pytest.mark.parametrize(
    "scaler", [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler(), StandardScaler(with_mean=False)]
)
def test_affine_preprocessing_reconstructs_predictions(scaler):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 3)) * [3.0, 10.0, 1.0] + [5.0, -2.0, 0.0]
    y = 1 + 2 * X[:, 0] - 0.3 * X[:, 1]
    model = make_pipeline(scaler, LinearRegression()).fit(X, y)
    weights, intercept = affine_preprocessing(model).to_original(model[-1].coef_, model[-1].intercept_)
    np.testing.assert_allclose(X @ weights + intercept, model.predict(X), atol=1e-9)
    explain_lr_prediction(model, X, y, x_query=X[:1])


def test_multiclass_affine_preprocessing_and_unsupported_pipelines():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 2)) * [3.0, 1.0]
    y = np.digitize(X[:, 0], [-1.0, 1.0])
    model = make_pipeline(MinMaxScaler(), LogisticRegression()).fit(X, y)
    weights, intercepts = affine_preprocessing(model).to_original(model[-1].coef_, model[-1].intercept_)
    np.testing.assert_allclose(X @ weights.T + intercepts, model.decision_function(X), atol=1e-9)
    explain_logistic_prediction(model, X, y, x_query=X[:1])
    with pytest.raises(ValueError, match="rescales each feature"):
        explain_lr_prediction(make_pipeline(PCA(2), LinearRegression()).fit(X, X[:, 0]), X, X[:, 0], x_query=X[:1])


def test_neural_binary_metrics_use_the_declared_output_kind():
    torch = pytest.importorskip("torch")
    logits = np.array([0.1, 0.2])
    targets = np.array([1.0, 1.0])
    assert output_kind(torch.nn.BCEWithLogitsLoss()) == "logits"
    assert output_kind(torch.nn.BCELoss()) == "probabilities"
    declared = infer_performance_metrics(logits, targets, task="classification", output="logits")
    assert declared["accuracy"] == 1.0


def test_optimizer_formulas_follow_configuration():
    assert r"\theta_t=\theta_{t-1}-\eta g_t" in _optimizer_operation("SGD", {"momentum": 0})
    assert "(1-d)" in _optimizer_operation("SGD", {"momentum": 0.9})
    assert r"\eta(g_t+\mu v_t)" in _optimizer_operation("SGD", {"momentum": 0.9, "nesterov": True})
    assert r"(1-\eta\lambda)" in _optimizer_operation("AdamW", {"weight_decay": 0.01})
    assert r"\operatorname{NAdam}" in _optimizer_operation("NAdam", {})
    assert r"\gamma^{\lfloor t/s\rfloor}" in _scheduler_operation("StepLR")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"\left(-0.0000\right)", r"\left(0.0000\right)"),
        (r"\begin{bmatrix}-0.000 & 1\end{bmatrix}", r"\begin{bmatrix}0.000 & 1\end{bmatrix}"),
        (r"x_{1}-0.000", r"x_{1}-0.000"),
        ("(x)-0.0000", "(x)-0.0000"),
        ("-0.0001", "-0.0001"),
    ],
)
def test_rounded_zero_has_no_sign(text, expected):
    assert _strip_negative_zero(text) == expected
