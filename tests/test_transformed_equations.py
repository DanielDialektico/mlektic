"""Models linear in transformed features u = φ(x) are written in u, in every view."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from mlektic import explain_logistic_prediction, explain_lr_prediction, visualize_logistic, visualize_lr
from mlektic.visualization.feature_terms import rename_features_latex
from mlektic.visualization.linear.prediction import _fmt
from mlektic.visualization.skins.effects import _NUMBER

RNG = np.random.default_rng(0)
X1 = np.linspace(-2.0, 2.0, 50).reshape(-1, 1)
X2 = RNG.normal(size=(70, 2))
X3 = RNG.normal(size=(70, 3))
BS = chr(92)


def _poly(estimator):
    return make_pipeline(PolynomialFeatures(2, include_bias=False), estimator)


def _equations(figure):
    frame = figure.frames[-1] if figure.frames else None
    annotations = frame.layout.annotations if frame is not None and frame.layout.annotations else ()
    texts = [str(a.text) for a in (*figure.layout.annotations, *annotations) if a.text]
    for trace in (*figure.data, *(frame.data if frame is not None else ())):
        text = getattr(trace, "text", None)
        if isinstance(text, str):
            texts.append(text)
        elif text is not None and not hasattr(text, "dtype"):
            texts.extend(str(value) for value in text)
    return " ".join(text for text in texts if "$" in text)


CASES = [
    ("linear 1d", visualize_lr, _poly(LinearRegression()), X1, X1[:, 0] ** 2, "x_{1}^{2}"),
    ("linear 2d", visualize_lr, _poly(LinearRegression()), X2, X2[:, 0] * X2[:, 1], "x_{2}^{2}"),
    ("linear 3d", visualize_lr, _poly(LinearRegression()), X3, X3[:, 0] * X3[:, 1], "x_{3}^{2}"),
    ("binary 1d", visualize_logistic, _poly(LogisticRegression()), X1, (np.abs(X1[:, 0]) > 1).astype(int), "x_{1}^{2}"),
    (
        "binary 2d",
        visualize_logistic,
        _poly(LogisticRegression()),
        X2,
        (X2[:, 0] ** 2 + X2[:, 1] ** 2 > 1).astype(int),
        "x_{2}^{2}",
    ),
    ("binary 3d", visualize_logistic, _poly(LogisticRegression()), X3, (X3[:, 0] ** 2 > 1).astype(int), "x_{3}^{2}"),
    (
        "multiclass 1d",
        visualize_logistic,
        _poly(LogisticRegression(max_iter=500)),
        X1,
        np.digitize(X1[:, 0] ** 2, [1.0, 2.5]),
        "x_{1}^{2}",
    ),
    (
        "multiclass 2d",
        visualize_logistic,
        _poly(LogisticRegression(max_iter=500)),
        X2,
        (X2[:, 0] > 0).astype(int) + (X2[:, 1] > 0),
        "x_{2}^{2}",
    ),
    (
        "multiclass 3d",
        visualize_logistic,
        _poly(LogisticRegression(max_iter=500)),
        X3,
        (X3[:, 0] > 0).astype(int) + (X3[:, 1] > 0),
        "x_{3}^{2}",
    ),
    (
        "iterative binary 2d",
        visualize_logistic,
        _poly(SGDClassifier(loss="log_loss", max_iter=100, random_state=0)),
        X2,
        (X2[:, 0] ** 2 + X2[:, 1] ** 2 > 1).astype(int),
        "x_{2}^{2}",
    ),
]


@pytest.mark.parametrize(("name", "function", "estimator", "X", "y", "last_term"), CASES, ids=[c[0] for c in CASES])
def test_polynomial_pipelines_write_every_view_in_transformed_features(name, function, estimator, X, y, last_term):
    estimator.fit(X, y)
    figure = function(estimator, X, y, steps=3)
    equations = _equations(figure)
    # The last canonical term of φ is in the fitted equation, so no coefficient was dropped.
    assert last_term in equations, name
    assert "f(x_1" not in equations  # the placeholder used when coefficients were unavailable


def test_multiclass_matrix_has_the_transformed_shape():
    y = (X2[:, 0] > 0).astype(int) + (X2[:, 1] > 0)
    figure = visualize_logistic(_poly(LogisticRegression(max_iter=500)).fit(X2, y), X2, y, steps=3)
    equations = _equations(figure)
    assert r"\Theta\in\mathbb{R}^{5\times 3}" in equations
    assert r"\mathbf u=\phi(\mathbf x)" in equations
    assert r"\mathbb{R}^{2\times 3}" not in equations


def test_plain_models_keep_their_original_space_equations():
    y = (X2[:, 0] > 0).astype(int) + (X2[:, 1] > 0)
    equations = _equations(visualize_logistic(LogisticRegression().fit(X2, y), X2, y, steps=3))
    assert r"\Theta\in\mathbb{R}^{2\times 3}" in equations
    assert r"\mathbf{u}" not in equations


def test_renaming_spares_latex_commands():
    names = ["x_1", "x_2", "x_1^2"]
    text = "$" + BS + "max_{k}" + BS + "theta_jx_j+(w)x_{3}+" + BS + "mathbf{x}$"
    renamed = rename_features_latex(text, names)
    assert BS + "max_{k}" in renamed
    assert "u_{j}" in renamed and "x_{1}^{2}" in renamed and BS + "mathbf{u}" in renamed


def test_number_colouring_keeps_the_sign_after_a_row_break():
    text = "a & -1.5" + BS + BS + "-2.6930 & 0.5 " + BS + "x3.25"
    assert [m.group(0) for m in _NUMBER.finditer(text)] == ["-1.5", "-2.6930", "0.5"]


def _stage_texts(figure, label="Output"):
    button = next(b for menu in figure.layout.updatemenus for b in menu.buttons if b.label == label)
    return " ".join(str(a.get("text", "")) for a in button.args[1]["annotations"])


@pytest.mark.parametrize("X", [X1, X2, X3], ids=["d1", "d2", "d3"])
def test_linear_explainer_substitutes_transformed_features(X):
    y = X[:, 0] ** 2 + X.sum(axis=1) + RNG.normal(0, 0.1, len(X))
    estimator = _poly(LinearRegression()).fit(X, y)
    figure = explain_lr_prediction(estimator, X, y, x_query=X[:1])
    meta = figure.layout.meta["mlektic_prediction"]
    u = estimator[:-1].transform(X[:1]).ravel()
    assert meta["equation_space"] == "transformed"
    assert meta["transformed_query"] == pytest.approx(u.tolist())
    assert meta["substitution_matches_predict"] is True
    assert meta["displayed_value"] == pytest.approx(float(estimator.predict(X[:1])[0]))
    substitution = _stage_texts(figure, "Substitution")
    last = f"({_fmt(estimator[-1].coef_[-1], 4)})"
    assert last in substitution  # the last transformed coefficient enters the sum


@pytest.mark.parametrize(
    ("X", "y"),
    [
        (X1, (np.abs(X1[:, 0]) > 1).astype(int)),
        (X2, (X2[:, 0] ** 2 + X2[:, 1] ** 2 > 1).astype(int)),
        (X3, (X3[:, 0] ** 2 > 1).astype(int)),
        (X1, np.digitize(X1[:, 0] ** 2, [1.0, 2.5])),
        (X2, (X2[:, 0] > 0).astype(int) + (X2[:, 1] > 0)),
    ],
    ids=["binary-d1", "binary-d2", "binary-d3", "multiclass-d1", "multiclass-d2"],
)
def test_logistic_explainer_substitutes_transformed_features(X, y):
    estimator = _poly(LogisticRegression(max_iter=500)).fit(X, y)
    figure = explain_logistic_prediction(estimator, X, y, x_query=X[:1])
    meta = figure.layout.meta["mlektic_prediction"]
    assert meta["displayed_class_probabilities"] == pytest.approx(estimator.predict_proba(X[:1]).ravel().tolist())
    texts = _stage_texts(figure, "Substitution") + _stage_texts(figure, "Input")
    assert "phi" in texts  # u = φ(x) is named
    assert r"\mathbb{R}^{2	imes" not in texts


def test_binary_explainer_uses_every_raw_feature():
    y = (X3 @ [1.0, -1.0, 2.0] > 0).astype(int)
    estimator = LogisticRegression().fit(X3, y)
    figure = explain_logistic_prediction(estimator, X3, y, x_query=X3[:1])
    substitution = _stage_texts(figure, "Substitution")
    # The third coefficient multiplies the third input; the 2-D view used to drop it.
    assert f"({_fmt(estimator.coef_[0, 2], 4)})" in substitution and f"({_fmt(X3[0, 2], 4)})" in substitution
