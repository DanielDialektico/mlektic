"""Default titles describe the fitted model: transformation, degree, d, D, K."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

from mlektic import explain_logistic_prediction, explain_lr_prediction, visualize_logistic, visualize_lr
from mlektic.visualization.titles import linear_title, logistic_title

RNG = np.random.default_rng(0)
X1 = np.linspace(-2.5, 2.5, 40).reshape(-1, 1)
Y1 = 1.0 - 0.7 * X1[:, 0] + 1.6 * X1[:, 0] ** 2
X2 = RNG.normal(size=(60, 2))
Y2 = X2[:, 0] + X2[:, 0] * X2[:, 1]
X3 = RNG.normal(size=(60, 3))
BINARY = (X2[:, 0] ** 2 + X2[:, 1] ** 2 > 1).astype(int)
CLASSES = (X2[:, 0] > 0).astype(int) + (X2[:, 1] > 0)


def _first_line(figure):
    return figure.layout.title.text.split("<br>")[0]


@pytest.mark.parametrize(
    ("estimator", "X", "y", "expected"),
    [
        (LinearRegression(), X1, Y1, "Linear Regression (Simple, 1 variable)"),
        (LinearRegression(), X2, Y2, "Linear Regression (2 variables)"),
        (
            make_pipeline(StandardScaler(), LinearRegression()),
            X3,
            X3 @ [1, 2, 3],
            "Multivariable Linear Regression Model (3 variables)",
        ),
        (Ridge(), X1, Y1, "Ridge Regression (Simple, 1 variable)"),
        (
            make_pipeline(PolynomialFeatures(2, include_bias=False), LinearRegression()),
            X1,
            Y1,
            "Polynomial Regression (degree 2, 1 variable → 2 features)",
        ),
        (
            make_pipeline(StandardScaler(), PolynomialFeatures(3, include_bias=False), Ridge()),
            X2,
            Y2,
            "Polynomial Ridge Regression (degree 3, 2 variables → 9 features)",
        ),
        (
            make_pipeline(SplineTransformer(), LinearRegression()),
            X1,
            Y1,
            "Linear Regression on Transformed Features (1 variable → 7 features)",
        ),
    ],
)
def test_linear_titles_follow_the_fitted_model(estimator, X, y, expected):
    estimator.fit(X, y)
    assert linear_title(estimator, X) == expected
    assert _first_line(visualize_lr(estimator, X, y, steps=3)) == expected


@pytest.mark.parametrize(
    ("estimator", "y", "expected"),
    [
        (LogisticRegression(), BINARY, "Binary Logistic Regression (2 variables)"),
        (LogisticRegression(), CLASSES, "Multiclass Logistic Regression (K=3, d=2)"),
        (
            make_pipeline(PolynomialFeatures(2, include_bias=False), LogisticRegression()),
            BINARY,
            "Binary Logistic Regression, degree 2 (d=2 → 5 features)",
        ),
        (
            make_pipeline(PolynomialFeatures(2, include_bias=False), LogisticRegression(max_iter=500)),
            CLASSES,
            "Multiclass Logistic Regression, degree 2 (K=3, d=2 → 5 features)",
        ),
    ],
)
def test_logistic_titles_follow_the_fitted_model(estimator, y, expected):
    estimator.fit(X2, y)
    assert logistic_title(estimator, X2) == expected
    assert _first_line(visualize_logistic(estimator, X2, y, steps=3)) == expected


def test_prediction_titles_name_the_model():
    ridge = Ridge().fit(X2, Y2)
    figure = explain_lr_prediction(ridge, X2, Y2, x_query=X2[:1])
    assert _first_line(figure) == "Prediction (Ridge Regression, 2 variables)"
    multiclass = LogisticRegression().fit(X2, CLASSES)
    figure = explain_logistic_prediction(multiclass, X2, CLASSES, x_query=X2[:1])
    assert _first_line(figure) == "Prediction (Multiclass Logistic Regression, K=3, d=2) · original space"


def test_long_titles_shrink_to_clear_the_controls():
    estimator = make_pipeline(PolynomialFeatures(2, include_bias=False), LogisticRegression(max_iter=500))
    estimator.fit(X2, CLASSES)
    compact = visualize_logistic(estimator, X2, CLASSES, steps=3, size="compact")
    wide = visualize_logistic(estimator, X2, CLASSES, steps=3, size="wide")
    assert 14 <= compact.layout.title.font.size < wide.layout.title.font.size
