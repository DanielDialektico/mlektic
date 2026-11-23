r"""Default figure titles that describe the fitted model, not only its input width.

A pipeline ``PolynomialFeatures(degree=2) -> LinearRegression`` on one input
column is a degree-2 polynomial regression with two transformed features, not
a "simple" one-variable linear regression. Titles therefore read the fitted
estimator: its preprocessing (affine scaling is ignored, any other map is
named), the polynomial degree, the input dimension :math:`d`, the transformed
dimension :math:`D_\phi`, the regularized family, and, for classification, the
number of classes :math:`K`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Preprocessing steps that only scale or shift each feature: the model stays
# linear in the original variables, so the title keeps its plain form.
_AFFINE_STEPS = {"StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler"}
_LINEAR_NAMES = {
    "Ridge": "Ridge Regression",
    "RidgeCV": "Ridge Regression",
    "Lasso": "Lasso Regression",
    "LassoCV": "Lasso Regression",
    "LassoLars": "Lasso Regression",
    "ElasticNet": "Elastic-Net Regression",
    "ElasticNetCV": "Elastic-Net Regression",
}


@dataclass(frozen=True)
class ModelDescription:
    """What a default title needs to know about a fitted estimator."""

    final: Any
    input_dimension: int
    transformed_dimension: int
    polynomial_degree: int | None
    transformed: bool
    classes: int | None

    @property
    def variables(self) -> str:
        """``"1 variable"`` or ``"d variables"``."""
        d = self.input_dimension
        return f"{d} variable" if d == 1 else f"{d} variables"

    @property
    def mapping(self) -> str:
        """``"d variables → D features"`` for a transformed feature space."""
        return f"{self.variables} → {self.transformed_dimension} features"


def describe_model(estimator: Any, X: Any) -> ModelDescription:
    """Inspect a fitted estimator or pipeline on its training inputs."""
    data = np.asarray(X)
    d = 1 if data.ndim == 1 else int(data.shape[1])
    steps = list(getattr(estimator, "steps", []) or [])
    final = steps[-1][1] if steps else estimator
    degree = None
    transformed = False
    D = d
    if len(steps) > 1:
        for _, step in steps[:-1]:
            if step is None or step == "passthrough":
                continue
            kind = step.__class__.__name__
            if kind == "PolynomialFeatures":
                raw = getattr(step, "degree", 1)
                step_degree = int(raw[1] if isinstance(raw, tuple) else raw)
                degree = max(degree or 1, step_degree)
            if kind not in _AFFINE_STEPS:
                transformed = True
        try:
            sample = data.reshape(-1, d)[:1]
            D = int(np.asarray(estimator[:-1].transform(sample)).reshape(1, -1).shape[1])
        except Exception:  # pragma: no cover - exotic preprocessing keeps the input width
            D = d
        transformed = transformed or D != d
    if degree is not None and degree < 2:
        degree = None
    classes = getattr(final, "classes_", None)
    return ModelDescription(
        final=final,
        input_dimension=d,
        transformed_dimension=D,
        polynomial_degree=degree,
        transformed=transformed,
        classes=None if classes is None else int(len(classes)),
    )


def _linear_parts(model: ModelDescription) -> tuple[str, str]:
    """Return ``(name, details)`` for a linear-regression family model."""
    name = _LINEAR_NAMES.get(model.final.__class__.__name__, "Linear Regression")
    if model.polynomial_degree is not None:
        polynomial = "Polynomial Regression" if name == "Linear Regression" else f"Polynomial {name}"
        return polynomial, f"degree {model.polynomial_degree}, {model.mapping}"
    if model.transformed:
        return f"{name} on Transformed Features", model.mapping
    if model.input_dimension == 1:
        return name, "Simple, 1 variable"
    return name, model.variables


def linear_title(estimator: Any, X: Any) -> str:
    """Default title of ``visualize_lr``."""
    model = describe_model(estimator, X)
    name, details = _linear_parts(model)
    if (
        name == "Linear Regression"
        and model.input_dimension > 2
        and not model.transformed
    ):
        return f"Multivariable Linear Regression Model ({details})"
    if name.endswith("Regression") and not model.transformed and model.input_dimension > 2:
        return f"Multivariable {name} ({details})"
    return f"{name} ({details})"


def linear_prediction_title(estimator: Any, X: Any) -> str:
    """Default title of ``explain_lr_prediction``."""
    name, details = _linear_parts(describe_model(estimator, X))
    return f"Prediction ({name}, {details.replace('Simple, ', '')})"


def _logistic_parts(model: ModelDescription) -> tuple[str, str]:
    multiclass = (model.classes or 2) > 2
    name = "Multiclass Logistic Regression" if multiclass else "Binary Logistic Regression"
    prefix = f"K={model.classes}, " if multiclass else ""
    mapping = f"d={model.input_dimension} → {model.transformed_dimension} features"
    if model.polynomial_degree is not None:
        return f"{name}, degree {model.polynomial_degree}", f"{prefix}{mapping}"
    if model.transformed:
        return f"{name}, transformed features", f"{prefix}{mapping}"
    if multiclass:
        return name, f"K={model.classes}, d={model.input_dimension}"
    return name, model.variables


def logistic_title(estimator: Any, X: Any) -> str:
    """Default title of ``visualize_logistic``."""
    name, details = _logistic_parts(describe_model(estimator, X))
    return f"{name} ({details})"


def logistic_prediction_title(estimator: Any, X: Any, *, space: str) -> str:
    """Default title of ``explain_logistic_prediction`` in ``space`` display."""
    name, details = _logistic_parts(describe_model(estimator, X))
    return f"Prediction ({name}, {details}) · {space} space"


__all__ = [
    "ModelDescription",
    "describe_model",
    "linear_prediction_title",
    "linear_title",
    "logistic_prediction_title",
    "logistic_title",
]
