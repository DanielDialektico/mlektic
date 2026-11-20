"""Exact back-transformation of per-feature affine preprocessing.

Prediction explanations show coefficients in the units the student typed.
That is only exact when the pipeline's preprocessing maps every feature
independently as ``u_j = a_j x_j + c_j`` (StandardScaler, MinMaxScaler,
MaxAbsScaler, RobustScaler, and chains of them).  The map is measured from the
fitted transformers themselves and then verified, so a pipeline that mixes or
bends features is rejected instead of being explained with wrong numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

UNSUPPORTED_PREPROCESSING = (
    "Prediction explanations need preprocessing that rescales each feature on its own "
    "(for example StandardScaler, MinMaxScaler, MaxAbsScaler, or RobustScaler). "
    "This pipeline's preprocessing mixes or transforms features nonlinearly, so its "
    "coefficients cannot be written in the original units; use the training animation instead."
)


@dataclass(frozen=True)
class AffinePreprocessing:
    """The verified map ``u = scale * x + shift`` applied before the final estimator."""

    scale: np.ndarray
    shift: np.ndarray

    def transform(self, x: Any) -> np.ndarray:
        """Map original-unit features to the units the model was fitted in."""
        return np.asarray(x, dtype=float) * self.scale + self.shift

    def to_original(self, weights: Any, intercept: Any) -> tuple[np.ndarray, Any]:
        """Rewrite ``w . u + b`` as ``w_o . x + b_o`` (``weights`` may be ``(d,)`` or ``(K, d)``)."""
        weights = np.asarray(weights, dtype=float)
        original_weights = weights * self.scale
        original_intercept = np.asarray(intercept, dtype=float) + np.sum(weights * self.shift, axis=-1)
        if original_intercept.ndim == 0:
            original_intercept = float(original_intercept)
        return original_weights, original_intercept


def affine_preprocessing(estimator: Any) -> AffinePreprocessing | None:
    """Return the pipeline's per-feature affine preprocessing, or ``None`` if it has none.

    Raises ``ValueError`` when preprocessing exists but is not per-feature affine.
    """
    steps = [step for _, step in getattr(estimator, "steps", [])[:-1] if step not in (None, "passthrough")]
    if not steps:
        return None
    n_features = getattr(estimator, "n_features_in_", None)
    if n_features is None:
        n_features = getattr(steps[0], "n_features_in_", None)
    if n_features is None:
        raise ValueError(UNSUPPORTED_PREPROCESSING)

    def apply(values: np.ndarray) -> np.ndarray:
        for step in steps:
            values = step.transform(values)
        return np.asarray(values, dtype=float)

    d = int(n_features)
    try:
        shift = apply(np.zeros((1, d)))
        scale = apply(np.ones((1, d))) - shift
        probe = np.random.default_rng(0).normal(scale=3.0, size=(8, d))
        observed = apply(probe)
    except Exception as exc:  # noqa: BLE001 - any transformer failure means "unsupported"
        raise ValueError(UNSUPPORTED_PREPROCESSING) from exc
    if shift.shape != (1, d) or observed.shape != probe.shape:
        raise ValueError(UNSUPPORTED_PREPROCESSING)
    expected = probe * scale + shift
    if not np.allclose(observed, expected, rtol=1e-9, atol=1e-9 * max(1.0, float(np.max(np.abs(expected))))):
        raise ValueError(UNSUPPORTED_PREPROCESSING)
    return AffinePreprocessing(scale=scale.ravel(), shift=shift.ravel())


__all__ = ["UNSUPPORTED_PREPROCESSING", "AffinePreprocessing", "affine_preprocessing"]
