"""Adapter for Scikit-Learn estimators and pipelines."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from ..utils.math import _sigmoid
from ..utils.probability import infer_multiclass_link, multiclass_probabilities
from .base import BaseModelAdapter


class SklearnAdapter(BaseModelAdapter):
    """Adapter for Scikit-Learn models and pipelines."""

    def __init__(self, estimator):
        """Create an adapter over a fitted or unfitted Scikit-Learn estimator."""
        self.estimator = estimator
        self.final_estimator = self._get_final()
        self.is_pipeline = isinstance(self.estimator, Pipeline)

    def _get_final(self):
        if isinstance(self.estimator, Pipeline):
            return self.estimator.steps[-1][1]
        return self.estimator

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values or class labels through the wrapped estimator."""
        return np.asarray(self.estimator.predict(X))

    def predict_proba(self, X: np.ndarray, classes: np.ndarray = None) -> np.ndarray:
        """Predict probabilities with graceful fallback."""
        if hasattr(self.estimator, "predict_proba"):
            P = self.estimator.predict_proba(X)
            return np.asarray(P, dtype=float)

        if hasattr(self.final_estimator, "predict_proba"):
            P = self.final_estimator.predict_proba(X)
            return np.asarray(P, dtype=float)

        if hasattr(self.estimator, "decision_function"):
            S = self.estimator.decision_function(X)
        elif hasattr(self.final_estimator, "decision_function"):
            S = self.final_estimator.decision_function(X)
        else:
            preds = np.asarray(self.estimator.predict(X)).ravel()
            _classes = np.unique(preds) if classes is None else np.asarray(classes)
            K = len(_classes)
            if K == 2:
                p1 = (preds == _classes[1]).astype(float)
                return np.column_stack([1.0 - p1, p1])
            Y = np.zeros((len(preds), K), dtype=float)
            idx = np.searchsorted(_classes, preds)
            Y[np.arange(len(preds)), idx] = 1.0
            return Y

        S = np.asarray(S, dtype=float)
        if S.ndim == 1:
            p1 = _sigmoid(S)
            return np.column_stack([1.0 - p1, p1])
        return multiclass_probabilities(S, self.resolve_multiclass_link(X))

    def decision_function(self, X: np.ndarray) -> np.ndarray | None:
        """Return decision scores when the wrapped estimator exposes them."""
        if hasattr(self.estimator, "decision_function"):
            return np.asarray(self.estimator.decision_function(X), dtype=float)
        if hasattr(self.final_estimator, "decision_function"):
            values = self.transform_X(X) if self.is_pipeline else X
            return np.asarray(self.final_estimator.decision_function(values), dtype=float)
        return None

    def resolve_multiclass_link(self, X: np.ndarray, requested: str = "auto") -> str:
        """Resolve the multiclass probability link used by the estimator."""
        if requested not in {"auto", "softmax", "ovr"}:
            raise ValueError("multiclass_link must be 'auto', 'softmax', or 'ovr'.")
        if requested != "auto":
            return requested

        sample = np.asarray(X)[: min(len(X), 64)]
        scores = self.decision_function(sample)
        if scores is None or scores.ndim != 2:
            return "softmax"

        if hasattr(self.estimator, "predict_proba"):
            probabilities = np.asarray(self.estimator.predict_proba(sample), dtype=float)
        elif hasattr(self.final_estimator, "predict_proba"):
            values = self.transform_X(sample) if self.is_pipeline else sample
            probabilities = np.asarray(self.final_estimator.predict_proba(values), dtype=float)
        else:
            estimator_name = self.final_estimator.__class__.__name__
            return "ovr" if estimator_name in {"SGDClassifier", "OneVsRestClassifier"} else "softmax"
        return infer_multiclass_link(scores, probabilities)

    def extract_linear_theta(self, d_expected=None):
        """Extract a flat linear-regression coefficient vector and intercept."""
        if not (hasattr(self.final_estimator, "coef_") and hasattr(self.final_estimator, "intercept_")):
            return None, None

        w = np.asarray(self.final_estimator.coef_, dtype=float).ravel()
        b_raw = np.asarray(self.final_estimator.intercept_, dtype=float).ravel()
        b = float(b_raw[0]) if b_raw.size else float(self.final_estimator.intercept_)

        if d_expected is not None and w.size != int(d_expected):
            return None, None
        return w, b

    def extract_logistic_theta(self, d_expected=None):
        """Extract logistic coefficients in the binary or multiclass schema."""
        if not (hasattr(self.final_estimator, "coef_") and hasattr(self.final_estimator, "intercept_")):
            return None

        W = np.asarray(self.final_estimator.coef_, dtype=float)
        b = np.asarray(self.final_estimator.intercept_, dtype=float).ravel()

        if W.ndim == 1:
            W = W.reshape(1, -1)

        K_eff, d = W.shape
        if d_expected is not None and d != int(d_expected):
            return None

        if K_eff == 1:
            return {
                "task": "binary",
                "w": W[0].copy(),
                "b": float(b[0]) if b.size else 0.0,
            }

        return {
            "task": "multiclass",
            "W": W.T.copy(),  # (d, K)
            "b": b.copy(),  # (K,)
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the wrapped estimator."""
        self.estimator.fit(X, y)

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Run one incremental update, bypassing pipeline transforms when needed."""
        if self.is_pipeline:
            # Assumes X is already transformed if it's a pipeline
            self.final_estimator.partial_fit(X, y)
        else:
            self.estimator.partial_fit(X, y)

    @property
    def is_iterative(self) -> bool:
        """Whether the final estimator supports incremental training."""
        return hasattr(self.final_estimator, "partial_fit")

    @property
    def classes(self) -> np.ndarray:
        """Return learned class labels when available."""
        if hasattr(self.final_estimator, "classes_"):
            return np.asarray(self.final_estimator.classes_)
        return None

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        """Transform X through every pipeline step except the final estimator."""
        if not self.is_pipeline:
            return X
        Xt = X
        for _, step in self.estimator.steps[:-1]:
            if hasattr(step, "transform"):
                Xt = step.transform(Xt)
        return Xt

    def _find_scaler(self):
        if not self.is_pipeline:
            return None
        for _, step in self.estimator.steps:
            has_transform = hasattr(step, "transform")
            has_mean = hasattr(step, "mean_")
            has_scale = hasattr(step, "scale_") or hasattr(step, "var_")
            if has_transform and has_mean and has_scale:
                return step
        return None

    def get_scaler_params(self):
        """Return the mean and scale of a pipeline scaler, if one exists."""
        scaler = self._find_scaler()
        if not scaler:
            return None, None

        mu = getattr(scaler, "mean_", None)
        scale = getattr(scaler, "scale_", None)
        if scale is None:
            var = getattr(scaler, "var_", None)
            if var is not None:
                scale = np.sqrt(np.asarray(var, dtype=float))

        return mu, scale

    def clone_for_replay(self):
        """Clone and configure for iterative replay."""
        est = clone(self.estimator)
        pref = self.estimator.steps[-1][0] if self.is_pipeline else None

        def p(name):
            return f"{pref}__{name}" if pref is not None else name

        for param_name, value in (
            ("warm_start", True),
            ("max_iter", 1),
            ("tol", 0.0),
            ("shuffle", False),
        ):
            try:
                est.set_params(**{p(param_name): value})
            except ValueError:
                continue

        return SklearnAdapter(est)
