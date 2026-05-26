"""Adapter for Scikit-Learn estimators and pipelines."""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from ..utils.math import _sigmoid, _softmax

from .base import BaseModelAdapter

class SklearnAdapter(BaseModelAdapter):
    """Adapter for Scikit-Learn models and pipelines."""
    
    def __init__(self, estimator):
        self.estimator = estimator
        self.final_estimator = self._get_final()
        self.is_pipeline = isinstance(self.estimator, Pipeline)
        
    def _get_final(self):
        if isinstance(self.estimator, Pipeline):
            return self.estimator.steps[-1][1]
        return self.estimator

    def predict(self, X: np.ndarray) -> np.ndarray:
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
        return _softmax(S)

    def extract_linear_theta(self, d_expected=None):
        if not (hasattr(self.final_estimator, "coef_") and hasattr(self.final_estimator, "intercept_")):
            return None, None

        w = np.asarray(self.final_estimator.coef_, dtype=float).ravel()
        b_raw = np.asarray(self.final_estimator.intercept_, dtype=float).ravel()
        b = float(b_raw[0]) if b_raw.size else float(self.final_estimator.intercept_)

        if d_expected is not None and w.size != int(d_expected):
            return None, None
        return w, b

    def extract_logistic_theta(self, d_expected=None):
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
            "b": b.copy(),    # (K,)
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.estimator.fit(X, y)
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.is_pipeline:
            # Assumes X is already transformed if it's a pipeline
            self.final_estimator.partial_fit(X, y)
        else:
            self.estimator.partial_fit(X, y)

    @property
    def is_iterative(self) -> bool:
        return hasattr(self.final_estimator, "partial_fit") or hasattr(self.final_estimator, "warm_start")
        
    @property
    def classes(self) -> np.ndarray:
        if hasattr(self.final_estimator, "classes_"):
            return np.asarray(self.final_estimator.classes_)
        return None

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        if not self.is_pipeline:
            return X
        Xt = X
        for name, step in self.estimator.steps[:-1]:
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
            
        try: est.set_params(**{p("warm_start"): True})
        except: pass
        try: est.set_params(**{p("max_iter"): 1})
        except: pass
        try: est.set_params(**{p("tol"): None})
        except: pass
        try: est.set_params(**{p("shuffle"): False})
        except: pass
        
        return SklearnAdapter(est)
