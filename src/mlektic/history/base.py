"""Strategy interface for capturing training history."""

import abc
from typing import Any, Dict, Tuple

import numpy as np

from ..adapters.base import BaseModelAdapter


class HistoryCaptureStrategy(abc.ABC):
    """Abstract strategy to capture training history."""

    @abc.abstractmethod
    def capture_linear(self, adapter: BaseModelAdapter, X: np.ndarray, y: np.ndarray, config: Any) -> Dict[str, Any]:
        """Capture history for linear regression."""
        pass

    @abc.abstractmethod
    def capture_logistic(self, adapter: BaseModelAdapter, X: np.ndarray, y: np.ndarray, config: Any) -> Dict[str, Any]:
        """Capture history for logistic regression."""
        pass


def _scale_linear_theta(w_s, b_s, scaler_params: Tuple[np.ndarray, np.ndarray]):
    """Convert linear coefficients from scaled space to original space."""
    if scaler_params is None or scaler_params[0] is None:
        return w_s.copy(), float(b_s)

    mu, scale = scaler_params
    dloc = w_s.size

    if scale is None:
        scale = np.ones(dloc, dtype=float)
    if mu is None:
        mu = np.zeros(dloc, dtype=float)

    denom = scale + 1e-12
    w_o = w_s / denom
    b_o = float(b_s - np.sum(w_s * mu / denom))
    return w_o, b_o


def _scale_logistic_binary_theta(w_s, b_s, scaler_params: Tuple[np.ndarray, np.ndarray]):
    """Convert binary logistic coefficients from scaled space to original space."""
    return _scale_linear_theta(w_s, b_s, scaler_params)


def _scale_logistic_multiclass_theta(W_s, b_s, scaler_params: Tuple[np.ndarray, np.ndarray]):
    """Convert multiclass logistic coefficients from scaled space to original space."""
    if scaler_params is None or scaler_params[0] is None:
        return W_s.copy(), b_s.copy()

    mu, scale = scaler_params
    dloc, K = W_s.shape

    if scale is None:
        scale = np.ones(dloc, dtype=float)
    if mu is None:
        mu = np.zeros(dloc, dtype=float)

    denom = (scale + 1e-12)[:, None]
    W_o = W_s / denom
    b_o = b_s - np.sum(W_s * (mu[:, None] / denom), axis=0)
    return W_o, b_o
