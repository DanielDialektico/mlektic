"""Shared pure helpers reused by linear and logistic services."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline

def _first_not_none(*args):
    """Return the first non-None argument from the given list of arguments."""
    for a in args:
        if a is not None:
            return a
    return None


def _as_2d(X):
    """Ensure that the input array X is 2-dimensional."""
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def _as_1d(y):
    """Ensure that the input array y is 1-dimensional."""
    return np.asarray(y).ravel()


def _get_final_estimator(estimator):
    """Extract the final estimator from a Pipeline, or return the estimator itself."""
    if isinstance(estimator, Pipeline):
        return estimator.steps[-1][1]
    return estimator


def _last_step_prefix(estimator):
    """Extract the string prefix of the last step in a Pipeline, or None."""
    if isinstance(estimator, Pipeline):
        return estimator.steps[-1][0]
    return None


def _try_set_params(estimator, **params):
    """Safely attempt to set parameters on an estimator, catching any exceptions."""
    try:
        estimator.set_params(**params)
        return True
    except Exception:
        return False


def _ema_smooth(arr, beta=0.85):
    """Apply Exponential Moving Average smoothing to a 1D array."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = beta * out[i - 1] + (1 - beta) * arr[i]
    return out


def _is_iterative(estimator):
    """Check if the estimator supports iterative training (partial_fit or warm_start)."""
    last = _get_final_estimator(estimator)
    return hasattr(last, "partial_fit") or hasattr(last, "warm_start")


def _find_standard_scaler(estimator):
    """
    Search for a StandardScaler-like step inside a Pipeline estimator.

    Checks if there is a step with `mean_`, `scale_` (or `var_`), and `transform`.
    """
    if not isinstance(estimator, Pipeline):
        return None

    for _, step in estimator.steps:
        has_transform = hasattr(step, "transform")
        has_mean = hasattr(step, "mean_")
        has_scale = hasattr(step, "scale_") or hasattr(step, "var_")
        if has_transform and has_mean and has_scale:
            return step
    return None


def _safe_get_scale(scaler):
    """Extract mean and scale properties safely from a scaler object."""
    if scaler is None:
        return None, None, True, True

    mu = getattr(scaler, "mean_", None)

    scale = getattr(scaler, "scale_", None)
    if scale is None:
        var = getattr(scaler, "var_", None)
        if var is not None:
            scale = np.sqrt(np.asarray(var, dtype=float))
        else:
            scale = None

    with_mean = bool(getattr(scaler, "with_mean", True))
    with_std = bool(getattr(scaler, "with_std", True))

    return mu, scale, with_mean, with_std


def _transform_up_to_last(pipeline, X):
    """
    Apply all steps of a Pipeline EXCEPT the final estimator.

    Returns the transformed input matrix X_transformed.
    """
    Xt = X
    for _, step in pipeline.steps[:-1]:
        if hasattr(step, "transform"):
            Xt = step.transform(Xt)
    return Xt


def _make_iterative_replay_estimator(estimator):
    """
    Clone the estimator and try to force it into iterative training mode.

    Sets warm_start=True, max_iter=1, tol=None, and shuffle=False on a
    best-effort basis.
    """
    est = clone(estimator)

    pref = _last_step_prefix(est)

    def p(name):
        return f"{pref}__{name}" if pref is not None else name

    _try_set_params(est, **{p("warm_start"): True})
    _try_set_params(est, **{p("max_iter"): 1})
    _try_set_params(est, **{p("tol"): None})
    _try_set_params(est, **{p("shuffle"): False})
    return est


__all__ = [
    "_first_not_none",
    "_as_2d",
    "_as_1d",
    "_get_final_estimator",
    "_last_step_prefix",
    "_try_set_params",
    "_ema_smooth",
    "_is_iterative",
    "_find_standard_scaler",
    "_safe_get_scale",
    "_transform_up_to_last",
    "_make_iterative_replay_estimator",
]
