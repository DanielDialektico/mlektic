"""Math utilities for the Mlektic library."""

import numpy as np

# Probabilities are clipped to [eps, 1 - eps] before taking logarithms, exactly
# as sklearn.metrics.log_loss does, so every panel reports the same empirical loss.
LOG_LOSS_EPS = float(np.finfo(float).eps)


def _sigmoid(z):
    """Compute a numerically stable sigmoid."""
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(Z):
    """Compute row-wise softmax probabilities."""
    Z = np.asarray(Z, dtype=float)
    Z = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=1, keepdims=True)


def _binary_log_loss_from_p(p, y, eps=LOG_LOSS_EPS):
    """Compute binary cross-entropy from class-1 probabilities."""
    p = np.clip(np.asarray(p, dtype=float).ravel(), eps, 1.0 - eps)
    y = np.asarray(y).ravel().astype(float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _multiclass_cross_entropy(P, y, classes, eps=LOG_LOSS_EPS):
    """Compute multiclass cross-entropy from probability matrix."""
    P = np.clip(np.asarray(P, dtype=float), eps, 1.0 - eps)
    Y = _one_hot(y, classes)
    return float(-np.mean(np.sum(Y * np.log(P), axis=1)))


def r2_finite(ss_res, ss_tot):
    """Coefficient of determination with sklearn's finite convention.

    ``1 - SS_res/SS_tot`` when the target varies.  For a constant target the
    ratio is undefined; like ``sklearn.metrics.r2_score(force_finite=True)`` a
    perfect fit scores 1 and any other fit scores 0.
    """
    ss_res = np.asarray(ss_res, dtype=float)
    if ss_tot > 0.0:
        return 1.0 - ss_res / ss_tot
    return np.where(ss_res == 0.0, 1.0, 0.0)


def _one_hot(y, classes):
    """Encode labels into one-hot matrix using provided class ordering."""
    y = np.asarray(y).ravel()
    classes = np.asarray(classes)
    idx = np.searchsorted(classes, y)
    Y = np.zeros((len(y), len(classes)), dtype=float)
    Y[np.arange(len(y)), idx] = 1.0
    return Y


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
