"""Math utilities for the Mlektic library."""

import numpy as np


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


def _binary_log_loss_from_p(p, y, eps=1e-12):
    """Compute binary cross-entropy from class-1 probabilities."""
    p = np.clip(np.asarray(p, dtype=float).ravel(), eps, 1.0 - eps)
    y = np.asarray(y).ravel().astype(float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _multiclass_cross_entropy(P, y, classes, eps=1e-12):
    """Compute multiclass cross-entropy from probability matrix."""
    P = np.clip(np.asarray(P, dtype=float), eps, 1.0)
    Y = _one_hot(y, classes)
    return float(-np.mean(np.sum(Y * np.log(P), axis=1)))


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
