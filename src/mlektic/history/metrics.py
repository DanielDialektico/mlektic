"""Metric history builders shared by linear and logistic captures."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from ..utils.math import _sigmoid

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def build_linear_metrics(
    y: np.ndarray,
    y_pred_hist: np.ndarray,
    loss_hist: np.ndarray,
    metric_config: Any = None,
    *,
    max_metrics: int = 5,
) -> dict[str, np.ndarray]:
    """Build metric histories for linear regression frames."""
    y = np.asarray(y).ravel()
    mse_hist = np.mean((y.reshape(-1, 1) - y_pred_hist) ** 2, axis=0)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot > 1e-12:
        ss_res = np.sum((y.reshape(-1, 1) - y_pred_hist) ** 2, axis=0)
        r2_hist = 1.0 - (ss_res / ss_tot)
    else:
        r2_hist = np.zeros_like(mse_hist)

    builtins = {
        "loss": ("Loss", np.asarray(loss_hist, dtype=float)),
        "mse": ("MSE", mse_hist),
        "r2": ("R²", r2_hist),
        "mae": ("MAE", _metric_by_frame(y, y_pred_hist, mean_absolute_error)),
    }
    return _select_metrics(builtins, metric_config, y, y_pred_hist, max_metrics=max_metrics)


def build_logistic_metrics(
    y: np.ndarray,
    X_eval: np.ndarray,
    w_hist: np.ndarray,
    b_hist: np.ndarray,
    loss_hist: np.ndarray,
    classes: np.ndarray,
    metric_config: Any = None,
    *,
    is_multiclass: bool,
    max_metrics: int = 5,
) -> dict[str, np.ndarray]:
    """Build metric histories for logistic regression frames."""
    y = np.asarray(y).ravel()
    y_pred_hist = _predict_logistic_labels(X_eval, w_hist, b_hist, classes, is_multiclass=is_multiclass)
    builtins = {
        "loss": ("Log-loss", np.asarray(loss_hist, dtype=float)),
        "log_loss": ("Log-loss", np.asarray(loss_hist, dtype=float)),
        "accuracy": ("Accuracy", _metric_by_frame(y, y_pred_hist, accuracy_score)),
        "f1": (
            "F1 Score",
            _f1_by_frame(y, y_pred_hist, classes=classes, is_multiclass=is_multiclass),
        ),
    }
    return _select_metrics(builtins, metric_config, y, y_pred_hist, max_metrics=max_metrics)


def _predict_logistic_labels(
    X_eval: np.ndarray,
    w_hist: np.ndarray,
    b_hist: np.ndarray,
    classes: np.ndarray,
    *,
    is_multiclass: bool,
) -> np.ndarray:
    """Predict class labels for each frame using the captured parameters."""
    classes = np.asarray(classes)
    if is_multiclass:
        scores = np.stack([X_eval @ w_hist[t] + b_hist[t] for t in range(w_hist.shape[0])], axis=2)
        class_indices = np.argmax(scores, axis=1)
        return classes[class_indices]

    weights = w_hist.reshape(w_hist.shape[0], -1)
    scores = X_eval @ weights.T + np.asarray(b_hist).reshape(1, -1)
    positive = _sigmoid(scores) >= 0.5
    return np.where(positive, classes[1], classes[0])


def _metric_by_frame(y_true: np.ndarray, y_pred_hist: np.ndarray, fn: MetricFn) -> np.ndarray:
    """Evaluate a metric over each prediction-history column."""
    return np.array([fn(y_true, y_pred_hist[:, frame]) for frame in range(y_pred_hist.shape[1])], dtype=float)


def _f1_by_frame(
    y_true: np.ndarray,
    y_pred_hist: np.ndarray,
    *,
    classes: np.ndarray,
    is_multiclass: bool,
) -> np.ndarray:
    """Evaluate F1 over each prediction-history column."""
    average = "macro" if is_multiclass else "binary"
    kwargs = {} if is_multiclass else {"pos_label": np.asarray(classes)[1]}
    return np.array(
        [
            f1_score(y_true, y_pred_hist[:, frame], average=average, zero_division=0, **kwargs)
            for frame in range(y_pred_hist.shape[1])
        ],
        dtype=float,
    )


def _select_metrics(
    builtins: Mapping[str, tuple[str, np.ndarray]],
    metric_config: Any,
    y_true: np.ndarray,
    y_pred_hist: np.ndarray,
    *,
    max_metrics: int,
) -> dict[str, np.ndarray]:
    """Select built-in metrics and append custom callable metric histories."""
    selected: dict[str, np.ndarray] = {}

    if metric_config is None:
        requested = _default_metric_keys(builtins)
        custom: Mapping[str, MetricFn] = {}
    elif isinstance(metric_config, Mapping):
        requested = _default_metric_keys(builtins)
        custom = metric_config
    elif isinstance(metric_config, str):
        requested = [metric_config]
        custom = {}
    elif isinstance(metric_config, Sequence):
        requested = [str(item) for item in metric_config]
        custom = {}
    else:
        raise TypeError("metrics must be None, a metric-name sequence, or a dict of callables.")

    for metric_key in requested:
        normalized = _normalize_metric_name(metric_key)
        if normalized in builtins:
            label, values = builtins[normalized]
            selected[label] = values

    for label, fn in custom.items():
        selected[str(label)] = _metric_by_frame(y_true, y_pred_hist, fn)

    if len(selected) > max_metrics:
        raise ValueError(
            f"The current figure format can display at most {max_metrics} metrics; "
            f"the configuration produced {len(selected)}. Select fewer metrics or custom callables."
        )
    return selected


def _default_metric_keys(builtins: Mapping[str, tuple[str, np.ndarray]]) -> list[str]:
    """Return unique built-in metric keys in declaration order."""
    keys: list[str] = []
    labels_seen: set[str] = set()
    for key, (label, _) in builtins.items():
        if label not in labels_seen:
            keys.append(key)
            labels_seen.add(label)
    return keys


def _normalize_metric_name(name: str) -> str:
    """Normalize user-facing metric names to built-in metric keys."""
    normalized = name.lower().replace("-", "_").replace("²", "2")
    aliases = {
        "logloss": "log_loss",
        "log_loss": "log_loss",
        "r_2": "r2",
        "r2": "r2",
        "f1_score": "f1",
    }
    return aliases.get(normalized, normalized)
