"""Lightweight metric inference for neural-network training histories."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..utils.math import r2_finite


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _looks_like_classification(predictions: np.ndarray, targets: np.ndarray) -> bool:
    if predictions.ndim > 1 and predictions.shape[-1] > 1:
        return True
    if targets.dtype.kind in {"b", "i", "u"}:
        return True
    unique_targets = np.unique(targets)
    return unique_targets.size <= 2 and np.all(np.isin(unique_targets, [0.0, 1.0]))


# Criteria whose inputs are logits (decide at 0) or probabilities (decide at 0.5).
_LOGIT_CRITERIA = {"BCEWithLogitsLoss", "CrossEntropyLoss", "MultiLabelSoftMarginLoss", "SoftMarginLoss"}
_PROBABILITY_CRITERIA = {"BCELoss", "NLLLoss"}


def output_kind(loss_fn: Any) -> str | None:
    """Return ``"logits"`` or ``"probabilities"`` from a PyTorch criterion, if known."""
    name = type(loss_fn).__name__ if loss_fn is not None else ""
    if name in _LOGIT_CRITERIA:
        return "logits"
    if name in _PROBABILITY_CRITERIA:
        return "probabilities"
    return None


def _classification_labels(predictions: np.ndarray, targets: np.ndarray, output: str | None = None):
    if targets.ndim > 1 and targets.shape[-1] > 1:
        true_labels = np.argmax(targets, axis=-1)
    else:
        true_labels = targets.ravel()
    if predictions.ndim > 1 and predictions.shape[-1] > 1:
        predicted_labels = np.argmax(predictions, axis=-1)
    else:
        scores = predictions.ravel()
        if output == "logits":
            threshold = 0.0
        elif output == "probabilities":
            threshold = 0.5
        else:
            # Unknown criterion: values outside [0, 1] can only be logits.
            # A declared criterion is preferred because logits may also fall in [0, 1].
            threshold = 0.5 if np.all((scores >= 0.0) & (scores <= 1.0)) else 0.0
        predicted_labels = (scores >= threshold).astype(int)
    return true_labels, predicted_labels


def _classification_metrics(
    predictions: np.ndarray, targets: np.ndarray, output: str | None = None
) -> Dict[str, float]:
    true_labels, predicted_labels = _classification_labels(predictions, targets, output)
    labels = np.unique(np.concatenate([true_labels, predicted_labels]))
    index = {label: position for position, label in enumerate(labels.tolist())}
    confusion = np.zeros((labels.size, labels.size), dtype=float)
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        confusion[index[true_label], index[predicted_label]] += 1.0
    diagonal = np.diag(confusion)
    precision = diagonal / np.maximum(confusion.sum(axis=0), 1e-12)
    recall = diagonal / np.maximum(confusion.sum(axis=1), 1e-12)
    return {
        "accuracy": float(np.mean(true_labels == predicted_labels)),
        "precision": float(np.mean(precision)),
        "recall": float(np.mean(recall)),
    }


def _regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    predicted_values = predictions.ravel().astype(float)
    true_values = targets.ravel().astype(float)
    residual = true_values - predicted_values
    mse = float(np.mean(residual**2))
    mae = float(np.mean(np.abs(residual)))
    denominator = float(np.sum((true_values - np.mean(true_values)) ** 2))
    r2 = float(r2_finite(float(np.sum(residual**2)), denominator))
    return {"mse": mse, "mae": mae, "r2": r2}


def infer_performance_metrics(
    predictions: Any,
    targets: Any,
    *,
    task: str = "auto",
    output: str | None = None,
) -> Dict[str, float]:
    """Infer three compact classification or regression metrics.

    ``output`` declares whether binary scores are ``"logits"`` or
    ``"probabilities"``; when omitted it is inferred from their range.
    """
    if task not in {"auto", "classification", "regression"}:
        raise ValueError("task must be 'auto', 'classification', or 'regression'.")
    prediction_values = _as_numpy(predictions)
    target_values = _as_numpy(targets)
    classification = task == "classification" or (
        task == "auto" and _looks_like_classification(prediction_values, target_values)
    )
    if classification:
        return _classification_metrics(prediction_values, target_values, output)
    return _regression_metrics(prediction_values, target_values)


__all__ = ["infer_performance_metrics"]
