"""Probability links used by logistic-regression visualizations."""

from __future__ import annotations

import numpy as np

from .math import _sigmoid, _softmax

MULTICLASS_LINKS = {"softmax", "ovr"}


def multiclass_probabilities(scores: np.ndarray, link: str) -> np.ndarray:
    """Map class scores to probabilities with the selected multiclass link."""
    values = np.asarray(scores, dtype=float)
    one_dimensional = values.ndim == 1
    values = np.atleast_2d(values)
    if link == "softmax":
        probabilities = _softmax(values)
    elif link == "ovr":
        independent = _sigmoid(values)
        probabilities = independent / np.maximum(independent.sum(axis=1, keepdims=True), 1e-15)
    else:
        raise ValueError("multiclass_link must be 'auto', 'softmax', or 'ovr'.")
    return probabilities[0] if one_dimensional else probabilities


def infer_multiclass_link(scores: np.ndarray, probabilities: np.ndarray) -> str:
    """Infer Softmax versus normalized one-vs-rest sigmoids from model outputs."""
    scores = np.asarray(scores, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if scores.ndim != 2 or probabilities.shape != scores.shape:
        return "softmax"
    errors = {
        link: float(np.mean(np.abs(multiclass_probabilities(scores, link) - probabilities)))
        for link in sorted(MULTICLASS_LINKS)
    }
    return min(errors, key=errors.get)


def multiclass_link_latex(link: str, classes: int | str = "K") -> str:
    """Return a compact rigorous LaTeX definition for a multiclass link."""
    if link == "ovr":
        return (
            r"q_k=\sigma(z_k)=\frac{1}{1+e^{-z_k}},\qquad "
            rf"\hat{{p}}_k=\frac{{q_k}}{{\sum_{{j=1}}^{{{classes}}}q_j}}"
        )
    return rf"\hat{{p}}_k=\frac{{e^{{z_k}}}}{{\sum_{{j=1}}^{{{classes}}}e^{{z_j}}}}"


def multiclass_link_name(link: str) -> str:
    """Return the pedagogical display name of a multiclass probability link."""
    return "normalized OvR sigmoids" if link == "ovr" else "Softmax"


__all__ = [
    "MULTICLASS_LINKS",
    "infer_multiclass_link",
    "multiclass_link_latex",
    "multiclass_link_name",
    "multiclass_probabilities",
]
