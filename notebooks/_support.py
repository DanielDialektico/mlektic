"""Small deterministic datasets and fitted models used by project notebooks.

This module intentionally contains no assertions.  Unit tests validate machine
invariants; the notebooks that import these helpers are for human inspection of
real Plotly figures and for guided learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

SEED = 17


@dataclass(frozen=True)
class Case:
    """A fitted estimator and the data used to fit it."""

    model: Any
    X: np.ndarray
    y: np.ndarray


def linear_case(dimensions: int = 1, *, estimator: str = "closed_form", noise: float = 0.18) -> Case:
    """Return a deterministic linear-regression case."""
    rng = np.random.default_rng(SEED + dimensions)
    X = rng.normal(size=(72, dimensions))
    theta = np.linspace(2.2, -0.6, dimensions)
    y = 1.25 + X @ theta + rng.normal(scale=noise, size=len(X))
    if estimator == "sgd":
        model = SGDRegressor(max_iter=80, tol=1e-5, random_state=SEED, learning_rate="invscaling")
    elif estimator == "closed_form":
        model = LinearRegression()
    else:
        raise ValueError("estimator must be 'closed_form' or 'sgd'.")
    return Case(model.fit(X, y), X, y)


def scaled_linear_case(dimensions: int = 2) -> Case:
    """Return a fitted affine preprocessing pipeline."""
    case = linear_case(dimensions)
    model = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]).fit(case.X, case.y)
    return Case(model, case.X, case.y)


def polynomial_linear_case() -> Case:
    """Return a model linear in transformed coefficients and curved in raw x."""
    X = np.linspace(-2.5, 2.5, 72).reshape(-1, 1)
    y = 1.0 - 0.7 * X[:, 0] + 1.6 * X[:, 0] ** 2
    model = Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)), ("model", LinearRegression())]).fit(
        X, y
    )
    return Case(model, X, y)


def binary_case(
    dimensions: int = 1,
    *,
    estimator: str = "logistic",
    string_labels: bool = False,
    imbalanced: bool = False,
) -> Case:
    """Return a deterministic binary-classification case."""
    if dimensions == 1:
        X = np.linspace(-3.2, 3.2, 90).reshape(-1, 1)
        score = 1.5 * X[:, 0] - 0.2
        y = (score + 0.35 * np.sin(3 * X[:, 0]) > 0).astype(int)
    else:
        weights = np.linspace(1.6, -0.7, dimensions)
        rng = np.random.default_rng(SEED + 20 + dimensions)
        X = rng.normal(size=(100, dimensions))
        y = (X @ weights + rng.normal(scale=0.35, size=len(X)) > 0).astype(int)
    if imbalanced:
        keep = (y == 0) | (np.arange(len(y)) % 4 == 0)
        X, y = X[keep], y[keep]
    if string_labels:
        y = np.where(y == 1, "accepted", "rejected")
    if estimator == "sgd":
        model = SGDClassifier(loss="log_loss", max_iter=120, tol=1e-5, random_state=SEED)
    elif estimator == "logistic":
        model = LogisticRegression(max_iter=1000, random_state=SEED)
    else:
        raise ValueError("estimator must be 'logistic' or 'sgd'.")
    return Case(model.fit(X, y), X, y)


def scaled_binary_case(dimensions: int = 2) -> Case:
    """Return a fitted scaled logistic pipeline."""
    case = binary_case(dimensions)
    model = Pipeline(
        [("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=SEED))]
    ).fit(case.X, case.y)
    return Case(model, case.X, case.y)


def multiclass_case(dimensions: int = 2, *, classes: int = 3, string_labels: bool = False) -> Case:
    """Return a deterministic multiclass case with informative geometry."""
    if dimensions == 1:
        X = np.linspace(-4.0, 4.0, 120).reshape(-1, 1)
        edges = np.linspace(-4.0, 4.0, classes + 1)[1:-1]
        y = np.digitize(X[:, 0], edges)
    else:
        X, y = make_classification(
            n_samples=120,
            n_features=dimensions,
            n_informative=max(2, min(dimensions, 4)),
            n_redundant=0,
            n_classes=classes,
            n_clusters_per_class=1,
            class_sep=1.4,
            random_state=SEED + dimensions + classes,
        )
    if string_labels:
        names = np.asarray([f"group-{index}" for index in range(classes)])
        y = names[y]
    model = LogisticRegression(max_iter=1200, random_state=SEED).fit(X, y)
    return Case(model, np.asarray(X), np.asarray(y))


def torch_xor_case(*, activation: str = "tanh", optimizer_name: str = "sgd", steps: int = 8):
    """Train a tiny XOR network and return model, inputs, and recorded history."""
    import torch

    from mlektic import TorchTrainingRecorder

    torch.manual_seed(SEED)
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    activation_layer = {"tanh": torch.nn.Tanh(), "relu": torch.nn.ReLU(), "sigmoid": torch.nn.Sigmoid()}[activation]
    model = torch.nn.Sequential(torch.nn.Linear(2, 4), activation_layer, torch.nn.Linear(4, 1), torch.nn.Sigmoid())
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.25, momentum=0.1)
    criterion = torch.nn.BCELoss()
    recorder = TorchTrainingRecorder(model, optimizer=optimizer, loss_fn=criterion)
    for step in range(steps):
        optimizer.zero_grad()
        prediction = model(X)
        loss = criterion(prediction, y)
        loss.backward()
        optimizer.step()
        recorder.record(step + 1, loss=loss, predictions=prediction, targets=y, task="classification")
    recorder.close()
    return model, X, recorder.to_history()


def case_heading(case_id: str, purpose: str) -> None:
    """Print a compact human-QA marker above a displayed figure."""
    print(f"CASE {case_id} | {purpose}")
