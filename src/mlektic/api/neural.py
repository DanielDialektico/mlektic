"""Public API for PyTorch neural-network visualizations."""

from __future__ import annotations

from typing import Any, Dict

from ..neural.recorder import TorchTrainingRecorder
from ..visualization.neural.architecture import build_nn_architecture_figure
from ..visualization.neural.math_view import build_nn_prediction_figure
from ..visualization.neural.training import (
    build_nn_activation_figure,
    build_nn_training_figure,
    build_nn_weight_figure,
)


def visualize_nn(
    model: Any,
    input_sample: Any | None = None,
    *,
    history: Dict[str, Any] | None = None,
    view: str = "architecture",
    title: str | None = None,
    max_neurons: int = 10,
    frame_duration: int = 120,
):
    """Visualize a PyTorch model architecture, training dynamics, or activations.

    Args:
        model: A ``torch.nn.Module``.
        input_sample: One sample or a batch used to infer layer output shapes.
        history: Payload returned by :meth:`TorchTrainingRecorder.to_history`.
        view: One of ``"architecture"``, ``"training"``, or ``"activations"``.
        title: Optional figure title.
        max_neurons: Maximum individual nodes drawn per layer in the architecture view.
        frame_duration: Milliseconds per animation frame in the training view.
    """
    if view == "architecture":
        return build_nn_architecture_figure(model, input_sample, title=title, max_neurons=max_neurons)
    if history is None:
        raise ValueError(f"view='{view}' requires a history from TorchTrainingRecorder.")
    if view == "training":
        return build_nn_training_figure(history, title=title, frame_duration=frame_duration)
    if view == "activations":
        return build_nn_activation_figure(history, title=title)
    raise ValueError("view must be 'architecture', 'training', or 'activations'.")


def visualize_nn_training(history: Dict[str, Any], *, title: str | None = None, frame_duration: int = 120):
    """Animate loss and parameter norms from a PyTorch training history."""
    return build_nn_training_figure(history, title=title, frame_duration=frame_duration)


def visualize_nn_weights(
    history: Dict[str, Any],
    *,
    parameter: str | None = None,
    title: str | None = None,
    frame_duration: int = 120,
):
    """Animate a captured matrix of weights as a diverging heatmap."""
    return build_nn_weight_figure(history, parameter=parameter, title=title, frame_duration=frame_duration)


def explain_nn_prediction(
    model: Any,
    x_query: Any,
    *,
    title: str | None = None,
    dec: int = 4,
    max_layers_math: int = 6,
    max_neurons_math: int = 8,
):
    """Explain a small PyTorch forward pass using formula composition and values."""
    return build_nn_prediction_figure(
        model,
        x_query,
        title=title,
        dec=dec,
        max_layers_math=max_layers_math,
        max_neurons_math=max_neurons_math,
    )


__all__ = [
    "TorchTrainingRecorder",
    "explain_nn_prediction",
    "visualize_nn",
    "visualize_nn_training",
    "visualize_nn_weights",
]
