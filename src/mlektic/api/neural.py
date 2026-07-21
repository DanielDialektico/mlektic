"""Public API for PyTorch neural-network visualizations."""

from __future__ import annotations

from typing import Any, Dict

from ..neural.recorder import TorchTrainingRecorder
from ..neural.report import build_nn_math_report, display_nn_math_report, export_nn_math_report
from ..visualization.neural.architecture import build_nn_architecture_figure
from ..visualization.neural.graph import build_nn_graph_figure
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
    max_frames: int | None = 20,
    frame_duration: int = 120,
    node_color_mode: str = "value",
    edge_color_mode: str = "weight",
):
    """Visualize PyTorch architecture, graph mathematics, training, or activations.

    Args:
        model: A ``torch.nn.Module``.
        input_sample: One sample or a batch used to infer layer output shapes.
        history: Payload returned by :meth:`TorchTrainingRecorder.to_history`.
        view: One of ``"architecture"``, ``"graph"``, ``"training"``,
            ``"weights"``, or ``"activations"``.
        title: Optional figure title.
        max_neurons: Maximum individual nodes drawn per layer in the architecture view.
        max_frames: Maximum recorded training steps retained by animated views.
        frame_duration: Milliseconds per animation frame in the training view.
        node_color_mode: ``"value"`` for exact globally scaled outputs or
            ``"relative"`` for per-layer contrast.
        edge_color_mode: ``"weight"`` for globally scaled parameters or
            ``"signal"`` for ``w_ji * a_i``.

    Returns:
        A Plotly figure for the selected view.
    """
    if view == "architecture":
        return build_nn_architecture_figure(
            model,
            input_sample,
            history=history,
            title=title,
            max_neurons=max_neurons,
        )
    if history is None:
        raise ValueError(f"view='{view}' requires a history from TorchTrainingRecorder.")
    if view == "graph":
        if input_sample is None:
            raise ValueError("view='graph' requires input_sample for node activations.")
        return build_nn_graph_figure(
            model,
            input_sample,
            history,
            title=title,
            max_neurons=max_neurons,
            max_frames=max_frames,
            frame_duration=frame_duration,
            node_color_mode=node_color_mode,
            edge_color_mode=edge_color_mode,
        )
    if view == "training":
        return build_nn_training_figure(
            history,
            title=title,
            frame_duration=frame_duration,
            max_frames=max_frames,
        )
    if view == "weights":
        return build_nn_weight_figure(
            history,
            title=title,
            frame_duration=frame_duration,
            max_frames=max_frames,
        )
    if view == "activations":
        return build_nn_activation_figure(
            model,
            history,
            input_sample=input_sample,
            title=title,
            frame_duration=frame_duration,
            max_frames=max_frames,
        )
    raise ValueError("view must be 'architecture', 'graph', 'training', 'weights', or 'activations'.")


def visualize_nn_architecture(
    model: Any,
    input_sample: Any | None = None,
    *,
    history: Dict[str, Any] | None = None,
    title: str | None = None,
    max_layers: int = 8,
):
    """Show layer roles, formulas, tensor dimensions, and hyperparameters.

    Args:
        model: A ``torch.nn.Module``.
        input_sample: Sample or batch used to infer tensor dimensions.
        history: Optional recorder payload used to enrich the architecture.
        title: Optional figure title.
        max_layers: Maximum number of leaf layers rendered individually.

    Returns:
        A static Plotly architecture figure.
    """
    return build_nn_architecture_figure(
        model,
        input_sample,
        history=history,
        title=title,
        max_layers=max_layers,
    )


def visualize_nn_graph(
    model: Any,
    input_sample: Any,
    history: Dict[str, Any],
    *,
    title: str | None = None,
    max_neurons: int = 8,
    max_frames: int | None = 20,
    frame_duration: int = 180,
    node_color_mode: str = "value",
    edge_color_mode: str = "weight",
):
    """Animate node outputs, edge values, and backpropagation gradients.

    The default modes use exact values and one global color scale per quantity
    across every retained frame. Node and edge scales intentionally differ:
    nodes encode activations while edges encode parameters. ``"relative"``
    increases per-layer node contrast, and ``"signal"`` colors each edge by
    ``w_ji * a_i`` instead of by its weight.

    Args:
        model: A ``torch.nn.Module``.
        input_sample: Sample or batch used to compute node activations.
        history: Payload returned by :meth:`TorchTrainingRecorder.to_history`.
        title: Optional figure title.
        max_neurons: Maximum visible nodes per layer.
        max_frames: Maximum retained animation frames, or ``None`` for all.
        frame_duration: Milliseconds per animation frame.
        node_color_mode: ``"value"`` or ``"relative"``.
        edge_color_mode: ``"weight"`` or ``"signal"``.

    Returns:
        An animated Plotly graph figure.
    """
    return build_nn_graph_figure(
        model,
        input_sample,
        history,
        title=title,
        max_neurons=max_neurons,
        max_frames=max_frames,
        frame_duration=frame_duration,
        node_color_mode=node_color_mode,
        edge_color_mode=edge_color_mode,
    )


def visualize_nn_training(
    history: Dict[str, Any],
    *,
    title: str | None = None,
    frame_duration: int = 120,
    max_metrics: int = 3,
    max_frames: int | None = 30,
):
    """Animate a compact 2x2 panel with loss and up to three metrics.

    Metrics may be supplied explicitly to the recorder or inferred from
    predictions and targets during :meth:`TorchTrainingRecorder.record`.
    """
    return build_nn_training_figure(
        history,
        title=title,
        frame_duration=frame_duration,
        max_metrics=max_metrics,
        max_frames=max_frames,
    )


def visualize_nn_weights(
    history: Dict[str, Any],
    *,
    parameter: str | None = None,
    title: str | None = None,
    frame_duration: int = 120,
    max_rows: int = 4,
    max_cols: int = 5,
    max_parameters: int = 6,
    max_frames: int | None = 30,
):
    """Animate captured parameter tensors using truncated LaTeX matrices.

    ``max_rows``, ``max_cols`` and ``max_parameters`` bound the mathematical
    display without modifying the values stored in the recorder history.
    """
    return build_nn_weight_figure(
        history,
        parameter=parameter,
        title=title,
        frame_duration=frame_duration,
        max_rows=max_rows,
        max_cols=max_cols,
        max_parameters=max_parameters,
        max_frames=max_frames,
    )


def explain_nn_prediction(
    model: Any,
    x_query: Any,
    *,
    history: Dict[str, Any] | None = None,
    title: str | None = None,
    dec: int = 4,
    max_layers_math: int = 6,
    max_neurons_math: int = 8,
    max_frames: int | None = 12,
    frame_duration: int = 220,
):
    """Explain and optionally animate a PyTorch forward pass mathematically.

    When ``history`` is provided, substitutions use retained parameter snapshots
    and evolve over training. Large models are summarized with the configured
    layer, neuron and frame limits.
    """
    return build_nn_prediction_figure(
        model,
        x_query,
        history=history,
        title=title,
        dec=dec,
        max_layers_math=max_layers_math,
        max_neurons_math=max_neurons_math,
        max_frames=max_frames,
        frame_duration=frame_duration,
    )


__all__ = [
    "TorchTrainingRecorder",
    "build_nn_math_report",
    "display_nn_math_report",
    "explain_nn_prediction",
    "export_nn_math_report",
    "visualize_nn",
    "visualize_nn_architecture",
    "visualize_nn_graph",
    "visualize_nn_training",
    "visualize_nn_weights",
]
