"""Training, weight, and activation Plotly builders for neural networks."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._style import NEURAL_COLORS, neural_layout


def _history_array(history: Dict[str, Any], key: str) -> np.ndarray:
    values = np.asarray(history.get(key, []), dtype=float)
    if values.size == 0:
        raise ValueError(f"History does not contain '{key}'.")
    return values


def _animation_controls(steps: np.ndarray, frame_duration: int) -> Tuple[list, list]:
    controls = [
        {
            "type": "buttons",
            "direction": "left",
            "x": 0.0,
            "y": 1.16,
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": frame_duration, "redraw": True}, "fromcurrent": True}],
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}],
                },
            ],
        }
    ]
    slider_steps = [
        {
            "label": str(step),
            "method": "animate",
            "args": [[str(index)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
        }
        for index, step in enumerate(steps)
    ]
    sliders = [{"active": 0, "currentvalue": {"prefix": "Step: "}, "pad": {"t": 45}, "steps": slider_steps}]
    return controls, sliders


def _finite_range(values: Iterable[np.ndarray], padding: float = 0.08) -> list[float] | None:
    valid = [np.asarray(value, dtype=float).ravel() for value in values]
    combined = np.concatenate(valid) if valid else np.array([])
    combined = combined[np.isfinite(combined)]
    if not combined.size:
        return None
    low, high = float(combined.min()), float(combined.max())
    span = max(high - low, 1e-9)
    return [low - span * padding, high + span * padding]


def build_nn_training_figure(
    history: Dict[str, Any],
    *,
    title: str | None = None,
    frame_duration: int = 120,
    max_norm_series: int = 6,
) -> go.Figure:
    """Animate loss and parameter norms captured by ``TorchTrainingRecorder``."""
    steps = _history_array(history, "steps")
    loss = np.asarray(history.get("loss", np.full(steps.size, np.nan)), dtype=float)
    norms = history.get("parameter_norms", {})
    if not np.isfinite(loss).any() and not norms:
        raise ValueError("History needs a loss or parameter norms to visualize training.")
    if title is None:
        title = "Training dynamics"
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Loss", "Parameter L2 norm"),
        horizontal_spacing=0.12,
    )
    series = list(norms.items())[:max_norm_series]
    fig.add_trace(
        go.Scatter(x=steps[:1], y=loss[:1], mode="lines+markers", name="loss", line={"color": NEURAL_COLORS["output"], "width": 3}),
        row=1,
        col=1,
    )
    palette = [NEURAL_COLORS["linear"], NEURAL_COLORS["input"], NEURAL_COLORS["activation"], NEURAL_COLORS["regularization"]]
    for index, (name, values) in enumerate(series):
        fig.add_trace(
            go.Scatter(
                x=steps[:1], y=np.asarray(values)[:1], mode="lines", name=name,
                line={"color": palette[index % len(palette)], "width": 2},
            ),
            row=1,
            col=2,
        )
    frames = []
    for frame_index in range(steps.size):
        frame_data = [go.Scatter(x=steps[: frame_index + 1], y=loss[: frame_index + 1])]
        frame_data.extend(
            go.Scatter(x=steps[: frame_index + 1], y=np.asarray(values)[: frame_index + 1])
            for _, values in series
        )
        frames.append(go.Frame(name=str(frame_index), data=frame_data, traces=list(range(len(frame_data)))))
    fig.frames = frames
    controls, sliders = _animation_controls(steps, frame_duration)
    fig.update_layout(**neural_layout(title, height=590), updatemenus=controls, sliders=sliders, legend={"y": -0.26, "orientation": "h"})
    fig.update_xaxes(title_text="Training step", gridcolor=NEURAL_COLORS["grid"])
    fig.update_yaxes(gridcolor=NEURAL_COLORS["grid"])
    return fig


def _matrix_for_display(matrix: np.ndarray, max_rows: int, max_cols: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        return matrix.reshape(1, -1)
    row_indices = np.linspace(0, matrix.shape[0] - 1, min(max_rows, matrix.shape[0]), dtype=int)
    col_indices = np.linspace(0, matrix.shape[1] - 1, min(max_cols, matrix.shape[1]), dtype=int)
    return matrix[np.ix_(row_indices, col_indices)]


def build_nn_weight_figure(
    history: Dict[str, Any],
    *,
    parameter: str | None = None,
    title: str | None = None,
    frame_duration: int = 120,
    max_rows: int = 32,
    max_cols: int = 32,
) -> go.Figure:
    """Animate a weight or gradient tensor as a stable, diverging heatmap."""
    tensors = history.get("parameters", {})
    if parameter is None:
        parameter = next((name for name, values in tensors.items() if values and np.asarray(values[0]).ndim >= 2), None)
    if parameter is None or parameter not in tensors or not tensors[parameter]:
        raise ValueError("No captured matrix parameter is available. Increase max_tensor_elements if needed.")
    snapshots = [_matrix_for_display(value, max_rows, max_cols) for value in tensors[parameter]]
    steps = _history_array(history, "steps")[: len(snapshots)]
    scale = max(float(np.max(np.abs(value))) for value in snapshots) or 1.0
    if title is None:
        title = f"Weight evolution: {parameter}"
    fig = go.Figure(
        go.Heatmap(
            z=snapshots[0],
            colorscale=[[0, NEURAL_COLORS["negative"]], [0.5, NEURAL_COLORS["background"]], [1, NEURAL_COLORS["positive"]]],
            zmid=0,
            zmin=-scale,
            zmax=scale,
            colorbar={"title": "weight"},
            hovertemplate="output %{y}, input %{x}<br>weight=%{z:.4f}<extra></extra>",
        )
    )
    fig.frames = [go.Frame(name=str(index), data=[go.Heatmap(z=value)]) for index, value in enumerate(snapshots)]
    controls, sliders = _animation_controls(steps, frame_duration)
    fig.update_layout(**neural_layout(title, height=620), updatemenus=controls, sliders=sliders)
    fig.update_xaxes(title_text="Input feature")
    fig.update_yaxes(title_text="Output neuron", autorange="reversed")
    return fig


def build_nn_activation_figure(history: Dict[str, Any], *, title: str | None = None) -> go.Figure:
    """Plot mean activation and dispersion for every recorded leaf layer."""
    steps = _history_array(history, "steps")
    activations = history.get("activations", {})
    if not activations:
        raise ValueError("No activation summaries were captured. Enable capture_activations in the recorder.")
    if title is None:
        title = "Activation flow"
    fig = go.Figure()
    for index, (name, statistics) in enumerate(activations.items()):
        mean = np.asarray(statistics["mean"], dtype=float)
        std = np.asarray(statistics["std"], dtype=float)
        color = [NEURAL_COLORS["linear"], NEURAL_COLORS["input"], NEURAL_COLORS["activation"], NEURAL_COLORS["regularization"]][index % 4]
        fig.add_trace(
            go.Scatter(
                x=steps[: mean.size], y=mean, mode="lines", name=name,
                line={"color": color, "width": 2},
                error_y={"type": "data", "array": std, "visible": True, "color": color, "thickness": 1},
            )
        )
    fig.update_layout(**neural_layout(title, height=560), legend={"y": -0.25, "orientation": "h"})
    fig.update_xaxes(title_text="Training step", gridcolor=NEURAL_COLORS["grid"])
    fig.update_yaxes(title_text="Mean activation (+/- std)", gridcolor=NEURAL_COLORS["grid"])
    return fig
