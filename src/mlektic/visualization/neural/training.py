"""Training and parameter-mathematics Plotly builders for neural networks."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...neural.introspection import describe_torch_model
from ...neural.taxonomy import select_with_ellipsis, shape_tex
from ._style import NEURAL_COLORS, animation_button_style, neural_layout
from .math_format import matrix_latex, parameter_snapshot, vector_latex


def _history_array(history: Dict[str, Any], key: str) -> np.ndarray:
    values = np.asarray(history.get(key, []), dtype=float)
    if values.size == 0:
        raise ValueError(f"History does not contain '{key}'.")
    return values


def _frame_indices(frame_count: int, max_frames: int | None) -> np.ndarray:
    if max_frames is None or frame_count <= max_frames:
        return np.arange(frame_count, dtype=int)
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1 or None.")
    return np.unique(np.linspace(0, frame_count - 1, max_frames, dtype=int))


def _animation_controls(
    steps: np.ndarray,
    frame_duration: int,
    *,
    frame_names: Sequence[str] | None = None,
) -> Tuple[list, list]:
    names = list(frame_names) if frame_names is not None else [str(index) for index in range(steps.size)]
    controls = [
        {
            "type": "buttons",
            "direction": "left",
            "x": 0.34,
            "y": 1.08,
            **animation_button_style(),
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
            "args": [[name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
        }
        for step, name in zip(steps, names)
    ]
    sliders = [{"active": 0, "currentvalue": {"prefix": "Step: "}, "pad": {"t": 45}, "steps": slider_steps}]
    return controls, sliders


def _stable_range(values: Sequence[np.ndarray], padding: float = 0.08) -> list[float] | None:
    finite = [np.asarray(value, dtype=float).ravel() for value in values if np.asarray(value).size]
    combined = np.concatenate(finite) if finite else np.array([])
    combined = combined[np.isfinite(combined)]
    if not combined.size:
        return None
    low, high = float(combined.min()), float(combined.max())
    span = max(high - low, 1e-9)
    return [low - padding * span, high + padding * span]


def _metric_annotation(step: int, loss: float, metrics: Sequence[tuple[str, float]]) -> Dict[str, Any]:
    parts = [rf"t={step}", rf"\mathcal{{L}}={loss:.5f}"]
    parts.extend(rf"\mathrm{{{name}}}={value:.4f}" for name, value in metrics if np.isfinite(value))
    return {
        "x": 0.99,
        "y": 1.08,
        "xref": "paper",
        "yref": "paper",
        "text": "$" + r"\quad".join(parts) + "$",
        "showarrow": False,
        "xanchor": "right",
        "font": {"size": 13, "color": NEURAL_COLORS["text"]},
    }


def build_nn_training_figure(
    history: Dict[str, Any],
    *,
    title: str | None = None,
    frame_duration: int = 120,
    max_metrics: int = 3,
    max_frames: int | None = 30,
) -> go.Figure:
    """Animate loss above up to three independent performance-metric plots."""
    steps = _history_array(history, "steps").astype(int)
    loss = np.asarray(history.get("loss", np.full(steps.size, np.nan)), dtype=float)
    metrics = [
        (name, np.asarray(values, dtype=float))
        for name, values in history.get("metrics", {}).items()
        if np.asarray(values).size
    ][:max_metrics]
    if not np.isfinite(loss).any() and not metrics:
        raise ValueError("History needs a loss or at least one performance metric.")
    rows = 1 + len(metrics)
    subplot_titles = [r"Training objective: $\mathcal{L}(\theta_t)$"]
    subplot_titles.extend(name.replace("_", " ").title() for name, _ in metrics)
    if metrics:
        metric_height = 0.62 / len(metrics)
        row_heights = [0.38, *[metric_height] * len(metrics)]
    else:
        row_heights = [1.0]
    figure = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.055 if metrics else 0.08,
        row_heights=row_heights,
    )
    figure.add_trace(
        go.Scatter(
            x=steps,
            y=loss,
            mode="lines",
            line={"color": NEURAL_COLORS["output"], "width": 1.5},
            opacity=0.20,
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=steps[:1],
            y=loss[:1],
            mode="lines+markers",
            name=r"$\mathcal{L}$",
            line={"color": NEURAL_COLORS["output"], "width": 3},
            marker={"size": 6},
            hovertemplate="step=%{x}<br>loss=%{y:.6f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    palette = [
        NEURAL_COLORS["linear"],
        NEURAL_COLORS["input"],
        NEURAL_COLORS["activation"],
        NEURAL_COLORS["regularization"],
        NEURAL_COLORS["output"],
    ]
    for index, (name, values) in enumerate(metrics):
        metric_row = index + 2
        figure.add_trace(
            go.Scatter(
                x=steps,
                y=values,
                mode="lines",
                line={"color": palette[index % len(palette)], "width": 1.5},
                opacity=0.20,
                hoverinfo="skip",
                showlegend=False,
            ),
            row=metric_row,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=steps[:1],
                y=values[:1],
                mode="lines+markers",
                name=name,
                line={"color": palette[index % len(palette)], "width": 2.5},
                marker={"size": 5},
                hovertemplate=f"step=%{{x}}<br>{name}=%{{y:.5f}}<extra></extra>",
                showlegend=False,
            ),
            row=metric_row,
            col=1,
        )
    dynamic_trace_indices = [1, *[3 + 2 * index for index in range(len(metrics))]]
    selected_frames = _frame_indices(steps.size, max_frames)
    frames = []
    for frame_index in selected_frames:
        frame_data = [go.Scatter(x=steps[: frame_index + 1], y=loss[: frame_index + 1])]
        frame_data.extend(
            go.Scatter(x=steps[: frame_index + 1], y=values[: frame_index + 1]) for _, values in metrics
        )
        current_metrics = [
            (name, float(values[frame_index])) for name, values in metrics if frame_index < values.size
        ]
        frames.append(
            go.Frame(
                name=str(int(frame_index)),
                data=frame_data,
                traces=dynamic_trace_indices,
                layout=go.Layout(
                    annotations=[
                        *list(figure.layout.annotations),
                        _metric_annotation(int(steps[frame_index]), float(loss[frame_index]), current_metrics),
                    ]
                ),
            )
        )
    figure.frames = frames
    controls, sliders = _animation_controls(
        steps[selected_frames],
        frame_duration,
        frame_names=[str(int(index)) for index in selected_frames],
    )
    if title is None:
        title = "Learning performance"
    layout = neural_layout(title, height=530 + 145 * len(metrics))
    layout["margin"] = {"t": 110, "r": 35, "b": 100, "l": 75}
    first_metrics = [(name, float(values[0])) for name, values in metrics]
    figure.update_layout(
        **layout,
        updatemenus=controls,
        sliders=sliders,
        annotations=[
            *list(figure.layout.annotations),
            _metric_annotation(int(steps[0]), float(loss[0]), first_metrics),
        ],
        showlegend=False,
    )
    figure.update_xaxes(title_text="Training step", gridcolor=NEURAL_COLORS["grid"], row=rows, col=1)
    if steps.size > 1:
        figure.update_xaxes(range=[float(steps[0]), float(steps[-1])], row=rows, col=1)
    figure.update_yaxes(title_text=r"$\mathcal{L}$", gridcolor=NEURAL_COLORS["grid"], row=1, col=1)
    loss_range = _stable_range([loss])
    if loss_range:
        figure.update_yaxes(range=loss_range, row=1, col=1)
    if metrics:
        for metric_index, (name, values) in enumerate(metrics, start=2):
            figure.update_yaxes(
                title_text=name.replace("_", " "),
                gridcolor=NEURAL_COLORS["grid"],
                row=metric_index,
                col=1,
            )
            metric_range = _stable_range([values])
            if metric_range:
                figure.update_yaxes(range=metric_range, row=metric_index, col=1)
    return figure


def _parameter_symbol(name: str, array: np.ndarray, index: int) -> str:
    if name.endswith(".weight"):
        base = "W" if array.ndim <= 2 else "K"
        return rf"{base}^{{({index})}}_t"
    if name.endswith(".bias"):
        return rf"\mathbf{{b}}^{{({index})}}_t"
    return rf"\theta^{{({index})}}_{{\mathrm{{{name.replace('.', '_')}}},t}}"


def _weight_annotations(
    history: Dict[str, Any],
    frame_index: int,
    selected_names: Sequence[str | None],
    step: int,
    dec: int,
    max_rows: int,
    max_cols: int,
) -> List[Dict[str, Any]]:
    snapshot = parameter_snapshot(history, frame_index)
    annotations: List[Dict[str, Any]] = [
        {
            "x": 0.5,
            "y": 1.10,
            "xref": "paper",
            "yref": "paper",
            "text": (
                r"$\mathbf{z}^{(\ell)}_t=W^{(\ell)}_t\mathbf{a}^{(\ell-1)}_t+\mathbf{b}^{(\ell)}_t,\quad "
                r"W^{(\ell)}_t\in\mathbb{R}^{d_\ell\times d_{\ell-1}},\quad "
                r"\mathbf{b}^{(\ell)}_t\in\mathbb{R}^{d_\ell}$"
            ),
            "showarrow": False,
            "font": {"size": 16, "color": NEURAL_COLORS["text"]},
        },
        {
            "x": 0.5,
            "y": 1.02,
            "xref": "paper",
            "yref": "paper",
            "text": rf"$t={step}\quad\text{{Each row of }}W^{{(\ell)}}\text{{ produces one output pre-activation }}z_j^{{(\ell)}}.$",
            "showarrow": False,
            "font": {"size": 13, "color": NEURAL_COLORS["muted"]},
        },
    ]
    y_positions = np.linspace(0.88, 0.13, max(len(selected_names), 1))
    layer_number = 0
    for y_position, name in zip(y_positions, selected_names):
        if name is None:
            annotations.append(
                {
                    "x": 0.5,
                    "y": y_position,
                    "xref": "paper",
                    "yref": "paper",
                    "text": r"$\vdots$",
                    "showarrow": False,
                    "font": {"size": 24, "color": NEURAL_COLORS["muted"]},
                }
            )
            continue
        if name not in snapshot:
            continue
        array = snapshot[name]
        if name.endswith(".weight"):
            layer_number += 1
        effective_layer = max(layer_number, 1)
        symbol = _parameter_symbol(name, array, effective_layer)
        dimensions = shape_tex(array.shape, drop_batch=False)
        matrix = matrix_latex(array, dec=dec, max_rows=max_rows, max_cols=max_cols)
        role = (
            r"\text{rows }\leftrightarrow\text{ output units; columns }\leftrightarrow\text{ input coordinates}"
            if array.ndim >= 2
            else r"\text{one additive offset per output unit}"
        )
        annotations.append(
            {
                "x": 0.5,
                "y": y_position,
                "xref": "paper",
                "yref": "paper",
                "text": rf"${symbol}={matrix}\in\mathbb{{R}}^{{{dimensions}}}\quad {role}$",
                "showarrow": False,
                "font": {"size": 13, "color": NEURAL_COLORS["text"]},
            }
        )
    return annotations


def build_nn_weight_figure(
    history: Dict[str, Any],
    *,
    parameter: str | None = None,
    title: str | None = None,
    frame_duration: int = 150,
    max_rows: int = 4,
    max_cols: int = 5,
    max_parameters: int = 6,
    max_frames: int | None = 30,
) -> go.Figure:
    """Animate parameter tensors entirely as mathematical LaTeX definitions."""
    parameters = history.get("parameters", {})
    available = [name for name, values in parameters.items() if values]
    if parameter is not None:
        if parameter not in parameters or not parameters[parameter]:
            raise ValueError(f"Parameter '{parameter}' was not captured by the recorder.")
        available = [parameter]
    if not available:
        raise ValueError("No parameter tensors were captured. Increase max_tensor_elements if needed.")
    selected_names = select_with_ellipsis(available, max_parameters)
    steps = _history_array(history, "steps").astype(int)
    frame_count = min(steps.size, min(len(parameters[name]) for name in available))
    steps = steps[:frame_count]
    selected_frames = _frame_indices(frame_count, max_frames)
    figure = go.Figure(go.Scatter(x=[], y=[], hoverinfo="skip", showlegend=False))
    frame_names = [str(int(index)) for index in selected_frames]
    figure.frames = [
        go.Frame(
            name=str(frame_index),
            data=[go.Scatter(x=[], y=[])],
            traces=[0],
            layout=go.Layout(
                annotations=_weight_annotations(
                    history,
                    frame_index,
                    selected_names,
                    int(steps[frame_index]),
                    3,
                    max_rows,
                    max_cols,
                )
            ),
        )
        for frame_index in selected_frames
    ]
    controls, sliders = _animation_controls(
        steps[selected_frames],
        frame_duration,
        frame_names=frame_names,
    )
    if title is None:
        title = "Parameter evolution in mathematical notation"
    layout = neural_layout(title, height=max(650, 210 + 115 * len(selected_names)))
    layout["margin"] = {"t": 115, "r": 35, "b": 100, "l": 35}
    figure.update_layout(
        **layout,
        annotations=_weight_annotations(history, 0, selected_names, int(steps[0]), 3, max_rows, max_cols),
        updatemenus=controls,
        sliders=sliders,
        showlegend=False,
    )
    figure.update_xaxes(visible=False, range=[0, 1])
    figure.update_yaxes(visible=False, range=[0, 1])
    return figure


def _activation_annotations(
    layers: Sequence[Dict[str, Any]],
    history: Dict[str, Any],
    frame_index: int,
    step: int,
) -> List[Dict[str, Any]]:
    selected = select_with_ellipsis(layers, 6)
    annotations: List[Dict[str, Any]] = [
        {
            "x": 0.5,
            "y": 1.07,
            "xref": "paper",
            "yref": "paper",
            "text": rf"$t={step}\quad\mathbf{{a}}^{{(\ell)}}=\phi_\ell(\mathbf{{z}}^{{(\ell)}})$",
            "showarrow": False,
            "font": {"size": 17, "color": NEURAL_COLORS["text"]},
        }
    ]
    y_positions = np.linspace(0.88, 0.15, max(len(selected), 1))
    for y_position, layer in zip(y_positions, selected):
        if layer is None:
            text = r"$\vdots$"
        else:
            stats = history["activations"][layer["name"]]
            mean = float(stats["mean"][frame_index])
            std = float(stats["std"][frame_index])
            minimum = float(stats["min"][frame_index])
            maximum = float(stats["max"][frame_index])
            dimension = shape_tex(layer.get("output_shape"))
            vector_values = history.get("activation_vectors", {}).get(layer["name"], [])
            vector = (
                vector_latex(vector_values[frame_index], dec=3, limit=6)
                if frame_index < len(vector_values)
                else r"[\text{aggregated}]"
            )
            text = (
                rf"$\text{{{layer['name']} ({layer['type']})}}:\quad {layer['formula']}"
                rf"\quad\mathbf{{a}}_t={vector}\in\mathbb{{R}}^{{{dimension}}}"
                rf"\quad\mu={mean:.3f},\;\sigma={std:.3f},\;[\min,\max]=[{minimum:.3f},{maximum:.3f}]$"
            )
        annotations.append(
            {
                "x": 0.5,
                "y": y_position,
                "xref": "paper",
                "yref": "paper",
                "text": text,
                "showarrow": False,
                "font": {"size": 13, "color": NEURAL_COLORS["text"]},
            }
        )
    return annotations


def build_nn_activation_figure(
    model: Any,
    history: Dict[str, Any],
    *,
    input_sample: Any | None = None,
    title: str | None = None,
    frame_duration: int = 150,
    max_frames: int | None = 30,
) -> go.Figure:
    """Explain activation evolution as LaTeX definitions and compact statistics."""
    activation_names = set(history.get("activations", {}))
    layers = [layer for layer in describe_torch_model(model, input_sample) if layer["name"] in activation_names]
    if not layers:
        raise ValueError("No activation summaries were captured. Enable capture_activations in the recorder.")
    steps = _history_array(history, "steps").astype(int)
    frame_count = min(steps.size, min(len(history["activations"][layer["name"]]["mean"]) for layer in layers))
    steps = steps[:frame_count]
    selected_frames = _frame_indices(frame_count, max_frames)
    figure = go.Figure(go.Scatter(x=[], y=[], hoverinfo="skip", showlegend=False))
    figure.frames = [
        go.Frame(
            name=str(frame_index),
            data=[go.Scatter(x=[], y=[])],
            traces=[0],
            layout=go.Layout(
                annotations=_activation_annotations(layers, history, frame_index, int(steps[frame_index]))
            ),
        )
        for frame_index in selected_frames
    ]
    controls, sliders = _animation_controls(
        steps[selected_frames],
        frame_duration,
        frame_names=[str(int(index)) for index in selected_frames],
    )
    if title is None:
        title = "Activation mathematics"
    layout = neural_layout(title, height=max(610, 230 + 105 * min(len(layers), 6)))
    layout["margin"] = {"t": 105, "r": 30, "b": 100, "l": 30}
    figure.update_layout(
        **layout,
        annotations=_activation_annotations(layers, history, 0, int(steps[0])),
        updatemenus=controls,
        sliders=sliders,
        showlegend=False,
    )
    figure.update_xaxes(visible=False, range=[0, 1])
    figure.update_yaxes(visible=False, range=[0, 1])
    return figure


__all__ = ["build_nn_activation_figure", "build_nn_training_figure", "build_nn_weight_figure"]
