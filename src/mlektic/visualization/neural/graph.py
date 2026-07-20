"""Animated mathematical graph for small dense PyTorch networks."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go

from ...neural.taxonomy import composed_dense_function, dense_stages
from ._style import NEURAL_COLORS, neural_layout
from .math_format import (
    compact_parameter_line,
    display_indices,
    gradient_snapshot,
    parameter_snapshot,
    vector_latex,
)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _mix_color(value: float, scale: float) -> str:
    """Map a signed value to a neutral-coral-teal color without hiding zero."""
    neutral = np.array([75, 78, 86], dtype=float)
    positive = np.array([85, 214, 190], dtype=float)
    negative = np.array([255, 125, 142], dtype=float)
    ratio = min(abs(float(value)) / max(scale, 1e-12), 1.0)
    target = positive if value >= 0 else negative
    rgb = np.rint(neutral * (1 - ratio) + target * ratio).astype(int)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def _sample_indices(count: int, max_neurons: int) -> List[int]:
    return display_indices(count, max_neurons)


def _frame_indices(frame_count: int, max_frames: int | None) -> np.ndarray:
    if max_frames is None or frame_count <= max_frames:
        return np.arange(frame_count, dtype=int)
    return np.unique(np.linspace(0, frame_count - 1, max_frames, dtype=int))


def _activation_vector(history: Dict[str, Any], stage: Dict[str, Any], frame_index: int) -> np.ndarray:
    key = stage.get("activation_name") or stage["name"]
    vectors = history.get("activation_vectors", {}).get(key, [])
    if frame_index < len(vectors):
        return np.asarray(vectors[frame_index], dtype=float).ravel()
    statistics = history.get("activations", {}).get(key, {})
    means = statistics.get("mean", [])
    fallback = float(means[frame_index]) if frame_index < len(means) else 0.0
    return np.full(stage["out_features"], fallback, dtype=float)


def _stage_bias(stage: Dict[str, Any], parameters: Dict[str, np.ndarray]) -> np.ndarray:
    bias_name = stage.get("bias_name")
    if bias_name and bias_name in parameters:
        return np.asarray(parameters[bias_name], dtype=float).ravel()
    return np.zeros(stage["out_features"], dtype=float)


def _global_scales(history: Dict[str, Any], stages: Sequence[Dict[str, Any]]) -> Tuple[float, float, float]:
    weight_values: List[np.ndarray] = []
    gradient_values: List[np.ndarray] = []
    activation_values: List[np.ndarray] = []
    for stage in stages:
        weight_values.extend(np.asarray(value).ravel() for value in history["parameters"].get(stage["weight_name"], []))
        gradient_values.extend(np.asarray(value).ravel() for value in history["gradients"].get(stage["weight_name"], []))
        key = stage.get("activation_name") or stage["name"]
        activation_values.extend(
            np.asarray(value).ravel() for value in history.get("activation_vectors", {}).get(key, [])
        )

    def scale(values: Sequence[np.ndarray]) -> float:
        return max((float(np.max(np.abs(value))) for value in values if value.size), default=1.0) or 1.0

    return scale(weight_values), scale(gradient_values), scale(activation_values)


def _graph_geometry(stages: Sequence[Dict[str, Any]], max_neurons: int):
    dimensions = [stages[0]["in_features"], *[stage["out_features"] for stage in stages]]
    indices = [_sample_indices(dimension, max_neurons) for dimension in dimensions]
    x_positions = np.linspace(0.07, 0.93, len(dimensions))
    y_positions = [np.linspace(0.22, 0.80, len(column)).tolist() for column in indices]
    edges = []
    for stage_position, stage in enumerate(stages):
        for source_position, source_index in enumerate(indices[stage_position]):
            for target_position, target_index in enumerate(indices[stage_position + 1]):
                edges.append(
                    {
                        "stage": stage,
                        "source_index": source_index,
                        "target_index": target_index,
                        "x": [x_positions[stage_position], x_positions[stage_position + 1]],
                        "y": [y_positions[stage_position][source_position], y_positions[stage_position + 1][target_position]],
                    }
                )
    return dimensions, indices, x_positions, y_positions, edges


def _node_values(
    history: Dict[str, Any],
    stages: Sequence[Dict[str, Any]],
    input_values: np.ndarray,
    frame_index: int,
) -> List[np.ndarray]:
    return [input_values, *[_activation_vector(history, stage, frame_index) for stage in stages]]


def _node_deltas(
    stages: Sequence[Dict[str, Any]],
    current: Dict[str, np.ndarray],
    previous: Dict[str, np.ndarray],
    gradients: Dict[str, np.ndarray],
) -> List[np.ndarray]:
    columns: List[np.ndarray] = []
    first_stage = stages[0]
    first_weights = current[first_stage["weight_name"]]
    first_previous = previous.get(first_stage["weight_name"], first_weights)
    columns.append(np.mean(first_weights - first_previous, axis=0))
    for stage in stages:
        weights = current[stage["weight_name"]]
        if stage["weight_name"] in previous:
            change = weights - previous[stage["weight_name"]]
        else:
            change = -gradients.get(stage["weight_name"], np.zeros_like(weights))
        columns.append(np.mean(change, axis=1))
    return columns


def _phase_annotations(
    stages: Sequence[Dict[str, Any]],
    dimensions: Sequence[int],
    x_positions: Sequence[float],
    parameters: Dict[str, np.ndarray],
    step: int,
    phase: str,
    dec: int,
) -> List[Dict[str, Any]]:
    phase_tex = (
        r"\text{Feed forward: }\mathbf{z}^{(\ell)}=W^{(\ell)}\mathbf{a}^{(\ell-1)}+\mathbf{b}^{(\ell)},\;"
        r"\mathbf{a}^{(\ell)}=\phi_\ell(\mathbf{z}^{(\ell)})"
        if phase == "forward"
        else r"\text{Backpropagation: }\nabla_{W^{(\ell)}}\mathcal{L}=\frac{\partial\mathcal{L}}{\partial W^{(\ell)}},\;"
        r"\Delta W_t^{(\ell)}=W_t^{(\ell)}-W_{t-1}^{(\ell)}"
    )
    annotations: List[Dict[str, Any]] = [
        {
            "x": 0.5,
            "y": 1.18,
            "xref": "paper",
            "yref": "paper",
            "text": f"${composed_dense_function(stages)}$",
            "showarrow": False,
            "font": {"size": 17, "color": NEURAL_COLORS["text"]},
        },
        {
            "x": 0.5,
            "y": 1.09,
            "xref": "paper",
            "yref": "paper",
            "text": f"${compact_parameter_line(stages, parameters, dec=dec)}$",
            "showarrow": False,
            "font": {"size": 13, "color": NEURAL_COLORS["muted"]},
        },
        {
            "x": 0.02,
            "y": 1.01,
            "xref": "paper",
            "yref": "paper",
            "text": rf"$t={step}\quad {phase_tex}$",
            "showarrow": False,
            "xanchor": "left",
            "font": {"size": 13, "color": NEURAL_COLORS["text"]},
        },
        {
            "x": 0.98,
            "y": 1.01,
            "xref": "paper",
            "yref": "paper",
            "text": r"$\color{#55d6be}{+}\;\text{positive}\qquad\color{#ff7d8e}{-}\;\text{negative}$",
            "showarrow": False,
            "xanchor": "right",
            "font": {"size": 12, "color": NEURAL_COLORS["muted"]},
        },
    ]
    for column_index, (x_position, dimension) in enumerate(zip(x_positions, dimensions)):
        if column_index == 0:
            dimension_tex = rf"\mathbf{{a}}^{{(0)}}=\mathbf{{x}}\in\mathbb{{R}}^{{{dimension}}}"
        else:
            stage = stages[column_index - 1]
            dimension_tex = (
                rf"\mathbf{{a}}^{{({column_index})}}\in\mathbb{{R}}^{{{dimension}}}"
                rf",\;W^{{({column_index})}}\in\mathbb{{R}}^{{{dimension}\times {stage['in_features']}}}"
            )
        annotations.append(
            {
                "x": x_position,
                "y": 0.08,
                "xref": "x",
                "yref": "y",
                "text": f"${dimension_tex}$",
                "showarrow": False,
                "font": {"size": 12, "color": NEURAL_COLORS["muted"]},
            }
        )
    return annotations


def build_nn_graph_figure(
    model: Any,
    input_sample: Any,
    history: Dict[str, Any],
    *,
    title: str | None = None,
    max_neurons: int = 8,
    max_frames: int | None = 20,
    frame_duration: int = 180,
    dec: int = 3,
) -> go.Figure:
    """Animate dense-network weights, activations, gradients, and parameter changes."""
    stages = dense_stages(model)
    if not stages:
        raise ValueError("The animated graph currently requires at least one torch.nn.Linear layer.")
    if max_neurons < 2:
        raise ValueError("max_neurons must be at least 2.")
    missing = [stage["weight_name"] for stage in stages if stage["weight_name"] not in history.get("parameters", {})]
    if missing:
        raise ValueError(
            "The recorder did not retain these weight tensors: "
            + ", ".join(missing)
            + ". Increase max_tensor_elements for a mathematical graph."
        )
    steps = np.asarray(history.get("steps", []), dtype=int)
    if not steps.size:
        raise ValueError("History has no recorded steps.")
    selected_frames = _frame_indices(steps.size, max_frames)
    input_array = _as_numpy(input_sample)
    if input_array.ndim == 1:
        input_values = input_array
    else:
        input_values = np.mean(input_array.reshape(input_array.shape[0], -1), axis=0)
    dimensions, indices, x_positions, y_positions, edges = _graph_geometry(stages, max_neurons)
    weight_scale, gradient_scale, activation_scale = _global_scales(history, stages)

    trace_indices = list(range(len(edges) + len(dimensions)))

    def frame_data(frame_index: int, phase: str):
        parameters = parameter_snapshot(history, frame_index)
        gradients = gradient_snapshot(history, frame_index)
        previous = parameter_snapshot(history, max(0, frame_index - 1)) if frame_index else {}
        node_values = _node_values(history, stages, input_values, frame_index)
        node_deltas = _node_deltas(stages, parameters, previous, gradients)
        data: List[go.Scatter] = []
        for edge in edges:
            stage = edge["stage"]
            weights = parameters[stage["weight_name"]]
            weight = float(weights[edge["target_index"], edge["source_index"]])
            gradient_matrix = gradients.get(stage["weight_name"], np.zeros_like(weights))
            gradient = float(gradient_matrix[edge["target_index"], edge["source_index"]])
            previous_weights = previous.get(stage["weight_name"], weights)
            delta = float(weight - previous_weights[edge["target_index"], edge["source_index"]])
            encoded = weight if phase == "forward" else (delta if frame_index else -gradient)
            scale = weight_scale if phase == "forward" else max(gradient_scale, weight_scale * 0.05)
            if phase == "forward":
                hover = (
                    f"<b>Feed forward</b><br>"
                    f"$w^{{({stage['index']})}}_{{{edge['target_index'] + 1},{edge['source_index'] + 1}}}={weight:.{dec}f}$<br>"
                    rf"$z_j^{{({stage['index']})}}=\sum_i w_{{ji}}a_i+b_j$"
                )
            else:
                hover = (
                    f"<b>Backpropagation</b><br>"
                    rf"$\partial\mathcal{{L}}/\partial w={gradient:.{dec}f}$<br>"
                    rf"$\Delta w={delta:.{dec}f}$<br>$w={weight:.{dec}f}$"
                )
            data.append(
                go.Scatter(
                    x=edge["x"],
                    y=edge["y"],
                    mode="lines",
                    line={"color": _mix_color(encoded, scale), "width": 1.0 + 2.5 * min(abs(encoded) / scale, 1)},
                    customdata=[hover, hover],
                    hovertemplate="%{customdata}<extra></extra>",
                    showlegend=False,
                )
            )
        for column_index, (column_indices, y_values) in enumerate(zip(indices, y_positions)):
            values = node_values[column_index]
            deltas = node_deltas[column_index]
            visible_values = [float(values[index]) if index < values.size else float(np.mean(values)) for index in column_indices]
            visible_deltas = [float(deltas[index]) if index < deltas.size else float(np.mean(deltas)) for index in column_indices]
            encoded_values = visible_values if phase == "forward" else visible_deltas
            scale = activation_scale if phase == "forward" else max(weight_scale * 0.05, 1e-9)
            colors = [_mix_color(value, scale) for value in encoded_values]
            hover_data = []
            for visible_position, node_index in enumerate(column_indices):
                if column_index == 0:
                    hover_data.append(
                        f"<b>Input node {node_index + 1}</b><br>$x_{{{node_index + 1}}}={visible_values[visible_position]:.{dec}f}$"
                    )
                    continue
                stage = stages[column_index - 1]
                bias = _stage_bias(stage, parameters)
                weight_row = parameters[stage["weight_name"]][node_index]
                gradient_rows = gradients.get(
                    stage["weight_name"],
                    np.zeros_like(parameters[stage["weight_name"]]),
                )
                gradient_row = gradient_rows[node_index]
                if phase == "forward":
                    hover_data.append(
                        f"<b>Feed forward · neuron {node_index + 1}</b><br>"
                        f"$a_{{{node_index + 1}}}^{{({column_index})}}={visible_values[visible_position]:.{dec}f}$<br>"
                        f"$b_{{{node_index + 1}}}^{{({column_index})}}={bias[node_index]:.{dec}f}$<br>"
                        rf"$W_{{{node_index + 1},:}}={vector_latex(weight_row, dec=dec, limit=6)}$<br>"
                        rf"$\mathbf{{a}}^{{({column_index})}}\in\mathbb{{R}}^{{{stage['out_features']}}}$"
                    )
                else:
                    hover_data.append(
                        f"<b>Backpropagation · neuron {node_index + 1}</b><br>"
                        rf"$\operatorname{{mean}}_i(\Delta w_{{{node_index + 1},i}})={visible_deltas[visible_position]:.{dec}f}$<br>"
                        rf"$\nabla W_{{{node_index + 1},:}}={vector_latex(gradient_row, dec=dec, limit=6)}$<br>"
                        "Exact weight and gradient values are shown on the incoming edges."
                    )
            data.append(
                go.Scatter(
                    x=[x_positions[column_index]] * len(y_values),
                    y=y_values,
                    mode="markers",
                    marker={"size": 24, "color": colors, "line": {"width": 1, "color": NEURAL_COLORS["text"]}},
                    customdata=hover_data,
                    hovertemplate="%{customdata}<extra></extra>",
                    showlegend=False,
                )
            )
        return data, parameters

    first_data, first_parameters = frame_data(int(selected_frames[0]), "forward")
    figure = go.Figure(data=first_data)
    frames = []
    slider_steps = []
    for source_index in selected_frames:
        for phase, short_phase in (("forward", "F"), ("backward", "B")):
            data, parameters = frame_data(int(source_index), phase)
            frame_name = f"{source_index}-{phase}"
            annotations = _phase_annotations(
                stages,
                dimensions,
                x_positions,
                parameters,
                int(steps[source_index]),
                phase,
                dec,
            )
            frames.append(
                go.Frame(
                    name=frame_name,
                    data=data,
                    traces=trace_indices,
                    layout=go.Layout(annotations=annotations),
                )
            )
            slider_steps.append(
                {
                    "label": f"{steps[source_index]} {short_phase}",
                    "method": "animate",
                    "args": [[frame_name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                }
            )
    figure.frames = frames
    if title is None:
        title = "Mathematical network graph"
    layout = neural_layout(title, height=730)
    layout["margin"] = {"t": 155, "r": 45, "b": 115, "l": 45}
    figure.update_layout(
        **layout,
        annotations=_phase_annotations(
            stages,
            dimensions,
            x_positions,
            first_parameters,
            int(steps[selected_frames[0]]),
            "forward",
            dec,
        ),
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": -0.05,
                "bgcolor": NEURAL_COLORS["panel"],
                "bordercolor": NEURAL_COLORS["grid"],
                "borderwidth": 1,
                "font": {"color": NEURAL_COLORS["text"], "size": 12},
                "showactive": False,
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
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Step / phase: "},
                "pad": {"t": 50},
                "steps": slider_steps,
            }
        ],
        showlegend=False,
    )
    figure.update_xaxes(visible=False, range=[0, 1])
    figure.update_yaxes(visible=False, range=[0, 1])
    return figure


__all__ = ["build_nn_graph_figure"]
