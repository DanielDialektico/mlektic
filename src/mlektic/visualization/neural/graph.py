"""Animated mathematical weight graph for small dense PyTorch networks."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go

from ...neural.introspection import run_torch_forward
from ...neural.taxonomy import composed_dense_function, dense_stages
from ._style import NEURAL_COLORS, animation_button_style, neural_layout
from .math_format import (
    compact_parameter_line,
    display_indices,
    gradient_snapshot,
    parameter_snapshot,
)

ACTIVATION_COLORSCALE = [
    [0.0, NEURAL_COLORS["activation_min"]],
    [0.5, NEURAL_COLORS["activation_mid"]],
    [1.0, NEURAL_COLORS["activation_max"]],
]


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _frame_indices(frame_count: int, max_frames: int | None) -> np.ndarray:
    if max_frames is None or frame_count <= max_frames:
        return np.arange(frame_count, dtype=int)
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1 or None.")
    return np.unique(np.linspace(0, frame_count - 1, max_frames, dtype=int))


def _interpolate_color(start: str, end: str, ratio: float) -> str:
    start_rgb = np.asarray([int(start[index : index + 2], 16) for index in (1, 3, 5)], dtype=float)
    end_rgb = np.asarray([int(end[index : index + 2], 16) for index in (1, 3, 5)], dtype=float)
    rgb = np.rint(start_rgb * (1.0 - ratio) + end_rgb * ratio).astype(int)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def _scaled_color(
    value: float,
    minimum: float,
    maximum: float,
    low_color: str,
    middle_color: str,
    high_color: str,
) -> str:
    if maximum <= minimum:
        return middle_color
    position = np.clip((float(value) - minimum) / (maximum - minimum), 0.0, 1.0)
    if position <= 0.5:
        return _interpolate_color(low_color, middle_color, float(position * 2.0))
    return _interpolate_color(middle_color, high_color, float((position - 0.5) * 2.0))


def _signed_color(
    value: float,
    minimum: float,
    maximum: float,
    low_color: str,
    middle_color: str,
    high_color: str,
) -> str:
    if minimum < 0.0 < maximum:
        if value <= 0.0:
            ratio = (float(value) - minimum) / (0.0 - minimum)
            return _interpolate_color(low_color, middle_color, float(np.clip(ratio, 0.0, 1.0)))
        ratio = float(value) / maximum
        return _interpolate_color(middle_color, high_color, float(np.clip(ratio, 0.0, 1.0)))
    if minimum >= 0.0:
        return _scaled_color(value, minimum, maximum, middle_color, middle_color, high_color)
    return _scaled_color(value, minimum, maximum, low_color, middle_color, middle_color)


def _signed_colorscale(
    minimum: float,
    maximum: float,
    low_color: str,
    middle_color: str,
    high_color: str,
) -> List[List[Any]]:
    if minimum < 0.0 < maximum:
        zero_position = (0.0 - minimum) / (maximum - minimum)
        return [[0.0, low_color], [zero_position, middle_color], [1.0, high_color]]
    if minimum >= 0.0:
        return [[0.0, middle_color], [1.0, high_color]]
    return [[0.0, low_color], [1.0, middle_color]]


def _weight_color(value: float, minimum: float, maximum: float) -> str:
    return _signed_color(
        value,
        minimum,
        maximum,
        NEURAL_COLORS["weight_min"],
        NEURAL_COLORS["weight_mid"],
        NEURAL_COLORS["weight_max"],
    )


def _activation_color(value: float, minimum: float, maximum: float) -> str:
    return _signed_color(
        value,
        minimum,
        maximum,
        NEURAL_COLORS["activation_min"],
        NEURAL_COLORS["activation_mid"],
        NEURAL_COLORS["activation_max"],
    )


def _current_parameters(model: Any) -> Dict[str, np.ndarray]:
    return {
        name: parameter.detach().float().cpu().numpy().copy()
        for name, parameter in model.named_parameters()
    }


def _parameter_snapshot_for_frame(
    history: Dict[str, Any],
    frame_index: int,
    final_frame_index: int,
    final_parameters: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    snapshot = parameter_snapshot(history, frame_index)
    if frame_index == final_frame_index:
        snapshot.update(final_parameters)
    return snapshot


def _global_limits(
    history: Dict[str, Any],
    stages: Sequence[Dict[str, Any]],
    final_parameters: Dict[str, np.ndarray],
) -> Tuple[float, float, float]:
    weights: List[np.ndarray] = []
    gradients: List[np.ndarray] = []
    for stage in stages:
        weights.extend(
            np.asarray(value, dtype=float).ravel()
            for value in history.get("parameters", {}).get(stage["weight_name"], [])
        )
        if stage["weight_name"] in final_parameters:
            weights.append(final_parameters[stage["weight_name"]].ravel())
        gradients.extend(
            np.asarray(value, dtype=float).ravel()
            for value in history.get("gradients", {}).get(stage["weight_name"], [])
        )
    weight_values = np.concatenate(weights) if weights else np.asarray([0.0])
    gradient_values = np.concatenate(gradients) if gradients else np.asarray([0.0])
    minimum = float(np.min(weight_values))
    maximum = float(np.max(weight_values))
    if maximum <= minimum:
        maximum = minimum + 1e-9
    gradient_scale = max(float(np.max(np.abs(gradient_values))), 1e-12)
    return minimum, maximum, gradient_scale


def _graph_geometry(stages: Sequence[Dict[str, Any]], max_neurons: int):
    dimensions = [stages[0]["in_features"], *[stage["out_features"] for stage in stages]]
    indices = [display_indices(dimension, max_neurons) for dimension in dimensions]
    x_positions = np.linspace(0.07, 0.93, len(dimensions))
    y_positions = [np.linspace(0.18, 0.70, len(column)).tolist() for column in indices]
    edges = []
    for stage_position, stage in enumerate(stages):
        for source_position, source_index in enumerate(indices[stage_position]):
            for target_position, target_index in enumerate(indices[stage_position + 1]):
                edges.append(
                    {
                        "stage": stage,
                        "stage_position": stage_position,
                        "source_index": source_index,
                        "target_index": target_index,
                        "x": [x_positions[stage_position], x_positions[stage_position + 1]],
                        "y": [
                            y_positions[stage_position][source_position],
                            y_positions[stage_position + 1][target_position],
                        ],
                    }
                )
    return dimensions, indices, x_positions, y_positions, edges


def _record_vector(records: Dict[str, Dict[str, np.ndarray]], name: str, size: int) -> np.ndarray:
    record = records.get(name, {})
    output = np.asarray(record.get("output", []), dtype=float)
    if not output.size:
        return np.zeros(size, dtype=float)
    if output.ndim == 1:
        return output.ravel()
    flattened = output.reshape(output.shape[0], -1)
    return np.mean(flattened, axis=0)


def _node_values(
    stages: Sequence[Dict[str, Any]],
    input_values: np.ndarray,
    records: Dict[str, Dict[str, np.ndarray]],
) -> List[np.ndarray]:
    return [
        input_values,
        *[
            _record_vector(
                records,
                stage.get("activation_name") or stage["name"],
                stage["out_features"],
            )
            for stage in stages
        ],
    ]


def _activation_limits(
    states: Sequence[Sequence[np.ndarray]],
    mode: str,
) -> List[Tuple[float, float]]:
    if not states:
        return [(0.0, 1.0)]
    if mode == "value":
        values = [np.asarray(vector, dtype=float).ravel() for state in states for vector in state]
        combined = np.concatenate(values) if values else np.asarray([0.0])
        minimum = float(np.min(combined))
        maximum = float(np.max(combined))
        if maximum <= minimum:
            maximum = minimum + 1e-9
        return [(minimum, maximum)] * len(states[0])
    limits: List[Tuple[float, float]] = []
    for column_index in range(len(states[0])):
        values = [np.asarray(state[column_index], dtype=float).ravel() for state in states]
        combined = np.concatenate(values) if values else np.asarray([0.0])
        minimum = float(np.min(combined))
        maximum = float(np.max(combined))
        if maximum <= minimum:
            maximum = minimum + 1e-9
        limits.append((minimum, maximum))
    return limits


def _edge_value(
    edge: Dict[str, Any],
    parameters: Dict[str, np.ndarray],
    node_values: Sequence[np.ndarray],
    mode: str,
) -> float:
    stage = edge["stage"]
    weight = float(parameters[stage["weight_name"]][edge["target_index"], edge["source_index"]])
    if mode == "weight":
        return weight
    source_values = node_values[edge["stage_position"]]
    source_activation = float(source_values[edge["source_index"]])
    return weight * source_activation


def _edge_limits(
    edges: Sequence[Dict[str, Any]],
    frame_states: Dict[int, tuple],
    mode: str,
    weight_limits: Tuple[float, float],
) -> Tuple[float, float]:
    if mode == "weight":
        return weight_limits
    values = [
        _edge_value(edge, state[0], state[3], mode)
        for state in frame_states.values()
        for edge in edges
    ]
    minimum = float(np.min(values)) if values else 0.0
    maximum = float(np.max(values)) if values else 0.0
    if maximum <= minimum:
        maximum = minimum + 1e-9
    return minimum, maximum


def _plain_vector(values: Any, dec: int, limit: int = 6) -> str:
    flat = np.asarray(values, dtype=float).ravel()
    if flat.size <= limit:
        cells = [f"{value:.{dec}f}" for value in flat]
    else:
        selected = display_indices(flat.size, limit)
        split = len(selected) // 2
        cells = [f"{flat[index]:.{dec}f}" for index in selected[:split]]
        cells.append("...")
        cells.extend(f"{flat[index]:.{dec}f}" for index in selected[split:])
    return "[" + ", ".join(cells) + "]"


def _format_output(value: float, dec: int, activation_type: str | None = None) -> str:
    precision = max(dec, 6)
    if value == 0.0:
        return "0 (ReLU inactive)" if activation_type == "ReLU" else "0 (exact)"
    if abs(value) < 10.0 ** (-dec):
        return f"{value:.3e}"
    return f"{value:.{precision}f}"


def _stage_bias(stage: Dict[str, Any], parameters: Dict[str, np.ndarray]) -> np.ndarray:
    bias_name = stage.get("bias_name")
    if bias_name and bias_name in parameters:
        return np.asarray(parameters[bias_name], dtype=float).ravel()
    return np.zeros(stage["out_features"], dtype=float)


def _graph_annotations(
    stages: Sequence[Dict[str, Any]],
    dimensions: Sequence[int],
    x_positions: Sequence[float],
    node_color_mode: str,
    edge_color_mode: str,
) -> List[Dict[str, Any]]:
    phase_tex = (
        r"\text{Feed forward: }\mathbf{z}^{(\ell)}=\Theta^{(\ell)}\mathbf{a}^{(\ell-1)}+\boldsymbol{\theta}_0^{(\ell)},\;"
        r"\mathbf{a}^{(\ell)}=\phi_\ell(\mathbf{z}^{(\ell)})\qquad"
        r"\text{Backpropagation: }\nabla_{\Theta^{(\ell)}}\mathcal{L}="
        r"\frac{\partial\mathcal{L}}{\partial\Theta^{(\ell)}}"
    )
    node_heatmap_tex = (
        r"\text{Node heatmap (exact): }a_j^{(\ell)}"
        if node_color_mode == "value"
        else (
            r"\text{Node heatmap (relative): }\widetilde a_j^{(\ell)}="
            r"\frac{a_j^{(\ell)}-a_{\min}^{(\ell)}}"
            r"{a_{\max}^{(\ell)}-a_{\min}^{(\ell)}}"
        )
    )
    edge_heatmap_tex = (
        r"\theta_{ji}^{(\ell)}"
        if edge_color_mode == "weight"
        else r"s_{ji}^{(\ell)}=\theta_{ji}^{(\ell)}a_i^{(\ell-1)}"
    )
    node_scale_tex = (
        r"a_j^{(\ell)}" if node_color_mode == "value" else r"\widetilde a_j^{(\ell)}"
    )
    annotations: List[Dict[str, Any]] = [
        {
            "x": 0.5,
            "y": 1.11,
            "xref": "paper",
            "yref": "paper",
            "text": f"${composed_dense_function(stages)}$",
            "showarrow": False,
            "font": {"size": 17, "color": NEURAL_COLORS["text"]},
        },
        {
            "x": 0.56,
            "y": 0.86,
            "xref": "paper",
            "yref": "paper",
            "text": f"${phase_tex}$",
            "showarrow": False,
            "font": {"size": 12, "color": NEURAL_COLORS["text"]},
        },
        {
            "x": 0.01,
            "y": 0.77,
            "xref": "paper",
            "yref": "paper",
            "text": (
                f"${node_heatmap_tex}"
                rf"\qquad\text{{Edge heatmap: }}{edge_heatmap_tex}$"
            ),
            "showarrow": False,
            "xanchor": "left",
            "font": {"size": 11, "color": NEURAL_COLORS["muted"]},
        },
        {
            "x": 0.99,
            "y": 0.77,
            "xref": "paper",
            "yref": "paper",
            "text": "<span style='color:#8f2942'>- -</span> backpropagation gradient",
            "showarrow": False,
            "xanchor": "right",
            "font": {"size": 11, "color": NEURAL_COLORS["muted"]},
        },
        {
            "x": 1.035,
            "y": 0.89,
            "xref": "paper",
            "yref": "paper",
            "text": f"${edge_heatmap_tex}$",
            "showarrow": False,
            "font": {"size": 12, "color": NEURAL_COLORS["text"]},
        },
        {
            "x": 1.035,
            "y": 0.48,
            "xref": "paper",
            "yref": "paper",
            "text": f"${node_scale_tex}$",
            "showarrow": False,
            "font": {"size": 12, "color": NEURAL_COLORS["text"]},
        },
    ]
    for column_index, (x_position, dimension) in enumerate(zip(x_positions, dimensions)):
        if column_index == 0:
            dimension_tex = rf"\mathbf{{a}}^{{(0)}}=\mathbf{{x}}\in\mathbb{{R}}^{{{dimension}}}"
        else:
            stage = stages[column_index - 1]
            dimension_tex = (
                rf"\mathbf{{a}}^{{({column_index})}}\in\mathbb{{R}}^{{{dimension}}}"
                rf",\;\Theta^{{({column_index})}}\in\mathbb{{R}}^{{{dimension}\times {stage['in_features']}}}"
            )
        annotations.append(
            {
                "x": x_position,
                "y": 0.05,
                "xref": "x",
                "yref": "y",
                "text": f"${dimension_tex}$",
                "showarrow": False,
                "font": {"size": 12, "color": NEURAL_COLORS["muted"]},
            }
        )
    return annotations


def _dynamic_label_traces(
    stages: Sequence[Dict[str, Any]],
    parameters: Dict[str, np.ndarray],
    step: int,
    dec: int,
    is_final: bool,
) -> List[go.Scatter]:
    final_tex = r"\quad\text{(final weights)}" if is_final else ""
    readout_decimals = max(dec, 4)
    return [
        go.Scatter(
            x=[0.5],
            y=[0.96],
            mode="text",
            text=[f"${compact_parameter_line(stages, parameters, dec=readout_decimals)}$"],
            textposition="middle center",
            textfont={"size": 13, "color": NEURAL_COLORS["muted"]},
            cliponaxis=False,
            hoverinfo="skip",
            showlegend=False,
            name="parameter readout",
        ),
        go.Scatter(
            x=[0.08],
            y=[0.86],
            mode="text",
            text=[rf"$t={step}{final_tex}$"],
            textposition="middle center",
            textfont={"size": 12, "color": NEURAL_COLORS["text"]},
            cliponaxis=False,
            hoverinfo="skip",
            showlegend=False,
            name="training step readout",
        ),
    ]


def _edge_traces(
    edges: Sequence[Dict[str, Any]],
    parameters: Dict[str, np.ndarray],
    previous: Dict[str, np.ndarray],
    gradients: Dict[str, np.ndarray],
    node_values: Sequence[np.ndarray],
    edge_minimum: float,
    edge_maximum: float,
    gradient_scale: float,
    edge_color_mode: str,
    dec: int,
) -> List[go.Scatter]:
    traces: List[go.Scatter] = []
    for edge in edges:
        stage = edge["stage"]
        weights = parameters[stage["weight_name"]]
        row = edge["target_index"]
        column = edge["source_index"]
        weight = float(weights[row, column])
        source_activation = float(node_values[edge["stage_position"]][column])
        signal = weight * source_activation
        encoded_value = weight if edge_color_mode == "weight" else signal
        previous_weights = previous.get(stage["weight_name"], weights)
        delta = float(weight - previous_weights[row, column])
        gradient_matrix = gradients.get(stage["weight_name"], np.zeros_like(weights))
        gradient = float(gradient_matrix[row, column])
        gradient_ratio = min(abs(gradient) / gradient_scale, 1.0)
        edge_hover = (
            f"<b>{'Weight evolution' if edge_color_mode == 'weight' else 'Forward signal'}</b>"
            f"<br>layer={stage['index']}<br>"
            f"theta[{row + 1},{column + 1}]={weight:.{dec}f}<br>"
            f"source output={source_activation:.{dec}f}<br>"
            f"w * a={signal:+.{dec}f}<br>"
            f"delta theta={delta:+.{dec}f}"
        )
        traces.append(
            go.Scatter(
                x=edge["x"],
                y=edge["y"],
                mode="lines",
                line={
                    "color": _weight_color(encoded_value, edge_minimum, edge_maximum),
                    "width": 3.2,
                },
                customdata=[edge_hover, edge_hover],
                hovertemplate="%{customdata}<extra></extra>",
                showlegend=False,
            )
        )
        gradient_hover = (
            f"<b>Backpropagation</b><br>layer={stage['index']}<br>"
            f"dL/dtheta[{row + 1},{column + 1}]={gradient:+.{dec}f}<br>"
            f"delta theta={delta:+.{dec}f}"
        )
        traces.append(
            go.Scatter(
                x=list(reversed(edge["x"])),
                y=list(reversed(edge["y"])),
                mode="lines",
                line={
                    "color": NEURAL_COLORS["backprop"],
                    "width": 0.5 + 1.6 * gradient_ratio,
                    "dash": "dot",
                },
                opacity=0.10 + 0.58 * gradient_ratio,
                customdata=[gradient_hover, gradient_hover],
                hovertemplate="%{customdata}<extra></extra>",
                showlegend=False,
            )
        )
    return traces


def _node_traces(
    stages: Sequence[Dict[str, Any]],
    indices: Sequence[Sequence[int]],
    x_positions: Sequence[float],
    y_positions: Sequence[Sequence[float]],
    node_values: Sequence[np.ndarray],
    parameters: Dict[str, np.ndarray],
    gradients: Dict[str, np.ndarray],
    activation_limits: Sequence[Tuple[float, float]],
    node_color_mode: str,
    dec: int,
) -> List[go.Scatter]:
    traces: List[go.Scatter] = []
    for column_index, (column_indices, y_values) in enumerate(zip(indices, y_positions)):
        values = node_values[column_index]
        activation_minimum, activation_maximum = activation_limits[column_index]
        hover_data = []
        visible_values = []
        for node_index in column_indices:
            value = float(values[node_index]) if node_index < values.size else float(np.mean(values))
            visible_values.append(value)
            normalized = (value - activation_minimum) / (activation_maximum - activation_minimum)
            if column_index == 0:
                hover_data.append(
                    f"<b>Input node {node_index + 1}</b><br>"
                    f"output x[{node_index + 1}] = {_format_output(value, dec)}<br>"
                    + (
                        f"relative heatmap value = {normalized:.3f}"
                        if node_color_mode == "relative"
                        else "heatmap uses the exact output"
                    )
                )
                continue
            stage = stages[column_index - 1]
            bias = _stage_bias(stage, parameters)
            weight_row = parameters[stage["weight_name"]][node_index]
            gradient_rows = gradients.get(
                stage["weight_name"],
                np.zeros_like(parameters[stage["weight_name"]]),
            )
            hover_data.append(
                f"<b>Neuron {node_index + 1}</b><br>"
                f"activation={stage.get('activation_type') or 'identity'}<br>"
                f"numerical output={_format_output(value, dec, stage.get('activation_type'))}<br>"
                + (
                    f"relative heatmap value={normalized:.3f}<br>"
                    if node_color_mode == "relative"
                    else "heatmap uses the exact output<br>"
                )
                + f"bias={bias[node_index]:.{dec}f}<br>"
                f"Theta[{node_index + 1},:]={_plain_vector(weight_row, dec)}<br>"
                f"grad Theta[{node_index + 1},:]={_plain_vector(gradient_rows[node_index], dec)}"
            )
        traces.append(
            go.Scatter(
                x=[x_positions[column_index]] * len(y_values),
                y=y_values,
                mode="markers",
                marker={
                    "size": 28,
                    "color": [
                        (
                            _scaled_color(
                                value,
                                activation_minimum,
                                activation_maximum,
                                NEURAL_COLORS["activation_min"],
                                NEURAL_COLORS["activation_mid"],
                                NEURAL_COLORS["activation_max"],
                            )
                            if node_color_mode == "relative"
                            else _activation_color(value, activation_minimum, activation_maximum)
                        )
                        for value in visible_values
                    ],
                    "line": {"width": 1.25, "color": NEURAL_COLORS["muted"]},
                },
                customdata=hover_data,
                hovertemplate="%{customdata}<extra></extra>",
                showlegend=False,
            )
        )
    return traces


def _colorbar_trace(
    minimum: float,
    maximum: float,
    *,
    colorscale: Sequence[Sequence[Any]],
    y: float,
) -> go.Scatter:
    return go.Scatter(
        x=[None, None],
        y=[None, None],
        mode="markers",
        marker={
            "size": 0,
            "color": [minimum, maximum],
            "cmin": minimum,
            "cmax": maximum,
            "colorscale": colorscale,
            "showscale": True,
            "colorbar": {
                "thickness": 12,
                "len": 0.34,
                "y": y,
                "x": 1.015,
                "xanchor": "left",
                "tickformat": ".3f",
            },
        },
        hoverinfo="skip",
        showlegend=False,
    )


def build_nn_graph_figure(
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
    dec: int = 3,
) -> go.Figure:
    """Animate exact node outputs and edge values with backpropagation cues."""
    stages = dense_stages(model)
    if not stages:
        raise ValueError("The animated graph currently requires at least one torch.nn.Linear layer.")
    if max_neurons < 2:
        raise ValueError("max_neurons must be at least 2.")
    if node_color_mode not in {"value", "relative"}:
        raise ValueError("node_color_mode must be 'value' or 'relative'.")
    if edge_color_mode not in {"weight", "signal"}:
        raise ValueError("edge_color_mode must be 'weight' or 'signal'.")
    missing = [
        stage["weight_name"]
        for stage in stages
        if stage["weight_name"] not in history.get("parameters", {})
    ]
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
    input_values = (
        input_array
        if input_array.ndim == 1
        else np.mean(input_array.reshape(input_array.shape[0], -1), axis=0)
    )
    dimensions, indices, x_positions, y_positions, edges = _graph_geometry(stages, max_neurons)
    final_parameters = _current_parameters(model)
    weight_minimum, weight_maximum, gradient_scale = _global_limits(
        history,
        stages,
        final_parameters,
    )

    frame_states: Dict[int, tuple] = {}
    for source_index in selected_frames:
        frame_index = int(source_index)
        parameters = _parameter_snapshot_for_frame(
            history,
            frame_index,
            steps.size - 1,
            final_parameters,
        )
        previous = parameter_snapshot(history, max(0, frame_index - 1))
        gradients = gradient_snapshot(history, frame_index)
        _, records = run_torch_forward(model, input_sample, parameters)
        node_values = _node_values(stages, input_values, records)
        frame_states[frame_index] = (parameters, previous, gradients, node_values)
    activation_limits = _activation_limits(
        [state[3] for state in frame_states.values()],
        node_color_mode,
    )
    edge_minimum, edge_maximum = _edge_limits(
        edges,
        frame_states,
        edge_color_mode,
        (weight_minimum, weight_maximum),
    )

    def frame_payload(frame_index: int):
        parameters, previous, gradients, node_values = frame_states[frame_index]
        data = _edge_traces(
            edges,
            parameters,
            previous,
            gradients,
            node_values,
            edge_minimum,
            edge_maximum,
            gradient_scale,
            edge_color_mode,
            dec,
        )
        data.extend(
            _node_traces(
                stages,
                indices,
                x_positions,
                y_positions,
                node_values,
                parameters,
                gradients,
                activation_limits,
                node_color_mode,
                dec,
            )
        )
        data.extend(
            _dynamic_label_traces(
                stages,
                parameters,
                int(steps[frame_index]),
                dec,
                frame_index == steps.size - 1,
            )
        )
        return data

    first_data = frame_payload(int(selected_frames[0]))
    annotations = _graph_annotations(
        stages,
        dimensions,
        x_positions,
        node_color_mode,
        edge_color_mode,
    )
    figure = go.Figure(
        data=[
            *first_data,
            _colorbar_trace(
                edge_minimum,
                edge_maximum,
                colorscale=_signed_colorscale(
                    edge_minimum,
                    edge_maximum,
                    NEURAL_COLORS["weight_min"],
                    NEURAL_COLORS["weight_mid"],
                    NEURAL_COLORS["weight_max"],
                ),
                y=0.70,
            ),
            _colorbar_trace(
                0.0 if node_color_mode == "relative" else activation_limits[0][0],
                1.0 if node_color_mode == "relative" else activation_limits[0][1],
                colorscale=(
                    ACTIVATION_COLORSCALE
                    if node_color_mode == "relative"
                    else _signed_colorscale(
                        activation_limits[0][0],
                        activation_limits[0][1],
                        NEURAL_COLORS["activation_min"],
                        NEURAL_COLORS["activation_mid"],
                        NEURAL_COLORS["activation_max"],
                    )
                ),
                y=0.29,
            ),
        ]
    )
    dynamic_trace_indices = list(range(len(first_data)))
    frames = []
    slider_steps = []
    for source_index in selected_frames:
        data = frame_payload(int(source_index))
        frame_name = str(int(source_index))
        frames.append(
            go.Frame(
                name=frame_name,
                data=data,
                traces=dynamic_trace_indices,
            )
        )
        is_final = int(source_index) == steps.size - 1
        slider_steps.append(
            {
                "label": f"{steps[source_index]}{' final' if is_final else ''}",
                "method": "animate",
                "args": [[frame_name], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}],
            }
        )
    figure.frames = frames
    if title is None:
        title = "Mathematical network: parameter and signal evolution"
    layout = neural_layout(title, height=740)
    layout["title"]["y"] = 0.985
    layout["title"]["yanchor"] = "top"
    layout["margin"] = {"t": 120, "r": 145, "b": 115, "l": 65}
    figure.update_layout(
        **layout,
        annotations=annotations,
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": -0.05,
                **animation_button_style(),
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_duration, "redraw": False},
                                "transition": {"duration": min(frame_duration, 180)},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {"mode": "immediate", "frame": {"duration": 0, "redraw": False}},
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Training step: "},
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
