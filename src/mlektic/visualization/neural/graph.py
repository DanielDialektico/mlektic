"""Animated mathematical weight graph for small dense PyTorch networks."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go

from ...neural.taxonomy import composed_dense_function, dense_stages
from ._style import NEURAL_COLORS, animation_button_style, neural_layout
from .math_format import (
    compact_parameter_line,
    display_indices,
    gradient_snapshot,
    parameter_snapshot,
    vector_latex,
)

WEIGHT_COLORSCALE = [
    [0.0, NEURAL_COLORS["weight_min"]],
    [0.5, NEURAL_COLORS["weight_mid"]],
    [1.0, NEURAL_COLORS["weight_max"]],
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


def _weight_color(value: float, minimum: float, maximum: float) -> str:
    if maximum <= minimum:
        return NEURAL_COLORS["weight_mid"]
    position = np.clip((float(value) - minimum) / (maximum - minimum), 0.0, 1.0)
    if position <= 0.5:
        return _interpolate_color(
            NEURAL_COLORS["weight_min"],
            NEURAL_COLORS["weight_mid"],
            float(position * 2.0),
        )
    return _interpolate_color(
        NEURAL_COLORS["weight_mid"],
        NEURAL_COLORS["weight_max"],
        float((position - 0.5) * 2.0),
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
                        "y": [
                            y_positions[stage_position][source_position],
                            y_positions[stage_position + 1][target_position],
                        ],
                    }
                )
    return dimensions, indices, x_positions, y_positions, edges


def _activation_vector(history: Dict[str, Any], stage: Dict[str, Any], frame_index: int) -> np.ndarray:
    key = stage.get("activation_name") or stage["name"]
    vectors = history.get("activation_vectors", {}).get(key, [])
    if frame_index < len(vectors):
        return np.asarray(vectors[frame_index], dtype=float).ravel()
    statistics = history.get("activations", {}).get(key, {})
    means = statistics.get("mean", [])
    fallback = float(means[frame_index]) if frame_index < len(means) else 0.0
    return np.full(stage["out_features"], fallback, dtype=float)


def _node_values(
    history: Dict[str, Any],
    stages: Sequence[Dict[str, Any]],
    input_values: np.ndarray,
    frame_index: int,
) -> List[np.ndarray]:
    return [input_values, *[_activation_vector(history, stage, frame_index) for stage in stages]]


def _stage_bias(stage: Dict[str, Any], parameters: Dict[str, np.ndarray]) -> np.ndarray:
    bias_name = stage.get("bias_name")
    if bias_name and bias_name in parameters:
        return np.asarray(parameters[bias_name], dtype=float).ravel()
    return np.zeros(stage["out_features"], dtype=float)


def _graph_annotations(
    stages: Sequence[Dict[str, Any]],
    dimensions: Sequence[int],
    x_positions: Sequence[float],
    parameters: Dict[str, np.ndarray],
    step: int,
    dec: int,
    is_final: bool,
) -> List[Dict[str, Any]]:
    phase_tex = (
        r"\text{Feed forward: }\mathbf{z}^{(\ell)}=W^{(\ell)}\mathbf{a}^{(\ell-1)}+\mathbf{b}^{(\ell)},\;"
        r"\mathbf{a}^{(\ell)}=\phi_\ell(\mathbf{z}^{(\ell)})\qquad"
        r"\text{Backpropagation: }\nabla_{W^{(\ell)}}\mathcal{L}="
        r"\frac{\partial\mathcal{L}}{\partial W^{(\ell)}}"
    )
    final_tex = r"\quad\text{(final weights)}" if is_final else ""
    annotations: List[Dict[str, Any]] = [
        {
            "x": 0.5,
            "y": 1.19,
            "xref": "paper",
            "yref": "paper",
            "text": f"${composed_dense_function(stages)}$",
            "showarrow": False,
            "font": {"size": 17, "color": NEURAL_COLORS["text"]},
        },
        {
            "x": 0.5,
            "y": 1.10,
            "xref": "paper",
            "yref": "paper",
            "text": f"${compact_parameter_line(stages, parameters, dec=dec)}$",
            "showarrow": False,
            "font": {"size": 13, "color": NEURAL_COLORS["muted"]},
        },
        {
            "x": 0.01,
            "y": 1.01,
            "xref": "paper",
            "yref": "paper",
            "text": rf"$t={step}{final_tex}\qquad {phase_tex}$",
            "showarrow": False,
            "xanchor": "left",
            "font": {"size": 12, "color": NEURAL_COLORS["text"]},
        },
        {
            "x": 0.99,
            "y": 0.94,
            "xref": "paper",
            "yref": "paper",
            "text": "<span style='color:#8f2942'>- -</span> backpropagation gradient",
            "showarrow": False,
            "xanchor": "right",
            "font": {"size": 11, "color": NEURAL_COLORS["muted"]},
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


def _edge_traces(
    edges: Sequence[Dict[str, Any]],
    parameters: Dict[str, np.ndarray],
    previous: Dict[str, np.ndarray],
    gradients: Dict[str, np.ndarray],
    weight_minimum: float,
    weight_maximum: float,
    gradient_scale: float,
    dec: int,
) -> List[go.Scatter]:
    traces: List[go.Scatter] = []
    for edge in edges:
        stage = edge["stage"]
        weights = parameters[stage["weight_name"]]
        row = edge["target_index"]
        column = edge["source_index"]
        weight = float(weights[row, column])
        previous_weights = previous.get(stage["weight_name"], weights)
        delta = float(weight - previous_weights[row, column])
        gradient_matrix = gradients.get(stage["weight_name"], np.zeros_like(weights))
        gradient = float(gradient_matrix[row, column])
        gradient_ratio = min(abs(gradient) / gradient_scale, 1.0)
        weight_hover = (
            f"<b>Weight evolution</b><br>layer={stage['index']}<br>"
            f"w[{row + 1},{column + 1}]={weight:.{dec}f}<br>"
            f"delta w={delta:+.{dec}f}"
        )
        traces.append(
            go.Scatter(
                x=edge["x"],
                y=edge["y"],
                mode="lines",
                line={
                    "color": _weight_color(weight, weight_minimum, weight_maximum),
                    "width": 3.2,
                },
                customdata=[weight_hover, weight_hover],
                hovertemplate="%{customdata}<extra></extra>",
                showlegend=False,
            )
        )
        gradient_hover = (
            f"<b>Backpropagation</b><br>layer={stage['index']}<br>"
            f"dL/dw[{row + 1},{column + 1}]={gradient:+.{dec}f}<br>"
            f"delta w={delta:+.{dec}f}"
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
    dec: int,
) -> List[go.Scatter]:
    traces: List[go.Scatter] = []
    for column_index, (column_indices, y_values) in enumerate(zip(indices, y_positions)):
        values = node_values[column_index]
        hover_data = []
        for node_index in column_indices:
            value = float(values[node_index]) if node_index < values.size else float(np.mean(values))
            if column_index == 0:
                hover_data.append(f"<b>Input {node_index + 1}</b><br>x[{node_index + 1}]={value:.{dec}f}")
                continue
            stage = stages[column_index - 1]
            bias = _stage_bias(stage, parameters)
            weight_row = parameters[stage["weight_name"]][node_index]
            gradient_rows = gradients.get(
                stage["weight_name"],
                np.zeros_like(parameters[stage["weight_name"]]),
            )
            hover_data.append(
                f"<b>Neuron {node_index + 1}</b><br>recorded activation={value:.{dec}f}<br>"
                f"bias={bias[node_index]:.{dec}f}<br>"
                rf"$W_{{{node_index + 1},:}}={vector_latex(weight_row, dec=dec, limit=6)}$<br>"
                rf"$\nabla W_{{{node_index + 1},:}}="
                rf"{vector_latex(gradient_rows[node_index], dec=dec, limit=6)}$"
            )
        traces.append(
            go.Scatter(
                x=[x_positions[column_index]] * len(y_values),
                y=y_values,
                mode="markers",
                marker={
                    "size": 24,
                    "color": NEURAL_COLORS["panel"],
                    "line": {"width": 1.5, "color": NEURAL_COLORS["text"]},
                },
                customdata=hover_data,
                hovertemplate="%{customdata}<extra></extra>",
                showlegend=False,
            )
        )
    return traces


def _colorbar_trace(minimum: float, maximum: float) -> go.Scatter:
    return go.Scatter(
        x=[None, None],
        y=[None, None],
        mode="markers",
        marker={
            "size": 0,
            "color": [minimum, maximum],
            "cmin": minimum,
            "cmax": maximum,
            "colorscale": WEIGHT_COLORSCALE,
            "showscale": True,
            "colorbar": {
                "title": {"text": "Weight value", "side": "right"},
                "thickness": 12,
                "len": 0.52,
                "y": 0.51,
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
    dec: int = 3,
) -> go.Figure:
    """Animate weights as a stable heatmap with simultaneous backpropagation cues."""
    stages = dense_stages(model)
    if not stages:
        raise ValueError("The animated graph currently requires at least one torch.nn.Linear layer.")
    if max_neurons < 2:
        raise ValueError("max_neurons must be at least 2.")
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

    def frame_payload(frame_index: int):
        parameters = _parameter_snapshot_for_frame(
            history,
            frame_index,
            steps.size - 1,
            final_parameters,
        )
        previous_index = max(0, frame_index - 1)
        previous = parameter_snapshot(history, previous_index)
        gradients = gradient_snapshot(history, frame_index)
        node_values = _node_values(history, stages, input_values, frame_index)
        data = _edge_traces(
            edges,
            parameters,
            previous,
            gradients,
            weight_minimum,
            weight_maximum,
            gradient_scale,
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
                dec,
            )
        )
        annotations = _graph_annotations(
            stages,
            dimensions,
            x_positions,
            parameters,
            int(steps[frame_index]),
            dec,
            frame_index == steps.size - 1,
        )
        return data, annotations

    first_data, first_annotations = frame_payload(int(selected_frames[0]))
    figure = go.Figure(data=[*first_data, _colorbar_trace(weight_minimum, weight_maximum)])
    dynamic_trace_indices = list(range(len(first_data)))
    frames = []
    slider_steps = []
    for source_index in selected_frames:
        data, annotations = frame_payload(int(source_index))
        frame_name = str(int(source_index))
        frames.append(
            go.Frame(
                name=frame_name,
                data=data,
                traces=dynamic_trace_indices,
                layout=go.Layout(annotations=annotations),
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
        title = "Mathematical network: weight evolution"
    layout = neural_layout(title, height=740)
    layout["margin"] = {"t": 165, "r": 105, "b": 115, "l": 45}
    figure.update_layout(
        **layout,
        annotations=first_annotations,
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
