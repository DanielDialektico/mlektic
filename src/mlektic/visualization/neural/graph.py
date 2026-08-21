"""Animated mathematical weight graph for small dense PyTorch networks."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go

from ...neural.introspection import _leaf_modules, run_torch_forward
from ...neural.taxonomy import composed_dense_function, dense_stages
from ._style import NEURAL_COLORS, animation_button_style, neural_layout
from .math_format import (
    buffer_snapshot,
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

# Every dense-graph variant shares these paper-space teaching rows. Dynamic
# readouts are mapped back into data coordinates so changing subplot domains
# cannot move them into a different semantic section.
GRAPH_PAPER_ROWS = {
    "model": 1.045,
    "parameters": 0.96,
    "phase": 0.88,
    "backpropagation": 0.82,
    "legend": 0.80,
    "activity": 0.70,
    "update": 0.64,
    "disclosure": 0.59,
}

# The uppermost node is a marker rather than text, so comparing only its center
# with the Update-halo baseline is insufficient. This paper-space clearance
# leaves room for both the marker radius and MathJax's descenders at every
# supported figure height.
UPDATE_TO_NETWORK_MINIMUM_GAP = 0.065


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


def _global_limits(
    history: Dict[str, Any],
    stages: Sequence[Dict[str, Any]],
) -> Tuple[float, float, float]:
    weights: List[np.ndarray] = []
    gradients: List[np.ndarray] = []
    for stage in stages:
        weights.extend(
            np.asarray(value, dtype=float).ravel()
            for value in history.get("parameters", {}).get(stage["weight_name"], [])
        )
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


def _stage_parameter_names(stages: Sequence[Dict[str, Any]]) -> List[str]:
    """Return unique weight and bias tensors represented by the dense graph."""
    names: List[str] = []
    for stage in stages:
        for key in ("weight_name", "bias_name"):
            name = stage.get(key)
            if name and name not in names:
                names.append(name)
    return names


def _interpolate_snapshots(
    start: Dict[str, np.ndarray],
    end: Dict[str, np.ndarray],
    ratio: float,
) -> Dict[str, np.ndarray]:
    """Create a perceptual parameter state between two recorded checkpoints."""
    return {
        name: np.asarray(start.get(name, value), dtype=float) * (1.0 - ratio) + np.asarray(value, dtype=float) * ratio
        for name, value in end.items()
    }


def _parameter_delta_values(
    names: Sequence[str],
    current: Dict[str, np.ndarray],
    reference: Dict[str, np.ndarray],
) -> np.ndarray:
    values = [
        (np.asarray(current[name], dtype=float) - np.asarray(reference.get(name, current[name]), dtype=float)).ravel()
        for name in names
        if name in current
    ]
    return np.concatenate(values) if values else np.asarray([0.0])


def _global_update_limit(
    history: Dict[str, Any],
    stages: Sequence[Dict[str, Any]],
    reference_mode: str,
    frame_indices: Sequence[int],
) -> float:
    names = _stage_parameter_names(stages)
    if len(frame_indices) < 2:
        return 1e-12
    initial = parameter_snapshot(history, 0)
    values = []
    for position, frame_index in enumerate(frame_indices):
        current = parameter_snapshot(history, frame_index)
        reference = (
            initial if reference_mode == "initial" else parameter_snapshot(history, frame_indices[max(0, position - 1)])
        )
        values.append(np.abs(_parameter_delta_values(names, current, reference)))
    combined = np.concatenate(values) if values else np.asarray([0.0])
    return max(float(np.max(combined)), 1e-12)


def _update_metrics(
    stages: Sequence[Dict[str, Any]],
    current: Dict[str, np.ndarray],
    reference: Dict[str, np.ndarray],
    gradients: Dict[str, np.ndarray],
    *,
    alignment_is_exact: bool,
) -> Dict[str, float]:
    names = _stage_parameter_names(stages)
    theta_parts = [np.asarray(current[name], dtype=float).ravel() for name in names if name in current]
    reference_parts = [
        np.asarray(reference.get(name, current[name]), dtype=float).ravel() for name in names if name in current
    ]
    delta = _parameter_delta_values(names, current, reference)
    gradient_parts = [np.asarray(gradients[name], dtype=float).ravel() for name in names if name in gradients]
    theta = np.concatenate(theta_parts) if theta_parts else np.asarray([0.0])
    reference_values = np.concatenate(reference_parts) if reference_parts else np.asarray([0.0])
    gradient = np.concatenate(gradient_parts) if gradient_parts else np.asarray([])
    theta_norm = float(np.linalg.norm(theta))
    reference_norm = float(np.linalg.norm(reference_values))
    delta_norm = float(np.linalg.norm(delta))
    gradient_norm = float(np.linalg.norm(gradient)) if gradient.size else float("nan")
    alignment = float("nan")
    if alignment_is_exact and gradient.size == delta.size and delta_norm > 0.0 and gradient_norm > 0.0:
        alignment = float(np.dot(delta, -gradient) / (delta_norm * gradient_norm))
    return {
        "theta_norm": theta_norm,
        "delta_norm": delta_norm,
        "relative_update": delta_norm / max(reference_norm, 1e-12),
        "gradient_norm": gradient_norm,
        "update_gradient_alignment": alignment,
    }


def _graph_geometry(
    stages: Sequence[Dict[str, Any]],
    max_neurons: int,
    *,
    y_minimum: float = 0.18,
    y_maximum: float = 0.70,
):
    dimensions = [stages[0]["in_features"], *[stage["out_features"] for stage in stages]]
    indices = [display_indices(dimension, max_neurons) for dimension in dimensions]
    x_positions = np.linspace(0.07, 0.93, len(dimensions))
    y_positions = [np.linspace(y_minimum, y_maximum, len(column)).tolist() for column in indices]
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


def _first_dense_input_vector(
    records: Dict[str, Dict[str, np.ndarray]],
    first_stage_name: str,
    fallback: np.ndarray,
) -> np.ndarray:
    """Return the actual representation consumed by the first visible Linear layer."""
    record = records.get(first_stage_name, {})
    inputs = record.get("input")
    if inputs is None:
        return np.asarray(fallback, dtype=float).ravel()
    values = np.asarray(inputs[0], dtype=float)
    return values.ravel() if values.size else np.asarray(fallback, dtype=float).ravel()


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
    values = [_edge_value(edge, state[0], state[3], mode) for state in frame_states.values() for edge in edges]
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
    evolution_mode: str,
    update_reference: str,
    update_scale_mode: str,
    show_update_panel: bool,
    show_loss_panel: bool,
    loss_name: str,
    show_backpropagation: bool,
) -> List[Dict[str, Any]]:
    deep_graph = len(stages) > 4
    has_bottom_panel = bool(show_update_panel or show_loss_panel)
    graph_axis_domain = (0.40, 1.0) if has_bottom_panel else (0.0, 1.0)
    graph_content_domain = (0.40, 0.92) if has_bottom_panel else (0.0, 1.0)
    dimension_y = 0.05 * (
        (graph_content_domain[1] - graph_content_domain[0])
        / (graph_axis_domain[1] - graph_axis_domain[0])
    )
    if deep_graph:
        depth = len(stages)
        model_formula = (
            rf"\hat{{\mathbf{{y}}}}=\mathbf{{a}}^{{({depth})}},\quad "
            rf"\mathbf{{z}}^{{(\ell)}}=\Theta^{{(\ell)}}\mathbf{{a}}^{{(\ell-1)}}+"
            rf"\theta_0^{{(\ell)}},\quad "
            rf"\mathbf{{a}}^{{(\ell)}}=\phi_\ell(\mathbf{{z}}^{{(\ell)}}),\;"
            rf"\ell=1,\ldots,{depth}"
        )
    else:
        model_formula = composed_dense_function(stages)
    forward_tex = (
        r"\text{Feed forward: }\mathbf{z}^{(\ell)}=\Theta^{(\ell)}\mathbf{a}^{(\ell-1)}+\boldsymbol{\theta}_0^{(\ell)},\;"
        r"\mathbf{a}^{(\ell)}=\phi_\ell(\mathbf{z}^{(\ell)})"
    )
    backprop_tex = (
        r"\text{Backpropagation overlay: }\nabla_{\Theta^{(\ell)}}\mathcal{L}="
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
        r"\text{neutral base}"
        if evolution_mode == "updates"
        else (
            r"\theta_{ji}^{(\ell)}"
            if edge_color_mode == "weight"
            else r"s_{ji}^{(\ell)}=\theta_{ji}^{(\ell)}a_i^{(\ell-1)}"
        )
    )
    edge_title_tex = "" if evolution_mode == "updates" else f"${edge_heatmap_tex}$"
    node_scale_tex = r"a_j^{(\ell)}" if node_color_mode == "value" else r"\widetilde a_j^{(\ell)}"
    shows_updates = evolution_mode in {"updates", "hybrid"}
    edge_title_y = 0.98 if shows_updates and show_update_panel else (0.93 if shows_updates else 0.89)
    node_title_y = 0.52 if show_update_panel else (0.31 if shows_updates else 0.48)
    # Header content occupies invariant paper rows. Mode-specific content may
    # add a row, but it never moves feed-forward mathematics beside or through
    # heatmap definitions.
    legend_y = GRAPH_PAPER_ROWS["legend"] + (0.02 if shows_updates else 0.0)
    activity_legend_y = GRAPH_PAPER_ROWS["activity"] + (0.03 if shows_updates else 0.0)
    if show_backpropagation:
        # Backpropagation owns a separate row.  Keeping it out of the forward
        # annotation prevents the combined MathJax box from reaching the
        # right-hand colorbar on notebook-width displays.
        legend_y -= 0.04
        activity_legend_y -= 0.02
    annotations: List[Dict[str, Any]] = [
        {
            "x": 0.5,
            "y": GRAPH_PAPER_ROWS["model"],
            "xref": "paper",
            "yref": "paper",
            "text": f"${model_formula}$",
            "showarrow": False,
            "font": {"size": 18, "color": NEURAL_COLORS["text"]},
        },
        {
            "x": 0.56,
            "y": GRAPH_PAPER_ROWS["phase"],
            "xref": "paper",
            "yref": "paper",
            "text": f"${forward_tex}$",
            "showarrow": False,
            "font": {"size": 14, "color": NEURAL_COLORS["text"]},
        },
        {
            "x": 0.01,
            "y": legend_y,
            "xref": "paper",
            "yref": "paper",
            "text": (
                f"${node_heatmap_tex}"
                rf"\qquad\text{{Edge heatmap: }}{edge_heatmap_tex}$"
            ),
            "showarrow": False,
            "xanchor": "left",
            "font": {"size": 13, "color": NEURAL_COLORS["muted"]},
        },
        {
            "x": 0.01,
            "y": activity_legend_y,
            "xref": "paper",
            "yref": "paper",
            "text": (
                r"$\text{Activity glow: }s_{ji}^{(\ell)}="
                r"\theta_{ji}^{(\ell)}a_i^{(\ell-1)},\quad"
                r"\text{thickness}\propto|s_{ji}^{(\ell)}|$"
            ),
            "showarrow": False,
            "xanchor": "left",
            "font": {"size": 13, "color": NEURAL_COLORS["muted"]},
        },
        {
            "x": 0.995,
            "y": edge_title_y,
            "xref": "paper",
            "yref": "paper",
            "text": edge_title_tex,
            "showarrow": False,
            "xanchor": "right",
            "font": {"size": 14, "color": NEURAL_COLORS["text"]},
        },
        {
            "x": 0.995,
            "y": node_title_y,
            "xref": "paper",
            "yref": "paper",
            "text": f"${node_scale_tex}$",
            "showarrow": False,
            "xanchor": "right",
            "font": {"size": 14, "color": NEURAL_COLORS["text"]},
        },
    ]
    if show_backpropagation:
        annotations.extend(
            [
                {
                    "x": 0.56,
                    "y": GRAPH_PAPER_ROWS["backpropagation"],
                    "xref": "paper",
                    "yref": "paper",
                    "text": f"${backprop_tex}$",
                    "showarrow": False,
                    "font": {"size": 14, "color": NEURAL_COLORS["text"]},
                },
                {
                    "x": 0.99,
                    "y": legend_y,
                    "xref": "paper",
                    "yref": "paper",
                    "text": "<span style='color:#8f2942'>- -</span> recorded backpropagation gradient",
                    "showarrow": False,
                    "xanchor": "right",
                    "font": {"size": 11, "color": NEURAL_COLORS["muted"]},
                },
            ]
        )
    if shows_updates:
        reference_text = (
            r"\text{previous displayed checkpoint}" if update_reference == "previous" else r"\text{initial checkpoint}"
        )
        annotations.extend(
            [
                {
                    "x": 0.01,
                    "y": GRAPH_PAPER_ROWS["update"],
                    "xref": "paper",
                    "yref": "paper",
                    "text": (
                        r"$\text{Update halo: }\Delta\theta_t="
                        r"\theta_t-\theta_{\mathrm{ref}}"
                        rf",\quad \mathrm{{ref}}={reference_text}"
                        r",\quad \text{width and opacity}\propto|\Delta\theta_t|$"
                    ),
                    "showarrow": False,
                    "xanchor": "left",
                    "font": {"size": 13, "color": NEURAL_COLORS["muted"]},
                },
                {
                    "x": 0.995,
                    "y": 0.77 if show_update_panel else 0.62,
                    "xref": "paper",
                    "yref": "paper",
                    "text": (
                        r"$\widetilde{\Delta\theta}\;\text{(per frame)}$"
                        if update_scale_mode == "frame"
                        else r"$\Delta\theta\;\text{(global scale)}$"
                    ),
                    "showarrow": False,
                    "xanchor": "right",
                    "font": {"size": 14, "color": NEURAL_COLORS["text"]},
                },
            ]
        )
    if show_update_panel:
        annotations.append(
            {
                "x": 0.76 if show_loss_panel else 0.54,
                "y": 0.195,
                "xref": "paper",
                "yref": "paper",
                "text": "<b>Recorded update diagnostics</b>",
                "showarrow": False,
                "xanchor": "center",
                "yanchor": "middle",
                "font": {"size": 12, "color": NEURAL_COLORS["text"]},
            }
        )
    if show_loss_panel:
        annotations.append(
            {
                "x": 0.31 if show_update_panel else 0.50,
                "y": 0.195,
                "xref": "paper",
                "yref": "paper",
                "text": rf"$\text{{Objective trajectory: }}\mathcal{{L}}_t\quad\mathrm{{{loss_name}}}$",
                "showarrow": False,
                "xanchor": "center",
                "font": {"size": 14, "color": NEURAL_COLORS["text"]},
            }
        )
    for column_index, (x_position, dimension) in enumerate(zip(x_positions, dimensions)):
        if column_index == 0:
            dimension_tex = rf"\mathbf{{a}}^{{(0)}}=\mathbf{{x}}\in\mathbb{{R}}^{{{dimension}}}"
        else:
            stage = stages[column_index - 1]
            dimension_tex = rf"\mathbf{{a}}^{{({column_index})}}\in\mathbb{{R}}^{{{dimension}}}"
            if not deep_graph:
                dimension_tex += (
                    rf",\;\Theta^{{({column_index})}}\in\mathbb{{R}}^{{{dimension}\times {stage['in_features']}}}"
                )
        annotations.append(
            {
                "x": x_position,
                "y": dimension_y,
                "xref": "x",
                "yref": "y",
                "text": f"${dimension_tex}$",
                "showarrow": False,
                "font": {"size": 12 if deep_graph else 14, "color": NEURAL_COLORS["muted"]},
            }
        )
    return annotations


def _dynamic_label_traces(
    stages: Sequence[Dict[str, Any]],
    parameters: Dict[str, np.ndarray],
    step: str,
    dec: int,
    is_final: bool,
    metrics: Dict[str, float],
    show_update_panel: bool,
    show_loss_panel: bool,
    frame_context: str,
) -> List[go.Scatter]:
    final_tex = r"\quad\text{(final weights)}" if is_final else ""
    readout_decimals = max(dec, 4)
    graph_axis_domain = (0.40, 1.0) if (show_update_panel or show_loss_panel) else (0.0, 1.0)

    def paper_to_data(value: float) -> float:
        return (value - graph_axis_domain[0]) / (graph_axis_domain[1] - graph_axis_domain[0])

    traces = [
        go.Scatter(
            x=[0.5],
            y=[paper_to_data(GRAPH_PAPER_ROWS["parameters"])],
            mode="text",
            text=[f"${compact_parameter_line(stages, parameters, dec=readout_decimals)}$"],
            textposition="middle center",
            textfont={"size": 15, "color": NEURAL_COLORS["muted"]},
            cliponaxis=False,
            hoverinfo="skip",
            showlegend=False,
            name="parameter readout",
        ),
        go.Scatter(
            x=[0.07],
            y=[paper_to_data(GRAPH_PAPER_ROWS["phase"])],
            mode="text",
            text=[
                rf"$t={step}{final_tex}$"
                f"<br><span style='font-size:10px;color:{NEURAL_COLORS['muted']}'>{frame_context}</span>"
            ],
            textposition="middle center",
            textfont={"size": 14, "color": NEURAL_COLORS["text"]},
            cliponaxis=False,
            hoverinfo="skip",
            showlegend=False,
            name="training step readout",
        ),
    ]
    if show_update_panel:
        gradient_text = f"{metrics['gradient_norm']:.3e}" if np.isfinite(metrics["gradient_norm"]) else "not recorded"
        alignment_text = (
            f"{metrics['update_gradient_alignment']:+.3f}"
            if np.isfinite(metrics["update_gradient_alignment"])
            else "n/a (aggregate/reference)"
        )
        traces.append(
            go.Scatter(
                x=[0.50],
                y=[0.42],
                mode="text",
                text=[
                    f"parameter norm ‖Θₜ‖₂={metrics['theta_norm']:.4f} · "
                    f"update norm ‖ΔΘₜ‖₂={metrics['delta_norm']:.3e} · "
                    f"relative update={metrics['relative_update']:.3e}<br>"
                    f"gradient norm ‖∇ΘL‖₂={gradient_text} · "
                    f"direction cosine cos(ΔΘₜ, −∇ΘL)={alignment_text}"
                ],
                textposition="middle center",
                textfont={"size": 12, "color": NEURAL_COLORS["text"]},
                cliponaxis=False,
                hoverinfo="skip",
                showlegend=False,
                name="update summary",
                xaxis="x3" if show_loss_panel else "x2",
                yaxis="y3" if show_loss_panel else "y2",
            )
        )
    return traces


def _loss_panel_traces(
    recorded_steps: np.ndarray,
    recorded_loss: np.ndarray,
    state: Dict[str, Any],
) -> List[go.Scatter]:
    """Render recorded objective values plus one synchronized state marker."""
    marker_symbol = "circle" if state["semantic"] else "circle-open"
    marker_name = "recorded loss" if state["semantic"] else "perceptual loss marker"
    return [
        go.Scatter(
            x=recorded_steps,
            y=recorded_loss,
            mode="lines+markers",
            line={"color": NEURAL_COLORS["output"], "width": 2.0},
            marker={"size": 5},
            hovertemplate="recorded step=%{x}<br>loss=%{y:.6g}<extra></extra>",
            showlegend=False,
            name="recorded objective curve",
            xaxis="x2",
            yaxis="y2",
        ),
        go.Scatter(
            x=[state["loss_step"]],
            y=[state["loss_value"]],
            mode="markers",
            marker={
                "size": 12,
                "symbol": marker_symbol,
                "color": NEURAL_COLORS["activation"],
                "line": {"color": NEURAL_COLORS["text"], "width": 1},
            },
            hovertemplate=(f"{marker_name}<br>step=%{{x:.3g}}<br>loss=%{{y:.6g}}<extra></extra>"),
            showlegend=False,
            name=marker_name,
            xaxis="x2",
            yaxis="y2",
        ),
    ]


def _edge_traces(
    edges: Sequence[Dict[str, Any]],
    parameters: Dict[str, np.ndarray],
    reference: Dict[str, np.ndarray],
    gradients: Dict[str, np.ndarray],
    node_values: Sequence[np.ndarray],
    edge_minimum: float,
    edge_maximum: float,
    signal_minimum: float,
    signal_maximum: float,
    gradient_scale: float,
    edge_color_mode: str,
    evolution_mode: str,
    update_limit: float,
    update_reference: str,
    top_k_updates: int | None,
    show_backpropagation: bool,
    dec: int,
) -> List[go.Scatter]:
    traces: List[go.Scatter] = []
    edge_deltas = []
    for edge in edges:
        stage = edge["stage"]
        weights = parameters[stage["weight_name"]]
        reference_weights = reference.get(stage["weight_name"], weights)
        edge_deltas.append(
            float(
                weights[edge["target_index"], edge["source_index"]]
                - reference_weights[edge["target_index"], edge["source_index"]]
            )
        )
    threshold = 0.0
    if top_k_updates is not None and edge_deltas:
        retained = min(top_k_updates, len(edge_deltas))
        threshold = float(np.sort(np.abs(edge_deltas))[-retained])
    for edge in edges:
        stage = edge["stage"]
        weights = parameters[stage["weight_name"]]
        row = edge["target_index"]
        column = edge["source_index"]
        weight = float(weights[row, column])
        source_activation = float(node_values[edge["stage_position"]][column])
        signal = weight * source_activation
        signal_scale = max(abs(signal_minimum), abs(signal_maximum), 1e-12)
        activity_ratio = min(abs(signal) / signal_scale, 1.0)
        encoded_value = weight if edge_color_mode == "weight" else signal
        reference_weights = reference.get(stage["weight_name"], weights)
        delta = float(weight - reference_weights[row, column])
        delta_ratio = min(abs(delta) / max(update_limit, 1e-12), 1.0)
        has_gradient = stage["weight_name"] in gradients
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
        activity_hover = (
            f"<b>Forward activity glow</b><br>layer={stage['index']}<br>"
            f"theta[{row + 1},{column + 1}]={weight:.{dec}f}<br>"
            f"source output={source_activation:.{dec}f}<br>"
            f"signal theta * a={signal:+.{max(dec, 5)}f}<br>"
            f"global relative magnitude={activity_ratio:.3f}"
        )
        traces.append(
            go.Scatter(
                x=edge["x"],
                y=edge["y"],
                mode="lines",
                line={
                    "color": _signed_color(
                        signal,
                        signal_minimum,
                        signal_maximum,
                        NEURAL_COLORS["weight_min"],
                        NEURAL_COLORS["weight_mid"],
                        NEURAL_COLORS["weight_max"],
                    ),
                    "width": 4.0 + 10.0 * activity_ratio,
                },
                opacity=0.06 + 0.48 * activity_ratio,
                customdata=[activity_hover, activity_hover],
                hovertemplate="%{customdata}<extra></extra>",
                showlegend=False,
                name="neural graph activity glow",
            )
        )
        if evolution_mode in {"updates", "hybrid"}:
            emphasized = top_k_updates is None or abs(delta) >= threshold
            halo_opacity = (0.12 + 0.88 * delta_ratio) * (1.0 if emphasized else 0.10)
            update_hover = (
                f"<b>Actual parameter update</b><br>layer={stage['index']}<br>"
                f"reference={'previous displayed checkpoint' if update_reference == 'previous' else 'initial checkpoint'}<br>"
                f"theta={weight:+.{dec}f}<br>delta theta={delta:+.{max(dec, 5)}f}<br>"
                f"normalized magnitude={delta_ratio:.3f}"
            )
            traces.append(
                go.Scatter(
                    x=edge["x"],
                    y=edge["y"],
                    mode="lines",
                    line={
                        "color": _signed_color(
                            delta,
                            -update_limit,
                            update_limit,
                            NEURAL_COLORS["update_negative"],
                            NEURAL_COLORS["update_mid"],
                            NEURAL_COLORS["update_positive"],
                        ),
                        "width": 3.0 + 12.0 * delta_ratio,
                    },
                    opacity=halo_opacity,
                    customdata=[update_hover, update_hover],
                    hovertemplate="%{customdata}<extra></extra>",
                    showlegend=False,
                    name="parameter update halo",
                )
            )
        base_color = (
            NEURAL_COLORS["grid"]
            if evolution_mode == "updates"
            else _weight_color(encoded_value, edge_minimum, edge_maximum)
        )
        traces.append(
            go.Scatter(
                x=edge["x"],
                y=edge["y"],
                mode="lines",
                line={
                    "color": base_color,
                    "width": 1.4 if evolution_mode == "updates" else 2.4,
                },
                customdata=[edge_hover, edge_hover],
                hovertemplate="%{customdata}<extra></extra>",
                showlegend=False,
                name="absolute parameter or signal",
            )
        )
        gradient_hover = (
            f"<b>Backpropagation</b><br>layer={stage['index']}<br>"
            + (
                f"dL/dtheta[{row + 1},{column + 1}]={gradient:+.{dec}f}<br>"
                if has_gradient
                else "gradient hidden: perceptual interpolation frame<br>"
            )
            + f"delta theta={delta:+.{dec}f}"
        )
        if show_backpropagation:
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
                    opacity=(0.10 + 0.58 * gradient_ratio) if has_gradient else 0.0,
                    customdata=[gradient_hover, gradient_hover],
                    hovertemplate="%{customdata}<extra></extra>",
                    showlegend=False,
                    name="recorded backpropagation gradient",
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
    show_backpropagation: bool,
    dec: int,
    marker_size: float,
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
                f"Theta[{node_index + 1},:]={_plain_vector(weight_row, dec)}"
                + (
                    f"<br>grad Theta[{node_index + 1},:]={_plain_vector(gradient_rows[node_index], dec)}"
                    if show_backpropagation
                    else ""
                )
            )
        traces.append(
            go.Scatter(
                x=[x_positions[column_index]] * len(y_values),
                y=y_values,
                mode="markers",
                marker={
                    "size": marker_size,
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
                name="neural graph activations",
            )
        )
    return traces


def _colorbar_trace(
    minimum: float,
    maximum: float,
    *,
    colorscale: Sequence[Sequence[Any]],
    y: float,
    length: float = 0.34,
    x: float = 1.015,
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
                "len": length,
                "y": y,
                "x": x,
                "xanchor": "left",
                "tickformat": ".3f",
                "tickfont": {"size": 11, "color": NEURAL_COLORS["text"]},
            },
        },
        hoverinfo="skip",
        showlegend=False,
        name="neural graph color scale",
    )


def _adaptive_node_marker_size(
    indices: Sequence[Sequence[int]],
    y_positions: Sequence[Sequence[float]],
    *,
    graph_pixel_height: float,
) -> float:
    """Fit node diameters to the narrowest visible vertical separation.

    Plotly marker diameters are expressed in screen pixels while graph
    positions use data coordinates. A count-only rule therefore overlaps
    nodes whenever a loss/update panel compresses the graph domain. This rule
    derives a conservative pixel clearance from the actual geometry and keeps
    a visible gap between every pair of nodes in every rendered column.
    """
    largest_visible_column = max((len(column) for column in indices), default=1)
    count_cap = 28.0 if largest_visible_column <= 5 else max(11.0, 42.0 - 3.0 * largest_visible_column)
    separations = [
        abs(float(current) - float(previous))
        for column in y_positions
        for previous, current in zip(column, column[1:])
    ]
    if not separations:
        return count_cap
    # Reserve 30% for a true inter-node gap and another 20% for compact-size
    # variants whose final plot domain is shorter than the builder baseline.
    spacing_cap = min(separations) * max(float(graph_pixel_height), 1.0) * 0.56
    return max(9.0, min(count_cap, float(np.floor(spacing_cap))))


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
    evolution_mode: str = "absolute",
    update_reference: str = "previous",
    update_scale: str = "global",
    show_update_panel: bool | None = None,
    show_loss_panel: bool = False,
    show_backpropagation: bool = False,
    top_k_updates: int | None = None,
    interpolation_frames: int = 0,
    dec: int = 3,
) -> go.Figure:
    """Animate exact node outputs and edge values with backpropagation cues."""
    stages = dense_stages(model)
    if not stages:
        raise ValueError("The animated graph currently requires at least one torch.nn.Linear layer.")
    leaf_modules = list(_leaf_modules(model))
    represented_module_names = {
        name for stage in stages for name in (stage.get("name"), stage.get("activation_name")) if name
    }
    dropout_modules = [
        module
        for _name, module in leaf_modules
        if module.__class__.__name__ in {"Dropout", "Dropout1d", "Dropout2d", "Dropout3d"}
    ]
    omitted_module_types: List[str] = []
    for name, module in leaf_modules:
        module_type = module.__class__.__name__
        if name in represented_module_names or module in dropout_modules:
            continue
        if module_type not in omitted_module_types:
            omitted_module_types.append(module_type)
    if max_neurons < 2:
        raise ValueError("max_neurons must be at least 2.")
    if node_color_mode not in {"value", "relative"}:
        raise ValueError("node_color_mode must be 'value' or 'relative'.")
    if edge_color_mode not in {"weight", "signal"}:
        raise ValueError("edge_color_mode must be 'weight' or 'signal'.")
    if evolution_mode not in {"absolute", "updates", "hybrid"}:
        raise ValueError("evolution_mode must be 'absolute', 'updates', or 'hybrid'.")
    if update_reference not in {"previous", "initial"}:
        raise ValueError("update_reference must be 'previous' or 'initial'.")
    if update_scale not in {"global", "frame"}:
        raise ValueError("update_scale must be 'global' or 'frame'.")
    if top_k_updates is not None and top_k_updates < 1:
        raise ValueError("top_k_updates must be at least 1 or None.")
    if interpolation_frames < 0 or interpolation_frames > 20:
        raise ValueError("interpolation_frames must be between 0 and 20.")
    if show_update_panel is None:
        show_update_panel = evolution_mode in {"updates", "hybrid"}
    recorded_loss = np.asarray(history.get("loss", []), dtype=float)
    if show_loss_panel and (not recorded_loss.size or not np.isfinite(recorded_loss).any()):
        raise ValueError("show_loss_panel=True requires finite recorded history['loss'] values.")
    loss_name = str(history.get("training_config", {}).get("loss", "recorded loss"))
    observation_phases = {
        str(frame.get("observation_phase", "unspecified")) for frame in history.get("frame_semantics", [])
    }
    loss_phase = next(iter(observation_phases)) if len(observation_phases) == 1 else "mixed/unspecified"
    shows_updates = evolution_mode in {"updates", "hybrid"}
    reserves_update_space = shows_updates or show_update_panel or show_loss_panel
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
    input_values = (
        input_array if input_array.ndim == 1 else np.mean(input_array.reshape(input_array.shape[0], -1), axis=0)
    )
    has_disclosures = bool(omitted_module_types or dropout_modules)
    has_bottom_panel = bool(show_update_panel or show_loss_panel)
    # The invisible graph axis reaches the top of the plotting canvas so
    # animated readouts can occupy invariant paper rows above the network.
    # Node/edge content remains bounded to the historical 0.92 ceiling.
    graph_axis_domain = (0.40, 1.0) if has_bottom_panel else (0.0, 1.0)
    graph_content_domain = (0.40, 0.92) if has_bottom_panel else (0.0, 1.0)
    graph_content_scale = (
        (graph_content_domain[1] - graph_content_domain[0])
        / (graph_axis_domain[1] - graph_axis_domain[0])
    )
    baseline_network_y_maximum = (
        (0.32 if has_disclosures else 0.40)
        if has_bottom_panel
        else (0.52 if has_disclosures else 0.60)
    )
    baseline_network_top = graph_content_domain[0] + (
        graph_content_domain[1] - graph_content_domain[0]
    ) * baseline_network_y_maximum
    network_top = (
        min(
            baseline_network_top,
            GRAPH_PAPER_ROWS["update"] - UPDATE_TO_NETWORK_MINIMUM_GAP,
        )
        if shows_updates
        else baseline_network_top
    )
    network_y_maximum = (network_top - graph_axis_domain[0]) / (
        graph_axis_domain[1] - graph_axis_domain[0]
    )
    dimensions, indices, x_positions, y_positions, edges = _graph_geometry(
        stages,
        max_neurons,
        y_minimum=(0.14 if reserves_update_space else 0.18) * graph_content_scale,
        y_maximum=network_y_maximum,
    )
    base_figure_height = 1000 if has_bottom_panel else 760
    base_top_margin = 165
    base_bottom_margin = 75 if has_bottom_panel else 115
    graph_pixel_height = (
        base_figure_height - base_top_margin - base_bottom_margin
    ) * (graph_content_domain[1] - graph_content_domain[0])
    node_marker_size = _adaptive_node_marker_size(
        indices,
        y_positions,
        graph_pixel_height=graph_pixel_height,
    )
    weight_minimum, weight_maximum, gradient_scale = _global_limits(
        history,
        stages,
    )

    initial_parameters = parameter_snapshot(history, 0)
    semantic_states: List[Dict[str, Any]] = []
    frame_states: Dict[int, tuple] = {}
    for position, source_index in enumerate(selected_frames):
        frame_index = int(source_index)
        parameters = parameter_snapshot(history, frame_index)
        buffers = buffer_snapshot(history, frame_index)
        reference = (
            initial_parameters
            if update_reference == "initial"
            else parameter_snapshot(history, int(selected_frames[max(0, position - 1)]))
        )
        gradients = gradient_snapshot(history, frame_index)
        _, records = run_torch_forward(model, input_sample, parameters, buffers)
        dense_input_values = _first_dense_input_vector(
            records,
            stages[0]["name"],
            input_values,
        )
        node_values = _node_values(stages, dense_input_values, records)
        previous_is_adjacent = position > 0 and frame_index == int(selected_frames[position - 1]) + 1
        frame_semantics = history.get("frame_semantics", [])
        temporal_semantics_are_exact = (
            frame_index < len(frame_semantics)
            and frame_semantics[frame_index].get("parameter_phase") == "post_step"
            and frame_semantics[frame_index].get("gradient_phase") == "post_backward"
        )
        metrics = _update_metrics(
            stages,
            parameters,
            reference,
            gradients,
            alignment_is_exact=(
                update_reference == "previous" and previous_is_adjacent and temporal_semantics_are_exact
            ),
        )
        state = {
            "name": f"step_{frame_index}",
            "frame_index": frame_index,
            "step": str(int(steps[frame_index])),
            "parameters": parameters,
            "buffers": buffers,
            "reference": reference,
            "gradients": gradients,
            "node_values": node_values,
            "metrics": metrics,
            "context": "recorded checkpoint",
            "semantic": True,
            "loss_step": float(steps[frame_index]),
            "loss_value": float(recorded_loss[frame_index]) if recorded_loss.size else float("nan"),
        }
        semantic_states.append(state)
        frame_states[frame_index] = (parameters, reference, gradients, node_values)
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
    signal_minimum, signal_maximum = _edge_limits(
        edges,
        frame_states,
        "signal",
        (weight_minimum, weight_maximum),
    )
    global_update_limit = _global_update_limit(
        history,
        stages,
        update_reference,
        [int(value) for value in selected_frames],
    )
    names = _stage_parameter_names(stages)
    display_states: List[Dict[str, Any]] = []
    for position, state in enumerate(semantic_states):
        if position > 0 and interpolation_frames:
            start = semantic_states[position - 1]
            for subframe in range(1, interpolation_frames + 1):
                ratio = subframe / (interpolation_frames + 1)
                parameters = _interpolate_snapshots(
                    start["parameters"],
                    state["parameters"],
                    ratio,
                )
                reference = initial_parameters if update_reference == "initial" else start["parameters"]
                _, records = run_torch_forward(
                    model,
                    input_sample,
                    parameters,
                    state["buffers"],
                )
                dense_input_values = _first_dense_input_vector(
                    records,
                    stages[0]["name"],
                    input_values,
                )
                node_values = _node_values(stages, dense_input_values, records)
                display_states.append(
                    {
                        "name": f"transition_{position}_{subframe}",
                        "frame_index": state["frame_index"],
                        "step": f"{start['step']}→{state['step']}",
                        "parameters": parameters,
                        "buffers": state["buffers"],
                        "reference": reference,
                        "gradients": {},
                        "node_values": node_values,
                        "metrics": _update_metrics(
                            stages,
                            parameters,
                            reference,
                            {},
                            alignment_is_exact=False,
                        ),
                        "context": f"perceptual interpolation · α={ratio:.2f} · not an optimizer step",
                        "semantic": False,
                        "loss_step": (float(start["loss_step"]) * (1.0 - ratio) + float(state["loss_step"]) * ratio),
                        "loss_value": (float(start["loss_value"]) * (1.0 - ratio) + float(state["loss_value"]) * ratio),
                    }
                )
        display_states.append(state)

    def frame_update_limit(state: Dict[str, Any]) -> float:
        if update_scale == "global":
            return global_update_limit
        values = _parameter_delta_values(
            names,
            state["parameters"],
            state["reference"],
        )
        return max(float(np.max(np.abs(values))), 1e-12)

    def frame_payload(state: Dict[str, Any]):
        parameters = state["parameters"]
        reference = state["reference"]
        gradients = state["gradients"]
        node_values = state["node_values"]
        data = _edge_traces(
            edges,
            parameters,
            reference,
            gradients,
            node_values,
            edge_minimum,
            edge_maximum,
            signal_minimum,
            signal_maximum,
            gradient_scale,
            edge_color_mode,
            evolution_mode,
            frame_update_limit(state),
            update_reference,
            top_k_updates,
            show_backpropagation,
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
                show_backpropagation,
                dec,
                node_marker_size,
            )
        )
        data.extend(
            _dynamic_label_traces(
                stages,
                parameters,
                state["step"],
                dec,
                state["semantic"] and state["frame_index"] == steps.size - 1,
                state["metrics"],
                show_update_panel,
                show_loss_panel,
                state["context"],
            )
        )
        if show_loss_panel:
            data.extend(_loss_panel_traces(steps, recorded_loss, state))
        return data

    first_data = frame_payload(display_states[0])
    annotations = _graph_annotations(
        stages,
        dimensions,
        x_positions,
        node_color_mode,
        edge_color_mode,
        evolution_mode,
        update_reference,
        update_scale,
        show_update_panel,
        show_loss_panel,
        loss_name,
        show_backpropagation,
    )
    disclosures = []
    if omitted_module_types:
        disclosures.append(
            "Dense replay shows Linear stages and their attached activations only; "
            f"omitted prefix/intermediate modules: {', '.join(omitted_module_types)}. "
            "Use view='blocks' for the complete executed topology."
        )
    if dropout_modules:
        probabilities = ", ".join(f"p={float(getattr(module, 'p', 0.5)):.3g}" for module in dropout_modules)
        disclosures.append(
            f"Dropout present ({probabilities}). Dense replay uses evaluation mode: "
            "historical stochastic masks were not recorded; activity glow is signal, not dropout."
        )
    disclosure_top = GRAPH_PAPER_ROWS["disclosure"]
    for disclosure_index, disclosure in enumerate(disclosures):
        annotations.append(
            {
                "x": 0.01,
                "y": disclosure_top - 0.04 * disclosure_index,
                "xref": "paper",
                "yref": "paper",
                "text": disclosure,
                "showarrow": False,
                "xanchor": "left",
                "font": {"size": 11, "color": NEURAL_COLORS["muted"]},
            }
        )
    colorbar_traces = []
    if evolution_mode != "updates":
        colorbar_traces.append(
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
                y=0.86 if show_update_panel else (0.80 if shows_updates else 0.70),
                length=0.20 if show_update_panel else (0.24 if shows_updates else 0.34),
            )
        )
    if shows_updates:
        displayed_update_limit = 1.0 if update_scale == "frame" else global_update_limit
        colorbar_traces.append(
            _colorbar_trace(
                -displayed_update_limit,
                displayed_update_limit,
                colorscale=_signed_colorscale(
                    -displayed_update_limit,
                    displayed_update_limit,
                    NEURAL_COLORS["update_negative"],
                    NEURAL_COLORS["update_mid"],
                    NEURAL_COLORS["update_positive"],
                ),
                y=0.65 if show_update_panel else 0.50,
                length=0.20 if show_update_panel else 0.22,
            )
        )
    colorbar_traces.append(
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
            y=0.39 if show_update_panel else (0.20 if shows_updates else 0.29),
            length=0.20 if show_update_panel else (0.24 if shows_updates else 0.34),
        )
    )
    figure = go.Figure(data=[*first_data, *colorbar_traces])
    dynamic_trace_indices = list(range(len(first_data)))
    frames = []
    slider_steps = []
    for state in display_states:
        data = frame_payload(state)
        frame_name = state["name"]
        frames.append(
            go.Frame(
                name=frame_name,
                data=data,
                traces=dynamic_trace_indices,
            )
        )
    for state in semantic_states:
        is_final = state["frame_index"] == steps.size - 1
        slider_steps.append(
            {
                "label": f"{state['step']}{' final' if is_final else ''}",
                "method": "animate",
                "args": [[state["name"]], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}],
            }
        )
    figure.frames = frames
    if title is None:
        title = "Mathematical network: parameter and signal evolution"
    layout = neural_layout(title, height=base_figure_height)
    layout["title"]["x"] = 0.5
    layout["title"]["xanchor"] = "center"
    layout["title"]["y"] = 0.985
    layout["title"]["yanchor"] = "top"
    layout["margin"] = {
        "t": 165,
        "r": 165 if shows_updates else 145,
        "b": 75 if has_bottom_panel else 115,
        "l": 65,
    }
    layout["meta"] = {
        "mlektic_neural_evolution": {
            "schema_version": 1,
            "evolution_mode": evolution_mode,
            "edge_color_mode": edge_color_mode,
            "node_color_mode": node_color_mode,
            "update_reference": update_reference,
            "update_scale": update_scale,
            "global_update_limit": global_update_limit,
            "activity_glow": {
                "quantity": "theta_ji * a_i",
                "scale": "global",
                "minimum": signal_minimum,
                "maximum": signal_maximum,
                "magnitude_channels": ["width", "opacity"],
                "sign_channel": "color",
            },
            "show_update_panel": show_update_panel,
            "show_loss_panel": show_loss_panel,
            "show_backpropagation": show_backpropagation,
            "section_layout": {
                "coordinate_system": "paper rows mapped to graph data coordinates",
                "rows": dict(GRAPH_PAPER_ROWS),
                "variant_invariant": True,
                "graph_axis_domain": list(graph_axis_domain),
                "network_content_domain": list(graph_content_domain),
                "network_y_max_reduced_for_disclosures": has_disclosures,
                "network_top": network_top,
                "minimum_update_to_network_gap": UPDATE_TO_NETWORK_MINIMUM_GAP,
            },
            "loss_panel": {
                "loss_name": loss_name,
                "observation_phase": loss_phase,
                "recorded_values_only": True,
                "perceptual_markers_are_evaluations": False,
            },
            "dropout": {
                "present": bool(dropout_modules),
                "probabilities": [float(getattr(module, "p", 0.5)) for module in dropout_modules],
                "historical_masks_recorded": False,
                "dense_replay_mode": "evaluation",
                "activity_glow_represents_dropout": False,
            },
            "dense_scope": {
                "represented_linear_stages": len(stages),
                "visible_neuron_counts": [len(column) for column in indices],
                "max_visible_neurons_per_layer": max_neurons,
                "node_marker_size": node_marker_size,
                "node_marker_policy": "actual pixel spacing with a non-overlap gap",
                "input_representation": (
                    "first represented Linear input after omitted modules" if omitted_module_types else "model input"
                ),
                "omitted_module_types": omitted_module_types,
                "full_topology_view": "blocks",
            },
            "top_k_updates": top_k_updates,
            "semantic_frames": len(semantic_states),
            "perceptual_frames_per_transition": interpolation_frames,
            "display_frames": len(display_states),
            "perceptual_frames_are_optimizer_steps": False,
            "gradient_alignment_policy": (
                "adjacent recorded checkpoints with post-step parameters and post-backward gradients only"
            ),
        }
    }
    if has_bottom_panel:
        loss_domain = [0.08, 0.54] if show_update_panel and show_loss_panel else [0.23, 0.77]
        update_domain = [0.58, 0.94] if show_loss_panel else [0.23, 0.77]
        layout.update(
            {
                "yaxis": {"domain": list(graph_axis_domain)},
                "xaxis2": {
                    "domain": loss_domain if show_loss_panel else update_domain,
                    "anchor": "y2",
                    "visible": bool(show_loss_panel),
                    "fixedrange": True,
                    "title": {"text": "Recorded training step"} if show_loss_panel else None,
                    "gridcolor": NEURAL_COLORS["grid"],
                },
                "yaxis2": {
                    "domain": [0.035, 0.17],
                    "anchor": "x2",
                    "visible": bool(show_loss_panel),
                    "fixedrange": True,
                    "title": {"text": "Loss"} if show_loss_panel else None,
                    "gridcolor": NEURAL_COLORS["grid"],
                },
            }
        )
        if show_update_panel and show_loss_panel:
            layout.update(
                {
                    "xaxis3": {
                        "domain": update_domain,
                        "range": [0.0, 1.0],
                        "anchor": "y3",
                        "visible": False,
                        "fixedrange": True,
                    },
                    "yaxis3": {
                        "domain": [0.035, 0.17],
                        "range": [0.0, 1.0],
                        "anchor": "x3",
                        "visible": False,
                        "fixedrange": True,
                    },
                }
            )
    panel_shapes = []
    if has_bottom_panel:
        panel_x0 = 0.06 if show_update_panel and show_loss_panel else 0.20
        panel_x1 = 0.95 if show_update_panel and show_loss_panel else 0.80
        panel_shapes.append(
            {
                "type": "rect",
                "xref": "paper",
                "yref": "paper",
                "x0": panel_x0,
                "x1": panel_x1,
                "y0": 0.025,
                "y1": 0.225,
                "line": {"color": NEURAL_COLORS["grid"], "width": 1},
                "fillcolor": NEURAL_COLORS["panel"],
                "layer": "below",
            }
        )
    rendered_frame_duration = max(30, frame_duration // (interpolation_frames + 1))
    animation_menu = {
        "type": "buttons",
        "direction": "left",
        "x": 0.0,
        "y": 0.445 if has_bottom_panel else -0.02,
        **animation_button_style(),
        "buttons": [
            {
                "label": "Play",
                "method": "animate",
                "args": [
                    None,
                    {
                        "frame": {"duration": rendered_frame_duration, "redraw": False},
                        "transition": {"duration": min(rendered_frame_duration, 120)},
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
    slider = {
        "active": 0,
        "currentvalue": {"prefix": "Training step: "},
        "pad": {"t": 12 if has_bottom_panel else 50},
        "steps": slider_steps,
    }
    if has_bottom_panel:
        slider.update(
            {
                "x": 0.08,
                "len": 0.92,
                "y": 0.375,
                "yanchor": "top",
                "pad": {"t": 22},
            }
        )
    figure.update_layout(
        **layout,
        annotations=annotations,
        shapes=panel_shapes,
        updatemenus=[animation_menu],
        sliders=[slider],
        showlegend=False,
    )
    figure.layout.xaxis.update(visible=False, range=[0, 1])
    figure.layout.yaxis.update(visible=False, range=[0, 1])
    return figure


__all__ = ["build_nn_graph_figure"]
