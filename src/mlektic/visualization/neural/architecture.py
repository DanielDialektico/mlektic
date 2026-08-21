"""Mathematical architecture diagram builder for PyTorch neural networks."""

from __future__ import annotations

from typing import Any, Dict, List

import plotly.graph_objects as go

from ...neural.introspection import describe_torch_model
from ...neural.taxonomy import (
    composed_dense_function,
    dense_stages,
    format_hyperparameters,
    select_with_ellipsis,
    shape_tex,
)
from ._style import NEURAL_COLORS, layer_color, neural_layout


def _wrapped_hyperparameters(
    values: Dict[str, Any],
    *,
    max_chars: int,
    limit: int = 4,
) -> str:
    """Return a bounded, explicit multiline summary for one module column.

    Plotly annotations do not constrain plain text to the visual column that
    surrounds their anchor. Architecture diagrams therefore wrap at semantic
    ``name=value`` boundaries instead of relying on the browser to wrap (or
    clip) a long configuration string.
    """
    items = list(values.items())
    parts = [f"{name}={value}" for name, value in items[:limit]]
    if len(items) > limit:
        parts.append("...")
    if not parts:
        return "no configurable<br>hyperparameters"

    lines: List[str] = []
    current = ""
    for part in parts:
        if len(part) > max_chars:
            part = f"{part[: max_chars - 1]}…"
        candidate = f"{current}, {part}" if current else part
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = part
        else:
            current = candidate
    if current:
        lines.append(current)
    return "<br>".join(lines)


def _half_width_for_role(role: str, scale: float) -> float:
    """Return the semantic node half-width after density-aware scaling."""
    base = 0.045 if role == "activation" else 0.055 if role in {"regularization", "reshape"} else 0.065
    return base * scale


def _shape_for_role(role: str, x: float, y: float, color: str, scale: float) -> Dict[str, Any]:
    """Use geometry to distinguish transformations, activations, and reshaping."""
    half_width = _half_width_for_role(role, scale)
    if role == "activation":
        return {
            "type": "circle",
            "x0": x - half_width,
            "x1": x + half_width,
            "y0": y - 0.105 * scale,
            "y1": y + 0.105 * scale,
            "line": {"color": color, "width": 2},
            "fillcolor": "rgba(255,255,255,0.035)",
        }
    if role in {"regularization", "reshape"}:
        return {
            "type": "rect",
            "x0": x - half_width,
            "x1": x + half_width,
            "y0": y - 0.11 * scale,
            "y1": y + 0.11 * scale,
            "line": {"color": color, "width": 2, "dash": "dot"},
            "fillcolor": "rgba(255,255,255,0.035)",
        }
    return {
        "type": "rect",
        "x0": x - half_width,
        "x1": x + half_width,
        "y0": y - 0.12 * scale,
        "y1": y + 0.12 * scale,
        "line": {"color": color, "width": 2},
        "fillcolor": "rgba(255,255,255,0.035)",
    }


def _architecture_formula(model: Any, layer_count: int) -> str:
    stages = dense_stages(model)
    if stages:
        if len(stages) > 4:
            depth = len(stages)
            return (
                rf"\hat{{\mathbf{{y}}}}=\mathbf{{a}}^{{({depth})}},\quad "
                rf"\mathbf{{z}}^{{(\ell)}}=\Theta^{{(\ell)}}\mathbf{{a}}^{{(\ell-1)}}+"
                rf"\theta_0^{{(\ell)}},\quad "
                rf"\mathbf{{a}}^{{(\ell)}}=\phi_\ell(\mathbf{{z}}^{{(\ell)}}),\;"
                rf"\ell=1,\ldots,{depth}"
            )
        return composed_dense_function(stages)
    operators = r"\circ".join(rf"\mathcal{{M}}_{{{index}}}" for index in range(layer_count, 0, -1))
    return rf"\hat{{\mathbf{{y}}}}=({operators})(\mathbf{{x}})"


def _training_line(history: Dict[str, Any] | None) -> str | None:
    if not history:
        return None
    config = history.get("training_config", {})
    optimizer = config.get("optimizer")
    loss = config.get("loss")
    hyperparameters = format_hyperparameters(config.get("optimizer_hyperparameters", {}), limit=4)
    parts = []
    if optimizer:
        parts.append(f"optimizer={optimizer}({hyperparameters})")
    if loss:
        parts.append(f"loss={loss}")
    return " | ".join(parts) if parts else None


def build_nn_architecture_figure(
    model: Any,
    input_sample: Any | None = None,
    *,
    history: Dict[str, Any] | None = None,
    title: str | None = None,
    max_neurons: int = 10,
    max_layers: int = 8,
) -> go.Figure:
    """Draw modules as semantic shapes with dimensions, formulas, and configuration."""
    del max_neurons  # Kept for backward compatibility with the first public release.
    if max_layers < 3:
        raise ValueError("max_layers must be at least 3.")
    layers = describe_torch_model(model, input_sample)
    selected = select_with_ellipsis(layers, max_layers)
    compact_details = len(selected) > 6
    x_positions = [0.07 + index * (0.86 / max(len(selected) - 1, 1)) for index in range(len(selected))]
    horizontal_gap = 0.86 / max(len(selected) - 1, 1)
    # Tie the visible character budget to the actual module-column pitch. This
    # remains safe after the classroom preset scales annotation typography.
    configuration_max_chars = max(20, min(36, round(horizontal_gap * 145)))
    # Preserve the established node dimensions for short networks.  Dense
    # summaries shrink only enough to reserve a real connector corridor between
    # the widest possible pair of learnable blocks.
    node_scale = min(1.0, max(0.45, (horizontal_gap - 0.024) / 0.13))
    center_y = 0.57
    shapes: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = [
        {
            "x": 0.5,
            "y": 1.05,
            "xref": "paper",
            "yref": "paper",
            "text": f"${_architecture_formula(model, len(layers))}$",
            "showarrow": False,
            "font": {"size": 17, "color": NEURAL_COLORS["text"]},
        }
    ]
    training_line = _training_line(history)
    if training_line:
        annotations.append(
            {
                "x": 0.5,
                "y": 0.95,
                "xref": "paper",
                "yref": "paper",
                "text": training_line,
                "showarrow": False,
                "font": {"size": 12, "color": NEURAL_COLORS["muted"]},
            }
        )

    node_half_widths = [
        0.018 if layer is None else _half_width_for_role(layer["role"], node_scale)
        for layer in selected
    ]
    for index in range(len(selected) - 1):
        connector_start = x_positions[index] + node_half_widths[index] + 0.004
        connector_end = x_positions[index + 1] - node_half_widths[index + 1] - 0.004
        if connector_end <= connector_start:
            continue
        annotations.append(
            {
                "x": connector_end,
                "y": center_y,
                "ax": connector_start,
                "ay": center_y,
                "xref": "x",
                "yref": "y",
                "axref": "x",
                "ayref": "y",
                "text": "",
                "showarrow": True,
                "arrowhead": 2,
                "arrowsize": 0.8,
                "arrowwidth": 1.5,
                "arrowcolor": NEURAL_COLORS["grid"],
            }
        )

    for displayed_index, (x_position, layer) in enumerate(zip(x_positions, selected)):
        if layer is None:
            annotations.append(
                {
                    "x": x_position,
                    "y": center_y,
                    "text": r"$\cdots$",
                    "showarrow": False,
                    "font": {"size": 28, "color": NEURAL_COLORS["muted"]},
                }
            )
            continue
        color = layer_color(layer["type"], layer["index"] == layers[-1]["index"])
        shapes.append(_shape_for_role(layer["role"], x_position, center_y, color, node_scale))
        input_dimension = shape_tex(layer.get("input_shape"))
        output_dimension = shape_tex(layer.get("output_shape"))
        parameter_shapes = ", ".join(
            f"{name}:{shape_tex(shape, drop_batch=False)}" for name, shape in layer["parameter_shapes"].items()
        )
        symbol = {
            "learnable": r"$W,b$",
            "activation": r"$\phi$",
            "regularization": r"$m$",
            "reshape": r"$\operatorname{shape}$",
        }.get(layer["role"], r"$\mathcal{M}$")
        annotations.extend(
            [
                {
                    "x": x_position,
                    "y": center_y + 0.04 * node_scale,
                    "text": symbol,
                    "showarrow": False,
                    "font": {"size": 15, "color": NEURAL_COLORS["text"]},
                },
                {
                    "x": x_position,
                    "y": center_y - 0.045 * node_scale,
                    "text": f"<b>{layer['type']}</b><br>{layer['name']}",
                    "showarrow": False,
                    "align": "center",
                    "font": {"size": 11, "color": NEURAL_COLORS["text"]},
                },
                {
                    "x": x_position,
                    "y": 0.84,
                    "text": rf"$\mathbb{{R}}^{{{input_dimension}}}\to\mathbb{{R}}^{{{output_dimension}}}$",
                    "showarrow": False,
                    "font": {"size": 13, "color": color},
                },
                {
                    "x": x_position,
                    "y": 0.31,
                    "text": "" if compact_details else f"${layer['formula']}$",
                    "showarrow": False,
                    "font": {"size": 14, "color": NEURAL_COLORS["text"]},
                },
                {
                    "x": x_position,
                    "y": 0.13,
                    "text": (
                        ""
                        if compact_details
                        else _wrapped_hyperparameters(
                            layer["hyperparameters"],
                            max_chars=configuration_max_chars,
                        )
                    ),
                    "showarrow": False,
                    "align": (
                        "left"
                        if displayed_index == 0
                        else "right"
                        if displayed_index == len(selected) - 1
                        else "center"
                    ),
                    "xanchor": (
                        "left"
                        if displayed_index == 0
                        else "right"
                        if displayed_index == len(selected) - 1
                        else "center"
                    ),
                    "font": {"size": 10, "color": NEURAL_COLORS["muted"]},
                },
            ]
        )
        hover = (
            f"<b>{layer['name']} · {layer['type']}</b><br>"
            f"input: {layer.get('input_shape')}<br>output: {layer.get('output_shape')}<br>"
            f"parameters: {layer['parameters']:,}<br>parameter shapes: {parameter_shapes or 'none'}<br>"
            f"{format_hyperparameters(layer['hyperparameters'], limit=20)}"
        )
        annotations[-1]["hovertext"] = hover

    if compact_details:
        annotations.append(
            {
                "x": 0.5,
                "y": 0.22,
                "text": "Per-module formulas and configuration are available on hover and in the dedicated mathematical views.",
                "showarrow": False,
                "font": {"size": 12, "color": NEURAL_COLORS["muted"]},
            }
        )

    if title is None:
        title = "Mathematical architecture"
    figure = go.Figure(
        data=[
            go.Scatter(
                x=x_positions,
                y=[center_y] * len(x_positions),
                mode="markers",
                marker={"size": max(38, 60 * node_scale), "opacity": 0},
                customdata=[
                    "Collapsed modules" if layer is None else format_hyperparameters(layer["hyperparameters"], limit=20)
                    for layer in selected
                ],
                hovertemplate="%{customdata}<extra></extra>",
                showlegend=False,
            )
        ]
    )
    layout = neural_layout(title, height=650)
    layout["margin"] = {"t": 130, "r": 35, "b": 50, "l": 35}
    layout["meta"] = {
        "mlektic_neural_architecture": {
            "displayed_nodes": len(selected),
            "node_scale": node_scale,
            "minimum_connector_gap": 0.024,
            "connectors_stop_at_node_boundaries": True,
            "configuration_layout": "semantic-multiline-columns",
            "configuration_max_chars": configuration_max_chars,
            "complete_configuration_on_hover": True,
        }
    }
    figure.update_layout(**layout, shapes=shapes, annotations=annotations, showlegend=False)
    figure.update_xaxes(visible=False, range=[0, 1])
    figure.update_yaxes(visible=False, range=[0, 1])
    return figure


__all__ = ["build_nn_architecture_figure"]
