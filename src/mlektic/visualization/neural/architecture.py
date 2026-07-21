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


def _shape_for_role(role: str, x: float, y: float, color: str) -> Dict[str, Any]:
    """Use geometry to distinguish transformations, activations, and reshaping."""
    if role == "activation":
        return {
            "type": "circle",
            "x0": x - 0.045,
            "x1": x + 0.045,
            "y0": y - 0.105,
            "y1": y + 0.105,
            "line": {"color": color, "width": 2},
            "fillcolor": "rgba(255,255,255,0.035)",
        }
    if role in {"regularization", "reshape"}:
        return {
            "type": "rect",
            "x0": x - 0.055,
            "x1": x + 0.055,
            "y0": y - 0.11,
            "y1": y + 0.11,
            "line": {"color": color, "width": 2, "dash": "dot"},
            "fillcolor": "rgba(255,255,255,0.035)",
        }
    return {
        "type": "rect",
        "x0": x - 0.065,
        "x1": x + 0.065,
        "y0": y - 0.12,
        "y1": y + 0.12,
        "line": {"color": color, "width": 2},
        "fillcolor": "rgba(255,255,255,0.035)",
    }


def _architecture_formula(model: Any, layer_count: int) -> str:
    stages = dense_stages(model)
    if stages:
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
    x_positions = [0.07 + index * (0.86 / max(len(selected) - 1, 1)) for index in range(len(selected))]
    center_y = 0.57
    shapes: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = [
        {
            "x": 0.5,
            "y": 1.14,
            "xref": "paper",
            "yref": "paper",
            "text": f"${_architecture_formula(model, len(layers))}$",
            "showarrow": False,
            "font": {"size": 18, "color": NEURAL_COLORS["text"]},
        }
    ]
    training_line = _training_line(history)
    if training_line:
        annotations.append(
            {
                "x": 0.5,
                "y": 1.05,
                "xref": "paper",
                "yref": "paper",
                "text": training_line,
                "showarrow": False,
                "font": {"size": 12, "color": NEURAL_COLORS["muted"]},
            }
        )

    for index in range(len(selected) - 1):
        shapes.append(
            {
                "type": "line",
                "x0": x_positions[index] + 0.065,
                "x1": x_positions[index + 1] - 0.065,
                "y0": center_y,
                "y1": center_y,
                "line": {"color": NEURAL_COLORS["grid"], "width": 2},
                "layer": "below",
            }
        )
        annotations.append(
            {
                "x": (x_positions[index] + x_positions[index + 1]) / 2,
                "y": center_y + 0.025,
                "text": "&#8594;",
                "showarrow": False,
                "font": {"size": 14, "color": NEURAL_COLORS["muted"]},
            }
        )

    for x_position, layer in zip(x_positions, selected):
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
        shapes.append(_shape_for_role(layer["role"], x_position, center_y, color))
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
                    "y": center_y + 0.04,
                    "text": symbol,
                    "showarrow": False,
                    "font": {"size": 15, "color": NEURAL_COLORS["text"]},
                },
                {
                    "x": x_position,
                    "y": center_y - 0.045,
                    "text": f"<b>{layer['type']}</b><br>{layer['name']}",
                    "showarrow": False,
                    "align": "center",
                    "font": {"size": 11, "color": NEURAL_COLORS["text"]},
                },
                {
                    "x": x_position,
                    "y": 0.88,
                    "text": rf"$\mathbb{{R}}^{{{input_dimension}}}\to\mathbb{{R}}^{{{output_dimension}}}$",
                    "showarrow": False,
                    "font": {"size": 12, "color": color},
                },
                {
                    "x": x_position,
                    "y": 0.31,
                    "text": f"${layer['formula']}$",
                    "showarrow": False,
                    "font": {"size": 11, "color": NEURAL_COLORS["text"]},
                },
                {
                    "x": x_position,
                    "y": 0.13,
                    "text": format_hyperparameters(layer["hyperparameters"], limit=4),
                    "showarrow": False,
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

    if title is None:
        title = "Mathematical architecture"
    figure = go.Figure(
        data=[
            go.Scatter(
                x=x_positions,
                y=[center_y] * len(x_positions),
                mode="markers",
                marker={"size": 60, "opacity": 0},
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
    figure.update_layout(**layout, shapes=shapes, annotations=annotations, showlegend=False)
    figure.update_xaxes(visible=False, range=[0, 1])
    figure.update_yaxes(visible=False, range=[0, 1])
    return figure


__all__ = ["build_nn_architecture_figure"]
