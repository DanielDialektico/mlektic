"""Layer-by-layer mathematical explanation for small PyTorch networks."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...neural.introspection import _leaf_modules, run_torch_forward
from ._style import NEURAL_COLORS, layer_color, neural_layout


def _vector_text(values: np.ndarray, dec: int, limit: int) -> str:
    flat = np.asarray(values, dtype=float).ravel()
    shown = [f"{value:.{dec}f}" for value in flat[:limit]]
    if flat.size > limit:
        shown.append(r"\ldots")
    return r"[" + r",\;".join(shown) + r"]"


def _formula_blocks(model: Any, records: Dict[str, Dict[str, np.ndarray]], dec: int, max_neurons: int) -> List[str]:
    blocks: List[str] = []
    linear_index = 0
    for name, module in _leaf_modules(model):
        record = records.get(name)
        if record is None:
            continue
        output = record["output"][0]
        if module.__class__.__name__ == "Linear":
            linear_index += 1
            input_values = record["input"][0]
            weight = module.weight.detach().cpu().numpy()
            bias = module.bias.detach().cpu().numpy() if module.bias is not None else np.zeros(weight.shape[0])
            if max(weight.shape) <= max_neurons and input_values.size <= max_neurons:
                rows = []
                for neuron_index in range(weight.shape[0]):
                    terms = " + ".join(
                        f"({weight[neuron_index, feature_index]:.{dec}f})({input_values[feature_index]:.{dec}f})"
                        for feature_index in range(weight.shape[1])
                    )
                    rows.append(f"z^{{({linear_index})}}_{{{neuron_index + 1}}} = {terms} + ({bias[neuron_index]:.{dec}f}) = {output[neuron_index]:.{dec}f}")
                body = r"\\".join(rows)
            else:
                body = (
                    rf"\mathbf{{z}}^{{({linear_index})}} = W^{{({linear_index})}}\mathbf{{a}}^{{({linear_index - 1})}} + \mathbf{{b}}^{{({linear_index})}}"
                    rf"\\\mathbf{{z}}^{{({linear_index})}} = {_vector_text(output, dec, max_neurons)}"
                )
            blocks.append(rf"\textbf{{{name}}}\quad\begin{{aligned}}{body}\end{{aligned}}")
        else:
            formula = {
                "ReLU": r"\max(0, \mathbf{z})",
                "Sigmoid": r"\sigma(\mathbf{z})",
                "Tanh": r"\tanh(\mathbf{z})",
                "GELU": r"\operatorname{GELU}(\mathbf{z})",
                "Softmax": r"\operatorname{softmax}(\mathbf{z})",
            }.get(module.__class__.__name__)
            if formula:
                blocks.append(
                    rf"\textbf{{{name}}}\quad\mathbf{{a}} = {formula} = {_vector_text(output, dec, max_neurons)}"
                )
    return blocks


def build_nn_prediction_figure(
    model: Any,
    x_query: Any,
    *,
    title: str | None = None,
    dec: int = 4,
    max_layers_math: int = 6,
    max_neurons_math: int = 8,
) -> go.Figure:
    """Explain one forward pass with composed functions and numeric substitutions."""
    if max_layers_math < 1 or max_neurons_math < 1:
        raise ValueError("max_layers_math and max_neurons_math must be positive.")
    output, records = run_torch_forward(model, x_query)
    layers = list(_leaf_modules(model))
    if len(layers) > max_layers_math:
        raise ValueError(
            f"This model has {len(layers)} leaf layers. Set max_layers_math higher or use the architecture view."
        )
    if title is None:
        title = "Forward-pass mathematics"
    output_values = output.detach().cpu().numpy()[0]
    blocks = _formula_blocks(model, records, dec, max_neurons_math)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.30, 0.70], horizontal_spacing=0.04)
    fig.add_trace(go.Scatter(x=[], y=[], showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[], y=[], showlegend=False, hoverinfo="skip"), row=1, col=2)
    for column in (1, 2):
        fig.update_xaxes(visible=False, range=[0, 1], row=1, col=column)
        fig.update_yaxes(visible=False, range=[0, 1], row=1, col=column)

    annotations = []
    layer_count = len(layers)
    for index, (name, module) in enumerate(layers):
        y = 0.92 - index * (0.80 / max(layer_count - 1, 1))
        color = layer_color(module.__class__.__name__, index == layer_count - 1)
        annotations.append(
            {
                "x": 0.50, "y": y, "xref": "x", "yref": "y", "text": f"<b>{name}</b><br>{module.__class__.__name__}",
                "showarrow": False, "align": "center", "font": {"size": 14, "color": color},
                "bgcolor": NEURAL_COLORS["panel"], "bordercolor": color, "borderwidth": 1, "borderpad": 8,
            }
        )
        if index < layer_count - 1:
            annotations.append(
                {
                    "x": 0.50, "y": y - 0.40 / max(layer_count - 1, 1), "xref": "x", "yref": "y",
                    "text": "v", "showarrow": False, "font": {"size": 18, "color": NEURAL_COLORS["muted"]},
                }
            )
    y_positions = np.linspace(0.91, 0.18, max(len(blocks), 1))
    for index, block in enumerate(blocks):
        annotations.append(
            {
                "x": 0.02, "y": y_positions[index], "xref": "x2", "yref": "y2", "text": f"${block}$",
                "showarrow": False, "xanchor": "left", "yanchor": "top", "align": "left",
                "font": {"size": 13, "color": NEURAL_COLORS["text"]},
            }
        )
    annotations.append(
        {
            "x": 0.02, "y": 0.04, "xref": "x2", "yref": "y2",
            "text": rf"$\hat{{y}} = {_vector_text(output_values, dec, max_neurons_math)}$",
            "showarrow": False, "xanchor": "left", "yanchor": "bottom",
            "font": {"size": 17, "color": NEURAL_COLORS["output"]},
        }
    )
    fig.update_layout(**neural_layout(title, height=max(600, 200 + 120 * len(blocks))), annotations=annotations)
    return fig
