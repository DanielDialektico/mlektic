"""Architecture diagram builder for PyTorch neural networks."""

from __future__ import annotations

from typing import Any, List

import numpy as np
import plotly.graph_objects as go

from ...neural.introspection import describe_torch_model
from ._style import NEURAL_COLORS, layer_color, neural_layout


def _display_units(layer: dict[str, Any], max_neurons: int) -> int:
    units = layer.get("units")
    if units is None:
        return 1
    return max(1, min(int(units), max_neurons))


def build_nn_architecture_figure(
    model: Any,
    input_sample: Any | None = None,
    *,
    title: str | None = None,
    max_neurons: int = 10,
) -> go.Figure:
    """Draw a compact layer graph, expanding small dense layers into neurons."""
    if max_neurons < 2:
        raise ValueError("max_neurons must be at least 2.")
    layers = describe_torch_model(model, input_sample)
    if title is None:
        title = "Neural network architecture"
    fig = go.Figure()
    x_positions = np.linspace(0.08, 0.92, len(layers) + 1)
    input_width = None
    if input_sample is not None:
        array = np.asarray(input_sample)
        input_width = int(array.shape[-1]) if array.ndim else 1
    columns: List[dict[str, Any]] = [
        {"name": "input", "type": "Input", "units": input_width, "parameters": 0, "formula": r"a^{(0)} = x"}
    ] + layers
    node_positions: List[List[float]] = []
    for column in columns:
        visible = _display_units(column, max_neurons)
        node_positions.append(np.linspace(0.17, 0.83, visible).tolist())

    edge_x: List[float | None] = []
    edge_y: List[float | None] = []
    for column_index in range(len(columns) - 1):
        left_x, right_x = x_positions[column_index], x_positions[column_index + 1]
        left_nodes, right_nodes = node_positions[column_index], node_positions[column_index + 1]
        for left_y in left_nodes:
            for right_y in right_nodes:
                edge_x.extend([left_x, right_x, None])
                edge_y.extend([left_y, right_y, None])
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"color": "rgba(174, 180, 189, 0.22)", "width": 1},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    for column_index, (column, y_values) in enumerate(zip(columns, node_positions)):
        is_output = column_index == len(columns) - 1
        color = NEURAL_COLORS["input"] if column_index == 0 else layer_color(column["type"], is_output)
        actual_units = column.get("units")
        display = _display_units(column, max_neurons)
        unit_text = "vector" if actual_units is None else f"{actual_units} units"
        if actual_units is not None and actual_units > display:
            unit_text += f" (showing {display})"
        label = f"<b>{column['name']}</b><br>{column['type']}<br>{unit_text}"
        if column_index:
            label += f"<br>{column['parameters']:,} params"
        fig.add_trace(
            go.Scatter(
                x=[x_positions[column_index]] * len(y_values),
                y=y_values,
                mode="markers",
                marker={"size": 22, "color": color, "line": {"width": 1, "color": NEURAL_COLORS["background"]}},
                customdata=[[label, column["formula"]]] * len(y_values),
                hovertemplate="%{customdata[0]}<br><i>%{customdata[1]}</i><extra></extra>",
                showlegend=False,
            )
        )
        fig.add_annotation(
            x=x_positions[column_index],
            y=1.04,
            text=label,
            showarrow=False,
            align="center",
            font={"size": 12, "color": NEURAL_COLORS["text"]},
        )

    fig.update_layout(**neural_layout(title), showlegend=False)
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1.16])
    return fig
