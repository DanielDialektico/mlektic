r"""Recorded weight matrices as Hinton diagrams.

Each entry :math:`\theta^{(\ell)}_{ji}` of :math:`\Theta^{(\ell)}` is a square
in row :math:`j` (output unit) and column :math:`i` (input unit). Its area is
proportional to :math:`|\theta^{(\ell)}_{ji}|` on one global scale for every
matrix and checkpoint, and its colour gives the sign. The intercepts
:math:`\boldsymbol\theta_0^{(\ell)}` form an extra column, set apart. Matrices
too large for LaTeX stay readable as a picture, and hover keeps the exact value.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._style import NEURAL_COLORS, neural_layout
from .geometry import format_number, playback_controls, recorded_steps, select_frames
from .math_format import display_indices

_ASSUMED_PLOT_WIDTH = 980.0


def _layer_pairs(history: Dict[str, Any], parameter: str | None) -> List[tuple[str, str | None]]:
    parameters = history.get("parameters", {})
    names = [
        name
        for name, values in parameters.items()
        if name.endswith("weight") and len(values) and np.asarray(values[0]).ndim == 2
    ]
    if parameter is not None:
        if parameter not in parameters:
            raise ValueError(f"parameter {parameter!r} was not recorded.")
        names = [parameter]
    if not names:
        raise ValueError("The history has no recorded two-dimensional weight matrices.")
    pairs = []
    for name in names:
        bias = name[: -len("weight")] + "bias"
        pairs.append((name, bias if parameters.get(bias) else None))
    return pairs


def build_nn_hinton_figure(
    history: Dict[str, Any],
    *,
    parameter: str | None = None,
    max_parameters: int = 6,
    max_rows: int = 32,
    max_cols: int = 32,
    max_frames: int | None = 30,
    frame_duration: int = 160,
    title: str | None = None,
) -> go.Figure:
    """Animate recorded weight matrices as Hinton diagrams."""
    steps = recorded_steps(history)
    pairs = [(ell, *pair) for ell, pair in enumerate(_layer_pairs(history, parameter), start=1)]
    omitted = max(0, len(pairs) - max_parameters)
    pairs = pairs[:max_parameters] if omitted == 0 else [*pairs[: max_parameters - 1], pairs[-1]]
    selected = select_frames(steps.size, max_frames)
    parameters = history["parameters"]
    loss = np.asarray(history.get("loss", []), dtype=float)

    blocks = []
    for ell, weight_name, bias_name in pairs:
        series = np.asarray(parameters[weight_name], dtype=float)
        rows = display_indices(series.shape[1], max_rows)
        cols = display_indices(series.shape[2], max_cols)
        weights = series[:, rows][:, :, cols]
        bias = np.asarray(parameters[bias_name], dtype=float)[:, rows] if bias_name else None
        blocks.append(
            {
                "ell": ell,
                "weight": weight_name,
                "bias": bias_name,
                "values": weights,
                "bias_values": bias,
                "rows": rows,
                "cols": cols,
                "shape": series.shape[1:],
            }
        )
    limit = max(
        max(float(np.max(np.abs(b["values"]))) for b in blocks),
        max((float(np.max(np.abs(b["bias_values"]))) for b in blocks if b["bias_values"] is not None), default=0.0),
        1e-12,
    )
    width_units = [len(b["cols"]) + (1.8 if b["bias"] else 0.0) for b in blocks]
    total_units = sum(width_units)
    max_height_units = max(len(b["rows"]) for b in blocks)
    spacing = 0.05
    available = 1.0 - spacing * (len(blocks) - 1)
    column_widths = [available * w / total_units for w in width_units]
    cell = min(
        _ASSUMED_PLOT_WIDTH * available / total_units,
        430.0 / max_height_units,
        34.0,
    )
    figure = make_subplots(
        rows=1,
        cols=len(blocks),
        column_widths=column_widths,
        horizontal_spacing=spacing,
        subplot_titles=[
            "$"
            + rf"\Theta^{{({b['ell']})}}\in\mathbb R^{{{b['shape'][0]}\times{b['shape'][1]}}}"
            + (rf"\quad\boldsymbol\theta_0^{{({b['ell']})}}" if b["bias"] else "")
            + (rf"\quad\text{{(+{omitted} more)}}" if omitted and k == len(blocks) - 2 else "")
            + "$"
            for k, b in enumerate(blocks)
        ],
    )
    for annotation in figure.layout.annotations:
        annotation.font = {"size": 14, "color": NEURAL_COLORS["muted"]}

    def marker_payload(block: Dict[str, Any], frame: int) -> Dict[str, Any]:
        values = block["values"][frame].ravel()
        if block["bias_values"] is not None:
            values = np.r_[values, block["bias_values"][frame]]
        sizes = np.round(3 + (cell - 3) * np.sqrt(np.abs(values) / limit), 2)
        colors = [NEURAL_COLORS["weight_max"] if v >= 0 else NEURAL_COLORS["weight_min"] for v in values]
        return {
            "marker": {"size": sizes.tolist(), "color": colors},
            "customdata": [format_number(v) for v in values],
        }

    for column, block in enumerate(blocks, start=1):
        rows, cols = len(block["rows"]), len(block["cols"])
        grid_r, grid_c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        xs, ys = grid_c.ravel().astype(float), grid_r.ravel().astype(float)
        text = [f"j={block['rows'][r] + 1}, i={block['cols'][c] + 1}" for r, c in zip(grid_r.ravel(), grid_c.ravel())]
        if block["bias"]:
            xs = np.r_[xs, np.full(rows, cols + 0.8)]
            ys = np.r_[ys, np.arange(rows)]
            text += [f"θ₀ for j={block['rows'][r] + 1}" for r in range(rows)]
        payload = marker_payload(block, int(selected[0]))
        figure.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker={"symbol": "square", "line": {"width": 0}, **payload["marker"]},
                customdata=payload["customdata"],
                text=text,
                hovertemplate="%{text}<br>value = %{customdata}<extra></extra>",
                name=f"neural hinton {block['weight']}",
                uid=f"NN_HINTON_{column}",
            ),
            row=1,
            col=column,
        )
        x_end = cols - 0.5 + (1.8 if block["bias"] else 0.0)
        figure.update_xaxes(range=[-0.6, x_end], visible=False, row=1, col=column)
        figure.update_yaxes(
            range=[max_height_units - 0.4, -0.6],
            visible=False,
            scaleanchor=f"x{column if column > 1 else ''}",
            row=1,
            col=column,
        )
        if len(block["rows"]) < block["shape"][0] or len(block["cols"]) < block["shape"][1]:
            figure.add_annotation(
                x=0.5,
                y=0.0,
                xref=f"x{column if column > 1 else ''} domain",
                yref="paper",
                yshift=-4,
                yanchor="top",
                showarrow=False,
                text=f"first and last rows/columns of {block['shape'][0]}×{block['shape'][1]}",
                font={"size": 11, "color": NEURAL_COLORS["muted"]},
            )

    static_annotations = list(figure.layout.annotations)

    def headline(index: int) -> Dict[str, Any]:
        text = rf"t={int(steps[index])}"
        if loss.size == steps.size:
            text += rf"\quad\mathcal L_t={format_number(loss[index])}"
        return {
            "x": 1.0,
            "y": 1.0,
            "yshift": 58,
            "yanchor": "bottom",
            "xref": "paper",
            "yref": "paper",
            "xanchor": "right",
            "showarrow": False,
            "text": f"${text}$",
            "font": {"size": 15, "color": NEURAL_COLORS["text"]},
        }

    caption = {
        "x": 0.0,
        "y": 1.0,
        "yshift": 58,
        "yanchor": "bottom",
        "xref": "paper",
        "yref": "paper",
        "xanchor": "left",
        "showarrow": False,
        "align": "left",
        "text": (
            "Area ∝ |θ<sub>ji</sub>| on one global scale · "
            f"<span style='color:{NEURAL_COLORS['weight_max']}'>■ positive</span> "
            f"<span style='color:{NEURAL_COLORS['weight_min']}'>■ negative</span> · "
            "rows j = output units, columns i = input units"
        ),
        "font": {"size": 12, "color": NEURAL_COLORS["muted"]},
    }
    frames = []
    for index in selected:
        frames.append(
            go.Frame(
                name=f"step_{int(index)}",
                data=[go.Scatter(**marker_payload(block, int(index))) for block in blocks],
                traces=list(range(len(blocks))),
                layout={"annotations": [*static_annotations, caption, headline(int(index))]},
            )
        )
    figure.frames = frames
    figure.layout.annotations = [*static_annotations, caption, headline(int(selected[0]))]
    if title is None:
        title = "Weights as a picture: Hinton diagrams"
    layout = neural_layout(title, height=min(780, max(480, int(cell * max_height_units) + 290)))
    layout["title"].update({"y": 0.98, "yanchor": "top"})
    layout["margin"] = {"t": 150, "r": 30, "b": 95, "l": 30}
    layout.update(
        playback_controls(
            [f.name for f in frames],
            [str(int(steps[i])) for i in selected],
            frame_duration=frame_duration,
            redraw=True,
            y=-0.06,
        )
    )
    layout["showlegend"] = False
    layout["meta"] = {
        "mlektic_neural_weights": {
            "schema_version": 2,
            "style": "hinton",
            "encoding": "square area proportional to |value|; colour gives the sign; one global scale",
            "global_abs_max": limit,
            "parameters": [b["weight"] for b in blocks],
            "biases": [b["bias"] for b in blocks],
            "omitted_parameter_count": omitted,
            "displayed_rows": [[r + 1 for r in b["rows"]] for b in blocks],
            "displayed_cols": [[c + 1 for c in b["cols"]] for b in blocks],
            "recorded_steps": [int(steps[i]) for i in selected],
        }
    }
    figure.update_layout(**layout)
    return figure


__all__ = ["build_nn_hinton_figure"]
