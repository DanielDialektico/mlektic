r"""Does the error signal reach the first layers?  Recorded gradient norms per layer.

For every dense weight matrix the recorder keeps :math:`\|\nabla_{\Theta^{(\ell)}}\mathcal L\|`
at each checkpoint. A heat map of :math:`\log_{10}` of that norm, layer by
layer and step by step, shows at a glance whether early layers receive a usable
gradient (vanishing or exploding gradients). Several recorded runs can be put
side by side on one shared colour scale, for example the same network with two
activations. The figure is static: it has no frames.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._style import NEURAL_COLORS, neural_layout
from .geometry import MAGNITUDE_SCALE, loss_label, recorded_steps

_RUN_COLORS = (
    NEURAL_COLORS["negative"],
    NEURAL_COLORS["positive"],
    NEURAL_COLORS["activation"],
    NEURAL_COLORS["linear"],
)


def _weight_norms(history: Dict[str, Any]) -> tuple[list[str], np.ndarray]:
    norms = history.get("gradient_norms", {})
    parameters = history.get("parameters", {})
    names = [
        name
        for name, values in norms.items()
        if name.endswith("weight")
        and len(values)
        and (not parameters.get(name) or np.asarray(parameters[name][0]).ndim >= 2)
    ]
    if not names:
        raise ValueError(
            "The history has no recorded weight-gradient norms. Record with "
            "TorchTrainingRecorder(capture_gradients=True)."
        )
    return names, np.stack([np.asarray(norms[name], dtype=float) for name in names])


def _superscript_power(exponent: float) -> str:
    return rf"10^{{{int(round(exponent))}}}"


def build_nn_gradient_flow_figure(
    histories: Dict[str, Any] | Mapping[str, Dict[str, Any]],
    *,
    labels: Sequence[str] | None = None,
    floor_decades: float = 8.0,
    title: str | None = None,
) -> go.Figure:
    """Heat maps of recorded weight-gradient norms per layer and step."""
    if isinstance(histories, Mapping) and "steps" in histories:
        runs = {"recorded run": histories}
    elif isinstance(histories, Mapping):
        runs = dict(histories)
    else:
        runs = {f"run {k + 1}": h for k, h in enumerate(histories)}
    if labels is not None:
        if len(labels) != len(runs):
            raise ValueError("labels must name every history.")
        runs = dict(zip(labels, runs.values()))
    if not 1 <= len(runs) <= 4:
        raise ValueError("Compare between one and four recorded histories.")
    prepared = []
    for name, history in runs.items():
        steps = recorded_steps(history)
        parameter_names, norms = _weight_norms(history)
        if norms.shape[1] != steps.size:
            raise ValueError(f"History {name!r} has gradient norms that do not match its steps.")
        loss = np.asarray(history.get("loss", []), dtype=float)
        prepared.append((name, steps, parameter_names, np.log10(np.maximum(norms, 1e-300)), loss, history))
    high = max(float(np.nanmax(p[3])) for p in prepared)
    low = max(min(float(np.nanmin(p[3])) for p in prepared), high - floor_decades)

    columns = len(prepared)
    titles = [rf"$\text{{{name}}}$" for name, *_ in prepared] + [""] * columns
    figure = make_subplots(
        rows=2,
        cols=columns,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.16,
        horizontal_spacing=0.08,
        subplot_titles=titles,
    )
    for annotation in figure.layout.annotations:
        annotation.font = {"size": 14, "color": NEURAL_COLORS["text"]}
    ratios = {}
    for column, (name, steps, parameter_names, log_norms, loss, history) in enumerate(prepared, start=1):
        depth = len(parameter_names)
        layer_labels = [f"ℓ = {ell}" for ell in range(1, depth + 1)]
        layer_labels[0] += " (input side)"
        layer_labels[-1] += " (output)"
        figure.add_trace(
            go.Heatmap(
                x=steps,
                y=layer_labels,
                z=np.clip(log_norms, low, high).astype(np.float32),
                zmin=low,
                zmax=high,
                colorscale=MAGNITUDE_SCALE,
                showscale=column == columns,
                colorbar={
                    "title": {"text": "log<sub>10</sub> ‖∇<sub>Θ</sub>𝓛‖", "side": "top"},
                    "len": 0.62,
                    "y": 0.69,
                    "thickness": 12,
                    "outlinewidth": 0,
                    "tickfont": {"size": 10, "color": NEURAL_COLORS["muted"]},
                },
                customdata=np.asarray([[p] * steps.size for p in parameter_names], dtype=object),
                hovertemplate="t=%{x}<br>%{y} · %{customdata}<br>log<sub>10</sub>‖∇‖=%{z:.2f}<extra></extra>",
                name=f"neural gradient flow {name}",
                uid=f"NN_GRADIENT_FLOW_{column}",
            ),
            row=1,
            col=column,
        )
        if loss.size == steps.size:
            figure.add_trace(
                go.Scatter(
                    x=steps,
                    y=loss,
                    mode="lines",
                    line={"color": _RUN_COLORS[(column - 1) % len(_RUN_COLORS)], "width": 2.5},
                    hovertemplate="t=%{x}<br>L=%{y:.4f}<extra></extra>",
                    name=f"neural gradient flow loss {name}",
                    uid=f"NN_GRADIENT_FLOW_LOSS_{column}",
                ),
                row=2,
                col=column,
            )
        exponent = float(np.nanmean(log_norms[0]) - np.nanmean(log_norms[-1]))
        ratios[name] = exponent
        verdict = (
            "the first layer barely learns"
            if exponent < -2
            else ("the first layer receives a larger gradient" if exponent > 2 else "the signal reaches every layer")
        )
        figure.add_annotation(
            x=0.5,
            y=1.0,
            xref=f"x{column + columns if column + columns > 1 else ''} domain",
            yref=f"y{column + columns if column + columns > 1 else ''} domain",
            yanchor="bottom",
            yshift=6,
            showarrow=False,
            text=(
                rf"$\overline{{\|\nabla_{{\Theta^{{(1)}}}}\mathcal L\|}}\,/\,"
                rf"\overline{{\|\nabla_{{\Theta^{{({depth})}}}}\mathcal L\|}}\approx{_superscript_power(exponent)}$"
                f"<br><span style='font-size:12px'>{verdict}</span>"
            ),
            font={"size": 13, "color": NEURAL_COLORS["activation"]},
        )
        figure.update_yaxes(autorange="reversed", row=1, col=column)
        figure.update_xaxes(showgrid=False, row=1, col=column)
        figure.update_xaxes(gridcolor=NEURAL_COLORS["grid"], title_text="$t$", row=2, col=column)
        figure.update_yaxes(
            gridcolor=NEURAL_COLORS["grid"], title_text=r"$\mathcal L_t$" if column == 1 else None, row=2, col=column
        )
    if title is None:
        title = "Does the error signal reach the first layers?"
    layout = neural_layout(title, height=740)
    layout["title"].update({"y": 0.975, "yanchor": "top"})
    layout["margin"] = {"t": 140, "r": 110, "b": 70, "l": 120}
    layout["showlegend"] = False
    layout["meta"] = {
        "mlektic_neural_gradient_flow": {
            "schema_version": 1,
            "semantics": "recorded weight-gradient norms per dense layer and checkpoint",
            "quantity": "log10 ||grad_Theta(l) L|| of the reduced batch loss",
            "runs": list(runs),
            "colour_range": [low, high],
            "floor_decades": floor_decades,
            "first_to_last_mean_log10_ratio": ratios,
            "losses": [loss_label(history) for *_, history in prepared],
            "animated": False,
        }
    }
    figure.update_layout(**layout)
    figure.add_annotation(
        x=0.0,
        y=1.0,
        yshift=40,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="bottom",
        showarrow=False,
        align="left",
        text="Recorded gradient norm of every weight matrix (rows, input side at the top) at every checkpoint (columns)."
        "<br>Dark cells: the layer receives almost no error signal. One colour scale is shared by all runs.",
        font={"size": 12, "color": NEURAL_COLORS["muted"]},
    )
    return figure


__all__ = ["build_nn_gradient_flow_figure"]
