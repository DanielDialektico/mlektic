r"""Decision surface of a dense network as a combination of hidden units.

The main panel evaluates the network output :math:`\hat y=f_\theta(\mathbf x)`
on the plane of two input features at every retained checkpoint. The small
panels show the units of the last hidden layer, :math:`a^{(L-1)}_j(\mathbf x)`,
and the bar panel the readout weights :math:`\Theta^{(L)}_{1j}` that combine
them. With one hidden layer every unit folds the plane along the exact line
:math:`z^{(1)}_j(\mathbf x)=0`, which is drawn on the main panel too.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._style import NEURAL_COLORS, neural_layout
from .geometry import (
    CLASS_COLORS,
    CLASS_SYMBOLS,
    SIGNED_SCALE,
    UNIT_COLORS,
    UNIT_SCALE,
    as_numpy,
    dense_chain,
    format_number,
    forward_states,
    loss_label,
    output_semantics,
    playback_controls,
    quantize,
    quantized_colorbar,
    recorded_metric,
    recorded_steps,
    rounded,
    select_frames,
    snapshots,
)

_CELLS = ((1, 2), (1, 3), (2, 2), (2, 3))


def _plane(
    inputs: np.ndarray,
    features: Sequence[int],
    reference: np.ndarray,
    size: int,
    padding: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid on the plane of two features; every other feature stays at ``reference``."""
    first, second = features
    low = inputs[:, [first, second]].min(axis=0)
    high = inputs[:, [first, second]].max(axis=0)
    margin = (high - low) * padding + 1e-9
    axis_x = np.linspace(low[0] - margin[0], high[0] + margin[0], size)
    axis_y = np.linspace(low[1] - margin[1], high[1] + margin[1], size)
    grid_x, grid_y = np.meshgrid(axis_x, axis_y)
    points = np.repeat(reference[None, :], size * size, axis=0)
    points[:, first] = grid_x.ravel()
    points[:, second] = grid_y.ravel()
    return axis_x, axis_y, points


def _fold_line(
    weight: np.ndarray,
    offset: float,
    features: Sequence[int],
    axis_x: np.ndarray,
    axis_y: np.ndarray,
) -> tuple[list, list]:
    """Exact line ``w1 x1 + w2 x2 + c = 0`` clipped to the visible plane."""
    w1, w2 = float(weight[features[0]]), float(weight[features[1]])
    if abs(w1) < 1e-12 and abs(w2) < 1e-12:
        return [None, None], [None, None]
    if abs(w2) >= abs(w1):
        xs = np.array([axis_x[0], axis_x[-1]])
        ys = -(w1 * xs + offset) / w2
    else:
        ys = np.array([axis_y[0], axis_y[-1]])
        xs = -(w2 * ys + offset) / w1
    return [float(v) for v in xs], [float(v) for v in ys]


def build_nn_decision_figure(
    model: Any,
    inputs: Any,
    history: Dict[str, Any],
    *,
    targets: Any | None = None,
    features: Sequence[int] = (0, 1),
    reference: Any | None = None,
    grid_size: int = 40,
    unit_grid_size: int = 20,
    max_units: int = 4,
    max_frames: int | None = 20,
    frame_duration: int = 260,
    padding: float = 0.12,
    title: str | None = None,
) -> go.Figure:
    """Animate the decision surface and the hidden units that build it."""
    chain = dense_chain(model, purpose="The decision-surface view")
    if len(chain) < 2:
        raise ValueError("The decision-surface view needs at least one hidden dense layer.")
    if not 16 <= grid_size <= 96:
        raise ValueError("grid_size must be between 16 and 96.")
    if not 12 <= unit_grid_size <= 64:
        raise ValueError("unit_grid_size must be between 12 and 64.")
    if not 0 <= max_units <= 4:
        raise ValueError("max_units must be between 0 and 4.")
    data = np.asarray(as_numpy(inputs), dtype=float)
    if data.ndim != 2 or data.shape[1] != chain[0].in_features:
        raise ValueError("inputs must be a 2-D batch with one column per model input feature.")
    if data.shape[1] < 2:
        raise ValueError("The decision-surface view needs at least two input features.")
    features = tuple(int(f) for f in features)
    if len(features) != 2 or len(set(features)) != 2 or not all(0 <= f < data.shape[1] for f in features):
        raise ValueError("features must name two distinct input columns.")
    fixed = np.median(data, axis=0) if reference is None else np.asarray(as_numpy(reference), dtype=float).reshape(-1)
    if fixed.size != data.shape[1]:
        raise ValueError("reference must contain one value per input feature.")
    labels = None if targets is None else np.asarray(as_numpy(targets))
    mode = output_semantics(chain, history, labels)
    steps = recorded_steps(history)
    selected = select_frames(steps.size, max_frames)
    loss = np.asarray(history.get("loss", []), dtype=float)
    accuracy = recorded_metric(history, "accuracy")

    axis_x, axis_y, grid = _plane(data, features, fixed, grid_size, padding)
    unit_x, unit_y, unit_grid = _plane(data, features, fixed, unit_grid_size, padding)
    hidden = chain[-2]
    readout = chain[-1]
    final_parameters, _ = snapshots(history, len(steps) - 1)
    readout_weight = np.asarray(final_parameters[readout.weight], dtype=float)
    unit_strength = np.linalg.norm(readout_weight, axis=0)
    shown_units = sorted(np.argsort(unit_strength)[::-1][:max_units].tolist())
    single_hidden = len(chain) == 2

    fields, unit_fields, readouts, folds = [], [], [], []
    for index in selected:
        parameters, buffers = snapshots(history, int(index))
        states = forward_states(model, chain, grid, parameters, buffers)
        unit_states = forward_states(model, chain, unit_grid, parameters, buffers)
        output = states["output"]
        if mode == "multiclass":
            probabilities = output
            if readout.activation_type not in {"Softmax"}:
                shifted = output - output.max(axis=1, keepdims=True)
                probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
            classes = probabilities.shape[1]
            confidence = (probabilities.max(axis=1) - 1 / classes) / (1 - 1 / classes)
            field = probabilities.argmax(axis=1) + np.clip(confidence, 0, 0.999)
        else:
            field = output[:, 0]
        fields.append(field.reshape(grid_size, grid_size))
        unit_fields.append([unit_states["a"][-2][:, j].reshape(unit_grid_size, unit_grid_size) for j in shown_units])
        weight = np.asarray(parameters[readout.weight], dtype=float)
        readouts.append(weight[:, shown_units])
        if single_hidden:
            w1 = np.asarray(parameters[hidden.weight], dtype=float)
            b1 = np.asarray(parameters[hidden.bias], dtype=float) if hidden.bias else np.zeros(w1.shape[0])
            others = [k for k in range(data.shape[1]) if k not in features]
            folds.append(
                [
                    _fold_line(w1[j], float(b1[j] + w1[j, others] @ fixed[others]), features, axis_x, axis_y)
                    for j in shown_units
                ]
            )

    classes = int(readout.out_features) if mode == "multiclass" else 2
    if mode == "probability":
        field_low, field_high, level = 0.0, 1.0, 0.5
    elif mode == "logit":
        bound = float(np.nanpercentile(np.abs(np.stack(fields)), 99)) or 1.0
        field_low, field_high, level = -bound, bound, 0.0
    elif mode == "multiclass":
        field_low, field_high, level = 0.0, float(classes), None
    else:
        stacked = np.stack(fields)
        field_low, field_high, level = float(np.nanmin(stacked)), float(np.nanmax(stacked)), None
    if hidden.activation_type in {"Tanh", "Softsign"}:
        unit_low, unit_high = -1.0, 1.0
    elif hidden.activation_type == "Sigmoid":
        unit_low, unit_high = 0.0, 1.0
    else:
        stacked_units = np.stack([np.stack(u) for u in unit_fields]) if shown_units else np.zeros(1)
        unit_low, unit_high = float(stacked_units.min()), float(stacked_units.max())
        if unit_low < 0 < unit_high:
            bound = max(-unit_low, unit_high)
            unit_low, unit_high = -bound, bound

    if mode == "multiclass":
        scale = []
        for k in range(classes):
            color = CLASS_COLORS[k % len(CLASS_COLORS)]
            scale += [[k / classes, "#2a2c33"], [(k + 0.999) / classes, color]]
        field_scale = scale
    elif mode == "regression":
        field_scale = UNIT_SCALE
    else:
        field_scale = SIGNED_SCALE

    L = len(chain)
    ell = L - 1
    out_tex = {
        "probability": r"\hat y=f_\theta(\mathbf x)",
        "logit": rf"z^{{({L})}}(\mathbf x)",
        "multiclass": r"\hat k=\arg\max_k\,\hat p_k(\mathbf x)",
        "regression": r"\hat y=f_\theta(\mathbf x)",
    }[mode]
    feature_names = [f"x_{{{f + 1}}}" for f in features]
    titles = [rf"${out_tex}$"]
    for slot in range(4):
        if slot < len(shown_units):
            j = shown_units[slot] + 1
            titles.append(rf"$a^{{({ell})}}_{{{j}}}=" + hidden.activation_tex + rf"\left(z^{{({ell})}}_{{{j}}}\right)$")
        else:
            titles.append("")
    titles += [r"$\text{Recorded loss: }\mathcal L_t$", rf"$\text{{readout weights }}\Theta^{{({L})}}_{{kj}}$"]
    figure = make_subplots(
        rows=3,
        cols=3,
        column_widths=[0.5, 0.25, 0.25],
        row_heights=[0.39, 0.39, 0.22],
        specs=[[{"rowspan": 2}, {}, {}], [None, {}, {}], [{}, {"colspan": 2}, None]],
        horizontal_spacing=0.06,
        vertical_spacing=0.13,
        subplot_titles=titles,
    )
    for annotation in figure.layout.annotations:
        annotation.font = {"size": 14, "color": NEURAL_COLORS["muted"]}

    raster = {"hoverinfo": "skip", "zsmooth": "fast", "zmin": 0, "zmax": 255, "zauto": False}
    figure.add_trace(
        go.Heatmap(
            x=axis_x,
            y=axis_y,
            z=np.asarray(quantize(fields[0], field_low, field_high), dtype=np.uint8),
            colorscale=field_scale,
            showscale=mode not in {"multiclass"},
            colorbar={**quantized_colorbar(field_low, field_high, title=""), "x": 0.455, "y": 0.6},
            name="neural decision field",
            uid="NN_DECISION_FIELD",
            **raster,
        ),
        row=1,
        col=1,
    )
    for slot, unit in enumerate(shown_units):
        row, col = _CELLS[slot]
        figure.add_trace(
            go.Heatmap(
                x=unit_x,
                y=unit_y,
                z=np.asarray(quantize(unit_fields[0][slot], unit_low, unit_high), dtype=np.uint8),
                colorscale=UNIT_SCALE,
                showscale=slot == len(shown_units) - 1,
                colorbar={**quantized_colorbar(unit_low, unit_high, title=""), "x": 1.0, "y": 0.6, "len": 0.5},
                name=f"neural decision unit {unit + 1}",
                uid=f"NN_DECISION_UNIT_{unit + 1}",
                **raster,
            ),
            row=row,
            col=col,
        )
    boundary_index = None
    if level is not None:
        boundary_index = len(figure.data)
        figure.add_trace(
            go.Contour(
                x=axis_x,
                y=axis_y,
                z=np.asarray(quantize(fields[0], field_low, field_high), dtype=np.uint8),
                contours={
                    "start": (level - field_low) / (field_high - field_low) * 255,
                    "end": (level - field_low) / (field_high - field_low) * 255,
                    "size": 1,
                    "coloring": "none",
                },
                line={"width": 3, "color": NEURAL_COLORS["text"]},
                showscale=False,
                hoverinfo="skip",
                name="neural decision boundary",
                uid="NN_DECISION_BOUNDARY",
            ),
            row=1,
            col=1,
        )
    if labels is not None:
        flat = labels.reshape(len(labels), -1)
        if mode == "regression" and flat.shape[1] == 1:
            figure.add_trace(
                go.Scatter(
                    x=data[:, features[0]],
                    y=data[:, features[1]],
                    mode="markers",
                    marker={
                        "size": 7,
                        "color": flat[:, 0].astype(float),
                        "colorscale": UNIT_SCALE,
                        "line": {"width": 1, "color": "#111111"},
                    },
                    hovertemplate=f"${feature_names[0]}$=%{{x:.4f}}<br>y=%{{marker.color:.4f}}<extra></extra>",
                    name="neural decision data",
                    uid="NN_DECISION_DATA",
                ),
                row=1,
                col=1,
            )
        else:
            codes = flat.argmax(axis=1) if flat.shape[1] > 1 else np.rint(flat[:, 0]).astype(int)
            for k in np.unique(codes):
                mask = codes == k
                figure.add_trace(
                    go.Scatter(
                        x=data[mask, features[0]],
                        y=data[mask, features[1]],
                        mode="markers",
                        marker={
                            "size": 7,
                            "color": CLASS_COLORS[int(k) % len(CLASS_COLORS)],
                            "symbol": CLASS_SYMBOLS[int(k) % len(CLASS_SYMBOLS)],
                            "line": {"width": 1, "color": "#111111"},
                        },
                        hovertemplate=f"class {int(k)}<br>x=%{{x:.4f}}, y=%{{y:.4f}}<extra></extra>",
                        name=f"neural decision class {int(k)}",
                        uid=f"NN_DECISION_CLASS_{int(k)}",
                    ),
                    row=1,
                    col=1,
                )
    fold_indices, own_fold_indices = [], []
    if single_hidden:
        for slot, unit in enumerate(shown_units):
            fold_indices.append(len(figure.data))
            xs, ys = folds[0][slot]
            figure.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line={"color": UNIT_COLORS[slot], "width": 1.8, "dash": "dot"},
                    hoverinfo="skip",
                    name=f"neural decision fold {unit + 1}",
                    uid=f"NN_DECISION_FOLD_{unit + 1}",
                ),
                row=1,
                col=1,
            )
        for slot, unit in enumerate(shown_units):
            row, col = _CELLS[slot]
            own_fold_indices.append(len(figure.data))
            xs, ys = folds[0][slot]
            figure.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line={"color": UNIT_COLORS[slot], "width": 2.6},
                    hoverinfo="skip",
                    name=f"neural decision unit fold {unit + 1}",
                    uid=f"NN_DECISION_UNIT_FOLD_{unit + 1}",
                ),
                row=row,
                col=col,
            )
    has_loss = loss.size == steps.size
    loss_index = None
    if has_loss:
        figure.add_trace(
            go.Scatter(
                x=steps,
                y=loss,
                mode="lines",
                line={"color": NEURAL_COLORS["grid"], "width": 2},
                hoverinfo="skip",
                name="neural decision loss ghost",
                uid="NN_DECISION_LOSS_GHOST",
            ),
            row=3,
            col=1,
        )
        loss_index = len(figure.data)
        first = int(selected[0])
        figure.add_trace(
            go.Scatter(
                x=steps[: first + 1],
                y=loss[: first + 1],
                mode="lines",
                line={"color": NEURAL_COLORS["positive"], "width": 2.5},
                hovertemplate="t=%{x}<br>L=%{y:.4f}<extra></extra>",
                name="neural decision loss",
                uid="NN_DECISION_LOSS",
            ),
            row=3,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=[steps[first]],
                y=[loss[first]],
                mode="markers",
                marker={"size": 9, "color": NEURAL_COLORS["positive"]},
                hoverinfo="skip",
                name="neural decision loss marker",
                uid="NN_DECISION_LOSS_MARKER",
            ),
            row=3,
            col=1,
        )
    readout_index = len(figure.data)
    unit_labels = [f"j={unit + 1}" for unit in shown_units]
    if readout.out_features == 1:
        figure.add_trace(
            go.Bar(
                x=unit_labels,
                y=rounded(readouts[0][0]),
                marker={"color": [UNIT_COLORS[s] for s in range(len(shown_units))], "line": {"width": 0}},
                text=[format_number(v, 3) for v in readouts[0][0]],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="Θ<sub>1j</sub> = %{y:.4f}<extra></extra>",
                name="neural decision readout",
                uid="NN_DECISION_READOUT",
            ),
            row=3,
            col=2,
        )
        readout_limit = float(np.max(np.abs(np.stack(readouts)))) * 1.3 or 1.0
    else:
        readout_limit = float(np.max(np.abs(np.stack(readouts)))) or 1.0
        figure.add_trace(
            go.Heatmap(
                x=unit_labels,
                y=[f"k={k}" for k in range(readout.out_features)],
                z=rounded(readouts[0]),
                zmin=-readout_limit,
                zmax=readout_limit,
                colorscale=UNIT_SCALE,
                showscale=False,
                hovertemplate="Θ<sub>kj</sub> = %{z:.4f}<extra></extra>",
                name="neural decision readout",
                uid="NN_DECISION_READOUT",
            ),
            row=3,
            col=2,
        )

    def headline(index: int) -> str:
        parts = [rf"t={int(steps[index])}"]
        if has_loss:
            parts.append(rf"\mathcal L_t={format_number(loss[index])}")
        if accuracy is not None and accuracy.size == steps.size:
            parts.append(rf"\text{{accuracy}}={format_number(accuracy[index])}")
        return {
            "x": 1.0,
            "y": 1.235,
            "xref": "paper",
            "yref": "paper",
            "xanchor": "right",
            "yanchor": "middle",
            "showarrow": False,
            "text": "$" + r"\quad ".join(parts) + "$",
            "font": {"size": 15, "color": NEURAL_COLORS["text"]},
        }

    static_annotations = list(figure.layout.annotations)
    figure.frames = []
    frames = []
    for position, index in enumerate(selected):
        payload, targets_index = [], []
        payload.append(go.Heatmap(z=quantize(fields[position], field_low, field_high)))
        targets_index.append(0)
        for slot in range(len(shown_units)):
            payload.append(go.Heatmap(z=quantize(unit_fields[position][slot], unit_low, unit_high)))
            targets_index.append(1 + slot)
        if boundary_index is not None:
            payload.append(go.Contour(z=quantize(fields[position], field_low, field_high)))
            targets_index.append(boundary_index)
        for slot in range(len(fold_indices)):
            xs, ys = folds[position][slot]
            payload.append(go.Scatter(x=xs, y=ys))
            targets_index.append(fold_indices[slot])
            payload.append(go.Scatter(x=xs, y=ys))
            targets_index.append(own_fold_indices[slot])
        if loss_index is not None:
            payload.append(go.Scatter(x=steps[: index + 1].tolist(), y=rounded(loss[: index + 1], 6)))
            targets_index.append(loss_index)
            payload.append(go.Scatter(x=[int(steps[index])], y=[float(loss[index])]))
            targets_index.append(loss_index + 1)
        if readout.out_features == 1:
            payload.append(
                go.Bar(y=rounded(readouts[position][0]), text=[format_number(v, 3) for v in readouts[position][0]])
            )
        else:
            payload.append(go.Heatmap(z=rounded(readouts[position])))
        targets_index.append(readout_index)
        frames.append(
            go.Frame(
                name=f"step_{int(index)}",
                data=payload,
                traces=targets_index,
                layout={"annotations": [*static_annotations, headline(int(index))]},
            )
        )
    figure.frames = frames
    figure.layout.annotations = [*static_annotations, headline(int(selected[0]))]

    explanation = (
        "Each hidden unit folds the plane along the line where its pre-activation is zero (dotted, in its colour); "
        "the readout weights combine the folds into the boundary."
        if single_hidden
        else f"Each panel is one unit of the last hidden layer as a function of the input; "
        f"the readout weights Θ<sup>({L})</sup> combine them."
    )
    disclosure = (
        f"Exact forward passes with recorded parameters · {len(shown_units)} of {hidden.out_features} units "
        f"(largest readout norm)"
    )
    if data.shape[1] > 2:
        disclosure += f" · other features fixed at the {'median' if reference is None else 'reference'}"
    figure.add_annotation(
        x=0.0,
        y=1.13,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        showarrow=False,
        align="left",
        text=f"{explanation}<br><span style='color:{NEURAL_COLORS['muted']}'>{disclosure}</span>",
        font={"size": 12, "color": NEURAL_COLORS["muted"]},
    )
    for frame in figure.frames:
        frame.layout.annotations = [*frame.layout.annotations, figure.layout.annotations[-1]]

    if title is None:
        title = "How a network builds a non-linear decision"
    layout = neural_layout(title, height=830)
    layout["title"].update({"y": 0.985, "yanchor": "top"})
    layout["margin"] = {"t": 160, "r": 70, "b": 95, "l": 60}
    layout.update(
        playback_controls(
            [f"step_{int(i)}" for i in selected],
            [str(int(steps[i])) for i in selected],
            frame_duration=frame_duration,
            redraw=True,
            y=-0.05,
        )
    )
    layout["showlegend"] = False
    layout["meta"] = {
        "mlektic_neural_decision": {
            "schema_version": 1,
            "semantics": "exact network output on a two-feature plane at recorded checkpoints",
            "output_semantics": mode,
            "features": [f + 1 for f in features],
            "other_features": "median of inputs" if reference is None else "reference",
            "fixed_values": [float(v) for v in fixed],
            "grid_size": grid_size,
            "unit_grid_size": unit_grid_size,
            "hidden_layer": hidden.index,
            "units_shown": [u + 1 for u in shown_units],
            "unit_selection": "largest readout column norm at the final checkpoint",
            "fold_lines_exact": single_hidden,
            "field_range": [field_low, field_high],
            "field_quantization": "integers 0-255 on one global range",
            "recorded_steps": [int(steps[i]) for i in selected],
            "loss": loss_label(history),
        }
    }
    figure.update_layout(**layout)
    figure.update_xaxes(showgrid=False, zeroline=False, constrain="domain")
    figure.update_yaxes(showgrid=False, zeroline=False)
    figure.update_xaxes(
        range=[axis_x[0], axis_x[-1]], title_text=f"${feature_names[0]}$", title_standoff=2, row=1, col=1
    )
    figure.update_yaxes(
        range=[axis_y[0], axis_y[-1]], title_text=f"${feature_names[1]}$", scaleanchor="x", row=1, col=1
    )
    for row, col in _CELLS:
        figure.update_xaxes(range=[unit_x[0], unit_x[-1]], showticklabels=False, row=row, col=col)
        figure.update_yaxes(range=[unit_y[0], unit_y[-1]], showticklabels=False, row=row, col=col)
    figure.update_xaxes(showgrid=True, gridcolor=NEURAL_COLORS["grid"], row=3, col=1)
    if has_loss:
        figure.update_yaxes(
            showgrid=True, gridcolor=NEURAL_COLORS["grid"], range=[0, float(np.nanmax(loss)) * 1.1], row=3, col=1
        )
    if readout.out_features == 1:
        figure.update_yaxes(
            range=[-readout_limit, readout_limit],
            showgrid=True,
            gridcolor=NEURAL_COLORS["grid"],
            zeroline=True,
            zerolinecolor=NEURAL_COLORS["muted"],
            row=3,
            col=2,
        )
    return figure


__all__ = ["build_nn_decision_figure"]
