r"""How each layer transforms the input plane.

A grid on the plane of two input features, the data, and the exact decision
boundary are pushed through the network one transformation at a time:
:math:`\mathbf a^{(0)}=\mathbf x`, then for every layer the affine map
:math:`\mathbf z^{(\ell)}=\Theta^{(\ell)}\mathbf a^{(\ell-1)}+\boldsymbol\theta_0^{(\ell)}`
and the activation :math:`\mathbf a^{(\ell)}=\phi_\ell(\mathbf z^{(\ell)})`.

Every stage is an exact forward pass. A stage with more than two coordinates is
shown through a disclosed linear projection:

* an affine image of the plane is itself a plane, so its two principal axes give
  an isometric view;
* the last hidden representation is read along the readout direction of
  :math:`\Theta^{(L)}` (horizontal axis), so the decision boundary of a
  one-output classifier is exactly a vertical line there;
* any other stage uses its two principal components, with the retained variance
  reported.

Stages are uniformly rescaled to fit the view. The motion between two stages is
a visual interpolation of the displayed coordinates; only the stages are states
of the network.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import plotly.graph_objects as go

from ._style import NEURAL_COLORS, neural_layout
from .geometry import (
    CLASS_COLORS,
    CLASS_SYMBOLS,
    UNIT_SCALE,
    as_numpy,
    dense_chain,
    forward_states,
    output_semantics,
    playback_controls,
    recorded_steps,
    rounded,
    snapshots,
)

_SUP = str.maketrans("0123456789()-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁽⁾⁻")


def _sup(index: int | str) -> str:
    return f"({index})".translate(_SUP)


def _principal_axes(points: np.ndarray, count: int = 2) -> tuple[np.ndarray, np.ndarray]:
    centered = points - points.mean(axis=0)
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    variance = singular**2
    share = variance / variance.sum() if variance.sum() > 0 else np.zeros_like(variance)
    axes = right[:count]
    if axes.shape[0] < count:  # a one-dimensional stage: complete with any orthogonal axis
        extra = np.zeros((count - axes.shape[0], points.shape[1]))
        axes = np.vstack([axes, extra])
    return axes.T, share


def _orthonormal_readout(weight: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, str]:
    """Basis whose first axis is a readout direction of ``Θ^(L)``."""
    if weight.shape[0] == 1:
        u = weight[0] / max(np.linalg.norm(weight[0]), 1e-12)
        residual = points - points.mean(axis=0)
        residual = residual - np.outer(residual @ u, u)
        _, _, right = np.linalg.svd(residual, full_matrices=False)
        v = right[0] - (right[0] @ u) * u
        v = v / max(np.linalg.norm(v), 1e-12)
        return np.c_[
            u, v
        ], "horizontal: readout direction of Θ (exact boundary); vertical: first principal axis orthogonal to it"
    _, _, right = np.linalg.svd(weight, full_matrices=False)
    note = "readout plane of Θ (top two right singular vectors)"
    if weight.shape[0] > 2:
        note += "; logits also depend on directions outside this plane"
    return right[:2].T, note


def _boundary_points(field: np.ndarray, axis_x: np.ndarray, axis_y: np.ndarray, level: float, limit: int) -> np.ndarray:
    """Points where the field crosses ``level`` along rows and columns."""
    values = field - level
    points: List[tuple[float, float]] = []
    for row in range(values.shape[0]):
        line = values[row]
        for j in np.where(np.signbit(line[:-1]) != np.signbit(line[1:]))[0]:
            t = line[j] / (line[j] - line[j + 1])
            points.append((axis_x[j] + t * (axis_x[j + 1] - axis_x[j]), axis_y[row]))
    for col in range(values.shape[1]):
        line = values[:, col]
        for j in np.where(np.signbit(line[:-1]) != np.signbit(line[1:]))[0]:
            t = line[j] / (line[j] - line[j + 1])
            points.append((axis_x[col], axis_y[j] + t * (axis_y[j + 1] - axis_y[j])))
    if not points:
        return np.empty((0, 2))
    array = np.asarray(points)
    return array[:: max(1, len(array) // limit)]


def build_nn_representation_figure(
    model: Any,
    inputs: Any,
    history: Dict[str, Any] | None = None,
    *,
    targets: Any | None = None,
    checkpoint: int = -1,
    features: Sequence[int] = (0, 1),
    reference: Any | None = None,
    grid_lines: int = 15,
    max_layers: int = 3,
    max_points: int = 400,
    stage_duration: int = 2600,
    transition: int = 1800,
    padding: float = 0.12,
    title: str | None = None,
) -> go.Figure:
    """Animate the input plane through every affine map and activation."""
    chain = dense_chain(model, purpose="The representation view")
    data = np.asarray(as_numpy(inputs), dtype=float)
    if data.ndim != 2 or data.shape[1] != chain[0].in_features or data.shape[1] < 2:
        raise ValueError("inputs must be a 2-D batch with at least two features, one column per model input.")
    features = tuple(int(f) for f in features)
    if len(features) != 2 or len(set(features)) != 2 or not all(0 <= f < data.shape[1] for f in features):
        raise ValueError("features must name two distinct input columns.")
    if not 5 <= grid_lines <= 31:
        raise ValueError("grid_lines must be between 5 and 31.")
    fixed = np.median(data, axis=0) if reference is None else np.asarray(as_numpy(reference), dtype=float).ravel()
    labels = None if targets is None else np.asarray(as_numpy(targets))
    rng = np.random.default_rng(0)
    keep = np.sort(rng.choice(len(data), size=min(max_points, len(data)), replace=False))
    data, labels = data[keep], (None if labels is None else labels[keep])

    if history is not None:
        steps = recorded_steps(history)
        index = checkpoint if checkpoint >= 0 else steps.size + checkpoint
        if not 0 <= index < steps.size:
            raise ValueError("checkpoint is outside the recorded history.")
        parameters, buffers = snapshots(history, index)
        step_label = f"recorded step t = {int(steps[index])}"
    else:
        parameters, buffers = None, None
        step_label = "current parameters"
        history = {}
    mode = output_semantics(chain, history, labels)
    readout = chain[-1]
    weight = (
        np.asarray(parameters[readout.weight], dtype=float)
        if parameters and readout.weight in parameters
        else readout_weight_from_model(model, readout)
    )
    bias = (
        np.asarray(parameters[readout.bias], dtype=float)
        if parameters and readout.bias and readout.bias in parameters
        else readout_bias_from_model(model, readout)
    )

    # Geometry in the input plane: grid lines, data, and the exact boundary.
    first, second = features
    low = data[:, [first, second]].min(axis=0)
    high = data[:, [first, second]].max(axis=0)
    margin = (high - low) * padding + 1e-9
    xs = np.linspace(low[0] - margin[0], high[0] + margin[0], grid_lines)
    ys = np.linspace(low[1] - margin[1], high[1] + margin[1], grid_lines)
    samples = 64
    tx = np.linspace(xs[0], xs[-1], samples)
    ty = np.linspace(ys[0], ys[-1], samples)

    def lift(plane: np.ndarray) -> np.ndarray:
        full = np.repeat(fixed[None, :], len(plane), axis=0)
        full[:, first], full[:, second] = plane[:, 0], plane[:, 1]
        return full

    horizontal = np.vstack([np.c_[tx, np.full(samples, y)] for y in ys])
    vertical = np.vstack([np.c_[np.full(samples, x), ty] for x in xs])
    sets: Dict[str, np.ndarray] = {"horizontal": horizontal, "vertical": vertical, "data": data[:, [first, second]]}
    classification = mode in {"probability", "logit"}
    level = 0.5 if mode == "probability" else 0.0
    if classification:
        fine = 160
        fx, fy = np.linspace(xs[0], xs[-1], fine), np.linspace(ys[0], ys[-1], fine)
        gx, gy = np.meshgrid(fx, fy)
        dense = forward_states(model, chain, lift(np.c_[gx.ravel(), gy.ravel()]), parameters, buffers)
        boundary = _boundary_points(dense["output"][:, 0].reshape(fine, fine), fx, fy, level, 320)
        if boundary.size:
            sets["boundary"] = boundary

    states = {key: forward_states(model, chain, lift(points), parameters, buffers) for key, points in sets.items()}

    # Stages: x, then z and a of each hidden layer, the readout, and the output.
    hidden = chain[:-1]
    shown_hidden = hidden if len(hidden) <= max_layers else [*hidden[: max_layers - 1], hidden[-1]]
    stages: List[Dict[str, Any]] = [{"key": "input", "label": "x"}]
    for layer in shown_hidden:
        stages.append({"key": ("z", layer.index), "label": f"z{_sup(layer.index)}", "layer": layer})
        if layer.activation_type != "Identity":
            stages.append({"key": ("a", layer.index), "label": f"a{_sup(layer.index)}", "layer": layer})
    if not hidden:
        raise ValueError("The representation view needs at least one hidden dense layer.")
    if readout.out_features == 1:
        stages.append({"key": "output", "label": "ŷ" if mode != "logit" else f"z{_sup(readout.index)}"})

    def raw(stage: Dict[str, Any], key: str) -> np.ndarray:
        if stage["key"] == "input":
            return sets[key]
        if stage["key"] == "output":
            return states[key]["output"][:, :1]
        kind, layer_index = stage["key"]
        return states[key][kind][layer_index - 1]

    last_hidden = hidden[-1].index
    coordinates: List[Dict[str, np.ndarray]] = []
    descriptions: List[str] = []
    for position, stage in enumerate(stages):
        if stage["key"] == "input":
            basis, note = None, "input plane"
        elif stage["key"] == "output":
            basis, note = None, "readout coordinate against the network output"
        else:
            kind, layer_index = stage["key"]
            cloud = np.vstack([raw(stage, key) for key in sets])
            if (
                kind == "a"
                and layer_index == last_hidden
                or (kind == "z" and layer_index == last_hidden and hidden[-1].activation_type == "Identity")
            ):
                basis, note = _orthonormal_readout(weight, cloud)
            elif cloud.shape[1] <= 2:
                basis, note = np.eye(cloud.shape[1], 2), "exact coordinates"
            else:
                basis, share = _principal_axes(cloud)
                exact_plane = layer_index == 1 and kind == "z"
                note = (
                    "isometric view of the affine image of the plane"
                    if exact_plane
                    else f"first two principal axes ({share[:2].sum() * 100:.0f}% of variance)"
                )
        stage["note"] = note
        projected: Dict[str, np.ndarray] = {}
        for key in sets:
            values = raw(stage, key)
            if stage["key"] == "input":
                projected[key] = values
            elif stage["key"] == "output":
                readout_coordinate = coordinates[-1][f"__readout_{key}"]
                projected[key] = np.c_[readout_coordinate, values[:, 0]]
            else:
                projected[key] = values @ basis
                if stage["key"] == ("a", last_hidden) or (
                    stage["key"] == ("z", last_hidden) and hidden[-1].activation_type == "Identity"
                ):
                    projected[f"__readout_{key}"] = projected[key][:, 0]
        coordinates.append(projected)
        descriptions.append(note)

    # Uniform rescale of each stage into [-1, 1]² (output stage: its own axes).
    normalized: List[Dict[str, np.ndarray]] = []
    separators: List[float | None] = []
    readout_norm = float(np.linalg.norm(weight[0])) if readout.out_features == 1 else 0.0
    readout_scale = None
    for position, (stage, projected) in enumerate(zip(stages, coordinates)):
        visible = {k: v for k, v in projected.items() if not k.startswith("__")}
        stacked = np.vstack(list(visible.values()))
        if stage["key"] == "output":
            center_u, scale_u = readout_scale
            values = stacked[:, 1]
            if mode == "probability":
                to_y = lambda v: 1.6 * v - 0.8  # noqa: E731
            else:
                spread = max(float(np.max(np.abs(values))), 1e-9)
                to_y = lambda v, s=spread: 0.9 * v / s  # noqa: E731
            normalized.append({k: np.c_[(v[:, 0] - center_u) / scale_u, to_y(v[:, 1])] for k, v in visible.items()})
        else:
            center = visible["data"].mean(axis=0)
            scale = max(float(np.max(np.abs(stacked - center))), 1e-9) / 0.98
            normalized.append({k: (v - center) / scale for k, v in visible.items()})
            if stage["key"] == ("a", last_hidden) or (
                stage["key"] == ("z", last_hidden) and hidden[-1].activation_type == "Identity"
            ):
                readout_scale = (center[0], scale)
        separator = None
        if classification and readout_scale is not None and readout.out_features == 1 and position >= 1:
            is_readout = (
                stage["key"] == "output"
                or stage["key"] == ("a", last_hidden)
                or (stage["key"] == ("z", last_hidden) and hidden[-1].activation_type == "Identity")
            )
            if is_readout:
                u_star = (0.0 - float(bias[0])) / max(readout_norm, 1e-12)
                separator = (u_star - readout_scale[0]) / readout_scale[1]
        separators.append(separator)

    def as_lines(points: np.ndarray, count: int) -> tuple[list, list]:
        rows = points.reshape(count, samples, 2)
        xs_out: list = []
        ys_out: list = []
        for row in rows:
            xs_out += rounded(row[:, 0]) + [None]
            ys_out += rounded(row[:, 1]) + [None]
        return xs_out, ys_out

    activation_names = {layer.index: layer.activation_text for layer in hidden}

    def stage_text(stage: Dict[str, Any]) -> str:
        key = stage["key"]
        if key == "input":
            return (
                "<b>Input plane</b>  <b>a</b><sup>(0)</sup> = <b>x</b><br>The grid is the plane of the two input features"
                + ("<br>The white curve is the exact decision boundary" if classification else "")
            )
        if key == "output":
            name = {"probability": "ŷ = σ(z", "logit": "z", "regression": "ŷ = φ(z"}[mode]
            close = ")" if mode != "logit" else ""
            return (
                f"<b>Output</b>  {name}<sup>({readout.index})</sup>{close}"
                f" with z<sup>({readout.index})</sup> = Θ<sup>({readout.index})</sup><b>a</b><sup>({readout.index - 1})</sup>"
                f" + θ<sub>0</sub><sup>({readout.index})</sup><br>Every point lands on one curve of its readout coordinate"
            )
        kind, index = key
        if kind == "z":
            return (
                f"<b>Affine map</b>  <b>z</b><sup>({index})</sup> = Θ<sup>({index})</sup><b>a</b><sup>({index - 1})</sup>"
                f" + <b>θ</b><sub>0</sub><sup>({index})</sup><br>Rotates, stretches, and shifts the plane: straight lines stay straight"
            )
        text = (
            f"<b>Activation</b>  <b>a</b><sup>({index})</sup> = {activation_names[index]}(<b>z</b><sup>({index})</sup>)"
            "<br>Acts on each coordinate separately, so the grid bends"
        )
        if index == last_hidden and classification:
            text += "; along the readout direction the boundary is a vertical line"
        return text

    traces: List[go.Scatter] = [
        go.Scatter(
            mode="lines",
            line={"color": "rgba(91,143,249,0.55)", "width": 1},
            hoverinfo="skip",
            name="neural representation grid rows",
            uid="NN_REPRESENTATION_ROWS",
        ),
        go.Scatter(
            mode="lines",
            line={"color": "rgba(246,189,96,0.55)", "width": 1},
            hoverinfo="skip",
            name="neural representation grid columns",
            uid="NN_REPRESENTATION_COLUMNS",
        ),
    ]
    boundary_trace = "boundary" in sets
    if boundary_trace:
        traces.append(
            go.Scatter(
                mode="markers",
                marker={"size": 3.4, "color": NEURAL_COLORS["text"]},
                hoverinfo="skip",
                name="neural representation boundary",
                uid="NN_REPRESENTATION_BOUNDARY",
            )
        )
    class_masks: List[tuple[int, np.ndarray]] = []
    if labels is not None and mode != "regression":
        flat = labels.reshape(len(labels), -1)
        codes = flat.argmax(axis=1) if flat.shape[1] > 1 else np.rint(flat[:, 0]).astype(int)
        for k in np.unique(codes):
            class_masks.append((int(k), codes == k))
    if class_masks:
        for k, _ in class_masks:
            traces.append(
                go.Scatter(
                    mode="markers",
                    marker={
                        "size": 7,
                        "color": CLASS_COLORS[k % len(CLASS_COLORS)],
                        "symbol": CLASS_SYMBOLS[k % len(CLASS_SYMBOLS)],
                        "line": {"width": 1, "color": "#111111"},
                    },
                    hoverinfo="skip",
                    name=f"neural representation class {k}",
                    uid=f"NN_REPRESENTATION_CLASS_{k}",
                )
            )
    else:
        colors = None if labels is None else labels.reshape(len(labels), -1)[:, 0].astype(float)
        traces.append(
            go.Scatter(
                mode="markers",
                marker={
                    "size": 7,
                    "color": NEURAL_COLORS["linear"] if colors is None else rounded(colors),
                    "colorscale": UNIT_SCALE if colors is not None else None,
                    "line": {"width": 1, "color": "#111111"},
                },
                hoverinfo="skip",
                name="neural representation data",
                uid="NN_REPRESENTATION_DATA",
            )
        )
    traces.append(
        go.Scatter(
            mode="lines",
            line={"color": NEURAL_COLORS["text"], "width": 2, "dash": "dash"},
            hoverinfo="skip",
            name="neural representation readout boundary",
            uid="NN_REPRESENTATION_SEPARATOR",
        )
    )
    traces.append(
        go.Scatter(
            x=[-1.22],
            y=[1.2],
            cliponaxis=False,
            mode="text",
            textposition="middle right",
            textfont={"size": 14, "color": NEURAL_COLORS["text"]},
            hoverinfo="skip",
            name="neural representation stage",
            uid="NN_REPRESENTATION_STAGE",
        )
    )
    traces.append(
        go.Scatter(
            x=[-1.22],
            y=[-1.22],
            cliponaxis=False,
            mode="text",
            textposition="middle right",
            textfont={"size": 11, "color": NEURAL_COLORS["muted"]},
            hoverinfo="skip",
            name="neural representation projection",
            uid="NN_REPRESENTATION_PROJECTION",
        )
    )

    def stage_payload(position: int) -> List[go.Scatter]:
        stage, coords = stages[position], normalized[position]
        payload = []
        x_rows, y_rows = as_lines(coords["horizontal"], grid_lines)
        x_cols, y_cols = as_lines(coords["vertical"], grid_lines)
        payload += [go.Scatter(x=x_rows, y=y_rows), go.Scatter(x=x_cols, y=y_cols)]
        if boundary_trace:
            payload.append(go.Scatter(x=rounded(coords["boundary"][:, 0]), y=rounded(coords["boundary"][:, 1])))
        if class_masks:
            for _, mask in class_masks:
                payload.append(go.Scatter(x=rounded(coords["data"][mask, 0]), y=rounded(coords["data"][mask, 1])))
        else:
            payload.append(go.Scatter(x=rounded(coords["data"][:, 0]), y=rounded(coords["data"][:, 1])))
        separator = separators[position]
        if separator is None:
            payload.append(go.Scatter(x=[None, None], y=[None, None]))
        else:
            payload.append(go.Scatter(x=[float(separator)] * 2, y=[-1.1, 1.1]))
        payload.append(go.Scatter(text=[f"{position + 1}/{len(stages)} · " + stage_text(stage)]))
        payload.append(go.Scatter(text=[f"View: {descriptions[position]} · {step_label}"]))
        return payload

    initial = stage_payload(0)
    for trace, update in zip(traces, initial):
        trace.update(update.to_plotly_json())
    figure = go.Figure(data=traces)
    names = [f"stage_{position}" for position in range(len(stages))]
    figure.frames = [
        go.Frame(name=name, data=stage_payload(position), traces=list(range(len(traces))))
        for position, name in enumerate(names)
    ]
    if title is None:
        title = "How the layers transform the input plane"
    layout = neural_layout(title, height=760)
    layout["margin"] = {"t": 80, "r": 30, "b": 90, "l": 30}
    layout.update(
        playback_controls(
            names,
            [stage["label"] for stage in stages],
            frame_duration=stage_duration,
            transition=transition,
            redraw=False,
            prefix="Stage: ",
            y=-0.02,
        )
    )
    layout["showlegend"] = False
    layout["xaxis"] = {"range": [-1.25, 1.25], "visible": False}
    layout["yaxis"] = {"range": [-1.3, 1.34], "visible": False, "scaleanchor": "x"}
    layout["meta"] = {
        "mlektic_neural_representation": {
            "schema_version": 1,
            "semantics": "exact forward pass of a grid, the data, and the decision boundary, stage by stage",
            "parameters": step_label,
            "output_semantics": mode,
            "stages": [stage["label"] for stage in stages],
            "projections": descriptions,
            "uniform_rescale_per_stage": True,
            "between_stage_motion": "visual interpolation of displayed coordinates; not network states",
            "features": [f + 1 for f in features],
            "points_shown": int(len(data)),
            "hidden_layers_shown": [layer.index for layer in shown_hidden],
        }
    }
    figure.update_layout(**layout)
    return figure


def readout_weight_from_model(model: Any, layer: Any) -> np.ndarray:
    parameters = dict(model.named_parameters())
    return parameters[layer.weight].detach().cpu().numpy().astype(float)


def readout_bias_from_model(model: Any, layer: Any) -> np.ndarray:
    parameters = dict(model.named_parameters())
    if layer.bias is None:
        return np.zeros(layer.out_features)
    return parameters[layer.bias].detach().cpu().numpy().astype(float)


__all__ = ["build_nn_representation_figure"]
