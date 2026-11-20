r"""One training step in slow motion: forward, loss, backward, update.

The figure follows a single example through the network drawn as a graph. It
uses the course notation of a dense chain and of backpropagation for one sample
(lower-case :math:`l` is the per-sample loss):

.. math::

   \mathbf z^{(\ell)}=\Theta^{(\ell)}\mathbf a^{(\ell-1)}+\boldsymbol\theta_0^{(\ell)},\quad
   \boldsymbol\delta^{(\ell)}=\frac{\partial l}{\partial\mathbf z^{(\ell)}},\quad
   \nabla_{\Theta^{(\ell)}}l=\boldsymbol\delta^{(\ell)}(\mathbf a^{(\ell-1)})^\top,\quad
   \Theta\leftarrow\Theta-\eta\nabla_\Theta l .

Every number is computed exactly with PyTorch autograd on a copy of the model
loaded with one recorded checkpoint. The update is one plain gradient step on
that single example, an illustration of the rule rather than a replay of the
recorded optimizer (which may use batches, momentum, or Adam); the metadata says
so explicitly.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from ...neural.introspection import _require_torch
from ._style import NEURAL_COLORS, neural_layout
from .geometry import (
    GRADIENT_SCALE,
    SIGNED_SCALE,
    UNIT_SCALE,
    as_numpy,
    dense_chain,
    format_number,
    loss_label,
    playback_controls,
    recorded_steps,
    snapshots,
)
from .math_format import display_indices

_BINS = 7


def _bin_styles(scale: Sequence[Sequence[Any]]) -> tuple[List[str], List[float]]:
    centers = [(b + 0.5) / _BINS for b in range(_BINS)]
    colors = sample_colorscale([[float(s), str(c)] for s, c in scale], centers)
    widths = [1.0 + 3.6 * abs(2 * c - 1) for c in centers]
    return colors, widths


def _signed_colors(
    values: Sequence[float], limit: float, scale: Sequence[Sequence[Any]], alpha: float = 0.45
) -> List[str]:
    """Translucent fills: the node value stays readable on dark and light skins."""
    limit = limit or 1.0
    positions = [float(np.clip((v / limit + 1) / 2, 0, 1)) for v in values]
    colors = sample_colorscale([[float(s), str(c)] for s, c in scale], positions)
    return [c.replace("rgb(", "rgba(").replace(")", f", {alpha})") for c in colors]


def _per_sample_loss_text(loss_name: str, y_hat: np.ndarray, target: np.ndarray, value: float) -> str:
    v = format_number(value)
    if loss_name == "BCELoss":
        return f"<i>l</i> = −[y ln ŷ + (1 − y) ln(1 − ŷ)] = {v}"
    if loss_name == "BCEWithLogitsLoss":
        return f"<i>l</i> = −[y ln σ(z) + (1 − y) ln(1 − σ(z))] = {v}"
    if loss_name == "MSELoss":
        return (
            f"<i>l</i> = (ŷ − y)<sup>2</sup> = {v}"
            if y_hat.size == 1
            else f"<i>l</i> = mean<sub>k</sub>(ŷ<sub>k</sub> − y<sub>k</sub>)<sup>2</sup> = {v}"
        )
    if loss_name == "CrossEntropyLoss":
        return f"<i>l</i> = −ln p̂<sub>y</sub> = −ln softmax(<b>z</b>)<sub>y</sub> = {v}"
    if loss_name == "L1Loss":
        return f"<i>l</i> = |ŷ − y| = {v}"
    return f"<i>l</i> = {loss_name}(ŷ, y) = {v}"


def build_nn_training_step_figure(
    model: Any,
    input_sample: Any,
    target: Any,
    loss_fn: Any,
    history: Dict[str, Any] | None = None,
    *,
    checkpoint: int = 0,
    learning_rate: float | None = None,
    max_neurons: int = 6,
    dec: int = 3,
    frame_duration: int = 1500,
    transition: int = 900,
    title: str | None = None,
) -> go.Figure:
    """Follow one example through forward, loss, backward, and one update."""
    torch = _require_torch()
    chain = dense_chain(model, purpose="The training-step view")
    if max_neurons < 2:
        raise ValueError("max_neurons must be at least 2.")
    if loss_fn is None or not callable(loss_fn):
        raise ValueError("loss_fn must be the PyTorch criterion used for training.")
    work = copy.deepcopy(model)
    work.eval()
    step_text = "current parameters"
    if history is not None:
        steps = recorded_steps(history)
        index = checkpoint if checkpoint >= 0 else steps.size + checkpoint
        if not 0 <= index < steps.size:
            raise ValueError("checkpoint is outside the recorded history.")
        parameters, buffers = snapshots(history, index)
        with torch.no_grad():
            for name, parameter in work.named_parameters():
                if name in parameters and tuple(parameters[name].shape) == tuple(parameter.shape):
                    parameter.copy_(torch.as_tensor(parameters[name], dtype=parameter.dtype))
            for name, buffer in work.named_buffers():
                if buffers and name in buffers:
                    buffer.copy_(torch.as_tensor(buffers[name], dtype=buffer.dtype))
        step_text = f"recorded checkpoint t = {int(steps[index])}"
        if learning_rate is None:
            groups = history.get("optimizer_groups") or []
            if index < len(groups) and groups[index] and "lr" in groups[index][0]:
                learning_rate = float(groups[index][0]["lr"])
    if learning_rate is None:
        learning_rate = 0.1
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    parameter = next(work.parameters())
    x = torch.as_tensor(as_numpy(input_sample), dtype=parameter.dtype).reshape(1, -1)
    if x.shape[1] != chain[0].in_features:
        raise ValueError("input_sample must contain one value per model input feature.")
    y = torch.as_tensor(as_numpy(target))
    loss_name = loss_fn.__class__.__name__
    if loss_name in {"CrossEntropyLoss", "NLLLoss"}:
        y = y.to(dtype=torch.long).reshape(1)
    else:
        y = y.to(dtype=parameter.dtype).reshape(1, -1)

    retained: Dict[str, Any] = {}
    modules = dict(work.named_modules())

    def keep(name: str):
        def hook(_module, _inputs, output):
            output.retain_grad()
            retained[name] = output

        return hook

    hooks = [modules[layer.linear].register_forward_hook(keep(layer.linear)) for layer in chain]
    hooks += [
        modules[layer.activation].register_forward_hook(keep(layer.activation)) for layer in chain if layer.activation
    ]
    try:
        work.zero_grad()
        output = work(x)
        loss = loss_fn(output, y)
        if loss.numel() != 1:
            raise ValueError("loss_fn must reduce one example to a scalar.")
        loss.backward()
    finally:
        for hook in hooks:
            hook.remove()
    z_values = [retained[layer.linear].detach().numpy()[0] for layer in chain]
    a_values = [
        (retained[layer.activation] if layer.activation else retained[layer.linear]).detach().numpy()[0]
        for layer in chain
    ]
    deltas = [retained[layer.linear].grad.detach().numpy()[0] for layer in chain]
    named = dict(work.named_parameters())
    weights = [named[layer.weight].detach().numpy().copy() for layer in chain]
    gradients = [named[layer.weight].grad.detach().numpy().copy() for layer in chain]
    new_weights = [w - learning_rate * g for w, g in zip(weights, gradients)]
    loss_before = float(loss.detach())
    with torch.no_grad():
        for p in work.parameters():
            if p.grad is not None:
                p -= learning_rate * p.grad
        loss_after = float(loss_fn(work(x), y).detach())
    x_values = x.detach().numpy()[0]

    # ── geometry ────────────────────────────────────────────────────────────
    widths = [chain[0].in_features] + [layer.out_features for layer in chain]
    shown = [display_indices(width, max_neurons) for width in widths]
    L = len(chain)

    def node_y(count: int, position: int, total: int) -> float:
        if count == 1:
            return 0.0
        span = min(2.8, 0.75 * (count - 1))
        y = span / 2 - span * position / (count - 1)
        if total > count and position >= count // 2:  # leave a gap for the vertical ellipsis
            y -= 0.18
        return y

    positions = [
        [(float(layer), node_y(len(indices), p, widths[layer])) for p, _ in enumerate(indices)]
        for layer, indices in enumerate(shown)
    ]
    edges = []  # (layer ℓ, source position, target position, source unit i, target unit j)
    for ell in range(1, L + 1):
        for si, i in enumerate(shown[ell - 1]):
            for tj, j in enumerate(shown[ell]):
                edges.append((ell, si, tj, i, j))
    weight_values = np.array([weights[ell - 1][j, i] for ell, _, _, i, j in edges])
    new_weight_values = np.array([new_weights[ell - 1][j, i] for ell, _, _, i, j in edges])
    gradient_values = np.array([gradients[ell - 1][j, i] for ell, _, _, i, j in edges])
    activations = [x_values, *a_values]
    signals = np.array([weights[ell - 1][j, i] * activations[ell - 1][i] for ell, _, _, i, j in edges])
    back_signals = np.array([deltas[ell - 1][j] * activations[ell - 1][i] for ell, _, _, i, j in edges])
    weight_limit = float(np.max(np.abs(np.r_[weight_values, new_weight_values]))) or 1.0
    gradient_limit = float(np.max(np.abs(gradient_values))) or 1.0
    weight_colors, bin_widths = _bin_styles(UNIT_SCALE)
    gradient_colors, _ = _bin_styles(GRADIENT_SCALE)

    def edge_bins(values: np.ndarray, limit: float, colors: List[str]) -> List[go.Scatter]:
        codes = np.clip(((values / limit) + 1) / 2 * _BINS, 0, _BINS - 1e-9).astype(int)
        payload = []
        for b in range(_BINS):
            xs: List[Any] = []
            ys: List[Any] = []
            for e, (ell, si, tj, _, _) in enumerate(edges):
                if codes[e] == b:
                    (x0, y0), (x1, y1) = positions[ell - 1][si], positions[ell][tj]
                    xs += [x0, x1, None]
                    ys += [y0, y1, None]
            payload.append(
                go.Scatter(x=xs or [None], y=ys or [None], line={"color": colors[b], "width": bin_widths[b]})
            )
        return payload

    max_edges = max(sum(1 for e in edges if e[0] == ell) for ell in range(1, L + 1))

    def pulses(ell: int, fraction: float, values: np.ndarray, backward: bool) -> go.Scatter:
        xs, ys, sizes, colors = [], [], [], []
        limit = float(np.max(np.abs(values[[k for k, e in enumerate(edges) if e[0] == ell]]))) or 1.0
        for k, (layer, si, tj, _, _) in enumerate(edges):
            if layer != ell:
                continue
            (x0, y0), (x1, y1) = positions[ell - 1][si], positions[ell][tj]
            f = 0.16 + 0.68 * ((1 - fraction) if backward else fraction)
            xs.append(round(x0 + (x1 - x0) * f, 4))
            ys.append(round(y0 + (y1 - y0) * f, 4))
            sizes.append(round(6 + 14 * min(1.0, abs(values[k]) / limit), 2))
            if backward:
                colors.append(NEURAL_COLORS["update_positive"] if values[k] >= 0 else NEURAL_COLORS["update_negative"])
            else:
                colors.append(NEURAL_COLORS["positive"] if values[k] >= 0 else NEURAL_COLORS["negative"])
        while len(xs) < max_edges:
            xs.append(None)
            ys.append(None)
            sizes.append(0)
            colors.append(NEURAL_COLORS["grid"])
        return go.Scatter(x=xs, y=ys, marker={"size": sizes, "color": colors})

    hidden_pulses = go.Scatter(x=[None] * max_edges, y=[None] * max_edges)
    flat_positions = [p for layer in positions for p in layer]
    offsets = np.cumsum([0] + [len(s) for s in shown])
    node_count = int(offsets[-1])
    dim = "#2a2c33"
    activation_limit = max(float(np.max(np.abs(np.concatenate([np.ravel(a) for a in activations])))), 1e-9)
    delta_limit = max(float(np.max(np.abs(np.concatenate([np.ravel(d) for d in deltas])))), 1e-9)

    def node_state(lit_forward: int, lit_backward: int | None) -> go.Scatter:
        """Colour layers ``<= lit_forward`` by activation, ``>= lit_backward`` by δ."""
        colors, texts, side, hover = [dim] * node_count, [""] * node_count, [""] * node_count, [""] * node_count
        for layer, indices in enumerate(shown):
            for p, unit in enumerate(indices):
                k = int(offsets[layer] + p)
                if lit_backward is not None and layer >= max(lit_backward, 1):
                    value = float(deltas[layer - 1][unit])
                    colors[k] = _signed_colors([value], delta_limit, GRADIENT_SCALE)[0]
                    texts[k] = format_number(value, dec)
                    side[k] = f"δ<sub>{unit + 1}</sub><sup>({layer})</sup>"
                    hover[k] = f"δ<sub>{unit + 1}</sub><sup>({layer})</sup> = ∂<i>l</i>/∂z = {format_number(value)}"
                elif layer <= lit_forward and lit_backward is None:
                    value = float(activations[layer][unit])
                    colors[k] = _signed_colors([value], activation_limit, SIGNED_SCALE)[0]
                    texts[k] = format_number(value, dec)
                    if layer == 0:
                        side[k] = f"x<sub>{unit + 1}</sub>"
                        hover[k] = (
                            f"a<sub>{unit + 1}</sub><sup>(0)</sup> = x<sub>{unit + 1}</sub> = {format_number(value)}"
                        )
                    else:
                        z = float(z_values[layer - 1][unit])
                        side[k] = f"z = {format_number(z, dec)}"
                        hover[k] = (
                            f"z<sub>{unit + 1}</sub><sup>({layer})</sup> = {format_number(z)}<br>"
                            f"a<sub>{unit + 1}</sub><sup>({layer})</sup> = {format_number(value)}"
                        )
        return go.Scatter(marker={"color": colors}, text=texts, hovertext=hover), go.Scatter(text=side)

    # ── traces ──────────────────────────────────────────────────────────────
    figure = go.Figure()
    for b in range(_BINS):
        figure.add_trace(
            go.Scatter(
                mode="lines",
                hoverinfo="skip",
                line={"color": weight_colors[b], "width": bin_widths[b]},
                name=f"neural step edges {b}",
                uid=f"NN_STEP_EDGES_{b}",
            )
        )
    figure.add_trace(
        go.Scatter(
            mode="markers",
            hoverinfo="skip",
            marker={"size": [0] * max_edges, "line": {"width": 0}},
            name="neural step pulses",
            uid="NN_STEP_PULSES",
        )
    )
    # Opaque discs in the canvas colour keep edges from crossing the node values
    # seen through the translucent node fills (skins translate this colour too).
    backing_index = len(figure.data)
    figure.add_trace(
        go.Scatter(
            x=[p[0] for p in flat_positions],
            y=[p[1] for p in flat_positions],
            mode="markers",
            marker={"size": 46, "color": NEURAL_COLORS["background"], "line": {"width": 0}},
            hoverinfo="skip",
            name="neural step node backing",
            uid="NN_STEP_BACKING",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[p[0] for p in flat_positions],
            y=[p[1] for p in flat_positions],
            mode="markers+text",
            textposition="middle center",
            textfont={"size": 11, "color": NEURAL_COLORS["text"]},
            marker={"size": 46, "color": [dim] * node_count, "line": {"width": 2, "color": NEURAL_COLORS["muted"]}},
            hovertemplate="%{hovertext}<extra></extra>",
            name="neural step nodes",
            uid="NN_STEP_NODES",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[p[0] for p in flat_positions],
            y=[p[1] - 0.36 for p in flat_positions],
            mode="text",
            textfont={"size": 11, "color": NEURAL_COLORS["muted"]},
            hoverinfo="skip",
            name="neural step node labels",
            uid="NN_STEP_LABELS",
        )
    )
    loss_x = L + 0.85
    figure.add_trace(
        go.Scatter(
            x=[loss_x],
            y=[0],
            mode="markers+text",
            marker={"size": 74, "symbol": "square", "color": dim, "line": {"width": 2, "color": NEURAL_COLORS["grid"]}},
            text=["<i>l</i>"],
            textfont={"size": 13, "color": NEURAL_COLORS["text"]},
            hoverinfo="skip",
            name="neural step loss",
            uid="NN_STEP_LOSS",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[L + 0.22, loss_x - 0.25],
            y=[0, 0],
            mode="lines",
            line={"color": NEURAL_COLORS["grid"], "width": 2, "dash": "dot"},
            hoverinfo="skip",
            name="neural step loss link",
            uid="NN_STEP_LINK",
        )
    )
    phase_x = list(np.linspace(0, loss_x, 4))
    figure.add_trace(
        go.Scatter(
            x=phase_x,
            y=[-2.05] * 4,
            mode="markers+text",
            text=["1 · Forward", "2 · Loss", "3 · Backward", "4 · Update"],
            textposition="bottom center",
            textfont={"size": 12, "color": NEURAL_COLORS["muted"]},
            marker={"size": 14, "color": [NEURAL_COLORS["grid"]] * 4},
            hoverinfo="skip",
            name="neural step phases",
            uid="NN_STEP_PHASES",
        )
    )
    equation_index = len(figure.data)
    figure.add_trace(
        go.Scatter(
            x=[-0.45],
            y=[2.35],
            mode="text",
            textposition="middle right",
            textfont={"size": 14, "color": NEURAL_COLORS["text"]},
            cliponaxis=False,
            hoverinfo="skip",
            name="neural step equation",
            uid="NN_STEP_EQUATION",
        )
    )
    ellipses = [
        (float(layer), 0.0 if len(indices) % 2 == 0 else node_y(len(indices), len(indices) // 2, widths[layer]) + 0.18)
        for layer, indices in enumerate(shown)
        if widths[layer] > len(indices)
    ]
    if ellipses:
        figure.add_trace(
            go.Scatter(
                x=[e[0] for e in ellipses],
                y=[e[1] - 0.09 for e in ellipses],
                mode="text",
                text=["⋮"] * len(ellipses),
                textfont={"size": 18, "color": NEURAL_COLORS["muted"]},
                hoverinfo="skip",
                name="neural step omitted units",
                uid="NN_STEP_OMITTED",
            )
        )

    # ── frames ──────────────────────────────────────────────────────────────
    y_text = ", ".join(format_number(v, dec) for v in np.ravel(as_numpy(target)))
    x_text = ", ".join(format_number(v, dec) for v in x_values[: min(4, x_values.size)])
    if x_values.size > 4:
        x_text += ", …"
    output_layer = chain[-1]
    activation_name = {layer.index: layer.activation_text for layer in chain}
    loss_line = _per_sample_loss_text(loss_name, a_values[-1], np.ravel(as_numpy(target)), loss_before)
    if (
        (output_layer.activation_type == "Sigmoid" and loss_name == "BCELoss")
        or loss_name
        in {
            "BCEWithLogitsLoss",
            "CrossEntropyLoss",
        }
        or (output_layer.activation_type == "Identity" and loss_name == "MSELoss" and output_layer.out_features == 1)
    ):
        if loss_name == "MSELoss":
            delta_rule = "δ<sup>(L)</sup> = ∂<i>l</i>/∂z<sup>(L)</sup> = 2(ŷ − y)"
        elif loss_name == "CrossEntropyLoss":
            delta_rule = "<b>δ</b><sup>(L)</sup> = ∂<i>l</i>/∂<b>z</b><sup>(L)</sup> = <b>p̂</b> − <b>e</b><sub>y</sub>"
        else:
            delta_rule = "δ<sup>(L)</sup> = ∂<i>l</i>/∂z<sup>(L)</sup> = ŷ − y"
    else:
        delta_rule = "<b>δ</b><sup>(L)</sup> = ∂<i>l</i>/∂<b>z</b><sup>(L)</sup>"
    delta_rule = delta_rule.replace("(L)", f"({L})")
    out_delta = ", ".join(format_number(v) for v in deltas[-1][:4])

    frames: List[go.Frame] = []
    phase_colors = lambda active: [  # noqa: E731
        NEURAL_COLORS["activation"]
        if p == active
        else (NEURAL_COLORS["muted"] if p < active else NEURAL_COLORS["grid"])
        for p in range(4)
    ]

    def add(name, label, phase, edge_mode, pulse, nodes, equation, loss_value=None, loss_color=dim, link_active=False):
        if edge_mode == "weights":
            edge_payload = edge_bins(weight_values, weight_limit, weight_colors)
        elif edge_mode == "updated":
            edge_payload = edge_bins(new_weight_values, weight_limit, weight_colors)
        else:
            edge_payload = edge_bins(gradient_values, gradient_limit, gradient_colors)
        node_payload, side_payload = nodes
        payload = [
            *edge_payload,
            pulse,
            node_payload,
            side_payload,
            go.Scatter(
                text=["<i>l</i>" if loss_value is None else f"<i>l</i> = {format_number(loss_value, dec)}"],
                marker={"color": loss_color},
            ),
            go.Scatter(line={"color": NEURAL_COLORS["activation"] if link_active else NEURAL_COLORS["grid"]}),
            go.Scatter(marker={"color": phase_colors(phase)}),
            go.Scatter(text=[equation]),
        ]
        targets = [index for index in range(equation_index + 1) if index != backing_index]
        frames.append(go.Frame(name=name, data=payload, traces=targets))
        labels.append(label)

    labels: List[str] = []
    loss_hot = "rgba(255,125,142,0.32)"
    loss_cool = "rgba(85,214,190,0.32)"
    add(
        "start",
        "start",
        -1,
        "weights",
        hidden_pulses,
        node_state(-1, None),
        f"One example <b>x</b> = ({x_text}) with target y = {y_text} · edge colour = weight θ<sub>ji</sub><sup>(ℓ)</sup>"
        " (blue negative, gold positive)",
    )
    for ell in range(1, L + 1):
        layer = chain[ell - 1]
        add(
            f"forward_{ell}_out",
            f"a{ell - 1}→",
            0,
            "weights",
            pulses(ell, 0.0, signals, False),
            node_state(ell - 1, None),
            f"Forward · each edge carries s<sub>ji</sub><sup>({ell})</sup> = θ<sub>ji</sub><sup>({ell})</sup>"
            f"a<sub>i</sub><sup>({ell - 1})</sup> (pulse size ∝ |s|)",
        )
        add(
            f"forward_{ell}_in",
            f"→z{ell}",
            0,
            "weights",
            pulses(ell, 1.0, signals, False),
            node_state(ell - 1, None),
            f"Forward · each edge carries s<sub>ji</sub><sup>({ell})</sup> = θ<sub>ji</sub><sup>({ell})</sup>"
            f"a<sub>i</sub><sup>({ell - 1})</sup> (pulse size ∝ |s|)",
        )
        activation_text = (
            f", <b>a</b><sup>({ell})</sup> = {activation_name[ell]}(<b>z</b><sup>({ell})</sup>)"
            if layer.activation_type != "Identity"
            else f", <b>a</b><sup>({ell})</sup> = <b>z</b><sup>({ell})</sup>"
        )
        add(
            f"forward_{ell}_node",
            f"a{ell}",
            0,
            "weights",
            hidden_pulses,
            node_state(ell, None),
            f"Forward · <b>z</b><sup>({ell})</sup> = Θ<sup>({ell})</sup><b>a</b><sup>({ell - 1})</sup>"
            f" + <b>θ</b><sub>0</sub><sup>({ell})</sup>{activation_text}" + (" = ŷ" if ell == L else ""),
        )
    add(
        "loss",
        "l",
        1,
        "weights",
        hidden_pulses,
        node_state(L, None),
        f"Loss of this example · {loss_label({}, loss_fn)}: {loss_line}",
        loss_before,
        loss_hot,
        True,
    )
    add(
        "backward_output",
        f"δ{L}",
        2,
        "weights",
        hidden_pulses,
        node_state(-1, L),
        f"Backward · {delta_rule} = {out_delta}",
        loss_before,
        loss_hot,
        True,
    )
    for ell in range(L, 0, -1):
        add(
            f"backward_{ell}_out",
            f"←{ell}",
            2,
            "weights",
            pulses(ell, 0.0, back_signals, True),
            node_state(-1, ell),
            f"Backward · δ<sup>({ell})</sup> travels back through every weight of Θ<sup>({ell})</sup>",
            loss_before,
            loss_hot,
        )
        add(
            f"backward_{ell}_in",
            f"←{ell - 1}",
            2,
            "weights",
            pulses(ell, 1.0, back_signals, True),
            node_state(-1, ell),
            f"Backward · δ<sup>({ell})</sup> travels back through every weight of Θ<sup>({ell})</sup>",
            loss_before,
            loss_hot,
        )
        if ell > 1:
            prev = chain[ell - 2]
            derivative = {
                "Tanh": "tanh′(z) = 1 − tanh<sup>2</sup>z",
                "Sigmoid": "σ′(z) = σ(z)(1 − σ(z))",
                "ReLU": "ReLU′(z) = 1[z > 0]",
            }.get(prev.activation_type, f"φ′<sub>{ell - 1}</sub>")
            add(
                f"backward_{ell - 1}_node",
                f"δ{ell - 1}",
                2,
                "weights",
                hidden_pulses,
                node_state(-1, ell - 1),
                f"Backward · <b>δ</b><sup>({ell - 1})</sup> = ((Θ<sup>({ell})</sup>)<sup>⊤</sup><b>δ</b><sup>({ell})</sup>)"
                f" ⊙ φ′<sub>{ell - 1}</sub>(<b>z</b><sup>({ell - 1})</sup>) with {derivative}",
                loss_before,
                loss_hot,
            )
    add(
        "gradients",
        "∇l",
        2,
        "gradients",
        hidden_pulses,
        node_state(-1, 1),
        "Gradients · ∇<sub>Θ<sup>(ℓ)</sup></sub> <i>l</i> = <b>δ</b><sup>(ℓ)</sup>(<b>a</b><sup>(ℓ−1)</sup>)<sup>⊤</sup>:"
        " edge ji gets ∂<i>l</i>/∂θ<sub>ji</sub> = δ<sub>j</sub>a<sub>i</sub> (red negative, green positive)",
        loss_before,
        loss_hot,
    )
    add(
        "update",
        "Θ←",
        3,
        "updated",
        hidden_pulses,
        node_state(-1, None),
        f"Update · Θ ← Θ − η∇<sub>Θ</sub> <i>l</i> with η = {format_number(learning_rate, 4)} ·"
        f" this example's loss goes from {format_number(loss_before)} to {format_number(loss_after)}",
        loss_after,
        loss_cool,
    )
    for target, update in zip(frames[0].traces, frames[0].data):
        figure.data[target].update(update.to_plotly_json())
    figure.frames = frames

    if title is None:
        title = "One training step in slow motion"
    layout = neural_layout(title, height=720)
    layout["margin"] = {"t": 70, "r": 30, "b": 95, "l": 30}
    layout.update(
        playback_controls(
            [f.name for f in frames],
            labels,
            frame_duration=frame_duration,
            transition=transition,
            redraw=False,
            prefix="",
            y=-0.02,
        )
    )
    layout["showlegend"] = False
    layout["hovermode"] = "closest"
    layout["xaxis"] = {"range": [-0.5, loss_x + 0.5], "visible": False}
    layout["yaxis"] = {"range": [-2.55, 2.6], "visible": False}
    layer_titles = ["input"] + [
        f"layer {layer.index} · {layer.activation_text}"
        if layer.activation_type != "Identity"
        else f"layer {layer.index}"
        for layer in chain
    ]
    layout["annotations"] = [
        {
            "x": float(layer),
            "y": 1.9,
            "xref": "x",
            "yref": "y",
            "text": name,
            "showarrow": False,
            "font": {"size": 12, "color": NEURAL_COLORS["muted"]},
        }
        for layer, name in enumerate(layer_titles)
    ]
    layout["meta"] = {
        "mlektic_neural_training_step": {
            "schema_version": 1,
            "semantics": "exact forward, per-sample loss, backpropagation, and one plain gradient step for one example",
            "parameters": step_text,
            "loss": loss_name,
            "loss_before": loss_before,
            "loss_after": loss_after,
            "learning_rate": learning_rate,
            "update_rule": "Θ ← Θ − η ∇_Θ l on this single example (illustrative; not the recorded optimizer step)",
            "units_shown": [[int(i) + 1 for i in s] for s in shown],
            "edge_encoding": f"{_BINS} fixed colour/width bins",
            "pulse_motion": "Plotly transitions between keyframes",
        }
    }
    figure.update_layout(**layout)
    return figure


__all__ = ["build_nn_training_step_figure"]
