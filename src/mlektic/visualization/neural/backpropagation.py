"""Layerwise forward/backpropagation teaching view for recorded PyTorch runs."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import plotly.graph_objects as go

from ...neural.introspection import describe_torch_model
from ...neural.taxonomy import select_with_ellipsis
from ._style import NEURAL_COLORS, animation_button_style, neural_layout


def _frame_indices(count: int, maximum: int | None) -> np.ndarray:
    if maximum is None or count <= maximum:
        return np.arange(count, dtype=int)
    if maximum < 1:
        raise ValueError("max_frames must be at least 1 or None.")
    return np.unique(np.linspace(0, count - 1, maximum, dtype=int))


def _trainable_layers(model: Any, input_sample: Any | None) -> list[Dict[str, Any]]:
    layers = []
    for layer in describe_torch_model(model, input_sample):
        parameter_names = [
            name
            for name in layer.get("parameter_shapes", {})
            if name.endswith("weight") or name.endswith("bias")
        ]
        if parameter_names:
            layers.append({**layer, "parameter_names": parameter_names})
    if not layers:
        raise ValueError("The model has no trainable leaf layers to explain.")
    return layers


def _layer_gradient_norms(
    history: Dict[str, Any],
    layers: list[Dict[str, Any]],
    frame_index: int,
) -> np.ndarray:
    values = []
    norms = history.get("gradient_norms", {})
    for layer in layers:
        squares = []
        prefix = layer["name"] + "."
        for name, recorded in norms.items():
            if name.startswith(prefix) and frame_index < len(recorded):
                value = float(recorded[frame_index])
                if np.isfinite(value):
                    squares.append(value * value)
        values.append(float(np.sqrt(np.sum(squares))) if squares else float("nan"))
    return np.asarray(values, dtype=float)


def _layer_update_norms(
    history: Dict[str, Any],
    layers: list[Dict[str, Any]],
    frame_index: int,
) -> np.ndarray:
    """Return exact adjacent-checkpoint parameter changes per visible layer."""
    if frame_index <= 0:
        return np.zeros(len(layers), dtype=float)
    parameters = history.get("parameters", {})
    values = []
    for layer in layers:
        prefix = layer["name"] + "."
        squares = []
        for name, snapshots in parameters.items():
            if not name.startswith(prefix) or frame_index >= len(snapshots):
                continue
            delta = np.asarray(snapshots[frame_index], dtype=float) - np.asarray(
                snapshots[frame_index - 1], dtype=float
            )
            squares.append(float(np.sum(delta * delta)))
        values.append(float(np.sqrt(np.sum(squares))) if squares else float("nan"))
    return np.asarray(values, dtype=float)


def _layer_parameter_norms(
    history: Dict[str, Any],
    layers: list[Dict[str, Any]],
    frame_index: int,
) -> np.ndarray:
    """Aggregate recorded parameter L2 norms for every visible layer."""
    values = []
    norms = history.get("parameter_norms", {})
    for layer in layers:
        prefix = layer["name"] + "."
        squares = []
        for name, recorded in norms.items():
            if name.startswith(prefix) and frame_index < len(recorded):
                value = float(recorded[frame_index])
                if np.isfinite(value):
                    squares.append(value * value)
        values.append(float(np.sqrt(np.sum(squares))) if squares else float("nan"))
    return np.asarray(values, dtype=float)


def _backprop_data(
    layers: list[Dict[str, Any]],
    gradient_norms: np.ndarray,
    update_norms: np.ndarray,
    parameter_norms: np.ndarray,
    gradient_maximum: float,
    update_maximum: float,
    step: int,
    loss: float,
    loss_change: float,
) -> list[go.Scatter]:
    x = np.linspace(0.10, 0.90, len(layers) + 2)
    data: list[go.Scatter] = []
    for index in range(len(x) - 1):
        data.append(
            go.Scatter(
                x=[x[index], x[index + 1]],
                y=[0.70, 0.70],
                mode="lines",
                line={"color": NEURAL_COLORS["activation"], "width": 4},
                hoverinfo="skip",
                showlegend=False,
                name="forward signal",
            )
        )
    for index, norm in enumerate(gradient_norms):
        magnitude = 0.0 if not np.isfinite(norm) else float(norm) / max(gradient_maximum, 1e-12)
        data.append(
            go.Scatter(
                x=[x[index + 2], x[index + 1]],
                y=[0.38, 0.38],
                mode="lines",
                line={"color": NEURAL_COLORS["output"], "width": 2.0 + 10.0 * magnitude},
                opacity=0.35 + 0.65 * magnitude,
                hovertemplate=(
                    f"{layers[index]['name']}<br>recorded gradient norm="
                    + (f"{norm:.6g}" if np.isfinite(norm) else "not recorded")
                    + "<extra></extra>"
                ),
                showlegend=False,
                name="recorded backward gradient",
            )
        )
    for index, norm in enumerate(update_norms):
        magnitude = 0.0 if not np.isfinite(norm) else float(norm) / max(update_maximum, 1e-12)
        data.append(
            go.Scatter(
                x=[x[index + 1], x[index + 2]],
                y=[0.22, 0.22],
                mode="lines",
                line={"color": NEURAL_COLORS["update_positive"], "width": 2.0 + 10.0 * magnitude},
                opacity=0.30 + 0.70 * magnitude,
                hovertemplate=(
                    f"{layers[index]['name']}<br>adjacent parameter-update norm="
                    + (f"{norm:.6g}" if np.isfinite(norm) else "not recorded")
                    + "<extra></extra>"
                ),
                showlegend=False,
                name="recorded optimizer update",
            )
        )
    labels = ["input", *[layer["type"] for layer in layers], "loss"]
    roles = ["x", *[f"a({i + 1})" for i in range(len(layers))], "L"]
    data.append(
        go.Scatter(
            x=x,
            y=[0.57] * len(x),
            mode="markers+text",
            text=[f"<b>{label}</b><br>{role}" for label, role in zip(labels, roles)],
            textposition="top center",
            marker={
                "size": [28, *([34] * len(layers)), 30],
                "color": [NEURAL_COLORS["input"], *([NEURAL_COLORS["linear"]] * len(layers)), NEURAL_COLORS["output"]],
                "line": {"color": NEURAL_COLORS["text"], "width": 1},
            },
            hoverinfo="skip",
            showlegend=False,
            name="forward and backward layer graph",
        )
    )
    layer_readouts = []
    for layer, gradient, update, parameter in zip(
        layers, gradient_norms, update_norms, parameter_norms
    ):
        gradient_text = f"{gradient:.3e}" if np.isfinite(gradient) else "n/a"
        update_text = f"{update:.3e}" if np.isfinite(update) else "n/a"
        relative = (
            update / max(parameter, 1e-12)
            if np.isfinite(update) and np.isfinite(parameter)
            else float("nan")
        )
        relative_text = f"{relative:.3e}" if np.isfinite(relative) else "n/a"
        layer_readouts.append(
            f"<b>{layer['name']}</b><br>"
            f"gradient L2 = {gradient_text}<br>"
            f"update L2 = {update_text}<br>"
            f"relative update = {relative_text}"
        )
    crowded_readout = len(layers) > 3
    readout_y = [
        0.145 if index % 2 == 0 else 0.055
        for index in range(len(layers))
    ] if crowded_readout else [0.08] * len(layers)
    data.append(
        go.Scatter(
            x=x[1:-1],
            y=readout_y,
            mode="text",
            text=layer_readouts,
            textposition="middle center",
            textfont={"size": 12, "color": NEURAL_COLORS["text"]},
            hoverinfo="skip",
            showlegend=False,
            name="layerwise gradient and update values",
        )
    )
    change_text = f"{loss_change:+.3e}" if np.isfinite(loss_change) else "n/a at first checkpoint"
    data.append(
        go.Scatter(
            x=[0.5],
            y=[0.47],
            mode="text",
            text=[f"<b>Step {step}</b> · L={loss:.6g} · ΔL={change_text}"],
            textfont={"size": 15, "color": NEURAL_COLORS["text"]},
            hoverinfo="skip",
            showlegend=False,
            name="backpropagation step summary",
        )
    )
    return data


def build_nn_backpropagation_figure(
    model: Any,
    history: Dict[str, Any],
    *,
    input_sample: Any | None = None,
    max_layers: int = 8,
    max_frames: int | None = 20,
    frame_duration: int = 900,
    title: str | None = None,
) -> go.Figure:
    """Animate exact recorded gradient norms over the chain-rule equations."""
    all_layers = _trainable_layers(model, input_sample)
    selected = select_with_ellipsis(all_layers, max_layers)
    omitted_count = len(all_layers) - sum(layer is not None for layer in selected)
    layers = [layer for layer in selected if layer is not None]
    steps = np.asarray(history.get("steps", []), dtype=int)
    if not steps.size:
        raise ValueError("History has no recorded steps.")
    frame_indices = _frame_indices(steps.size, max_frames)
    all_gradient_norms = [
        _layer_gradient_norms(history, layers, int(index)) for index in frame_indices
    ]
    all_update_norms = [
        _layer_update_norms(history, layers, int(index)) for index in frame_indices
    ]
    all_parameter_norms = [
        _layer_parameter_norms(history, layers, int(index)) for index in frame_indices
    ]
    finite_gradients = (
        np.concatenate([values[np.isfinite(values)] for values in all_gradient_norms])
        if all_gradient_norms
        else np.asarray([])
    )
    finite_updates = (
        np.concatenate([values[np.isfinite(values)] for values in all_update_norms])
        if all_update_norms
        else np.asarray([])
    )
    gradient_maximum = max(
        float(np.max(finite_gradients)) if finite_gradients.size else 0.0,
        1e-12,
    )
    update_maximum = max(
        float(np.max(finite_updates)) if finite_updates.size else 0.0,
        1e-12,
    )
    loss_values = np.asarray(history.get("loss", np.full(steps.size, np.nan)), dtype=float)
    first_index = int(frame_indices[0])
    first_data = _backprop_data(
        layers,
        all_gradient_norms[0],
        all_update_norms[0],
        all_parameter_norms[0],
        gradient_maximum,
        update_maximum,
        int(steps[first_index]),
        float(loss_values[first_index]),
        float("nan")
        if first_index == 0
        else float(loss_values[first_index] - loss_values[first_index - 1]),
    )
    figure = go.Figure(first_data)
    dynamic = list(range(len(first_data)))
    figure.frames = [
        go.Frame(
            name=f"step_{index}",
            data=_backprop_data(
                layers,
                gradient_norms,
                update_norms,
                parameter_norms,
                gradient_maximum,
                update_maximum,
                int(steps[index]),
                float(loss_values[index]),
                float("nan")
                if index == 0
                else float(loss_values[index] - loss_values[index - 1]),
            ),
            traces=dynamic,
        )
        for index, gradient_norms, update_norms, parameter_norms in zip(
            frame_indices, all_gradient_norms, all_update_norms, all_parameter_norms
        )
    ]
    if title is None:
        title = "Backpropagation: chain rule and recorded gradients"
    loss_name = str(history.get("training_config", {}).get("loss", "selected loss"))
    optimizer_name = str(history.get("training_config", {}).get("optimizer", "optimizer"))
    scope_line = (
        f"{omitted_count} intermediate trainable layer"
        f"{'s' if omitted_count != 1 else ''} omitted from the display; complete recorded history retained.<br>"
        if omitted_count
        else ""
    )
    annotations = [
        {
            "x": 0.5,
            "y": 0.92,
            "xref": "paper",
            "yref": "paper",
            "text": (
                r"$\text{Forward: }\mathbf{z}^{(\ell)}=\Theta^{(\ell)}\mathbf{a}^{(\ell-1)}+"
                r"\mathbf{b}^{(\ell)},\quad\mathbf{a}^{(\ell)}=\phi_\ell(\mathbf{z}^{(\ell)})$"
            ),
            "showarrow": False,
            "font": {"size": 17, "color": NEURAL_COLORS["text"]},
        },
        {
            "x": 0.5,
            "y": 0.76,
            "xref": "paper",
            "yref": "paper",
            "text": "forward signal →",
            "showarrow": False,
            "font": {"size": 12, "color": NEURAL_COLORS["activation"]},
        },
        {
            "x": 0.5,
            "y": 0.42,
            "xref": "paper",
            "yref": "paper",
            "text": "← reverse-mode gradient",
            "showarrow": False,
            "font": {"size": 12, "color": NEURAL_COLORS["output"]},
        },
        {
            "x": 0.5,
            "y": 0.82,
            "xref": "paper",
            "yref": "paper",
            "text": (
                r"$\text{Backward: }\boldsymbol{\delta}^{(\ell)}="
                r"\left((\Theta^{(\ell+1)})^\top\boldsymbol{\delta}^{(\ell+1)}\right)\odot"
                r"\phi_\ell'(\mathbf{z}^{(\ell)}),\quad"
                r"\nabla_{\Theta^{(\ell)}}\mathcal{L}=\boldsymbol{\delta}^{(\ell)}"
                r"(\mathbf{a}^{(\ell-1)})^\top$"
            ),
            "showarrow": False,
            "font": {"size": 16, "color": NEURAL_COLORS["text"]},
        },
        {
            "x": 0.5,
            "y": -0.11,
            "xref": "paper",
            "yref": "paper",
            "text": (
                scope_line
                +
                f"Global width scale · objective: {loss_name} · optimizer: {optimizer_name} · "
                "optimizer converts gradients into updates · plain SGD: Δθ = −η∇θL"
            ),
            "showarrow": False,
            "font": {"size": 12, "color": NEURAL_COLORS["muted"]},
        },
        {
            "x": 0.5,
            "y": 0.26,
            "xref": "paper",
            "yref": "paper",
            "text": "optimizer update →",
            "showarrow": False,
            "font": {"size": 12, "color": NEURAL_COLORS["update_positive"]},
        },
    ]
    layout = neural_layout(title, height=760)
    layout.update(
        {
            "margin": {"t": 125, "r": 50, "b": 175, "l": 50},
            "annotations": annotations,
            "updatemenus": [
                {
                    "type": "buttons",
                    "direction": "left",
                    "x": 0.0,
                    "y": 1.04,
                    **animation_button_style(),
                    "buttons": [
                        {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": frame_duration, "redraw": False}, "fromcurrent": True}]},
                        {"label": "Pause", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]},
                    ],
                }
            ],
            "sliders": [
                {
                    "active": 0,
                    "currentvalue": {"prefix": "Recorded step: "},
                    "pad": {"t": 35},
                    "steps": [
                        {"label": str(int(steps[index])), "method": "animate", "args": [[f"step_{index}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]}
                        for index in frame_indices
                    ],
                }
            ],
            "meta": {
                "mlektic_neural_backpropagation": {
                    "schema_version": 1,
                    "gradient_quantity": "recorded per-layer parameter-gradient L2 norm",
                    "gradient_scale": "global across displayed frames",
                    "update_quantity": "adjacent recorded per-layer parameter-change L2 norm",
                    "update_scale": "global across displayed frames",
                    "numeric_layer_readout": [
                        "gradient L2 norm",
                        "adjacent parameter-update L2 norm",
                        "relative parameter-update norm",
                        "loss and adjacent loss change",
                    ],
                    "delta_values_recorded": False,
                    "chain_rule_equations": "canonical dense-layer identities",
                    "optimizer": optimizer_name,
                    "loss": loss_name,
                    "displayed_layer_count": len(layers),
                    "omitted_layer_count": omitted_count,
                    "crowded_readout_layout": "alternating-rows" if len(layers) > 3 else "single-row",
                    "scope_disclosure_row": "lower-caption",
                }
            },
        }
    )
    figure.update_layout(**layout)
    figure.update_xaxes(visible=False, range=[0, 1])
    figure.update_yaxes(visible=False, range=[0, 1])
    return figure


__all__ = ["build_nn_backpropagation_figure"]
