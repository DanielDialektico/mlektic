"""Layer-by-layer mathematical explanation for PyTorch forward passes."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...neural.introspection import _leaf_modules, describe_torch_model, run_torch_forward
from ...neural.taxonomy import composed_dense_function, dense_stages, select_with_ellipsis, shape_tex
from ._style import NEURAL_COLORS, layer_color, neural_layout
from .math_format import parameter_snapshot, vector_latex
from .training import _animation_controls


def _frame_indices(frame_count: int, max_frames: int | None) -> np.ndarray:
    if max_frames is None or frame_count <= max_frames:
        return np.arange(frame_count, dtype=int)
    return np.unique(np.linspace(0, frame_count - 1, max_frames, dtype=int))


def _selected_layers(layers: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any] | None]:
    return select_with_ellipsis(layers, limit)


def _linear_substitution(
    module: Any,
    record: Dict[str, np.ndarray],
    layer_number: int,
    dec: int,
    max_neurons: int,
    weight_values: np.ndarray | None = None,
    bias_values: np.ndarray | None = None,
) -> str:
    output = np.asarray(record["output"][0], dtype=float).ravel()
    input_values = np.asarray(record["input"][0], dtype=float).ravel()
    weight = (
        np.asarray(weight_values, dtype=float)
        if weight_values is not None
        else module.weight.detach().cpu().numpy()
    )
    bias = (
        np.asarray(bias_values, dtype=float)
        if bias_values is not None
        else module.bias.detach().cpu().numpy()
        if module.bias is not None
        else np.zeros(weight.shape[0])
    )
    if weight.shape[0] <= min(max_neurons, 4) and input_values.size <= max_neurons:
        rows = []
        for neuron_index in range(weight.shape[0]):
            terms = "+".join(
                f"({weight[neuron_index, feature_index]:.{dec}f})({input_values[feature_index]:.{dec}f})"
                for feature_index in range(weight.shape[1])
            )
            rows.append(
                rf"z^{{({layer_number})}}_{{{neuron_index + 1}}}={terms}"
                rf"+({bias[neuron_index]:.{dec}f})={output[neuron_index]:.{dec}f}"
            )
        return r"\begin{aligned}" + r"\\".join(rows) + r"\end{aligned}"
    return (
        rf"\mathbf{{z}}^{{({layer_number})}}=W^{{({layer_number})}}\mathbf{{a}}^{{({layer_number - 1})}}"
        rf"+\mathbf{{b}}^{{({layer_number})}}={vector_latex(output, dec=dec, limit=max_neurons)}"
    )


def _formula_blocks(
    model: Any,
    layers: Sequence[Dict[str, Any] | None],
    records: Dict[str, Dict[str, np.ndarray]],
    snapshot: Dict[str, np.ndarray],
    dec: int,
    max_neurons: int,
) -> List[str]:
    modules = dict(_leaf_modules(model))
    blocks: List[str] = []
    for layer in layers:
        if layer is None:
            blocks.append(r"\vdots\quad\text{intermediate layers summarized}\quad\vdots")
            continue
        name = layer["name"]
        module = modules[name]
        record = records.get(name)
        if record is None:
            continue
        if layer["type"] == "Linear":
            body = _linear_substitution(
                module,
                record,
                layer["math_index"],
                dec,
                max_neurons,
                snapshot.get(f"{name}.weight"),
                snapshot.get(f"{name}.bias"),
            )
        else:
            output = np.asarray(record["output"][0], dtype=float).ravel()
            body = rf"{layer['formula']}={vector_latex(output, dec=dec, limit=max_neurons)}"
        input_dimension = shape_tex(layer.get("input_shape"))
        output_dimension = shape_tex(layer.get("output_shape"))
        blocks.append(
            rf"\underbrace{{\text{{{name}: {layer['type']}}}}}_{{\mathbb{{R}}^{{{input_dimension}}}\to\mathbb{{R}}^{{{output_dimension}}}}}"
            rf"\quad {body}"
        )
    return blocks


def _explanation_annotations(
    model: Any,
    selected_layers: Sequence[Dict[str, Any] | None],
    records: Dict[str, Dict[str, np.ndarray]],
    output: Any,
    snapshot: Dict[str, np.ndarray],
    step: int | None,
    dec: int,
    max_neurons: int,
) -> List[Dict[str, Any]]:
    modules = dict(_leaf_modules(model))
    blocks = _formula_blocks(model, selected_layers, records, snapshot, dec, max_neurons)
    annotations: List[Dict[str, Any]] = [
        {
            "x": 0.5,
            "y": 1.10,
            "xref": "paper",
            "yref": "paper",
            "text": f"${composed_dense_function(dense_stages(model))}$",
            "showarrow": False,
            "font": {"size": 17, "color": NEURAL_COLORS["text"]},
        }
    ]
    if step is not None:
        annotations.append(
            {
                "x": 0.99,
                "y": 1.03,
                "xref": "paper",
                "yref": "paper",
                "text": rf"$t={step}$",
                "showarrow": False,
                "xanchor": "right",
                "font": {"size": 13, "color": NEURAL_COLORS["muted"]},
            }
        )
    visible_count = len(selected_layers)
    layer_y_positions = np.linspace(0.88, 0.18, max(visible_count, 1))
    for index, (layer, y_position) in enumerate(zip(selected_layers, layer_y_positions)):
        if layer is None:
            label = r"$\vdots$"
            color = NEURAL_COLORS["muted"]
        else:
            module = modules[layer["name"]]
            label = f"<b>{layer['name']}</b><br>{module.__class__.__name__}"
            color = layer_color(layer["type"], index == visible_count - 1)
        annotations.append(
            {
                "x": 0.50,
                "y": y_position,
                "xref": "x",
                "yref": "y",
                "text": label,
                "showarrow": False,
                "align": "center",
                "font": {"size": 13, "color": color},
                "bgcolor": NEURAL_COLORS["panel"] if layer is not None else NEURAL_COLORS["background"],
                "bordercolor": color,
                "borderwidth": 1 if layer is not None else 0,
                "borderpad": 7,
            }
        )
        if index < visible_count - 1:
            annotations.append(
                {
                    "x": 0.50,
                    "y": y_position - 0.35 / max(visible_count - 1, 1),
                    "xref": "x",
                    "yref": "y",
                    "text": "&#8595;",
                    "showarrow": False,
                    "font": {"size": 13, "color": NEURAL_COLORS["muted"]},
                }
            )
    block_y_positions = np.linspace(0.90, 0.18, max(len(blocks), 1))
    for block, y_position in zip(blocks, block_y_positions):
        annotations.append(
            {
                "x": 0.02,
                "y": y_position,
                "xref": "x2",
                "yref": "y2",
                "text": f"${block}$",
                "showarrow": False,
                "xanchor": "left",
                "yanchor": "top",
                "align": "left",
                "font": {"size": 12, "color": NEURAL_COLORS["text"]},
            }
        )
    output_values = output.detach().cpu().numpy()[0]
    annotations.append(
        {
            "x": 0.02,
            "y": 0.05,
            "xref": "x2",
            "yref": "y2",
            "text": rf"$\hat{{\mathbf{{y}}}}_t={vector_latex(output_values, dec=dec, limit=max_neurons)}$",
            "showarrow": False,
            "xanchor": "left",
            "font": {"size": 17, "color": NEURAL_COLORS["output"]},
        }
    )
    return annotations


def _history_is_complete(model: Any, history: Dict[str, Any]) -> bool:
    parameters = history.get("parameters", {})
    return all(name in parameters and len(parameters[name]) for name, _ in model.named_parameters())


def build_nn_prediction_figure(
    model: Any,
    x_query: Any,
    *,
    history: Dict[str, Any] | None = None,
    title: str | None = None,
    dec: int = 4,
    max_layers_math: int = 6,
    max_neurons_math: int = 8,
    max_frames: int | None = 12,
    frame_duration: int = 220,
) -> go.Figure:
    """Explain and optionally animate a forward pass with numerical substitutions."""
    if max_layers_math < 3 or max_neurons_math < 1:
        raise ValueError("max_layers_math must be at least 3 and max_neurons_math must be positive.")
    layers = describe_torch_model(model, x_query)
    selected_layers = _selected_layers(layers, max_layers_math)
    frame_specs: List[tuple[int | None, Dict[str, np.ndarray]]] = []
    steps: List[int] = []
    if history is not None and _history_is_complete(model, history):
        history_steps = np.asarray(history.get("steps", []), dtype=int)
        indices = _frame_indices(history_steps.size, max_frames)
        for frame_index in indices:
            frame_specs.append((int(frame_index), parameter_snapshot(history, int(frame_index))))
            steps.append(int(history_steps[frame_index]))
    else:
        frame_specs.append((None, {}))
    outputs_and_records = [run_torch_forward(model, x_query, snapshot or None) for _, snapshot in frame_specs]
    figure = make_subplots(rows=1, cols=2, column_widths=[0.25, 0.75], horizontal_spacing=0.04)
    figure.add_trace(go.Scatter(x=[], y=[], showlegend=False, hoverinfo="skip"), row=1, col=1)
    figure.add_trace(go.Scatter(x=[], y=[], showlegend=False, hoverinfo="skip"), row=1, col=2)
    for column in (1, 2):
        figure.update_xaxes(visible=False, range=[0, 1], row=1, col=column)
        figure.update_yaxes(visible=False, range=[0, 1], row=1, col=column)
    frame_names = [str(index) for index in range(len(frame_specs))]
    annotations_by_frame = []
    for frame_position, ((history_index, snapshot), (output, records)) in enumerate(
        zip(frame_specs, outputs_and_records)
    ):
        step = steps[frame_position] if history_index is not None else None
        annotations_by_frame.append(
            _explanation_annotations(
                model,
                selected_layers,
                records,
                output,
                snapshot,
                step,
                dec,
                max_neurons_math,
            )
        )
    if len(frame_specs) > 1:
        figure.frames = [
            go.Frame(
                name=name,
                data=[go.Scatter(x=[], y=[]), go.Scatter(x=[], y=[])],
                traces=[0, 1],
                layout=go.Layout(annotations=annotations),
            )
            for name, annotations in zip(frame_names, annotations_by_frame)
        ]
        controls, sliders = _animation_controls(
            np.asarray(steps, dtype=int),
            frame_duration,
            frame_names=frame_names,
        )
    else:
        controls, sliders = [], []
    if title is None:
        title = "Forward-pass mathematics"
    layout = neural_layout(title, height=max(650, 220 + 115 * len(selected_layers)))
    layout["margin"] = {"t": 115, "r": 35, "b": 100 if sliders else 45, "l": 35}
    figure.update_layout(
        **layout,
        annotations=annotations_by_frame[0],
        updatemenus=controls,
        sliders=sliders,
        showlegend=False,
    )
    return figure


__all__ = ["build_nn_prediction_figure"]
