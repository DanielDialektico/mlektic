"""Layer-by-layer mathematical explanation for PyTorch forward passes."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...neural.introspection import _leaf_modules, describe_torch_model, run_torch_forward
from ...neural.taxonomy import composed_dense_function, dense_stages, select_with_ellipsis, shape_tex
from ._style import NEURAL_COLORS, animation_button_style, layer_color, neural_layout
from .math_format import buffer_snapshot, display_indices, parameter_snapshot, vector_latex
from .training import _animation_controls

# Fitted prediction uses fixed paper regions.  MathJax annotations contain only
# text; card geometry is owned by shapes and therefore cannot resize after a
# stage button triggers a second typesetting pass in the notebook.
_PREDICTION_CARD_BOUNDS = {
    "Input": (0.020, 0.240, 0.690, 0.805),
    "Substitution": (0.265, 0.725, 0.610, 0.825),
    "Output": (0.750, 0.990, 0.675, 0.815),
}


def _frame_indices(frame_count: int, max_frames: int | None) -> np.ndarray:
    if max_frames is None or frame_count <= max_frames:
        return np.arange(frame_count, dtype=int)
    return np.unique(np.linspace(0, frame_count - 1, max_frames, dtype=int))


def _selected_layers(layers: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any] | None]:
    return select_with_ellipsis(layers, limit)


def _safe_vector_limit(dec: int, requested_limit: int) -> int:
    """Bound inline vector width for the narrowest supported derivation column.

    Coordinate count alone is not a safe width contract: every extra decimal
    increases the MathJax box.  This conservative character budget keeps
    vectors inside the derivation column while ellipsis preserves disclosure
    that additional coordinates exist.
    """
    estimated_coordinate_width = max(dec + 5, 1)
    width_limited_count = max(2, 40 // estimated_coordinate_width)
    return max(1, min(requested_limit, width_limited_count))


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
        np.asarray(weight_values, dtype=float) if weight_values is not None else module.weight.detach().cpu().numpy()
    )
    bias = (
        np.asarray(bias_values, dtype=float)
        if bias_values is not None
        else module.bias.detach().cpu().numpy()
        if module.bias is not None
        else np.zeros(weight.shape[0])
    )
    safe_vector_limit = _safe_vector_limit(dec, max_neurons)
    rows_per_neuron = 3 if input_values.size >= 4 else 1
    exact_row_count = int(weight.shape[0]) * rows_per_neuron
    if (
        weight.shape[0] <= min(safe_vector_limit, 4)
        and input_values.size <= safe_vector_limit
        and exact_row_count <= 4
    ):
        rows = []
        for neuron_index in range(weight.shape[0]):
            term_values = [
                f"({weight[neuron_index, feature_index]:.{dec}f})({input_values[feature_index]:.{dec}f})"
                for feature_index in range(weight.shape[1])
            ]
            if len(term_values) >= 4:
                chunks = [term_values[index : index + 2] for index in range(0, len(term_values), 2)]
                rows.append(rf"z^{{({layer_number})}}_{{{neuron_index + 1}}}&=" + "+".join(chunks[0]))
                for chunk_index, chunk in enumerate(chunks[1:], start=1):
                    suffix = (
                        rf"+({bias[neuron_index]:.{dec}f})={output[neuron_index]:.{dec}f}"
                        if chunk_index == len(chunks) - 1
                        else ""
                    )
                    rows.append(r"&\quad+" + "+".join(chunk) + suffix)
            else:
                rows.append(
                    rf"z^{{({layer_number})}}_{{{neuron_index + 1}}}&="
                    + "+".join(term_values)
                    + rf"+({bias[neuron_index]:.{dec}f})={output[neuron_index]:.{dec}f}"
                )
        return r"\begin{aligned}" + r"\\".join(rows) + r"\end{aligned}"
    return (
        r"\begin{aligned}"
        rf"\mathbf{{z}}^{{({layer_number})}}&="
        rf"\Theta^{{({layer_number})}}\mathbf{{a}}^{{({layer_number - 1})}}"
        rf"+\boldsymbol{{\theta}}_0^{{({layer_number})}}"
        rf"\\&={vector_latex(output, dec=dec, limit=safe_vector_limit)}"
        r"\end{aligned}"
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
    safe_vector_limit = _safe_vector_limit(dec, max_neurons)
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
            aligned_formula = str(layer["formula"]).replace("=", "&=", 1)
            body = (
                r"\begin{aligned}"
                rf"{aligned_formula}="
                rf"{vector_latex(output, dec=dec, limit=safe_vector_limit)}"
                r"\end{aligned}"
            )
        input_dimension = shape_tex(layer.get("input_shape"))
        output_dimension = shape_tex(layer.get("output_shape"))
        blocks.append(
            rf"\underbrace{{\text{{{name}: {layer['type']}}}}}_{{\mathbb{{R}}^{{{input_dimension}}}\to\mathbb{{R}}^{{{output_dimension}}}}}"
            rf"\quad {body}"
        )
    return blocks


def _block_vertical_layout(
    blocks: Sequence[str],
    *,
    prediction_stages: bool,
) -> tuple[List[float], List[int], float, float, int]:
    """Allocate layer formulas from their rendered row count.

    Plotly positions annotations by one anchor coordinate even when MathJax
    produces several rows.  Uniform anchor spacing therefore lets a four-row
    Linear expansion reach the following activation.  This allocator reserves
    a baseline pitch per rendered row and a separate inter-layer corridor.
    Only genuinely dense selections use the 13-point fallback.
    """
    if not blocks:
        return [], [], 0.0, 0.0, 14
    line_counts = [1 + block.count(r"\\") for block in blocks]
    line_steps = sum(max(line_count - 1, 0) for line_count in line_counts)
    gap_count = max(len(blocks) - 1, 0)
    block_top = 0.56 if prediction_stages else 0.84
    block_bottom = 0.06 if prediction_stages else 0.18
    available_height = block_top - block_bottom
    line_pitch = 0.034
    # This corridor is a structural invariant: formulas may become compact,
    # but one layer is never allowed to invade the next layer's section.
    layer_gap = 0.10 if prediction_stages else 0.14
    required_height = line_steps * line_pitch + gap_count * layer_gap
    if required_height > available_height and required_height > 0.0:
        scale = available_height / required_height
        line_pitch = max(0.024, line_pitch * scale)
        layer_gap = max(0.036, layer_gap * scale)
        required_height = line_steps * line_pitch + gap_count * layer_gap
        if required_height > available_height:
            final_scale = available_height / required_height
            line_pitch *= final_scale
            layer_gap *= final_scale
    positions: List[float] = []
    cursor = block_top
    for block_index, line_count in enumerate(line_counts):
        positions.append(cursor)
        cursor -= max(line_count - 1, 0) * line_pitch
        if block_index < len(blocks) - 1:
            cursor -= layer_gap
    font_size = 14 if line_pitch >= 0.030 else 13
    return positions, line_counts, line_pitch, layer_gap, font_size


def _numeric_substitution_preview(
    model: Any,
    records: Dict[str, Dict[str, np.ndarray]],
    snapshot: Dict[str, np.ndarray],
    dec: int,
    max_terms: int = 4,
) -> str:
    """Return one exact fitted-parameter substitution for the first dense unit."""
    stages = dense_stages(model)
    if not stages:
        return r"\text{No dense substitution is available for this model.}"
    dense_stage = stages[0]
    record = records.get(dense_stage["name"])
    if record is None:
        return r"\mathbf{z}^{(1)}=\Theta^{(1)}\mathbf{x}+\boldsymbol{\theta}^{(1)}_0"
    module = dict(_leaf_modules(model))[dense_stage["name"]]
    input_values = np.asarray(record["input"][0], dtype=float).ravel()
    output_values = np.asarray(record["output"][0], dtype=float).ravel()
    weight = np.asarray(
        snapshot.get(dense_stage["weight_name"], module.weight.detach().cpu().numpy()),
        dtype=float,
    )
    bias = np.asarray(
        snapshot.get(
            dense_stage.get("bias_name", ""),
            module.bias.detach().cpu().numpy() if module.bias is not None else np.zeros(weight.shape[0]),
        ),
        dtype=float,
    ).ravel()
    indices = display_indices(input_values.size, max_terms)
    terms = [
        f"({weight[0, feature_index]:.{dec}f})({input_values[feature_index]:.{dec}f})"
        for feature_index in indices
    ]
    rows: List[str] = [r"z^{(1)}_1&=" + terms[0]]
    if input_values.size > max_terms:
        rows.append(r"&\quad+\cdots+" + terms[-1])
    else:
        rows.extend(r"&\quad+" + term for term in terms[1:])
    rows.append(rf"&\quad+({bias[0]:.{dec}f})={output_values[0]:.{dec}f}")
    return r"\\".join(rows)


def _prediction_card_shapes(stage: int) -> List[Dict[str, Any]]:
    """Return cumulative, fixed-size summary regions for one lesson stage."""
    colors = {
        "Input": NEURAL_COLORS["input"],
        "Substitution": NEURAL_COLORS["linear"],
        "Output": NEURAL_COLORS["output"],
    }
    shapes: List[Dict[str, Any]] = []
    for card_stage, heading in enumerate(("Input", "Substitution", "Output"), start=1):
        if stage < card_stage:
            continue
        x0, x1, y0, y1 = _PREDICTION_CARD_BOUNDS[heading]
        shapes.append(
            {
                "type": "rect",
                "xref": "paper",
                "yref": "paper",
                "x0": x0,
                "x1": x1,
                "y0": y0,
                "y1": y1,
                "line": {"color": colors[heading], "width": 1},
                "fillcolor": NEURAL_COLORS["panel"],
                "layer": "below",
            }
        )
    return shapes


def _explanation_annotations(
    model: Any,
    x_query: Any,
    selected_layers: Sequence[Dict[str, Any] | None],
    records: Dict[str, Dict[str, np.ndarray]],
    output: Any,
    snapshot: Dict[str, np.ndarray],
    dec: int,
    max_neurons: int,
    loss_name: str | None,
    stage: int = 3,
    prediction_stages: bool = True,
) -> List[Dict[str, Any]]:
    modules = dict(_leaf_modules(model))
    blocks = _formula_blocks(model, selected_layers, records, snapshot, dec, max_neurons)
    query_values = np.asarray(x_query.detach().cpu().numpy() if hasattr(x_query, "detach") else x_query)
    if query_values.ndim > 1:
        query_values = query_values[0]
    output_values = output.detach().cpu().numpy()[0]
    # The summary row has three independent cards.  Wide input/output tensors
    # use a stricter two-value overview so the cards remain disjoint at the
    # supported 1024 px notebook width.  The detailed forward pass below still
    # honors max_neurons and therefore retains the requested mathematical depth.
    # Summary cards are a bounded overview, independent of detailed forward
    # density. Two representative coordinates remain readable in every public
    # size/theme; the derivation below retains max_neurons exact values.
    summary_dec = min(dec, 3)
    input_tex = vector_latex(query_values, dec=summary_dec, limit=min(query_values.size, 2))
    output_tex = vector_latex(output_values, dec=summary_dec, limit=min(output_values.size, 2))
    decision_tex = ""
    normalized_loss = str(loss_name or "").lower()
    if output_values.size == 1 and "bcewithlogits" in normalized_loss:
        winning_class = int(float(output_values.ravel()[0]) >= 0.0)
        decision_tex = rf"\\\hat{{k}}&=\mathbb{{1}}[\hat{{y}}\geq 0]={winning_class}"
    elif output_values.size == 1 and "bce" in normalized_loss:
        winning_class = int(float(output_values.ravel()[0]) >= 0.5)
        decision_tex = rf"\\\hat{{k}}&=\mathbb{{1}}[\hat{{y}}\geq 0.5]={winning_class}"
    elif output_values.size > 1 and "crossentropy" in normalized_loss:
        winning_class = int(np.argmax(output_values))
        decision_tex = rf"\\\hat{{k}}&=\arg\max_j\hat{{y}}_j={winning_class}"
    output_body = (
        rf"$\begin{{aligned}}\hat{{\mathbf{{y}}}}&={output_tex}"
        rf"{decision_tex}\end{{aligned}}$"
    )
    stages = dense_stages(model)
    if len(stages) > 4:
        depth = len(stages)
        model_formula = (
            rf"\hat{{\mathbf{{y}}}}=\mathbf{{a}}^{{({depth})}},\quad "
            rf"\mathbf{{a}}^{{(\ell)}}=\phi_\ell(\Theta^{{(\ell)}}\mathbf{{a}}^{{(\ell-1)}}+"
            rf"\boldsymbol{{\theta}}_0^{{(\ell)}}),\;\ell=1,\ldots,{depth}"
        )
    else:
        model_formula = composed_dense_function(stages)
    annotations: List[Dict[str, Any]] = [
        {
            "x": 0.5,
            "y": 0.955,
            "xref": "paper",
            "yref": "paper",
            "text": f"${model_formula}$",
            "showarrow": False,
            "font": {"size": 18, "color": NEURAL_COLORS["text"]},
        }
    ]
    if prediction_stages:
        substitution_tex = _numeric_substitution_preview(
            model,
            records,
            snapshot,
            dec,
            max_terms=min(max_neurons, 2),
        )
        input_bounds = _PREDICTION_CARD_BOUNDS["Input"]
        substitution_bounds = _PREDICTION_CARD_BOUNDS["Substitution"]
        output_bounds = _PREDICTION_CARD_BOUNDS["Output"]
        summary_cards = [
            (
                (input_bounds[0] + input_bounds[1]) / 2,
                input_bounds[3] + 0.015,
                "Input",
                rf"$\mathbf{{x}}={input_tex}$",
            ),
            (
                (substitution_bounds[0] + substitution_bounds[1]) / 2,
                substitution_bounds[3] + 0.015,
                "Substitution",
                rf"$\begin{{aligned}}{substitution_tex}\end{{aligned}}$",
            ),
            (
                (output_bounds[0] + output_bounds[1]) / 2,
                output_bounds[3] + 0.015,
                "Output",
                output_body,
            ),
        ]
        for card_stage, (
            x_position,
            heading_y,
            heading,
            body,
        ) in enumerate(
            summary_cards,
            start=1,
        ):
            if stage < card_stage:
                continue
            annotations.append(
                {
                    "x": x_position,
                    "y": heading_y,
                    "xref": "paper",
                    "yref": "paper",
                    "text": f"<b>{heading}</b>",
                    "showarrow": False,
                    "xanchor": "center",
                    "yanchor": "bottom",
                    "align": "center",
                    "font": {"size": 14, "color": NEURAL_COLORS["text"]},
                }
            )
            annotations.append(
                {
                    "x": x_position,
                    "y": ((_PREDICTION_CARD_BOUNDS[heading][2] + _PREDICTION_CARD_BOUNDS[heading][3]) / 2),
                    "xref": "paper",
                    "yref": "paper",
                    "text": body,
                    "showarrow": False,
                    "xanchor": "center",
                    "yanchor": "middle",
                    "align": "center",
                    "font": {"size": 14, "color": NEURAL_COLORS["text"]},
                }
            )
    (
        block_y_positions,
        block_line_counts,
        block_line_pitch,
        _block_layer_gap,
        block_font_size,
    ) = _block_vertical_layout(blocks, prediction_stages=prediction_stages)

    visible_count = len(selected_layers)
    if len(block_y_positions) == visible_count:
        layer_y_positions = [
            y_position - max(line_count - 1, 0) * block_line_pitch / 2.0
            for y_position, line_count in zip(block_y_positions, block_line_counts)
        ]
    else:
        block_top = 0.56 if prediction_stages else 0.84
        block_bottom = 0.06 if prediction_stages else 0.18
        layer_y_positions = np.linspace(block_top, block_bottom, max(visible_count, 1))
    for index, (layer, y_position) in enumerate(zip(selected_layers, layer_y_positions)):
        if stage < 1:
            continue
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
            arrow_y = (y_position + layer_y_positions[index + 1]) / 2.0
            annotations.append(
                {
                    "x": 0.50,
                    "y": arrow_y,
                    "xref": "x",
                    "yref": "y",
                    "text": "&#8595;",
                    "showarrow": False,
                    "font": {"size": 13, "color": NEURAL_COLORS["muted"]},
                }
            )
    for block, y_position in zip(blocks, block_y_positions):
        if stage < 2:
            continue
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
                "font": {"size": block_font_size, "color": NEURAL_COLORS["text"]},
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
    parameter_state: str = "final",
) -> go.Figure:
    """Explain a fitted forward pass or explicitly replay it during training."""
    if max_layers_math < 3 or max_neurons_math < 1:
        raise ValueError("max_layers_math must be at least 3 and max_neurons_math must be positive.")
    if parameter_state not in {"final", "training_replay"}:
        raise ValueError("parameter_state must be 'final' or 'training_replay'.")
    layers = describe_torch_model(model, x_query)
    selected_layers = _selected_layers(layers, max_layers_math)
    frame_specs: List[tuple[int | None, Dict[str, np.ndarray], Dict[str, np.ndarray]]] = []
    steps: List[int] = []
    if history is not None and _history_is_complete(model, history):
        history_steps = np.asarray(history.get("steps", []), dtype=int)
        if not history_steps.size:
            raise ValueError("History has no recorded steps.")
        indices = (
            _frame_indices(history_steps.size, max_frames)
            if parameter_state == "training_replay"
            else np.asarray([history_steps.size - 1], dtype=int)
        )
        for frame_index in indices:
            frame_specs.append(
                (
                    int(frame_index),
                    parameter_snapshot(history, int(frame_index)),
                    buffer_snapshot(history, int(frame_index)),
                )
            )
            steps.append(int(history_steps[frame_index]))
    else:
        if parameter_state == "training_replay":
            raise ValueError("parameter_state='training_replay' requires complete recorded parameter history.")
        frame_specs.append((None, {}, {}))
    outputs_and_records = [
        run_torch_forward(model, x_query, snapshot or None, buffers or None) for _, snapshot, buffers in frame_specs
    ]
    loss_name = str(history.get("training_config", {}).get("loss", "")) if history is not None else None
    figure = make_subplots(rows=1, cols=2, column_widths=[0.25, 0.75], horizontal_spacing=0.04)
    figure.add_trace(go.Scatter(x=[], y=[], showlegend=False, hoverinfo="skip"), row=1, col=1)
    figure.add_trace(go.Scatter(x=[], y=[], showlegend=False, hoverinfo="skip"), row=1, col=2)
    for column in (1, 2):
        figure.update_xaxes(visible=False, range=[0, 1], row=1, col=column)
        figure.update_yaxes(visible=False, range=[0, 1], row=1, col=column)
    frame_names = [str(index) for index in range(len(frame_specs))]
    annotations_by_frame = []
    for (_history_index, snapshot, _buffers), (output, records) in zip(frame_specs, outputs_and_records):
        annotations_by_frame.append(
            _explanation_annotations(
                model,
                x_query,
                selected_layers,
                records,
                output,
                snapshot,
                dec,
                max_neurons_math,
                loss_name,
                stage=3,
                prediction_stages=parameter_state == "final",
            )
        )
    staged_annotations: List[List[Dict[str, Any]]] = []
    staged_shapes: List[List[Dict[str, Any]]] = []
    if parameter_state == "final":
        _history_index, snapshot, _buffers = frame_specs[0]
        output, records = outputs_and_records[0]
        staged_annotations = [
            _explanation_annotations(
                model,
                x_query,
                selected_layers,
                records,
                output,
                snapshot,
                dec,
                max_neurons_math,
                loss_name,
                stage=stage,
                prediction_stages=True,
            )
            for stage in range(4)
        ]
        staged_shapes = [_prediction_card_shapes(stage) for stage in range(4)]
        controls = [
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.02,
                "y": 1.02,
                "xanchor": "left",
                **animation_button_style(),
                "buttons": [
                    {
                        "label": label,
                        "method": "relayout",
                        "args": [
                            {
                                "annotations": staged_annotations[stage],
                                "shapes": staged_shapes[stage],
                            }
                        ],
                    }
                    for label, stage in (
                        ("Input", 1),
                        ("Substitution", 2),
                        ("Output", 3),
                        ("Reset", 0),
                    )
                ],
            }
        ]
        sliders = []
        initial_annotations = staged_annotations[0]
        initial_shapes = staged_shapes[0]
    elif len(frame_specs) > 1:
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
        controls[0]["x"] = 0.02
        controls[0]["y"] = 1.02
        controls[0]["xanchor"] = "left"
        initial_annotations = annotations_by_frame[0]
        initial_shapes = []
    else:
        controls, sliders = [], []
        initial_annotations = annotations_by_frame[0]
        initial_shapes = []
    if title is None:
        title = (
            "Neural prediction: input, substitution, and output"
            if parameter_state == "final"
            else "Neural training replay: parameter and signal evolution"
        )
    layout = neural_layout(title, height=max(720, 275 + 125 * len(selected_layers)))
    layout["title"]["x"] = 0.5
    layout["title"]["xanchor"] = "center"
    layout["margin"] = {"t": 195, "r": 35, "b": 100 if sliders else 55, "l": 35}
    layout["meta"] = {
        "mlektic_neural_prediction": {
            "parameter_state": parameter_state,
            "training_replay": parameter_state == "training_replay",
            "staged_explanation": parameter_state == "final",
            "interaction": "staged prediction" if parameter_state == "final" else "training animation",
            "prediction_cards_visible": parameter_state == "final",
            "standalone_training_view": parameter_state == "training_replay",
            "line_aware_vertical_spacing": True,
            "section_layout": {
                "controls": "reserved upper-left band",
                "model_formula": "paper row 0.955",
                "summary_cards": dict(_PREDICTION_CARD_BOUNDS),
                "derivation": "independent two-column region from paper row 0.56 to 0.06",
                "cards_use_fixed_shapes": True,
                "initial_stage": "Reset",
            },
            "stages": (["Input", "Substitution", "Output", "Reset"] if parameter_state == "final" else []),
        }
    }
    figure.update_layout(
        **layout,
        annotations=initial_annotations,
        shapes=initial_shapes,
        updatemenus=controls,
        sliders=sliders,
        showlegend=False,
    )
    return figure


__all__ = ["build_nn_prediction_figure"]
