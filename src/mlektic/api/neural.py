"""Public API for PyTorch neural-network visualizations."""

from __future__ import annotations

from typing import Any, Dict

from ..neural.capture import capture_neural_graph
from ..neural.graph_ir import NeuralGraph
from ..neural.recorder import TorchTrainingRecorder
from ..neural.report import build_nn_math_report, display_nn_math_report, export_nn_math_report
from ..neural.semantics import register_neural_descriptor
from ..visualization.design import apply_visual_system, resolve_visual_spec
from ..visualization.neural.architecture import build_nn_architecture_figure
from ..visualization.neural.backpropagation import build_nn_backpropagation_figure
from ..visualization.neural.blocks import build_nn_block_figure
from ..visualization.neural.graph import build_nn_graph_figure
from ..visualization.neural.hyperparameters import build_nn_hyperparameter_figure
from ..visualization.neural.landscape import build_nn_loss_landscape_figure
from ..visualization.neural.math_view import build_nn_prediction_figure
from ..visualization.neural.training import (
    build_nn_activation_figure,
    build_nn_training_figure,
    build_nn_weight_figure,
)

_DENSE_REPLAY_MODULE_TYPES = {
    "Linear",
    "Identity",
    "ReLU",
    "LeakyReLU",
    "ELU",
    "SELU",
    "GELU",
    "SiLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "LogSoftmax",
}


def _requires_complete_topology(graph: NeuralGraph) -> bool:
    """Return whether a neuron-only dense replay would omit executed work."""
    for node in graph.nodes:
        if node.op in {"placeholder", "output", "get_attr"}:
            continue
        if node.op != "call_module":
            return True
        if node.module_type not in _DENSE_REPLAY_MODULE_TYPES:
            return True
    incoming: Dict[str, int] = {node.id: 0 for node in graph.nodes}
    outgoing: Dict[str, int] = {node.id: 0 for node in graph.nodes}
    for edge in graph.edges:
        incoming[edge.target] = incoming.get(edge.target, 0) + 1
        outgoing[edge.source] = outgoing.get(edge.source, 0) + 1
    return any(count > 1 for count in incoming.values()) or any(count > 1 for count in outgoing.values())


def _complete_topology_figure(
    model: Any,
    input_sample: Any,
    *,
    title: str | None,
    capture_backend: str,
    input_kwargs: Dict[str, Any] | None,
    max_nodes: int,
    show_formulas: bool,
) -> tuple[Any | None, NeuralGraph]:
    """Capture once and return a truthful block graph when dense replay is incomplete."""
    capture_input = input_sample
    first_leaf = next(
        (module for module in model.modules() if module is not model and not list(module.children())),
        None,
    )
    expected_sample_dimensions = {
        "Conv1d": 2,
        "BatchNorm1d": 2,
        "Conv2d": 3,
        "BatchNorm2d": 3,
        "Conv3d": 4,
        "BatchNorm3d": 4,
    }
    expected = expected_sample_dimensions.get(first_leaf.__class__.__name__) if first_leaf is not None else None
    if expected is not None and hasattr(capture_input, "dim") and int(capture_input.dim()) == expected:
        capture_input = capture_input.unsqueeze(0)
    graph = capture_neural_graph(
        model,
        capture_input,
        input_kwargs=input_kwargs,
        backend=capture_backend,
    )
    if not _requires_complete_topology(graph):
        return None, graph
    figure = build_nn_block_figure(
        graph,
        title=title,
        max_nodes=max_nodes,
        show_formulas=show_formulas,
    )
    metadata = dict(figure.layout.meta or {})
    metadata["mlektic_neural_graph_route"] = {
        "requested_view": "graph",
        "rendered_view": "complete execution blocks",
        "reason": "a neuron-only dense replay would omit executed modules, operations, or branches",
        "history_animation_applied": False,
        "topology_is_preferred_over_misleading_animation": True,
    }
    figure.update_layout(meta=metadata)
    return figure, graph


def _apply_nn_visual_system(
    figure: Any,
    *,
    theme: str | None,
    format: str,
    density: str,
    size: str,
    width: int | None,
    height: int | None,
    responsive: bool,
    reduced_motion: bool,
    math_font_scale: float = 1.0,
) -> Any:
    """Apply the shared additive visual contract to a neural figure."""
    spec = resolve_visual_spec(
        detail=density,
        theme=theme,
        format=format,
        size=size,
        width=width,
        height=height,
        responsive=responsive,
        reduced_motion=reduced_motion,
    )
    metadata = dict(figure.layout.meta or {}) if isinstance(figure.layout.meta, dict) else {}
    prediction_metadata = metadata.get("mlektic_neural_prediction")
    if (
        (spec.reduced_motion or spec.format == "report")
        and isinstance(prediction_metadata, dict)
        and prediction_metadata.get("staged_explanation")
    ):
        # Staged fitted predictions have no Plotly frames: their buttons replace
        # annotations directly.  Select the complete Output stage before the
        # shared static-mode contract removes those controls.
        for menu in figure.layout.updatemenus or ():
            output_button = next(
                (button for button in menu.buttons or () if str(button.label) == "Output"),
                None,
            )
            if output_button is None or not output_button.args:
                continue
            update = output_button.args[0]
            if isinstance(update, dict) and update.get("annotations"):
                figure.layout.annotations = update["annotations"]
                if "shapes" in update:
                    figure.layout.shapes = update["shapes"]
                prediction_metadata = dict(prediction_metadata)
                prediction_metadata.update(
                    {
                        "static_stage": "Output",
                        "static_contains_complete_forward_pass": True,
                    }
                )
                metadata["mlektic_neural_prediction"] = prediction_metadata
                figure.update_layout(meta=metadata)
                break
    figure = apply_visual_system(figure, spec, family="neural")
    hyperparameter_metadata = metadata.get("mlektic_neural_hyperparameters")
    if height is None and isinstance(hyperparameter_metadata, dict):
        minimum_height = hyperparameter_metadata.get("content_min_height")
        if isinstance(minimum_height, (int, float)):
            figure.layout.height = max(int(figure.layout.height or 0), int(minimum_height))
    if not 0.75 <= math_font_scale <= 2.0:
        raise ValueError("math_font_scale must be between 0.75 and 2.0.")
    if math_font_scale != 1.0:

        def scale_annotations(annotations: Any) -> None:
            for annotation in annotations or ():
                text = str(getattr(annotation, "text", ""))
                if "$" not in text and "\\(" not in text:
                    continue
                size_value = getattr(getattr(annotation, "font", None), "size", None)
                if size_value is not None:
                    annotation.font.size = max(8, round(float(size_value) * math_font_scale))

        scale_annotations(figure.layout.annotations)
        for frame in figure.frames or ():
            scale_annotations(getattr(frame.layout, "annotations", None))
            for trace in frame.data or ():
                text_values = getattr(trace, "text", None)
                if text_values and any("$" in str(value) for value in text_values):
                    size_value = getattr(getattr(trace, "textfont", None), "size", None)
                    if size_value is not None:
                        trace.textfont.size = max(8, round(float(size_value) * math_font_scale))
        for trace in figure.data:
            text_values = getattr(trace, "text", None)
            if text_values and any("$" in str(value) for value in text_values):
                size_value = getattr(getattr(trace, "textfont", None), "size", None)
                if size_value is not None:
                    trace.textfont.size = max(8, round(float(size_value) * math_font_scale))
    return figure


def visualize_nn(
    model: Any,
    input_sample: Any | None = None,
    *,
    history: Dict[str, Any] | None = None,
    optimizer: Any | None = None,
    loss_fn: Any | None = None,
    scheduler: Any | None = None,
    view: str = "architecture",
    title: str | None = None,
    max_neurons: int = 10,
    max_frames: int | None = 20,
    frame_duration: int = 120,
    node_color_mode: str = "value",
    edge_color_mode: str = "weight",
    evolution_mode: str = "absolute",
    update_reference: str = "previous",
    update_scale: str = "global",
    show_update_panel: bool | None = None,
    show_loss_panel: bool = False,
    show_backpropagation: bool = False,
    top_k_updates: int | None = None,
    interpolation_frames: int = 0,
    capture_backend: str = "auto",
    input_kwargs: Dict[str, Any] | None = None,
    max_nodes: int = 48,
    show_formulas: bool = True,
    theme: str | None = None,
    format: str = "dashboard",
    density: str = "essential",
    size: str = "default",
    width: int | None = None,
    height: int | None = None,
    responsive: bool = False,
    reduced_motion: bool = False,
    math_font_scale: float = 1.0,
):
    """Visualize PyTorch architecture, graph mathematics, training, or activations.

    Args:
        model: A ``torch.nn.Module``.
        input_sample: One sample or a batch used to infer layer output shapes.
        history: Payload returned by :meth:`TorchTrainingRecorder.to_history`.
        view: One of ``"architecture"``, ``"blocks"``, ``"hyperparameters"``,
            ``"graph"``, ``"training"``, ``"weights"``, ``"activations"``,
            or ``"backpropagation"``.
        optimizer: Optional live PyTorch optimizer used by the hyperparameter view.
        loss_fn: Optional live PyTorch objective used by the hyperparameter view.
        scheduler: Optional live PyTorch learning-rate scheduler used by the
            hyperparameter view.
        title: Optional figure title.
        max_neurons: Maximum representative neurons drawn per layer in an
            eligible pure-dense animated graph.
        max_frames: Maximum recorded training steps retained by animated views.
        frame_duration: Milliseconds per animation frame in the training view.
        node_color_mode: ``"value"`` for exact globally scaled outputs or
            ``"relative"`` for per-layer contrast.
        edge_color_mode: ``"weight"`` for globally scaled parameters or
            ``"signal"`` for ``w_ji * a_i``.
        evolution_mode: ``"absolute"`` preserves the classic graph,
            ``"updates"`` emphasizes parameter changes, and ``"hybrid"``
            overlays signed update halos on the absolute encoding.
        update_reference: Compare parameters with the ``"previous"`` displayed
            checkpoint or the ``"initial"`` recorded state.
        update_scale: Use one truthful ``"global"`` update scale across the
            animation or normalize contrast independently in each ``"frame"``.
        show_update_panel: Show update norms and gradient alignment. By default,
            it is enabled only for ``"updates"`` and ``"hybrid"``.
        show_loss_panel: Add the recorded selected-loss curve below the graph.
        show_backpropagation: Overlay recorded parameter gradients on graph edges.
        top_k_updates: Optionally emphasize only the largest visible edge updates.
        interpolation_frames: Perceptual frames inserted between recorded
            checkpoints; these are explicitly not optimizer steps.
        capture_backend: ``"auto"``, ``"fx"``, or ``"hooks"`` for the block view.
        input_kwargs: Optional keyword inputs passed to the model in the block view.
        max_nodes: Maximum semantic blocks rendered before a collapsed summary.
        show_formulas: Show alternating above/below formulas when the captured
            execution graph is small.
        theme: Additive visual theme; ``None`` preserves the classic default.
        format: ``"dashboard"``, ``"lesson"``, ``"compact"``, or ``"report"``.
        density: Mathematical information density recorded in visual metadata.
        size: Named canvas size preset.
        width: Optional explicit canvas width in pixels.
        height: Optional explicit canvas height in pixels.
        responsive: Scale the resolved composition with its container.
        reduced_motion: Show the exact final state without animation controls.
        math_font_scale: Scale mathematical annotations from 0.75 to 2.0.

    Returns:
        A Plotly figure for the selected view.
    """
    if view == "architecture":
        figure = build_nn_architecture_figure(
            model,
            input_sample,
            history=history,
            title=title,
            max_neurons=max_neurons,
        )
    elif view == "blocks":
        if input_sample is None:
            raise ValueError("view='blocks' requires input_sample for execution capture.")
        captured = capture_neural_graph(
            model,
            input_sample,
            input_kwargs=input_kwargs,
            backend=capture_backend,
        )
        figure = build_nn_block_figure(
            captured,
            title=title,
            max_nodes=max_nodes,
            show_formulas=show_formulas,
        )
    elif view == "hyperparameters":
        figure = build_nn_hyperparameter_figure(
            model,
            history=history,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            title=title,
        )
    elif history is None:
        raise ValueError(f"view='{view}' requires a history from TorchTrainingRecorder.")
    elif view == "graph":
        if input_sample is None:
            raise ValueError("view='graph' requires input_sample for node activations.")
        figure, _captured = _complete_topology_figure(
            model,
            input_sample,
            title=title,
            capture_backend=capture_backend,
            input_kwargs=input_kwargs,
            max_nodes=max_nodes,
            show_formulas=show_formulas,
        )
        if figure is None:
            figure = build_nn_graph_figure(
                model,
                input_sample,
                history,
                title=title,
                max_neurons=max_neurons,
                max_frames=max_frames,
                frame_duration=frame_duration,
                node_color_mode=node_color_mode,
                edge_color_mode=edge_color_mode,
                evolution_mode=evolution_mode,
                update_reference=update_reference,
                update_scale=update_scale,
                show_update_panel=show_update_panel,
                show_loss_panel=show_loss_panel,
                show_backpropagation=show_backpropagation,
                top_k_updates=top_k_updates,
                interpolation_frames=interpolation_frames,
            )
    elif view == "training":
        figure = build_nn_training_figure(
            history,
            title=title,
            frame_duration=frame_duration,
            max_frames=max_frames,
        )
    elif view == "weights":
        figure = build_nn_weight_figure(
            history,
            title=title,
            frame_duration=frame_duration,
            max_frames=max_frames,
        )
    elif view == "activations":
        figure = build_nn_activation_figure(
            model,
            history,
            input_sample=input_sample,
            title=title,
            frame_duration=frame_duration,
            max_frames=max_frames,
        )
    elif view == "backpropagation":
        figure = build_nn_backpropagation_figure(
            model,
            history,
            input_sample=input_sample,
            title=title,
            max_frames=max_frames,
            frame_duration=frame_duration,
        )
    else:
        raise ValueError(
            "view must be 'architecture', 'blocks', 'hyperparameters', 'graph', "
            "'training', 'weights', 'activations', or 'backpropagation'."
        )
    return _apply_nn_visual_system(
        figure,
        theme=theme,
        format=format,
        density=density,
        size=size,
        width=width,
        height=height,
        responsive=responsive,
        reduced_motion=reduced_motion,
        math_font_scale=math_font_scale,
    )


def visualize_nn_architecture(
    model: Any,
    input_sample: Any | None = None,
    *,
    history: Dict[str, Any] | None = None,
    title: str | None = None,
    max_layers: int = 8,
    architecture_mode: str = "legacy",
    capture_backend: str = "auto",
    input_kwargs: Dict[str, Any] | None = None,
    max_nodes: int = 48,
    show_formulas: bool = True,
    theme: str | None = None,
    format: str = "dashboard",
    density: str = "essential",
    size: str = "default",
    width: int | None = None,
    height: int | None = None,
    responsive: bool = False,
    reduced_motion: bool = False,
    math_font_scale: float = 1.0,
):
    """Show layer roles, formulas, tensor dimensions, and hyperparameters.

    Args:
        model: A ``torch.nn.Module``.
        input_sample: Sample or batch used to infer tensor dimensions.
        history: Optional recorder payload used to enrich the architecture.
        title: Optional figure title.
        max_layers: Maximum number of leaf layers rendered individually.
        architecture_mode: ``"legacy"`` preserves the established figure;
            ``"blocks"`` renders the captured execution graph.
        capture_backend: ``"auto"``, ``"fx"``, or ``"hooks"``.
        input_kwargs: Optional keyword inputs passed to the model.
        max_nodes: Maximum blocks before a collapsed summary node.
        show_formulas: Show formulas below nodes for small block graphs.
        theme: Additive visual theme; ``None`` preserves the classic default.
        format: Composition preset.
        density: Mathematical information density.
        size: Named canvas size preset.
        width: Optional explicit canvas width in pixels.
        height: Optional explicit canvas height in pixels.
        responsive: Scale the resolved composition with its container.
        reduced_motion: Remove motion controls and show the final state.
        math_font_scale: Scale mathematical annotations from 0.75 to 2.0.

    Returns:
        A static Plotly architecture figure.
    """
    if architecture_mode == "legacy":
        figure = build_nn_architecture_figure(
            model,
            input_sample,
            history=history,
            title=title,
            max_layers=max_layers,
        )
    elif architecture_mode == "blocks":
        if input_sample is None:
            raise ValueError("architecture_mode='blocks' requires input_sample.")
        figure = build_nn_block_figure(
            capture_neural_graph(
                model,
                input_sample,
                input_kwargs=input_kwargs,
                backend=capture_backend,
            ),
            title=title,
            max_nodes=max_nodes,
            show_formulas=show_formulas,
        )
    else:
        raise ValueError("architecture_mode must be 'legacy' or 'blocks'.")
    return _apply_nn_visual_system(
        figure,
        theme=theme,
        format=format,
        density=density,
        size=size,
        width=width,
        height=height,
        responsive=responsive,
        reduced_motion=reduced_motion,
        math_font_scale=math_font_scale,
    )


def inspect_nn(
    model: Any,
    inputs: Any,
    *,
    input_kwargs: Dict[str, Any] | None = None,
    backend: str = "auto",
) -> NeuralGraph:
    """Capture a renderer-independent neural execution graph."""
    return capture_neural_graph(model, inputs, input_kwargs=input_kwargs, backend=backend)


def visualize_nn_blocks(
    model: Any,
    inputs: Any,
    *,
    input_kwargs: Dict[str, Any] | None = None,
    backend: str = "auto",
    title: str | None = None,
    max_nodes: int = 48,
    show_formulas: bool = True,
    theme: str | None = None,
    format: str = "dashboard",
    density: str = "essential",
    size: str = "default",
    width: int | None = None,
    height: int | None = None,
    responsive: bool = False,
    reduced_motion: bool = False,
    math_font_scale: float = 1.0,
):
    """Capture and render a branch-aware semantic block graph."""
    figure = build_nn_block_figure(
        capture_neural_graph(model, inputs, input_kwargs=input_kwargs, backend=backend),
        title=title,
        max_nodes=max_nodes,
        show_formulas=show_formulas,
    )
    return _apply_nn_visual_system(
        figure,
        theme=theme,
        format=format,
        density=density,
        size=size,
        width=width,
        height=height,
        responsive=responsive,
        reduced_motion=reduced_motion,
        math_font_scale=math_font_scale,
    )


def visualize_nn_graph(
    model: Any,
    input_sample: Any,
    history: Dict[str, Any],
    *,
    title: str | None = None,
    max_neurons: int = 8,
    max_frames: int | None = 20,
    frame_duration: int = 180,
    node_color_mode: str = "value",
    edge_color_mode: str = "weight",
    evolution_mode: str = "absolute",
    update_reference: str = "previous",
    update_scale: str = "global",
    show_update_panel: bool | None = None,
    show_loss_panel: bool = False,
    show_backpropagation: bool = False,
    top_k_updates: int | None = None,
    interpolation_frames: int = 0,
    capture_backend: str = "auto",
    input_kwargs: Dict[str, Any] | None = None,
    max_nodes: int = 48,
    show_formulas: bool = True,
    theme: str | None = None,
    format: str = "dashboard",
    density: str = "essential",
    size: str = "default",
    width: int | None = None,
    height: int | None = None,
    responsive: bool = False,
    reduced_motion: bool = False,
    math_font_scale: float = 1.0,
):
    """Visualize a truthful graph, animating only complete pure-dense replays.

    A model with operations or branches that a dense replay would omit is
    routed to the complete execution-block topology. The route and absence of
    training animation are disclosed in figure metadata.

    For eligible pure-dense models, the default modes use exact values and one
    global color scale per quantity
    across every retained frame. Node and edge scales intentionally differ:
    nodes encode activations while edges encode parameters. ``"relative"``
    increases per-layer node contrast, and ``"signal"`` colors each edge by
    ``w_ji * a_i`` instead of by its weight.

    Args:
        model: A ``torch.nn.Module``.
        input_sample: Sample or batch used to compute node activations.
        history: Payload returned by :meth:`TorchTrainingRecorder.to_history`.
        title: Optional figure title.
        max_neurons: Maximum visible nodes per layer.
        max_frames: Maximum retained animation frames, or ``None`` for all.
        frame_duration: Milliseconds per animation frame.
        node_color_mode: ``"value"`` or ``"relative"``.
        edge_color_mode: ``"weight"`` or ``"signal"``.
        evolution_mode: ``"absolute"`` for the classic encoding, ``"updates"``
            for update-first encoding, or ``"hybrid"`` for both.
        update_reference: ``"previous"`` displayed checkpoint or ``"initial"``.
        update_scale: ``"global"`` for comparable magnitudes across time or
            ``"frame"`` for per-frame contrast normalization.
        show_update_panel: Show parameter/update norms and gradient alignment.
        show_loss_panel: Show the finite recorded objective below the graph.
        show_backpropagation: Overlay recorded reverse-mode gradients. The
            default is ``False`` because this adds one animated trace per edge.
        top_k_updates: Optionally emphasize only the largest visible edge updates.
        interpolation_frames: Perceptual frames between recorded checkpoints;
            they improve motion but never represent optimizer steps.
        capture_backend: Capture backend used when complete execution blocks
            are required to avoid omitting non-dense operations.
        input_kwargs: Optional keyword inputs forwarded during graph capture.
        max_nodes: Maximum execution blocks before a disclosed summary.
        show_formulas: Show alternating inline formulas when they fit safely.
        theme: Additive visual theme; ``None`` preserves the classic default.
        format: Composition preset.
        density: Mathematical information density.
        size: Named canvas size preset.
        width: Optional explicit canvas width in pixels.
        height: Optional explicit canvas height in pixels.
        responsive: Scale the resolved composition with its container.
        reduced_motion: Remove motion controls and show the final state.
        math_font_scale: Scale mathematical annotations from 0.75 to 2.0.

    Returns:
        A Plotly graph figure: animated for complete pure-dense replay, or a
        complete static execution topology when dense replay would omit work.
    """
    figure, _captured = _complete_topology_figure(
        model,
        input_sample,
        title=title,
        capture_backend=capture_backend,
        input_kwargs=input_kwargs,
        max_nodes=max_nodes,
        show_formulas=show_formulas,
    )
    if figure is None:
        figure = build_nn_graph_figure(
            model,
            input_sample,
            history,
            title=title,
            max_neurons=max_neurons,
            max_frames=max_frames,
            frame_duration=frame_duration,
            node_color_mode=node_color_mode,
            edge_color_mode=edge_color_mode,
            evolution_mode=evolution_mode,
            update_reference=update_reference,
            update_scale=update_scale,
            show_update_panel=show_update_panel,
            show_loss_panel=show_loss_panel,
            show_backpropagation=show_backpropagation,
            top_k_updates=top_k_updates,
            interpolation_frames=interpolation_frames,
        )
    return _apply_nn_visual_system(
        figure,
        theme=theme,
        format=format,
        density=density,
        size=size,
        width=width,
        height=height,
        responsive=responsive,
        reduced_motion=reduced_motion,
        math_font_scale=math_font_scale,
    )


def visualize_nn_training(
    history: Dict[str, Any],
    *,
    title: str | None = None,
    frame_duration: int = 120,
    max_metrics: int = 3,
    max_frames: int | None = 30,
    theme: str | None = None,
    format: str = "dashboard",
    density: str = "essential",
    size: str = "default",
    width: int | None = None,
    height: int | None = None,
    responsive: bool = False,
    reduced_motion: bool = False,
    math_font_scale: float = 1.0,
):
    """Animate a compact 2x2 panel with loss and up to three metrics.

    Metrics may be supplied explicitly to the recorder or inferred from
    predictions and targets during :meth:`TorchTrainingRecorder.record`.
    """
    figure = build_nn_training_figure(
        history,
        title=title,
        frame_duration=frame_duration,
        max_metrics=max_metrics,
        max_frames=max_frames,
    )
    return _apply_nn_visual_system(
        figure,
        theme=theme,
        format=format,
        density=density,
        size=size,
        width=width,
        height=height,
        responsive=responsive,
        reduced_motion=reduced_motion,
        math_font_scale=math_font_scale,
    )


def visualize_nn_hyperparameters(
    model: Any,
    *,
    history: Dict[str, Any] | None = None,
    optimizer: Any | None = None,
    loss_fn: Any | None = None,
    scheduler: Any | None = None,
    title: str | None = None,
    theme: str | None = None,
    format: str = "dashboard",
    density: str = "essential",
    size: str = "wide",
    width: int | None = None,
    height: int | None = None,
    responsive: bool = False,
    reduced_motion: bool = False,
    math_font_scale: float = 1.0,
):
    """Show every detected effective PyTorch hyperparameter and its mathematics.

    The figure is instance based: it reads every supported public configuration
    value from the supplied model and, when available, each optimizer parameter
    group, objective, and learning-rate scheduler.  Recorder history can supply
    the training configuration when live objects are unavailable.  Runtime-only
    implementation switches are retained but explicitly distinguished from
    arguments that change the mathematical map.
    """
    figure = build_nn_hyperparameter_figure(
        model,
        history=history,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        title=title,
    )
    return _apply_nn_visual_system(
        figure,
        theme=theme,
        format=format,
        density=density,
        size=size,
        width=width,
        height=height,
        responsive=responsive,
        reduced_motion=reduced_motion,
        math_font_scale=math_font_scale,
    )


def visualize_nn_loss_landscape(
    model: Any,
    inputs: Any,
    targets: Any,
    loss_fn: Any,
    history: Dict[str, Any],
    *,
    grid_size: int = 23,
    span: float | None = None,
    max_frames: int | None = 20,
    frame_duration: int = 180,
    seed: int = 17,
    title: str | None = None,
    theme: str | None = None,
    format: str = "dashboard",
    density: str = "essential",
    size: str = "default",
    width: int | None = None,
    height: int | None = None,
    responsive: bool = False,
    reduced_motion: bool = False,
    math_font_scale: float = 1.0,
):
    """Evaluate and animate an exact two-direction batch-loss slice.

    The returned surface is an affine 2-D section through the final recorded
    parameter state, not a claim about the full high-dimensional objective.
    History PCA supplies the directions when two independent recorded
    directions exist; otherwise a deterministic orthogonal complement is used.
    """
    figure = build_nn_loss_landscape_figure(
        model,
        inputs,
        targets,
        loss_fn,
        history,
        grid_size=grid_size,
        span=span,
        max_frames=max_frames,
        frame_duration=frame_duration,
        seed=seed,
        title=title,
    )
    return _apply_nn_visual_system(
        figure,
        theme=theme,
        format=format,
        density=density,
        size=size,
        width=width,
        height=height,
        responsive=responsive,
        reduced_motion=reduced_motion,
        math_font_scale=math_font_scale,
    )


def visualize_nn_backpropagation(
    model: Any,
    history: Dict[str, Any],
    *,
    input_sample: Any | None = None,
    max_layers: int = 8,
    max_frames: int | None = 20,
    frame_duration: int = 900,
    title: str | None = None,
    theme: str | None = None,
    format: str = "dashboard",
    density: str = "essential",
    size: str = "default",
    width: int | None = None,
    height: int | None = None,
    responsive: bool = False,
    reduced_motion: bool = False,
    math_font_scale: float = 1.0,
):
    """Explain recorded layerwise gradients, updates, and loss changes.

    Gradient values are aggregate L2 norms retained by the recorder. Update
    values are exact adjacent-checkpoint parameter differences; they are not
    presented as raw gradients or as plain-SGD updates when another optimizer
    is used. ``max_layers`` and ``max_frames`` bound presentation only. Any
    omitted trainable layers are counted in a dedicated lower caption and in
    ``layout.meta``; crowded numerical readouts use invariant alternating rows
    rather than overlapping.
    """
    figure = build_nn_backpropagation_figure(
        model,
        history,
        input_sample=input_sample,
        max_layers=max_layers,
        max_frames=max_frames,
        frame_duration=frame_duration,
        title=title,
    )
    return _apply_nn_visual_system(
        figure,
        theme=theme,
        format=format,
        density=density,
        size=size,
        width=width,
        height=height,
        responsive=responsive,
        reduced_motion=reduced_motion,
        math_font_scale=math_font_scale,
    )


def visualize_nn_weights(
    history: Dict[str, Any],
    *,
    parameter: str | None = None,
    title: str | None = None,
    frame_duration: int = 120,
    max_rows: int = 4,
    max_cols: int = 5,
    max_parameters: int = 6,
    max_frames: int | None = 30,
    theme: str | None = None,
    format: str = "dashboard",
    density: str = "essential",
    size: str = "default",
    width: int | None = None,
    height: int | None = None,
    responsive: bool = False,
    reduced_motion: bool = False,
    math_font_scale: float = 1.0,
):
    """Animate captured parameter tensors using truncated LaTeX matrices.

    ``max_rows``, ``max_cols`` and ``max_parameters`` bound the mathematical
    display without modifying the values stored in the recorder history.
    When intermediate tensors are omitted, their exact count occupies a
    matrix-height row with fixed clearance from both adjacent parameter blocks.
    The same row allocation is reused by every animation frame and disclosed
    in ``layout.meta``.
    """
    figure = build_nn_weight_figure(
        history,
        parameter=parameter,
        title=title,
        frame_duration=frame_duration,
        max_rows=max_rows,
        max_cols=max_cols,
        max_parameters=max_parameters,
        max_frames=max_frames,
    )
    return _apply_nn_visual_system(
        figure,
        theme=theme,
        format=format,
        density=density,
        size=size,
        width=width,
        height=height,
        responsive=responsive,
        reduced_motion=reduced_motion,
        math_font_scale=math_font_scale,
    )


def explain_nn_prediction(
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
    theme: str | None = None,
    format: str = "dashboard",
    density: str = "essential",
    size: str = "default",
    width: int | None = None,
    height: int | None = None,
    responsive: bool = False,
    reduced_motion: bool = False,
    math_font_scale: float = 1.0,
):
    """Explain a PyTorch forward pass mathematically.

    ``parameter_state='final'`` exposes prediction-only Input, numerical
    Substitution, Output, and Reset stages, with no training controls. Use
    ``parameter_state='training_replay'`` for the independent Play/Pause and
    checkpoint-slider view of parameter and signal evolution. That mode has no
    prediction-stage controls, summary cards, or duplicated prediction result.
    Large models are summarized with the configured layer, neuron and frame
    limits.
    """
    figure = build_nn_prediction_figure(
        model,
        x_query,
        history=history,
        title=title,
        dec=dec,
        max_layers_math=max_layers_math,
        max_neurons_math=max_neurons_math,
        max_frames=max_frames,
        frame_duration=frame_duration,
        parameter_state=parameter_state,
    )
    return _apply_nn_visual_system(
        figure,
        theme=theme,
        format=format,
        density=density,
        size=size,
        width=width,
        height=height,
        responsive=responsive,
        reduced_motion=reduced_motion,
        math_font_scale=math_font_scale,
    )


__all__ = [
    "TorchTrainingRecorder",
    "build_nn_math_report",
    "display_nn_math_report",
    "explain_nn_prediction",
    "export_nn_math_report",
    "inspect_nn",
    "register_neural_descriptor",
    "visualize_nn",
    "visualize_nn_architecture",
    "visualize_nn_blocks",
    "visualize_nn_graph",
    "visualize_nn_hyperparameters",
    "visualize_nn_loss_landscape",
    "visualize_nn_backpropagation",
    "visualize_nn_training",
    "visualize_nn_weights",
]
