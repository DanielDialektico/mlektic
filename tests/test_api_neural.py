import re

import numpy as np
import plotly.graph_objects as go
import pytest

from mlektic import (
    TorchTrainingRecorder,
    build_nn_math_report,
    explain_nn_prediction,
    export_nn_math_report,
    visualize_nn,
    visualize_nn_architecture,
    visualize_nn_backpropagation,
    visualize_nn_graph,
    visualize_nn_hyperparameters,
    visualize_nn_loss_landscape,
    visualize_nn_training,
    visualize_nn_weights,
)
from mlektic.neural.introspection import describe_torch_model

torch = pytest.importorskip("torch")


@pytest.fixture
def trained_small_network():
    torch.manual_seed(7)
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 4),
        torch.nn.Tanh(),
        torch.nn.Linear(4, 1),
        torch.nn.Sigmoid(),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25, momentum=0.1)
    criterion = torch.nn.BCELoss()
    recorder = TorchTrainingRecorder(model, optimizer=optimizer, loss_fn=criterion)
    for step in range(6):
        optimizer.zero_grad()
        prediction = model(X)
        loss = criterion(prediction, y)
        loss.backward()
        optimizer.step()
        predicted = prediction >= 0.5
        target = y.bool()
        true_positive = (predicted & target).float().sum()
        precision = true_positive / (predicted.float().sum() + 1e-8)
        recall = true_positive / (target.float().sum() + 1e-8)
        accuracy = (predicted == target).float().mean()
        recorder.record(
            step,
            loss=loss,
            metrics={"accuracy": accuracy, "precision": precision, "recall": recall},
        )
    recorder.close()
    return model, X, recorder.to_history()


def _annotation_text(figure):
    return " ".join(str(annotation.text) for annotation in figure.layout.annotations)


def _two_frame_parameter_history(model):
    parameters = {}
    gradient_norms = {}
    parameter_norms = {}
    for name, parameter in model.named_parameters():
        initial = parameter.detach().cpu().numpy().copy()
        final = initial + 1e-3
        parameters[name] = [initial, final]
        gradient_norms[name] = [0.1, 0.05]
        parameter_norms[name] = [
            float(np.linalg.norm(initial)),
            float(np.linalg.norm(final)),
        ]
    return {
        "steps": np.asarray([1, 2]),
        "loss": np.asarray([1.0, 0.8]),
        "parameters": parameters,
        "gradient_norms": gradient_norms,
        "parameter_norms": parameter_norms,
        "training_config": {"optimizer": "Adam", "loss": "CrossEntropyLoss"},
    }


def test_recorder_retains_math_and_training_metadata(trained_small_network):
    model, X, history = trained_small_network
    layers = describe_torch_model(model, X[:1])

    assert layers[0]["math_index"] == layers[1]["math_index"] == 1
    assert layers[2]["math_index"] == layers[3]["math_index"] == 2
    assert layers[0]["input_shape"] == (1, 2)
    assert layers[0]["parameter_shapes"]["weight"] == (4, 2)
    assert layers[1]["formula"] == r"\mathbf{a}^{(1)}=\tanh(\mathbf{z}^{(1)})"
    assert history["training_config"]["optimizer"] == "SGD"
    assert history["training_config"]["loss"] == "BCELoss"
    assert history["metrics"]["accuracy"].shape == (6,)
    assert history["activation_vectors"]["1"][0].shape == (4,)


def test_architecture_has_dimensions_formulas_and_hyperparameters(trained_small_network):
    model, X, history = trained_small_network
    figure = visualize_nn_architecture(model, X[:1], history=history)
    text = _annotation_text(figure)

    assert isinstance(figure, go.Figure)
    assert r"\mathbb{R}^{2}" in text
    assert r"\mathbf{a}^{(1)}=\tanh(\mathbf{z}^{(1)})" in text
    assert "in_features=2" in text
    assert "SGD" in text


def test_architecture_wraps_configuration_within_semantic_module_columns():
    class ConvNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(1, 4, 3, padding=1),
                torch.nn.BatchNorm2d(4),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
            )
            self.classifier = torch.nn.Linear(4 * 4 * 4, 10)

        def forward(self, values):
            return self.classifier(torch.flatten(self.features(values), 1))

    figure = visualize_nn_architecture(
        ConvNet(),
        torch.randn(1, 1, 8, 8),
        max_layers=8,
        theme="classroom",
        size="wide",
    )
    metadata = figure.layout.meta["mlektic_neural_architecture"]
    configuration = [
        annotation
        for annotation in figure.layout.annotations
        if annotation.y == pytest.approx(0.13) and str(annotation.text)
    ]

    assert metadata["configuration_layout"] == "semantic-multiline-columns"
    assert metadata["complete_configuration_on_hover"] is True
    assert "<br>" in str(configuration[0].text)
    assert configuration[0].xanchor == "left"
    assert configuration[-1].xanchor == "right"
    assert all(
        len(line) <= metadata["configuration_max_chars"]
        for annotation in configuration
        for line in str(annotation.text).split("<br>")
    )
    assert "padding_mode=zeros" in str(figure.data[0].customdata[0])


def test_compact_architecture_reserves_connector_corridors_between_nodes():
    modules = []
    widths = [128, 512, 512, 384, 256, 128, 64, 32, 10]
    for index, (input_width, output_width) in enumerate(zip(widths, widths[1:])):
        modules.append(torch.nn.Linear(input_width, output_width))
        modules.append(torch.nn.GELU())
        if index < len(widths) - 2:
            modules.append(torch.nn.Dropout(0.1))
    model = torch.nn.Sequential(*modules)
    figure = visualize_nn_architecture(model, torch.randn(1, 128), max_layers=8)
    metadata = figure.layout.meta["mlektic_neural_architecture"]
    connectors = [
        annotation
        for annotation in figure.layout.annotations
        if annotation.showarrow and annotation.axref == "x" and annotation.ayref == "y"
    ]
    node_shapes = list(figure.layout.shapes)

    assert metadata["node_scale"] < 1.0
    assert metadata["connectors_stop_at_node_boundaries"] is True
    assert connectors
    assert all(float(connector.x) > float(connector.ax) for connector in connectors)
    assert all(float(connector.x) - float(connector.ax) >= 0.015 for connector in connectors)
    for connector in connectors:
        for shape in node_shapes:
            assert not float(shape.x0) <= float(connector.ax) <= float(shape.x1)
            assert not float(shape.x0) <= float(connector.x) <= float(shape.x1)
    assert max(float(shape.x1) - float(shape.x0) for shape in node_shapes) < 0.11
    assert not any(str(annotation.text) == "&#8594;" for annotation in figure.layout.annotations)


def test_graph_animates_stable_weight_heatmap_without_optional_backprop_overlay(
    trained_small_network,
):
    model, X, history = trained_small_network
    figure = visualize_nn_graph(model, X[0], history, max_frames=3)

    assert isinstance(figure, go.Figure)
    assert len(figure.frames) == 3
    first_hover = " ".join(str(value) for trace in figure.frames[0].data for value in (trace.customdata or []))
    last_hover = " ".join(str(value) for trace in figure.frames[-1].data for value in (trace.customdata or []))
    assert "Weight evolution" in first_hover
    assert "Backpropagation" not in first_hover
    assert "Theta[1,:]=" in first_hover
    assert "grad Theta[1,:]=" not in first_hover
    assert r"\begin" not in first_hover
    assert f"theta[1,1]={model[0].weight[0, 0].item():.3f}" in last_hover
    first_parameter_text = next(trace.text[0] for trace in figure.frames[0].data if trace.name == "parameter readout")
    last_parameter_text = next(trace.text[0] for trace in figure.frames[-1].data if trace.name == "parameter readout")
    final_step_text = next(trace.text[0] for trace in figure.frames[-1].data if trace.name == "training step readout")
    assert first_parameter_text != last_parameter_text
    assert "final weights" in final_step_text
    assert "Feed forward" in _annotation_text(figure)
    assert "Backpropagation" not in _annotation_text(figure)
    assert "Node heatmap (exact)" in _annotation_text(figure)
    assert r"a_j^{(\ell)}" in _annotation_text(figure)
    assert r"\mathbb{R}" in _annotation_text(figure)
    first_node_colors = [tuple(trace.marker.color) for trace in figure.frames[0].data if trace.mode == "markers"]
    last_node_colors = [tuple(trace.marker.color) for trace in figure.frames[-1].data if trace.mode == "markers"]
    assert any(first != last for first, last in zip(first_node_colors[1:], last_node_colors[1:]))
    assert r"\theta_{ji}^{(\ell)}" in _annotation_text(figure)
    assert r"a_j^{(\ell)}" in _annotation_text(figure)
    assert figure.data[-1].marker.cmin < 0.0
    assert figure.data[-1].marker.cmax > 0.0
    assert figure.data[-1].marker.showscale is True
    assert figure.data[-2].marker.colorbar.title.text is None
    assert figure.data[-1].marker.colorbar.title.text is None
    assert figure.data[-2].marker.colorbar.tickfont.size == 11
    assert figure.data[-1].marker.colorbar.tickfont.size == 11
    assert figure.layout.margin.l == 65
    assert figure.layout.margin.r == 145
    step_readout = next(trace for trace in figure.data if trace.name == "training step readout")
    assert step_readout.x[0] == pytest.approx(0.07)
    assert step_readout.textposition == "middle center"
    assert figure.layout.updatemenus[0].buttons[0].args[1]["frame"]["redraw"] is False
    assert all(" F" not in step.label and " B" not in step.label for step in figure.layout.sliders[0].steps)
    evolution = figure.layout.meta["mlektic_neural_evolution"]
    assert evolution["evolution_mode"] == "absolute"
    assert evolution["show_update_panel"] is False
    assert evolution["show_backpropagation"] is False
    assert evolution["semantic_frames"] == evolution["display_frames"] == 3
    assert all(trace.name != "parameter update halo" for trace in figure.frames[0].data)
    assert all(trace.name != "update summary" for trace in figure.frames[0].data)


def test_graph_can_opt_into_recorded_backpropagation_overlay(trained_small_network):
    model, X, history = trained_small_network
    figure = visualize_nn_graph(
        model,
        X[0],
        history,
        max_frames=3,
        show_backpropagation=True,
    )
    first_hover = " ".join(str(value) for trace in figure.frames[0].data for value in (trace.customdata or []))

    assert "Backpropagation" in first_hover
    assert "grad Theta[1,:]=" in first_hover
    assert "Backpropagation overlay" in _annotation_text(figure)
    metadata = figure.layout.meta["mlektic_neural_evolution"]
    rows = metadata["section_layout"]["rows"]
    forward = next(annotation for annotation in figure.layout.annotations if "Feed forward" in str(annotation.text))
    backpropagation = next(
        annotation for annotation in figure.layout.annotations if "Backpropagation overlay" in str(annotation.text)
    )
    node_legend = next(annotation for annotation in figure.layout.annotations if "Node heatmap" in str(annotation.text))
    activity_legend = next(
        annotation for annotation in figure.layout.annotations if "Activity glow" in str(annotation.text)
    )
    assert metadata["show_backpropagation"] is True
    assert float(forward.y) == pytest.approx(rows["phase"])
    assert float(backpropagation.y) == pytest.approx(rows["backpropagation"])
    assert float(forward.y) > float(backpropagation.y) > float(node_legend.y) > float(activity_legend.y)
    assert any(trace.name == "recorded backpropagation gradient" for trace in figure.frames[0].data)


def test_graph_supports_relative_nodes_and_forward_signal_edges(trained_small_network):
    model, X, history = trained_small_network
    figure = visualize_nn(
        model,
        X[1],
        history=history,
        view="graph",
        node_color_mode="relative",
        edge_color_mode="signal",
        max_frames=2,
        theme="accessible",
    )
    hover = " ".join(str(value) for trace in figure.frames[0].data for value in (trace.customdata or []))

    assert r"s_{ji}^{(\ell)}=\theta_{ji}^{(\ell)}a_i^{(\ell-1)}" in _annotation_text(figure)
    assert r"\widetilde a_j^{(\ell)}" in _annotation_text(figure)
    assert figure.data[-1].marker.cmin == 0.0
    assert figure.data[-1].marker.cmax == 1.0
    assert "Forward signal" in hover
    assert "Forward activity glow" in hover
    assert "w * a=" in hover
    assert "Node heatmap (relative)" in _annotation_text(figure)
    first_nodes = [trace for trace in figure.frames[0].data if trace.name == "neural graph activations"]
    last_nodes = [trace for trace in figure.frames[-1].data if trace.name == "neural graph activations"]
    assert first_nodes and last_nodes
    assert all(not isinstance(trace.marker.color, str) for trace in first_nodes)
    assert any(tuple(first.marker.color) != tuple(last.marker.color) for first, last in zip(first_nodes, last_nodes))
    first_glows = [trace for trace in figure.frames[0].data if trace.name == "neural graph activity glow"]
    last_glows = [trace for trace in figure.frames[-1].data if trace.name == "neural graph activity glow"]
    assert first_glows and len(first_glows) == len(last_glows)
    assert any(
        first.line.width != last.line.width or first.opacity != last.opacity
        for first, last in zip(first_glows, last_glows)
    )
    assert figure.layout.meta["mlektic_neural_evolution"]["activity_glow"]["scale"] == "global"
    assert "Activity glow" in _annotation_text(figure)
    node_edge_legend = next(
        annotation for annotation in figure.layout.annotations if "Node heatmap" in str(annotation.text)
    )
    activity_legend = next(
        annotation for annotation in figure.layout.annotations if "Activity glow" in str(annotation.text)
    )
    assert float(node_edge_legend.y) - float(activity_legend.y) >= 0.10


def test_graph_loss_panel_uses_recorded_objective_and_discloses_perceptual_markers(
    trained_small_network,
):
    model, X, history = trained_small_network
    figure = visualize_nn_graph(
        model,
        X[1],
        history,
        max_frames=4,
        evolution_mode="hybrid",
        interpolation_frames=1,
        show_loss_panel=True,
    )
    metadata = figure.layout.meta["mlektic_neural_evolution"]

    assert metadata["show_loss_panel"] is True
    assert metadata["loss_panel"]["loss_name"] == "BCELoss"
    assert metadata["loss_panel"]["perceptual_markers_are_evaluations"] is False
    assert any(trace.name == "recorded objective curve" for trace in figure.data)
    assert tuple(figure.layout.xaxis2.domain) == pytest.approx((0.08, 0.54))
    assert tuple(figure.layout.xaxis3.domain) == pytest.approx((0.58, 0.94))
    transition_marker = next(
        trace
        for frame in figure.frames
        if str(frame.name).startswith("transition_")
        for trace in frame.data
        if trace.name == "perceptual loss marker"
    )
    assert transition_marker.marker.symbol == "circle-open"


def test_graph_loss_panel_preserves_the_evolving_parameter_readout(
    trained_small_network,
):
    model, X, history = trained_small_network
    figure = visualize_nn_graph(
        model,
        X[1],
        history,
        max_frames=None,
        interpolation_frames=1,
        show_loss_panel=True,
        show_backpropagation=False,
    )
    readouts = [
        next(trace for trace in frame.data if trace.name == "parameter readout")
        for frame in figure.frames
    ]

    assert tuple(figure.layout.yaxis.domain) == pytest.approx((0.40, 1.0))
    expected_parameter_y = (0.96 - 0.40) / (1.0 - 0.40)
    assert all(float(trace.y[0]) == pytest.approx(expected_parameter_y) for trace in readouts)
    assert all(0.0 <= float(trace.y[0]) <= 1.0 for trace in readouts)
    assert readouts[0].text != readouts[-1].text


def test_dropout_graph_prefers_complete_executed_topology():
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.25),
        torch.nn.Linear(3, 1),
    )
    X = torch.randn(8, 2)
    recorder = TorchTrainingRecorder(model)
    model(X)
    recorder.record(1, loss=torch.tensor(1.0))
    history = recorder.to_history()
    recorder.close()

    figure = visualize_nn_graph(model, X[0], history)
    route = figure.layout.meta["mlektic_neural_graph_route"]
    labels = " ".join(str(value) for trace in figure.data for value in (trace.text or []))

    assert route["rendered_view"] == "complete execution blocks"
    assert route["topology_is_preferred_over_misleading_animation"] is True
    assert route["history_animation_applied"] is False
    assert "Dropout" in labels


def test_dense_graph_uses_more_smaller_nodes_for_wide_layers():
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 12),
        torch.nn.ReLU(),
        torch.nn.Linear(12, 3),
    )
    X = torch.randn(4, 8)
    recorder = TorchTrainingRecorder(model)
    model(X)
    recorder.record(1, loss=torch.tensor(1.0))
    history = recorder.to_history()
    recorder.close()

    figure = visualize_nn_graph(model, X[0], history, max_neurons=8, max_frames=1)
    node_traces = [trace for trace in figure.data if trace.name == "neural graph activations"]
    scope = figure.layout.meta["mlektic_neural_evolution"]["dense_scope"]

    assert [len(trace.x) for trace in node_traces] == [8, 8, 3]
    marker_size = float(scope["node_marker_size"])
    assert all(float(trace.marker.size) == marker_size for trace in node_traces)
    assert 9 <= marker_size <= 18
    assert scope["visible_neuron_counts"] == [8, 8, 3]
    assert scope["node_marker_policy"] == "actual pixel spacing with a non-overlap gap"
    y_domain = figure.layout.yaxis.domain or (0.0, 1.0)
    graph_pixel_height = (
        float(figure.layout.height) - float(figure.layout.margin.t) - float(figure.layout.margin.b)
    ) * (float(y_domain[1]) - float(y_domain[0]))
    for trace in node_traces:
        positions = sorted(float(value) for value in trace.y)
        if len(positions) > 1:
            minimum_gap = min(current - previous for previous, current in zip(positions, positions[1:]))
            assert marker_size <= minimum_gap * graph_pixel_height * 0.60


def test_convolutional_graph_renders_every_executed_stage():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 4, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(4),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(4 * 4 * 4, 3),
    )
    X = torch.randn(2, 1, 8, 8)
    recorder = TorchTrainingRecorder(model)
    model(X)
    recorder.record(1, loss=torch.tensor(1.0))
    history = recorder.to_history()
    recorder.close()

    figure = visualize_nn_graph(model, X[0], history, max_neurons=8, max_frames=1)
    route = figure.layout.meta["mlektic_neural_graph_route"]
    labels = " ".join(str(value) for trace in figure.data for value in (trace.text or []))

    assert route["rendered_view"] == "complete execution blocks"
    for module_type in ("Conv2d", "BatchNorm2d", "ReLU", "MaxPool2d", "Flatten", "Linear"):
        assert module_type in labels


def test_classroom_theme_preserves_neural_animation_button_size(trained_small_network):
    model, X, history = trained_small_network
    figure = visualize_nn_graph(
        model,
        X[0],
        history,
        max_frames=2,
        theme="classroom",
    )

    assert figure.layout.updatemenus[0].font.size == 12


def test_graph_rejects_unknown_color_modes(trained_small_network):
    model, X, history = trained_small_network

    with pytest.raises(ValueError, match="node_color_mode"):
        visualize_nn_graph(model, X[0], history, node_color_mode="unknown")
    with pytest.raises(ValueError, match="edge_color_mode"):
        visualize_nn_graph(model, X[0], history, edge_color_mode="unknown")


def test_graph_hybrid_mode_makes_recorded_updates_visually_explicit(
    trained_small_network,
):
    model, X, history = trained_small_network
    figure = visualize_nn_graph(
        model,
        X[1],
        history,
        max_frames=4,
        frame_duration=180,
        evolution_mode="hybrid",
        update_reference="previous",
        update_scale="global",
        top_k_updates=3,
        interpolation_frames=2,
    )
    metadata = figure.layout.meta["mlektic_neural_evolution"]
    names = {trace.name for trace in figure.frames[-1].data}
    final_halos = [trace for trace in figure.frames[-1].data if trace.name == "parameter update halo"]
    transition_context = " ".join(
        str(value)
        for frame in figure.frames
        if str(frame.name).startswith("transition_")
        for trace in frame.data
        if trace.name == "training step readout"
        for value in trace.text
    )
    adjacent_summary = next(
        trace.text[0]
        for frame in figure.frames
        if frame.name == "step_1"
        for trace in frame.data
        if trace.name == "update summary"
    )
    update_summary = next(trace for trace in figure.data if trace.name == "update summary")

    assert metadata["evolution_mode"] == "hybrid"
    assert metadata["update_reference"] == "previous"
    assert metadata["update_scale"] == "global"
    assert metadata["semantic_frames"] == 4
    assert metadata["display_frames"] == 10
    assert metadata["perceptual_frames_per_transition"] == 2
    assert metadata["perceptual_frames_are_optimizer_steps"] is False
    assert len(figure.frames) == 10
    assert len(figure.layout.sliders[0].steps) == 4
    assert figure.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] == 60
    assert "parameter update halo" in names
    assert "absolute parameter or signal" in names
    assert "update summary" in names
    assert sum(trace.opacity > 0.12 for trace in final_halos) <= 3
    assert any(trace.opacity <= 0.12 for trace in final_halos)
    assert "not an optimizer step" in transition_context
    assert "previous displayed checkpoint" in _annotation_text(figure)
    assert "Recorded update diagnostics" in _annotation_text(figure)
    assert update_summary.xaxis == "x2"
    assert update_summary.yaxis == "y2"
    assert 0.0 < figure.layout.shapes[0].y0 < figure.layout.shapes[0].y1 < 0.26
    assert tuple(figure.layout.yaxis.domain) == pytest.approx((0.40, 1.0))
    assert metadata["section_layout"]["network_content_domain"] == pytest.approx([0.40, 0.92])
    assert figure.layout.sliders[0].x == pytest.approx(0.08)
    assert figure.layout.sliders[0].y == pytest.approx(0.375)
    assert figure.layout.margin.b == 75
    activity_legend = next(
        annotation for annotation in figure.layout.annotations if "Activity glow" in str(annotation.text)
    )
    update_legend = next(
        annotation for annotation in figure.layout.annotations if "Update halo" in str(annotation.text)
    )
    graph_top_data = max(
        max(float(value) for value in trace.y) for trace in figure.data if trace.name == "neural graph activations"
    )
    graph_top_in_paper = (
        figure.layout.yaxis.domain[0] + (figure.layout.yaxis.domain[1] - figure.layout.yaxis.domain[0]) * graph_top_data
    )
    minimum_gap = figure.layout.meta["mlektic_neural_evolution"]["section_layout"]["minimum_update_to_network_gap"]
    assert graph_top_in_paper <= update_legend.y - minimum_gap
    assert update_legend.y < activity_legend.y
    parameter_readout = next(trace for trace in figure.data if trace.name == "parameter readout")
    phase_definition = next(
        annotation for annotation in figure.layout.annotations if "Feed forward" in str(annotation.text)
    )
    parameter_readout_in_paper = (
        figure.layout.yaxis.domain[0]
        + (figure.layout.yaxis.domain[1] - figure.layout.yaxis.domain[0]) * parameter_readout.y[0]
    )
    assert parameter_readout_in_paper > phase_definition.y
    assert "n/a" not in adjacent_summary


def test_graph_updates_mode_can_use_initial_and_frame_normalized_references(
    trained_small_network,
):
    model, X, history = trained_small_network
    figure = visualize_nn(
        model,
        X[0],
        history=history,
        view="graph",
        evolution_mode="updates",
        update_reference="initial",
        update_scale="frame",
        max_frames=4,
    )
    metadata = figure.layout.meta["mlektic_neural_evolution"]
    base_edges = [trace for trace in figure.frames[-1].data if trace.name == "absolute parameter or signal"]

    assert metadata["evolution_mode"] == "updates"
    assert metadata["update_reference"] == "initial"
    assert metadata["update_scale"] == "frame"
    assert metadata["show_update_panel"] is True
    assert base_edges
    assert {trace.line.color for trace in base_edges} == {"#353841"}
    assert "per frame" in _annotation_text(figure)
    assert "initial checkpoint" in _annotation_text(figure)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("evolution_mode", "unknown"),
        ("update_reference", "unknown"),
        ("update_scale", "unknown"),
        ("top_k_updates", 0),
        ("interpolation_frames", -1),
        ("interpolation_frames", 21),
    ],
)
def test_graph_rejects_invalid_update_evolution_controls(
    trained_small_network,
    argument,
    value,
):
    model, X, history = trained_small_network

    with pytest.raises(ValueError, match=argument):
        visualize_nn_graph(model, X[0], history, **{argument: value})


def test_graph_distinguishes_exact_relu_zero_from_rounded_values():
    model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.ReLU())
    with torch.no_grad():
        model[0].weight.fill_(-1.0)
        model[0].bias.zero_()
    recorder = TorchTrainingRecorder(model)
    output = model(torch.tensor([[1.0, 1.0]]))
    recorder.record(0, loss=output.sum())
    history = recorder.to_history()
    recorder.close()

    figure = visualize_nn_graph(model, torch.tensor([1.0, 1.0]), history)
    hover = " ".join(str(value) for trace in figure.frames[0].data for value in (trace.customdata or []))

    assert "activation=ReLU" in hover
    assert "numerical output=0 (ReLU inactive)" in hover


def test_training_separates_loss_and_metrics_and_decimates_frames(trained_small_network):
    _model, _X, history = trained_small_network
    figure = visualize_nn_training(history, max_frames=3)

    assert isinstance(figure, go.Figure)
    assert len(figure.frames) == 3
    assert {"accuracy", "precision", "recall"}.issubset({trace.name for trace in figure.data})
    assert figure.layout.yaxis.title.text == r"$\mathcal{L}$"
    assert figure.layout.yaxis2.title.text == "accuracy"
    assert figure.layout.yaxis3.title.text == "precision"
    assert figure.layout.yaxis4.title.text == "recall"
    assert figure.layout.updatemenus[0].font.color == "#15171b"
    assert figure.layout.updatemenus[0].x == 0.0
    assert figure.layout.xaxis.domain != figure.layout.xaxis2.domain
    assert figure.layout.yaxis.domain != figure.layout.yaxis3.domain
    summary = max(figure.layout.annotations, key=lambda annotation: annotation.y)
    assert summary.y >= 1.15


def test_neural_lesson_format_never_adds_concept_stage_filters(trained_small_network):
    _model, _X, history = trained_small_network
    figure = visualize_nn_training(history, max_frames=3, format="lesson")

    labels = [button.label for menu in figure.layout.updatemenus for button in menu.buttons]
    assert labels == ["Play", "Pause"]
    assert not {"1 Data", "2 Model", "3 Objective", "4 Complete"}.intersection(labels)


def test_training_keeps_four_panels_when_metrics_are_missing():
    history = {"steps": np.arange(3), "loss": np.asarray([1.0, 0.8, 0.6]), "metrics": {}}
    figure = visualize_nn_training(history)
    text = _annotation_text(figure)
    frame_text = " ".join(str(annotation.text) for annotation in figure.frames[0].layout.annotations)

    assert figure.layout.xaxis4 is not None
    assert figure.layout.yaxis4 is not None
    assert text.count("Metric not recorded") == 3
    assert text.count("Pass predictions and targets to recorder.record()") == 3
    assert frame_text.count("Pass predictions and targets to recorder.record()") == 3


def test_weights_are_latex_matrices_with_dimensions_and_ellipsis(trained_small_network):
    _model, _X, history = trained_small_network
    figure = visualize_nn_weights(history, parameter="0.weight", max_rows=2, max_cols=1, max_frames=2)
    text = _annotation_text(figure)

    assert isinstance(figure, go.Figure)
    assert len(figure.frames) == 2
    assert r"\begin{bmatrix}" in text
    assert r"\mathbb{R}^{4 \times 2}" in text
    assert r"\cdots" in text or r"\vdots" in text
    assert figure.layout.updatemenus[0].x == 0.0
    assert figure.layout.title.x == 0.5
    assert figure.layout.title.xanchor == "center"


def test_weights_keep_tall_matrix_clear_of_matching_bias(trained_small_network):
    _model, _X, history = trained_small_network
    figure = visualize_nn_weights(history, max_frames=2)

    weight = next(
        annotation for annotation in figure.layout.annotations if str(annotation.text).startswith(r"$W^{(1)}_t=")
    )
    bias = next(
        annotation
        for annotation in figure.layout.annotations
        if str(annotation.text).startswith(r"$\mathbf{b}^{(1)}_t=")
    )

    assert float(weight.y) - float(bias.y) >= 0.30


def test_large_weights_reserve_an_independent_omission_row():
    model = torch.nn.Sequential(*[torch.nn.Linear(4, 4) for _ in range(7)])
    history = _two_frame_parameter_history(model)
    figure = visualize_nn_weights(
        history,
        max_rows=3,
        max_cols=4,
        max_parameters=6,
        max_frames=2,
        theme="academic",
        size="wide",
    )

    for annotations in [figure.layout.annotations, figure.frames[-1].layout.annotations]:
        notice = next(
            annotation
            for annotation in annotations
            if "intermediate parameter tensors" in str(annotation.text)
        )
        matrices = [annotation for annotation in annotations if str(annotation.text).startswith("$W")]
        above = min((matrix for matrix in matrices if matrix.y > notice.y), key=lambda matrix: matrix.y)
        below = max((matrix for matrix in matrices if matrix.y < notice.y), key=lambda matrix: matrix.y)

        assert float(above.y) - float(notice.y) >= 0.12
        assert float(notice.y) - float(below.y) >= 0.13
    metadata = figure.layout.meta["mlektic_neural_weights"]
    assert metadata["omission_notice_layout"] == "reserved-matrix-height-row"


def test_activations_and_forward_math_evolve_over_time(trained_small_network):
    model, X, history = trained_small_network
    activations = visualize_nn(model, X[:1], history=history, view="activations", max_frames=3)
    explanation = explain_nn_prediction(model, X[0], history=history, max_frames=3)
    replay = explain_nn_prediction(
        model,
        X[0],
        history=history,
        max_frames=3,
        parameter_state="training_replay",
    )
    report = explain_nn_prediction(model, X[0], history=history, format="report")
    reduced = explain_nn_prediction(model, X[0], history=history, reduced_motion=True)

    assert isinstance(activations, go.Figure)
    assert len(activations.frames) == 3
    assert r"\mathbf{a}^{(\ell)}" in _annotation_text(activations)
    assert isinstance(explanation, go.Figure)
    assert len(explanation.frames) == 0
    assert len(replay.frames) == 3
    assert "prediction" in explanation.layout.title.text
    assert replay.layout.title.text == "Neural training replay: parameter and signal evolution"
    substitution_annotations = explanation.layout.updatemenus[0].buttons[1].args[0]["annotations"]
    output_annotations = explanation.layout.updatemenus[0].buttons[2].args[0]["annotations"]
    substitution_text = " ".join(str(annotation["text"]) for annotation in substitution_annotations)
    output_text = " ".join(str(annotation["text"]) for annotation in output_annotations)
    output_card = next(
        annotation
        for annotation in output_annotations
        if str(annotation["text"]).startswith(r"$\begin{aligned}\hat{\mathbf{y}}")
    )
    first_linear_block = next(
        annotation
        for annotation in output_annotations
        if str(annotation["text"]).startswith(r"$\underbrace{\text{0: Linear}")
    )
    first_activation_block = next(
        annotation
        for annotation in output_annotations
        if str(annotation["text"]).startswith(r"$\underbrace{\text{1: Tanh}")
    )
    second_linear_block = next(
        annotation
        for annotation in output_annotations
        if str(annotation["text"]).startswith(r"$\underbrace{\text{2: Linear}")
    )
    assert r"\hat{\mathbf{y}}=f_\theta" in substitution_text
    assert "Input" not in _annotation_text(explanation)
    assert not explanation.layout.shapes
    assert "Substitution" in substitution_text
    assert "Output" in output_text
    assert r"\hat{k}&=\mathbb{1}" in output_text
    assert [button.label for button in explanation.layout.updatemenus[0].buttons] == [
        "Input",
        "Substitution",
        "Output",
        "Reset",
    ]
    assert explanation.layout.updatemenus[0].x == pytest.approx(0.02)
    assert explanation.layout.updatemenus[0].xanchor == "left"
    assert explanation.layout.title.x == 0.5
    assert explanation.layout.meta["mlektic_neural_prediction"]["parameter_state"] == "final"
    assert explanation.layout.meta["mlektic_neural_prediction"]["section_layout"]["initial_stage"] == "Reset"
    assert replay.layout.meta["mlektic_neural_prediction"]["training_replay"] is True
    assert replay.layout.meta["mlektic_neural_prediction"]["stages"] == []
    assert replay.layout.meta["mlektic_neural_prediction"]["prediction_cards_visible"] is False
    assert replay.layout.meta["mlektic_neural_prediction"]["standalone_training_view"] is True
    assert replay.layout.updatemenus[0].x == pytest.approx(0.02)
    replay_text = _annotation_text(replay)
    assert "Fixed query" not in replay_text
    assert "Recorded checkpoint" not in replay_text
    assert "<b>Input</b>" not in replay_text
    assert "<b>Substitution</b>" not in replay_text
    assert "<b>Output</b>" not in replay_text
    assert not any(annotation.bgcolor and annotation.xref == "paper" for annotation in replay.layout.annotations)
    assert all(
        not any(annotation.bgcolor and annotation.xref == "paper" for annotation in frame.layout.annotations)
        for frame in replay.frames
    )
    assert output_card["xanchor"] == "center"
    output_bounds = explanation.layout.meta["mlektic_neural_prediction"]["section_layout"]["summary_cards"]["Output"]
    assert output_card["x"] == pytest.approx((output_bounds[0] + output_bounds[1]) / 2)
    substitution_card = next(
        annotation for annotation in output_annotations if r"z^{(1)}_1&=" in str(annotation["text"])
    )
    assert ")(" in str(substitution_card["text"])
    assert output_card.get("borderwidth") is None
    assert substitution_card.get("borderwidth") is None
    assert r"\phantom{00000000}" not in str(substitution_card["text"])
    assert r"\\&\quad+" in str(substitution_card["text"])
    assert r"\\\hat{k}&=" in str(output_card["text"])
    assert r"\phantom" not in str(output_card["text"])
    first_linear_line_steps = str(first_linear_block["text"]).count(r"\\")
    first_linear_to_activation = float(first_linear_block["y"]) - float(first_activation_block["y"])
    assert first_linear_to_activation - first_linear_line_steps * 0.034 >= 0.08
    assert first_linear_block["font"]["size"] == 14
    assert first_activation_block["font"]["size"] == 14
    assert explanation.layout.meta["mlektic_neural_prediction"]["line_aware_vertical_spacing"] is True
    assert r"\\&\quad+" in str(second_linear_block["text"])
    stage_shapes = [button.args[0]["shapes"] for button in explanation.layout.updatemenus[0].buttons]
    assert [len(shapes) for shapes in stage_shapes] == [1, 2, 3, 0]
    assert all(shape["xref"] == "paper" and shape["yref"] == "paper" for shapes in stage_shapes[:3] for shape in shapes)
    assert not any(
        str(annotation["text"]).startswith(r"$\hat{\mathbf{y}}=\begin{bmatrix}") for annotation in output_annotations
    )
    for static_figure in (report, reduced):
        static_text = _annotation_text(static_figure)
        static_metadata = static_figure.layout.meta["mlektic_neural_prediction"]
        assert not static_figure.layout.updatemenus
        assert "Input" in static_text
        assert "Substitution" in static_text
        assert "Output" in static_text
        assert r"z^{(1)}_{1}" in static_text
        assert r"\hat{\mathbf{y}}&=\begin{bmatrix}" in static_text
        assert static_metadata["static_stage"] == "Output"
        assert static_metadata["static_contains_complete_forward_pass"] is True
        assert len(static_figure.layout.shapes) == 3


def test_short_regression_prediction_separates_input_and_contains_math_cards():
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 10),
        torch.nn.Tanh(),
        torch.nn.Linear(10, 1),
    )
    figure = explain_nn_prediction(model, torch.tensor([-0.47, 0.185]))
    output_annotations = figure.layout.updatemenus[0].buttons[2].args[0]["annotations"]
    input_linear = next(
        annotation for annotation in output_annotations if str(annotation["text"]) == "<b>0</b><br>Linear"
    )
    substitution_card = next(
        annotation for annotation in output_annotations if r"z^{(1)}_1&=" in str(annotation["text"])
    )
    output_card = next(
        annotation
        for annotation in output_annotations
        if str(annotation["text"]).startswith(r"$\begin{aligned}\hat{\mathbf{y}}")
    )

    assert 0.06 <= float(input_linear["y"]) <= 0.56
    assert substitution_card["xanchor"] == "center"
    assert output_card["xanchor"] == "center"
    output_shapes = figure.layout.updatemenus[0].buttons[2].args[0]["shapes"]
    assert len(output_shapes) == 3
    substitution_shape = output_shapes[1]
    substitution_bounds = figure.layout.meta["mlektic_neural_prediction"]["section_layout"]["summary_cards"][
        "Substitution"
    ]
    assert substitution_shape["x0"] == pytest.approx(substitution_bounds[0])
    assert substitution_shape["x1"] == pytest.approx(substitution_bounds[1])


@pytest.mark.parametrize(
    ("theme", "size"),
    [
        ("classic", "default"),
        ("academic", "wide"),
        ("classroom", "classroom"),
        ("accessible", "compact"),
    ],
)
def test_prediction_section_contract_is_invariant_across_public_visual_variants(theme, size):
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 4),
        torch.nn.Tanh(),
        torch.nn.Linear(4, 1),
        torch.nn.Sigmoid(),
    )
    figure = explain_nn_prediction(model, torch.tensor([0.0, 1.0]), theme=theme, size=size)
    metadata = figure.layout.meta["mlektic_neural_prediction"]["section_layout"]
    expected_bounds = metadata["summary_cards"]

    assert metadata["initial_stage"] == "Reset"
    assert not figure.layout.shapes
    assert figure.layout.updatemenus[0].y == pytest.approx(1.02)
    reset = figure.layout.updatemenus[0].buttons[3].args[0]
    assert not reset["shapes"]
    assert "Input" not in " ".join(str(annotation["text"]) for annotation in reset["annotations"])
    for button_index, visible_count in enumerate((1, 2, 3)):
        stage = figure.layout.updatemenus[0].buttons[button_index].args[0]
        assert len(stage["shapes"]) == visible_count
        for shape, heading in zip(stage["shapes"], ("Input", "Substitution", "Output")):
            assert (shape["x0"], shape["x1"], shape["y0"], shape["y1"]) == pytest.approx(expected_bounds[heading])
    model_formula = reset["annotations"][0]
    assert model_formula["y"] == pytest.approx(0.955)


def test_prediction_vectors_use_width_aware_truncation_for_high_dimension_and_precision():
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 24),
        torch.nn.GELU(),
        torch.nn.Linear(24, 5),
    )
    figure = explain_nn_prediction(
        model,
        torch.linspace(-1.0, 1.0, 32),
        dec=12,
        max_neurons_math=20,
        size="compact",
    )
    substitution = figure.layout.updatemenus[0].buttons[1].args[0]["annotations"]
    derivations = [
        str(annotation["text"]) for annotation in substitution if str(annotation["text"]).startswith(r"$\underbrace")
    ]

    assert derivations
    assert all(r"\cdots" in derivation for derivation in derivations)
    assert all(len(re.findall(r"-?\d+\.\d{12}", derivation)) <= 3 for derivation in derivations)


def test_large_prediction_summary_cards_are_bounded_without_shrinking_detail():
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, 10),
    )
    figure = explain_nn_prediction(model, torch.randn(128), max_neurons_math=6)
    output_annotations = figure.layout.updatemenus[0].buttons[2].args[0]["annotations"]
    input_card = next(
        annotation
        for annotation in output_annotations
        if str(annotation["text"]).startswith(r"$\mathbf{x}=\begin{bmatrix}")
    )
    substitution_card = next(
        annotation for annotation in output_annotations if r"z^{(1)}_1&=" in str(annotation["text"])
    )
    output_card = next(
        annotation
        for annotation in output_annotations
        if str(annotation["text"]).startswith(r"$\begin{aligned}\hat{\mathbf{y}}")
    )
    detailed_output = next(
        annotation
        for annotation in output_annotations
        if str(annotation["text"]).startswith(r"$\underbrace{\text{2: Linear}")
    )

    assert str(input_card["text"]).count("&") <= 2
    assert r"\cdots" in str(substitution_card["text"])
    assert r"\phantom" not in str(output_card["text"])
    assert output_card["xanchor"] == "center"
    output_bounds = figure.layout.meta["mlektic_neural_prediction"]["section_layout"]["summary_cards"]["Output"]
    assert output_card["x"] == pytest.approx((output_bounds[0] + output_bounds[1]) / 2)
    assert r"\begin{aligned}" in str(detailed_output["text"])
    assert r"\\&=" in str(detailed_output["text"])
    assert r"\cdots" in str(detailed_output["text"])


def test_crowded_forward_math_uses_bounded_blocks_and_compact_typography():
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 4),
        torch.nn.Tanh(),
        torch.nn.Linear(4, 4),
        torch.nn.Tanh(),
        torch.nn.Linear(4, 4),
        torch.nn.Tanh(),
    )
    figure = explain_nn_prediction(
        model,
        torch.tensor([0.0, 1.0]),
        max_layers_math=6,
    )
    output_annotations = figure.layout.updatemenus[0].buttons[2].args[0]["annotations"]
    blocks = [annotation for annotation in output_annotations if str(annotation["text"]).startswith(r"$\underbrace")]

    assert len(blocks) == 6
    assert all(str(block["text"]).count(r"\\") + 1 <= 4 for block in blocks)
    assert all(block["font"]["size"] == 13 for block in blocks)
    assert all(float(current["y"]) > float(following["y"]) for current, following in zip(blocks, blocks[1:]))


def test_backpropagation_view_teaches_chain_rule_with_recorded_gradient_norms(
    trained_small_network,
):
    model, X, history = trained_small_network
    figure = visualize_nn_backpropagation(model, history, input_sample=X[:1], max_frames=3)
    metadata = figure.layout.meta["mlektic_neural_backpropagation"]

    assert len(figure.frames) == 3
    assert "Backward" in _annotation_text(figure)
    assert "optimizer converts gradients into updates" in _annotation_text(figure)
    assert metadata["gradient_quantity"] == "recorded per-layer parameter-gradient L2 norm"
    assert metadata["update_quantity"] == "adjacent recorded per-layer parameter-change L2 norm"
    assert "relative parameter-update norm" in metadata["numeric_layer_readout"]
    assert any(trace.name == "layerwise gradient and update values" for trace in figure.frames[-1].data)
    assert metadata["loss"] == "BCELoss"
    assert metadata["optimizer"] == "SGD"


def test_large_backpropagation_separates_scope_disclosure_and_layer_readouts():
    layers = []
    for _ in range(7):
        layers.extend([torch.nn.Linear(4, 4), torch.nn.ReLU()])
    model = torch.nn.Sequential(*layers)
    history = _two_frame_parameter_history(model)
    figure = visualize_nn_backpropagation(
        model,
        history,
        input_sample=torch.randn(1, 4),
        max_layers=6,
        max_frames=2,
        theme="classroom",
        size="wide",
    )
    metadata = figure.layout.meta["mlektic_neural_backpropagation"]
    scope = next(
        annotation
        for annotation in figure.layout.annotations
        if "complete recorded history retained" in str(annotation.text)
    )
    readout = next(
        trace
        for trace in figure.data
        if trace.name == "layerwise gradient and update values"
    )

    assert metadata["omitted_layer_count"] == 2
    assert metadata["crowded_readout_layout"] == "alternating-rows"
    assert metadata["scope_disclosure_row"] == "lower-caption"
    assert float(scope.y) < 0.0
    assert len(set(float(value) for value in readout.y)) == 2
    assert readout.textfont.size == 12


def test_loss_landscape_is_disclosed_exact_two_direction_slice(trained_small_network):
    model, X, history = trained_small_network
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    figure = visualize_nn_loss_landscape(
        model,
        X,
        y,
        torch.nn.BCELoss(),
        history,
        grid_size=9,
        max_frames=3,
    )
    metadata = figure.layout.meta["mlektic_neural_loss_slice"]

    assert len(figure.frames) == 3
    assert figure.data[0].type == "surface"
    assert np.isfinite(np.asarray(figure.data[0].z, dtype=float)).all()
    assert metadata["trajectory_is_projected"] is True
    assert metadata["surface_is_global_loss_landscape"] is False
    assert metadata["convergence_claimed"] is False
    final_path, final_marker = figure.frames[-1].data
    assert metadata["last_recorded_loss"] == pytest.approx(float(final_path.z[-1]))
    assert float(final_marker.z[0]) > metadata["last_recorded_loss"]
    assert metadata["checkpoint_marker_z_offset_is_visual_only"] is True
    assert metadata["loss"] == "BCELoss"
    assert len(figure.data) == 3
    assert len(figure.data[1].x) == 1
    assert not any("checkpoint" in str(annotation.text).lower() for annotation in figure.layout.annotations)
    previous_path_length = 0
    for frame_index, frame in enumerate(figure.frames):
        path, checkpoint = frame.data
        checkpoint_annotations = [
            annotation for annotation in frame.layout.annotations if "checkpoint" in str(annotation.text).lower()
        ]
        assert len(path.x) >= 1
        assert len(path.x) >= previous_path_length
        previous_path_length = len(path.x)
        assert checkpoint.mode == "markers"
        assert checkpoint.marker.size == 10
        if frame_index < len(figure.frames) - 1:
            assert not checkpoint_annotations
        else:
            assert len(checkpoint_annotations) == 1
            final_annotation = checkpoint_annotations[0]
            assert str(final_annotation.text).startswith("<b>Final checkpoint")
            assert final_annotation.font.size == 15
            assert final_annotation.x == pytest.approx(0.70)
            assert final_annotation.y == pytest.approx(0.80)


def test_math_font_scale_enlarges_latex_without_changing_default(trained_small_network):
    model, X, history = trained_small_network
    default = visualize_nn_weights(history, parameter="0.weight", max_frames=1)
    enlarged = visualize_nn_weights(
        history,
        parameter="0.weight",
        max_frames=1,
        math_font_scale=1.5,
    )
    default_sizes = [annotation.font.size for annotation in default.layout.annotations]
    enlarged_sizes = [annotation.font.size for annotation in enlarged.layout.annotations]

    assert max(enlarged_sizes) > max(default_sizes)
    with pytest.raises(ValueError, match="math_font_scale"):
        visualize_nn_weights(history, math_font_scale=2.1)


def test_large_forward_view_summarizes_layers_instead_of_failing():
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 8),
        torch.nn.GELU(),
        torch.nn.Linear(8, 4),
        torch.nn.Tanh(),
        torch.nn.Linear(4, 2),
    )
    figure = explain_nn_prediction(model, torch.tensor([0.2, -0.1, 0.4]), max_layers_math=4)
    substitution_annotations = figure.layout.updatemenus[0].buttons[1].args[0]["annotations"]
    substitution_text = " ".join(str(annotation["text"]) for annotation in substitution_annotations)

    assert isinstance(figure, go.Figure)
    assert r"\vdots" in substitution_text
    assert r"\mathbf{z}^{(4)}" in substitution_text
    assert r"\begin{bmatrix}" in substitution_text


def test_prediction_training_replay_requires_complete_history():
    model = torch.nn.Sequential(torch.nn.Linear(2, 1))

    with pytest.raises(ValueError, match="complete recorded parameter history"):
        explain_nn_prediction(
            model,
            torch.tensor([0.2, -0.1]),
            parameter_state="training_replay",
        )
    with pytest.raises(ValueError, match="parameter_state"):
        explain_nn_prediction(model, torch.tensor([0.2, -0.1]), parameter_state="unknown")


def test_html_report_contains_complete_taxonomy_and_exports(trained_small_network, tmp_path):
    model, X, history = trained_small_network
    html = build_nn_math_report(model, X[:1], history=history, title="XOR mathematics")
    destination = export_nn_math_report(
        model,
        X[:1],
        history=history,
        path=tmp_path / "xor-report.html",
        title="XOR mathematics",
    )

    assert html.startswith("<!doctype html>")
    assert html.endswith("</html>")
    assert "mathjax@3" in html.lower()
    assert "Training configuration and evolution" in html
    assert "optimizer" in html.lower() and "SGD" in html
    assert r"\mathbf{a}^{(1)}=\tanh(\mathbf{z}^{(1)})" in html
    assert r"\mathbb{R}^{4 \times 2}" in html
    assert destination.read_text(encoding="utf-8") == html
    assert destination.stat().st_size < 200_000


def test_recorder_skips_oversized_tensors_without_losing_norms():
    model = torch.nn.Sequential(torch.nn.Linear(20, 20))
    recorder = TorchTrainingRecorder(model, max_tensor_elements=8, capture_activations=False)
    output = model(torch.ones(1, 20)).sum()
    output.backward()
    recorder.record(0, loss=output)
    history = recorder.to_history()
    recorder.close()

    assert history["parameters"] == {}
    assert np.isfinite(history["parameter_norms"]["0.weight"][0])


def test_recorder_keeps_initial_gradient_frame_aligned():
    model = torch.nn.Sequential(torch.nn.Linear(2, 1))
    recorder = TorchTrainingRecorder(model, capture_activations=False)
    initial_loss = model(torch.ones(1, 2)).sum()
    recorder.record(0, loss=initial_loss)
    initial_loss.backward()
    recorder.record(1, loss=initial_loss)
    history = recorder.to_history()
    recorder.close()

    assert len(history["gradients"]["0.weight"]) == 2
    assert np.allclose(history["gradients"]["0.weight"][0], 0.0)
    assert not np.allclose(history["gradients"]["0.weight"][1], 0.0)


def test_recorder_infers_three_classification_metrics():
    model = torch.nn.Sequential(torch.nn.Linear(2, 3))
    recorder = TorchTrainingRecorder(model, capture_activations=False)
    predictions = torch.tensor([[4.0, 1.0, 0.0], [0.0, 3.0, 1.0], [0.0, 2.0, 3.0]])
    targets = torch.tensor([0, 1, 1])
    recorder.record(0, predictions=predictions, targets=targets, task="classification")
    history = recorder.to_history()
    recorder.close()

    assert list(history["metrics"]) == ["accuracy", "precision", "recall"]
    assert history["metrics"]["accuracy"][0] == pytest.approx(2 / 3)


def test_hyperparameter_view_covers_model_optimizer_objective_and_scheduler():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8, bias=False),
        torch.nn.BatchNorm1d(8, eps=1e-4, momentum=0.2),
        torch.nn.LeakyReLU(negative_slope=0.15),
        torch.nn.Dropout(p=0.25),
        torch.nn.Linear(8, 3),
    )
    first_group = list(model[0].parameters())
    second_group = [parameter for name, parameter in model.named_parameters() if not name.startswith("0.")]
    optimizer = torch.optim.Adam(
        [
            {"params": first_group, "lr": 0.001},
            {"params": second_group, "lr": 0.002},
        ],
        lr=0.002,
        betas=(0.8, 0.95),
        weight_decay=0.01,
    )
    objective = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.4)

    figure = visualize_nn_hyperparameters(
        model,
        optimizer=optimizer,
        loss_fn=objective,
        scheduler=scheduler,
        theme="academic",
    )
    routed = visualize_nn(
        model,
        view="hyperparameters",
        optimizer=optimizer,
        loss_fn=objective,
        scheduler=scheduler,
    )
    metadata = figure.layout.meta["mlektic_neural_hyperparameters"]
    components = metadata["components"]
    optimizer_components = [component for component in components if component["scope"] == "optimizer"]
    component_names = {(component["scope"], component["type"]) for component in components}
    names_by_component = {
        (component["scope"], component["type"]): {item["name"] for item in component["hyperparameters"]}
        for component in components
    }

    assert metadata["coverage"].startswith("all detected effective configuration values")
    assert metadata["generic_definition_count"] == 0
    assert metadata["specialized_definition_count"] == metadata["hyperparameter_count"]
    assert metadata["hyperparameter_count"] == sum(len(component["hyperparameters"]) for component in components)
    assert {("optimizer", "Adam"), ("objective", "CrossEntropyLoss"), ("scheduler", "StepLR")} <= component_names
    assert len(optimizer_components) == len(optimizer.param_groups) == 2
    assert [component["hyperparameters"][0]["value"] for component in optimizer_components] == [
        "0.001",
        "0.002",
    ]
    assert {"in_features", "out_features", "bias"} <= names_by_component[("module", "Linear")]
    assert {"num_features", "eps", "momentum", "affine", "track_running_stats"} <= names_by_component[
        ("module", "BatchNorm1d")
    ]
    # Current PyTorch exposes BatchNorm bias as an explicit effective option.
    assert "bias" in names_by_component[("module", "BatchNorm1d")]
    assert {name for name in optimizer.param_groups[0] if name != "params"} == names_by_component[("optimizer", "Adam")]
    assert {"weight", "ignore_index", "reduction", "label_smoothing"} == names_by_component[
        ("objective", "CrossEntropyLoss")
    ]
    assert {"step_size", "gamma", "last_epoch"} == names_by_component[("scheduler", "StepLR")]
    assert (
        routed.layout.meta["mlektic_neural_hyperparameters"]["hyperparameter_count"] == metadata["hyperparameter_count"]
    )
    assert figure.layout.height >= metadata["content_min_height"]


def test_recorder_retains_scheduler_configuration_for_hyperparameter_lessons():
    model = torch.nn.Sequential(torch.nn.Linear(2, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.3)
    recorder = TorchTrainingRecorder(
        model,
        optimizer=optimizer,
        loss_fn=torch.nn.MSELoss(),
        scheduler=scheduler,
        capture_activations=False,
    )
    recorder.record(0, loss=0.0)
    history = recorder.to_history()
    recorder.close()

    assert history["training_config"]["scheduler"] == "StepLR"
    assert history["training_config"]["scheduler_hyperparameters"]["step_size"] == 4
    assert history["training_config"]["scheduler_hyperparameters"]["gamma"] == pytest.approx(0.3)


def test_graph_controls_keep_a_reserved_gap_above_training_step(trained_small_network):
    model, X, history = trained_small_network
    figure = visualize_nn_graph(model, X[1], history, show_loss_panel=False, show_update_panel=False)

    assert figure.layout.updatemenus[0].y == pytest.approx(-0.02)
