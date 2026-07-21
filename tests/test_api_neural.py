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
    visualize_nn_graph,
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


def test_graph_animates_stable_weight_heatmap_and_backprop_overlay(trained_small_network):
    model, X, history = trained_small_network
    figure = visualize_nn_graph(model, X[0], history, max_frames=3)

    assert isinstance(figure, go.Figure)
    assert len(figure.frames) == 3
    first_hover = " ".join(str(value) for trace in figure.frames[0].data for value in (trace.customdata or []))
    last_hover = " ".join(str(value) for trace in figure.frames[-1].data for value in (trace.customdata or []))
    assert "Weight evolution" in first_hover
    assert "Backpropagation" in first_hover
    assert "W[1,:]=" in first_hover
    assert "grad W[1,:]=" in first_hover
    assert r"\begin" not in first_hover
    assert f"w[1,1]={model[0].weight[0, 0].item():.3f}" in last_hover
    first_parameter_text = next(
        trace.text[0] for trace in figure.frames[0].data if trace.name == "parameter readout"
    )
    last_parameter_text = next(
        trace.text[0] for trace in figure.frames[-1].data if trace.name == "parameter readout"
    )
    final_step_text = next(
        trace.text[0] for trace in figure.frames[-1].data if trace.name == "training step readout"
    )
    assert first_parameter_text != last_parameter_text
    assert "final weights" in final_step_text
    assert "Feed forward" in _annotation_text(figure)
    assert "Backpropagation" in _annotation_text(figure)
    assert "Node heatmap (exact)" in _annotation_text(figure)
    assert r"a_j^{(\ell)}" in _annotation_text(figure)
    assert r"\mathbb{R}" in _annotation_text(figure)
    first_node_colors = [tuple(trace.marker.color) for trace in figure.frames[0].data if trace.mode == "markers"]
    last_node_colors = [tuple(trace.marker.color) for trace in figure.frames[-1].data if trace.mode == "markers"]
    assert any(first != last for first, last in zip(first_node_colors[1:], last_node_colors[1:]))
    assert figure.data[-2].marker.colorbar.title.text == r"$w_{ji}^{(\ell)}$"
    assert figure.data[-1].marker.colorbar.title.text == r"$a_j^{(\ell)}$"
    assert figure.data[-1].marker.cmin < 0.0
    assert figure.data[-1].marker.cmax > 0.0
    assert figure.data[-1].marker.showscale is True
    assert figure.layout.updatemenus[0].buttons[0].args[1]["frame"]["redraw"] is False
    assert all(" F" not in step.label and " B" not in step.label for step in figure.layout.sliders[0].steps)


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
    )
    hover = " ".join(
        str(value) for trace in figure.frames[0].data for value in (trace.customdata or [])
    )

    assert figure.data[-2].marker.colorbar.title.text == r"$w_{ji}^{(\ell)}a_i^{(\ell-1)}$"
    assert figure.data[-1].marker.colorbar.title.text == r"$\widetilde{a}_j^{(\ell)}$"
    assert figure.data[-1].marker.cmin == 0.0
    assert figure.data[-1].marker.cmax == 1.0
    assert "Forward signal" in hover
    assert "w * a=" in hover
    assert "Node heatmap (relative)" in _annotation_text(figure)


def test_graph_rejects_unknown_color_modes(trained_small_network):
    model, X, history = trained_small_network

    with pytest.raises(ValueError, match="node_color_mode"):
        visualize_nn_graph(model, X[0], history, node_color_mode="unknown")
    with pytest.raises(ValueError, match="edge_color_mode"):
        visualize_nn_graph(model, X[0], history, edge_color_mode="unknown")


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
    assert figure.layout.title.x == 0.16


def test_activations_and_forward_math_evolve_over_time(trained_small_network):
    model, X, history = trained_small_network
    activations = visualize_nn(model, X[:1], history=history, view="activations", max_frames=3)
    explanation = explain_nn_prediction(model, X[0], history=history, max_frames=3)

    assert isinstance(activations, go.Figure)
    assert len(activations.frames) == 3
    assert r"\mathbf{a}^{(\ell)}" in _annotation_text(activations)
    assert isinstance(explanation, go.Figure)
    assert len(explanation.frames) == 3
    assert "Forward" in explanation.layout.title.text
    assert r"\hat{\mathbf{y}}=f_\theta" in _annotation_text(explanation)
    assert explanation.layout.updatemenus[0].x == 0.0
    assert explanation.layout.title.x == 0.16


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

    assert isinstance(figure, go.Figure)
    assert r"\vdots" in _annotation_text(figure)
    assert r"z^{(4)}_{1}" in _annotation_text(figure)


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
