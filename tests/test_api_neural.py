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
        accuracy = ((prediction >= 0.5) == y.bool()).float().mean()
        recorder.record(step, loss=loss, metrics={"accuracy": accuracy})
        optimizer.step()
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


def test_graph_animates_forward_and_backprop_with_exact_values(trained_small_network):
    model, X, history = trained_small_network
    figure = visualize_nn_graph(model, X[0], history, max_frames=3)

    assert isinstance(figure, go.Figure)
    assert len(figure.frames) == 6
    assert [frame.name.split("-")[-1] for frame in figure.frames[:2]] == ["forward", "backward"]
    forward_hover = " ".join(str(value) for trace in figure.frames[0].data for value in (trace.customdata or []))
    backward_hover = " ".join(str(value) for trace in figure.frames[1].data for value in (trace.customdata or []))
    assert "Feed forward" in forward_hover
    assert "Backpropagation" in backward_hover
    assert r"W_{1,:}" in forward_hover
    assert r"\partial\mathcal{L}" in backward_hover
    assert r"\nabla W_{1,:}" in backward_hover
    assert r"\mathbb{R}" in _annotation_text(figure)


def test_training_separates_loss_and_metrics_and_decimates_frames(trained_small_network):
    _model, _X, history = trained_small_network
    figure = visualize_nn_training(history, max_frames=3)

    assert isinstance(figure, go.Figure)
    assert len(figure.frames) == 3
    assert "accuracy" in [trace.name for trace in figure.data]
    assert figure.layout.yaxis.title.text == r"$\mathcal{L}$"
    assert figure.layout.yaxis2.title.text == "Metric value"


def test_weights_are_latex_matrices_with_dimensions_and_ellipsis(trained_small_network):
    _model, _X, history = trained_small_network
    figure = visualize_nn_weights(history, parameter="0.weight", max_rows=2, max_cols=1, max_frames=2)
    text = _annotation_text(figure)

    assert isinstance(figure, go.Figure)
    assert len(figure.frames) == 2
    assert r"\begin{bmatrix}" in text
    assert r"\mathbb{R}^{4 \times 2}" in text
    assert r"\cdots" in text or r"\vdots" in text


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
