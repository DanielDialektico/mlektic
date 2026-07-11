import plotly.graph_objects as go
import pytest

from mlektic import (
    TorchTrainingRecorder,
    explain_nn_prediction,
    visualize_nn,
    visualize_nn_training,
    visualize_nn_weights,
)

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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
    criterion = torch.nn.BCELoss()
    recorder = TorchTrainingRecorder(model)
    for step in range(6):
        optimizer.zero_grad()
        prediction = model(X)
        loss = criterion(prediction, y)
        loss.backward()
        recorder.record(step, loss=loss)
        optimizer.step()
    recorder.close()
    return model, X, recorder.to_history()


def test_visualize_nn_architecture(trained_small_network):
    model, X, _history = trained_small_network
    fig = visualize_nn(model, X[:1])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_visualize_nn_training_and_weights(trained_small_network):
    _model, _X, history = trained_small_network
    training = visualize_nn_training(history)
    weights = visualize_nn_weights(history)
    assert isinstance(training, go.Figure)
    assert len(training.frames) == len(history["steps"])
    assert isinstance(weights, go.Figure)
    assert len(weights.frames) == len(history["steps"])


def test_visualize_nn_activations_and_prediction_math(trained_small_network):
    model, X, history = trained_small_network
    activations = visualize_nn(model, history=history, view="activations")
    explanation = explain_nn_prediction(model, X[0])
    assert isinstance(activations, go.Figure)
    assert isinstance(explanation, go.Figure)
    assert "Forward" in explanation.layout.title.text
