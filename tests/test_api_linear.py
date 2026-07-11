import numpy as np
import plotly.graph_objs as go
import pytest
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlektic import visualize_lr
from mlektic.api.linear import explain_lr_prediction


def _make_dummy_linear_data(n_samples=50, n_features=1):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    # y = sum(x_i * i) + noise
    weights = np.arange(1, n_features + 1)
    y = X.dot(weights) + np.random.randn(n_samples) * 0.1
    return X, y


@pytest.fixture
def trained_sgd_1d():
    X, y = _make_dummy_linear_data(50, 1)
    model = SGDRegressor(max_iter=10, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def trained_sgd_2d():
    X, y = _make_dummy_linear_data(50, 2)
    model = SGDRegressor(max_iter=10, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def trained_sgd_5d():
    X, y = _make_dummy_linear_data(50, 5)
    model = SGDRegressor(max_iter=10, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def trained_pipeline_1d():
    X, y = _make_dummy_linear_data(50, 1)
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('sgd', SGDRegressor(max_iter=10, random_state=42))
    ])
    model.fit(X, y)
    return model, X, y


class TestVisualizeLR:
    def test_visualize_lr_1d(self, trained_sgd_1d):
        model, X, y = trained_sgd_1d
        fig = visualize_lr(model, X, y, steps=5, show_loss=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 5

    def test_visualize_lr_2d(self, trained_sgd_2d):
        model, X, y = trained_sgd_2d
        fig = visualize_lr(model, X, y, steps=5, show_loss=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 5

    def test_visualize_lr_nd(self, trained_sgd_5d):
        model, X, y = trained_sgd_5d
        fig = visualize_lr(model, X, y, steps=5, show_loss=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 5

    def test_visualize_lr_no_loss(self, trained_sgd_1d):
        model, X, y = trained_sgd_1d
        fig = visualize_lr(model, X, y, steps=5, show_loss=False)
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 5

    def test_visualize_lr_pipeline_scaled(self, trained_pipeline_1d):
        model, X, y = trained_pipeline_1d
        fig = visualize_lr(model, X, y, steps=5, display_space="scaled")
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 5

    def test_visualize_lr_pipeline_original(self, trained_pipeline_1d):
        model, X, y = trained_pipeline_1d
        fig = visualize_lr(model, X, y, steps=5, display_space="original")
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 5



class TestExplainLRPrediction:
    def test_explain_prediction_1d(self, trained_sgd_1d):
        model, X, y = trained_sgd_1d
        fig = explain_lr_prediction(model, X, y, x_query=X[0])
        assert isinstance(fig, go.Figure)
        assert "x_1" in fig.layout.title.text or "Prediction" in fig.layout.title.text

    def test_explain_prediction_2d(self, trained_sgd_2d):
        model, X, y = trained_sgd_2d
        fig = explain_lr_prediction(model, X, y, x_query=X[1])
        assert isinstance(fig, go.Figure)

    def test_explain_prediction_nd(self, trained_sgd_5d):
        model, X, y = trained_sgd_5d
        fig = explain_lr_prediction(model, X, y, x_query=X[2])
        assert isinstance(fig, go.Figure)

    def test_explain_prediction_pipeline_scaled(self, trained_pipeline_1d):
        model, X, y = trained_pipeline_1d
        fig = explain_lr_prediction(model, X, y, x_query=X[0], display_space="scaled")
        assert isinstance(fig, go.Figure)

    def test_explain_prediction_pipeline_original(self, trained_pipeline_1d):
        model, X, y = trained_pipeline_1d
        fig = explain_lr_prediction(model, X, y, x_query=X[0], display_space="original")
        assert isinstance(fig, go.Figure)
