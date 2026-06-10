import numpy as np
import pytest
import plotly.graph_objs as go
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlektic import visualize_logistic, explain_logistic_prediction


def _make_dummy_classification_data(n_samples=50, n_features=1, n_classes=2):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    if n_classes == 2:
        y = (X[:, 0] > 0).astype(int)
    else:
        y = np.random.randint(0, n_classes, size=n_samples)
    return X, y


@pytest.fixture
def trained_binary_1d():
    X, y = _make_dummy_classification_data(50, 1, 2)
    model = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def trained_binary_2d():
    X, y = _make_dummy_classification_data(50, 2, 2)
    model = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def trained_binary_nd():
    X, y = _make_dummy_classification_data(50, 4, 2)
    model = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def trained_multiclass_1d():
    X, y = _make_dummy_classification_data(50, 1, 3)
    model = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def trained_multiclass_2d():
    X, y = _make_dummy_classification_data(50, 2, 3)
    model = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def trained_multiclass_nd():
    X, y = _make_dummy_classification_data(50, 5, 3)
    model = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def trained_pipeline_binary_1d():
    X, y = _make_dummy_classification_data(50, 1, 2)
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('sgd', SGDClassifier(loss='log_loss', max_iter=10, random_state=42))
    ])
    model.fit(X, y)
    return model, X, y


class TestVisualizeLogistic:
    def test_visualize_binary_1d(self, trained_binary_1d):
        model, X, y = trained_binary_1d
        fig = visualize_logistic(model, X, y, steps=5, show_loss=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 5

    def test_visualize_binary_2d(self, trained_binary_2d):
        model, X, y = trained_binary_2d
        fig = visualize_logistic(model, X, y, steps=5, show_loss=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 5

    def test_visualize_binary_nd(self, trained_binary_nd):
        model, X, y = trained_binary_nd
        fig = visualize_logistic(model, X, y, steps=5, show_loss=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 5

    def test_visualize_multiclass_1d(self, trained_multiclass_1d):
        model, X, y = trained_multiclass_1d
        fig = visualize_logistic(model, X, y, steps=5, show_loss=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 5

    def test_visualize_multiclass_2d(self, trained_multiclass_2d):
        model, X, y = trained_multiclass_2d
        fig = visualize_logistic(model, X, y, steps=5, show_loss=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 5

    def test_visualize_multiclass_nd(self, trained_multiclass_nd):
        model, X, y = trained_multiclass_nd
        fig = visualize_logistic(model, X, y, steps=5, show_loss=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 5

    def test_visualize_pipeline_scaled(self, trained_pipeline_binary_1d):
        model, X, y = trained_pipeline_binary_1d
        fig = visualize_logistic(model, X, y, steps=5, display_space="scaled")
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 5

    def test_visualize_pipeline_original(self, trained_pipeline_binary_1d):
        model, X, y = trained_pipeline_binary_1d
        fig = visualize_logistic(model, X, y, steps=5, display_space="original")
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 5

class TestExplainLogisticPrediction:
    def test_explain_prediction_binary_1d(self, trained_binary_1d):
        model, X, y = trained_binary_1d
        fig = explain_logistic_prediction(model, X, y, x_query=X[0])
        assert isinstance(fig, go.Figure)
        assert "Prediction" in fig.layout.title.text

    def test_explain_prediction_pipeline_scaled(self, trained_pipeline_binary_1d):
        model, X, y = trained_pipeline_binary_1d
        fig = explain_logistic_prediction(model, X, y, x_query=X[0], display_space="scaled")
        assert isinstance(fig, go.Figure)

    def test_explain_prediction_pipeline_original(self, trained_pipeline_binary_1d):
        model, X, y = trained_pipeline_binary_1d
        fig = explain_logistic_prediction(model, X, y, x_query=X[0], display_space="original")
        assert isinstance(fig, go.Figure)

    def test_explain_prediction_binary_2d(self, trained_binary_2d):
        model, X, y = trained_binary_2d
        fig = explain_logistic_prediction(model, X, y, x_query=X[0])
        assert isinstance(fig, go.Figure)
        assert "Prediction" in fig.layout.title.text

    def test_explain_prediction_multiclass_1d(self):
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        X = np.linspace(-5, 5, 100).reshape(-1, 1)
        y = np.zeros(100)
        y[X.ravel() > -1] = 1
        y[X.ravel() > 2] = 2
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        fig = explain_logistic_prediction(model, X, y, x_query=X[0])
        assert isinstance(fig, go.Figure)
        assert "Multiclass" in fig.layout.title.text

