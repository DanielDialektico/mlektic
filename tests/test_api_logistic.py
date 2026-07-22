import numpy as np
import plotly.graph_objs as go
import pytest
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlektic import explain_logistic_prediction, visualize_logistic
from mlektic.services.linear_history import fit_history_logistic
from mlektic.utils.probability import multiclass_probabilities
from mlektic.visualization.logistic.router import build_logistic_figure


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
        annotations = " ".join(str(item.text) for item in fig.layout.annotations)
        assert r"q_k=\sigma(z_k)" in annotations
        assert r"\Theta^\top\mathbf{x}+\boldsymbol{\theta}_0" in annotations

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

    def test_probability_link_matches_estimator_semantics(self):
        X, y = _make_dummy_classification_data(80, 2, 3)
        estimators = [
            SGDClassifier(loss="log_loss", max_iter=1000, random_state=42),
            LogisticRegression(max_iter=1000, random_state=42),
        ]
        expected_links = ["ovr", "softmax"]
        for estimator, expected_link in zip(estimators, expected_links):
            estimator.fit(X, y)
            history = fit_history_logistic(estimator, X, y, steps=3, mode="final_interp")
            scores = estimator.decision_function(X)
            reconstructed = multiclass_probabilities(scores, history["probability_link"])
            assert history["probability_link"] == expected_link
            assert np.allclose(reconstructed, estimator.predict_proba(X), atol=1e-10)

    def test_animation_transition_is_smooth_for_2d_traces(self, trained_binary_1d):
        model, X, y = trained_binary_1d
        fig = visualize_logistic(
            model,
            X,
            y,
            steps=4,
            frame_duration=100,
            transition_duration=75,
        )
        options = fig.layout.updatemenus[0].buttons[0].args[1]
        assert options["transition"]["duration"] == 75
        assert options["frame"]["redraw"] is True
        assert fig.layout.transition.ordering == "traces first"

    def test_high_dimensional_multiclass_layout_keeps_true_dimensions(self):
        rng = np.random.default_rng(8)
        steps, samples, dimensions, classes = 3, 40, 20, 12
        X = rng.normal(size=(samples, dimensions))
        y = np.arange(samples) % classes
        history = {
            "history_kind": "iterative",
            "classes": np.arange(classes),
            "probability_link": "softmax",
            "w_hist": np.zeros((steps, dimensions, classes)),
            "b_hist": np.zeros((steps, classes)),
            "loss_hist": np.linspace(2.5, 1.5, steps),
        }
        fig = build_logistic_figure(X, y, history=history, show_loss=True, max_theta_cols=5)
        annotations = " ".join(str(item.text) for item in fig.layout.annotations)
        assert rf"\mathbb{{R}}^{{{dimensions}\times {classes}}}" in annotations
        assert r"\cdots" in annotations
        assert len(fig.frames) == steps
        for annotation in fig.layout.annotations:
            if annotation.xref == "paper":
                assert -0.15 <= float(annotation.x) <= 1.05
                assert -0.15 <= float(annotation.y) <= 1.15

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
