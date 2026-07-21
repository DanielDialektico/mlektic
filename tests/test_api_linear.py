import numpy as np
import plotly.graph_objs as go
import pytest
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlektic import visualize_lr
from mlektic.api.linear import explain_lr_prediction
from mlektic.visualization.linear.prediction import _fmt
from mlektic.visualization.theme import get_button_highlight_script


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
        assert len(fig.frames) == 13
        assert len(fig.layout.sliders[0].steps) == 5
        assert fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["redraw"] is False

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
        annotations = " ".join(str(item.text) for item in fig.layout.annotations)
        assert r"\theta_j" in annotations
        first_math = " ".join(str(item.text) for item in fig.frames[0].layout.annotations)
        final_math = " ".join(str(item.text) for item in fig.frames[-1].layout.annotations)
        options = fig.layout.updatemenus[0].buttons[0].args[1]
        assert first_math != final_math
        assert options["frame"]["redraw"] is True
        assert fig.data[0].line.simplify is False

    def test_temporal_decimation_and_smooth_transition(self, trained_sgd_1d):
        model, X, y = trained_sgd_1d
        fig = visualize_lr(
            model,
            X,
            y,
            steps=20,
            max_frames=4,
            frame_duration=100,
            transition_duration=70,
            animation_mode="native",
        )
        options = fig.layout.updatemenus[0].buttons[0].args[1]
        assert len(fig.frames) == 4
        assert options["transition"]["duration"] == 70
        assert options["transition"]["easing"] == "linear"
        assert options["frame"]["redraw"] is True
        assert fig.layout.transition.ordering == "traces first"

    def test_default_transition_finishes_before_next_frame(self, trained_sgd_1d):
        model, X, y = trained_sgd_1d
        fig = visualize_lr(model, X, y, steps=5, frame_duration=80)
        options = fig.layout.updatemenus[0].buttons[0].args[1]

        assert options["frame"]["duration"] == 27
        assert options["transition"]["duration"] == 16
        assert options["transition"]["duration"] < options["frame"]["duration"]
        assert options["frame"]["redraw"] is False
        assert fig.data[1].line.simplify is False
        assert all(frame.data[0].line.simplify is False for frame in fig.frames)
        assert all(len(frame.data[0].x) == len(frame.data[0].y) for frame in fig.frames)
        assert all(not frame.layout.to_plotly_json() for frame in fig.frames)
        assert fig.frames[0].data[2].text != fig.frames[-1].data[2].text
        assert fig.data[4].mode == "markers+text"
        assert fig.data[4].marker.symbol == "square"
        assert fig.data[4].textfont.color == "black"
        assert fig.layout.xaxis3.visible is False
        assert len(set(fig.data[4].x)) == 1
        assert len(set(fig.data[4].y)) == len(fig.data[4].y)
        assert fig.layout.updatemenus[0].showactive is False

    def test_transition_is_bounded_below_frame_duration(self, trained_sgd_1d):
        model, X, y = trained_sgd_1d
        fig = visualize_lr(
            model,
            X,
            y,
            steps=5,
            frame_duration=80,
            transition_duration=80,
        )
        options = fig.layout.updatemenus[0].buttons[0].args[1]

        assert options["transition"]["duration"] == 16

    def test_hybrid_fps_and_visual_subframes(self, trained_sgd_1d):
        model, X, y = trained_sgd_1d
        fig = visualize_lr(
            model,
            X,
            y,
            steps=3,
            fps=25,
            interpolation_frames=4,
        )
        options = fig.layout.updatemenus[0].buttons[0].args[1]

        assert len(fig.frames) == 9
        assert len(fig.layout.sliders[0].steps) == 3
        assert options["frame"]["duration"] == 40
        assert options["frame"]["redraw"] is False

    def test_native_mode_keeps_one_frame_per_checkpoint(self, trained_sgd_1d):
        model, X, y = trained_sgd_1d
        fig = visualize_lr(model, X, y, steps=5, animation_mode="native")

        assert len(fig.frames) == 5
        assert fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["redraw"] is True

    def test_visualize_lr_no_loss(self, trained_sgd_1d):
        model, X, y = trained_sgd_1d
        fig = visualize_lr(model, X, y, steps=5, show_loss=False)
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 13

    def test_visualize_lr_pipeline_scaled(self, trained_pipeline_1d):
        model, X, y = trained_pipeline_1d
        fig = visualize_lr(model, X, y, steps=5, display_space="scaled")
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 13

    def test_visualize_lr_pipeline_original(self, trained_pipeline_1d):
        model, X, y = trained_pipeline_1d
        fig = visualize_lr(model, X, y, steps=5, display_space="original")
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 13



class TestExplainLRPrediction:
    def test_large_values_use_compact_scientific_latex(self):
        assert _fmt(16_609_008, 4) == r"1.6609\times 10^{7}"
        assert _fmt(-9_686_209, 4) == r"-9.6862\times 10^{6}"

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
        annotations = " ".join(str(item.text) for item in fig.layout.annotations)
        assert r"\hat{y}" in annotations

    def test_explain_prediction_pipeline_scaled(self, trained_pipeline_1d):
        model, X, y = trained_pipeline_1d
        fig = explain_lr_prediction(model, X, y, x_query=X[0], display_space="scaled")
        assert isinstance(fig, go.Figure)

    def test_explain_prediction_pipeline_original(self, trained_pipeline_1d):
        model, X, y = trained_pipeline_1d
        fig = explain_lr_prediction(model, X, y, x_query=X[0], display_space="original")
        assert isinstance(fig, go.Figure)

    def test_large_nd_prediction_displays_true_column_vectors(self):
        X, y = _make_dummy_linear_data(60, 15)
        model = SGDRegressor(max_iter=20, random_state=42).fit(X, y)
        fig = explain_lr_prediction(model, X, y, x_query=X[0])
        input_annotations = fig.layout.updatemenus[0].buttons[0].args[1]["annotations"]
        vector_text = next(item["text"] for item in input_annotations if r"\mathbf{x}=\begin{bmatrix}" in item["text"])

        assert " & " not in vector_text
        assert r"\vdots" in vector_text


def test_buttons_remain_white_without_animation_state_tracking():
    script = get_button_highlight_script()

    assert ".js-plotly-plot .updatemenu-button rect" in script
    assert "fill: white !important" in script
    assert ".js-plotly-plot .updatemenu-button text" in script
    assert "fill: black !important" in script
    assert "!important" in script
    assert "MutationObserver" not in script
    assert "data-mlektic-active-btn" not in script
    assert "mlektic-active-button" not in script
    assert "requestAnimationFrame" not in script
