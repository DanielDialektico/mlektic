import json

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from mlektic import (
    available_themes,
    explain_logistic_prediction,
    explain_lr_prediction,
    export_figure,
    get_theme_tokens,
    visualize_logistic,
    visualize_lr,
)


@pytest.fixture
def linear_case():
    X = np.linspace(-2.0, 2.0, 36).reshape(-1, 1)
    y = 1.25 + 2.4 * X[:, 0] + 0.08 * np.sin(5 * X[:, 0])
    return LinearRegression().fit(X, y), X, y


@pytest.fixture
def logistic_case():
    X = np.linspace(-2.5, 2.5, 50).reshape(-1, 1)
    y = (X[:, 0] > 0.15).astype(int)
    return LogisticRegression(random_state=7).fit(X, y), X, y


def _visual_meta(figure):
    return figure.layout.meta["mlektic_visual"]


def test_visual_tokens_are_public_immutable_and_complete():
    assert available_themes() == ("classic", "academic", "classroom", "compact", "accessible")
    academic = get_theme_tokens("academic")
    assert academic.title_size < get_theme_tokens("classroom").title_size
    assert academic.background == "#17181c"
    with pytest.raises((AttributeError, TypeError)):
        academic.title_size = 99


@pytest.mark.parametrize("width", [700, 1000, 1400])
def test_explicit_notebook_widths_are_deterministic(linear_case, width):
    model, X, y = linear_case
    figure = visualize_lr(
        model,
        X,
        y,
        steps=5,
        animation_mode="native",
        width=width,
        height=640,
    )
    assert figure.layout.width == width
    assert figure.layout.height == 640
    assert _visual_meta(figure)["resolved_width"] == width
    assert len(figure.frames) == 5


def test_classic_dashboard_keeps_existing_visual_contract(linear_case):
    model, X, y = linear_case
    figure = visualize_lr(model, X, y, steps=5, animation_mode="native")

    assert figure.layout.template.layout.paper_bgcolor == "rgb(17,17,17)"
    assert figure.layout.width == 1100
    assert figure.layout.height == 600
    assert figure.data[0].marker.color == "#7dd3fc"
    assert figure.data[1].line.color == "#EF553B"
    assert len(figure.frames) == 5
    assert _visual_meta(figure)["theme"] == "classic"
    assert _visual_meta(figure)["motion_preserved"] is True


def test_density_is_a_compatible_alias_for_phase1_detail(linear_case):
    model, X, y = linear_case
    figure = visualize_lr(
        model,
        X,
        y,
        steps=4,
        density="academic",
        animation_mode="native",
    )
    annotations = " ".join(str(annotation.text) for annotation in figure.layout.annotations)

    assert "Fitted-model derivation" in annotations
    assert _visual_meta(figure)["density"] == "academic"
    with pytest.raises(ValueError, match="detail and density"):
        visualize_lr(model, X, y, detail="complete", density="academic")


def test_academic_and_compact_are_opt_in_and_preserve_motion(linear_case):
    model, X, y = linear_case
    classic = visualize_lr(model, X, y, steps=6, animation_mode="native")
    styled = visualize_lr(
        model,
        X,
        y,
        steps=6,
        animation_mode="native",
        theme="academic",
        format="compact",
    )

    assert len(styled.frames) == len(classic.frames) == 6
    assert styled.layout.height < classic.layout.height
    assert styled.layout.paper_bgcolor == "#17181c"
    assert styled.layout.font.family == "Inter, Arial, sans-serif"
    assert _visual_meta(styled)["motion_preserved"] is True


def test_compact_academic_panel_keeps_slider_clearance(linear_case):
    model, X, y = linear_case
    dashboard = visualize_lr(
        model,
        X,
        y,
        steps=6,
        animation_mode="native",
        theme="academic",
        density="academic",
    )
    compact = visualize_lr(
        model,
        X,
        y,
        steps=6,
        animation_mode="native",
        theme="academic",
        format="compact",
        density="academic",
        size="notebook",
    )
    dashboard_plot_height = (
        dashboard.layout.height - dashboard.layout.margin.t - dashboard.layout.margin.b
    )
    compact_plot_height = compact.layout.height - compact.layout.margin.t - compact.layout.margin.b

    assert compact.layout.margin.b == dashboard.layout.margin.b
    assert compact_plot_height >= dashboard_plot_height
    assert compact.layout.height >= dashboard.layout.height - 50
    assert max(float(annotation.y) for annotation in compact.layout.annotations) < 1.1


def test_lesson_stages_concepts_without_decimating_frames(linear_case):
    model, X, y = linear_case
    figure = visualize_lr(
        model,
        X,
        y,
        steps=7,
        animation_mode="native",
        format="lesson",
    )
    stage_menu = figure.layout.updatemenus[-1]

    assert [button.label for button in stage_menu.buttons] == [
        "1 Data",
        "2 Model",
        "3 Objective",
        "4 Complete",
    ]
    assert stage_menu.direction == "down"
    assert figure.layout.margin.r >= 150
    assert len(figure.frames) == 7
    assert figure.data[0].visible is True
    assert any(trace.visible is False for trace in figure.data[1:])


def test_theme_does_not_reclassify_metric_cards_as_data(linear_case):
    model, X, y = linear_case
    figure = visualize_lr(model, X, y, steps=5, theme="classroom", format="lesson")
    metric_trace = next(trace for trace in figure.data if trace.uid == "METRIC_VALUES")

    assert metric_trace.marker.symbol == "square"
    assert metric_trace.visible is False


@pytest.mark.parametrize("option", ["report", "reduced_motion"])
def test_static_alternatives_use_the_exact_final_frame(linear_case, option):
    model, X, y = linear_case
    kwargs = {"format": "report"} if option == "report" else {"reduced_motion": True}
    animated = visualize_lr(model, X, y, steps=5, animation_mode="native")
    final_model_y = np.asarray(animated.frames[-1].data[0].y, dtype=float)
    static = visualize_lr(model, X, y, steps=5, animation_mode="native", **kwargs)

    assert len(static.frames) == 0
    assert len(static.layout.updatemenus or ()) == 0
    assert len(static.layout.sliders or ()) == 0
    assert np.allclose(np.asarray(static.data[1].y, dtype=float), final_model_y)
    assert _visual_meta(static)["motion_preserved"] is False


def test_accessible_theme_uses_color_redundancy(logistic_case):
    model, X, y = logistic_case
    figure = visualize_logistic(
        model,
        X,
        y,
        steps=5,
        show_loss=True,
        theme="accessible",
    )
    model_traces = [trace for trace in figure.data if trace.uid == "MODEL_LINE"]
    loss_traces = [trace for trace in figure.data if trace.uid == "LOSS_LINE"]

    assert figure.data[0].marker.symbol == "circle-open"
    assert model_traces[0].line.dash == "solid"
    assert loss_traces[0].line.dash == "dot"
    assert _visual_meta(figure)["accessibility"]["color_is_redundant"] is True


def test_size_responsive_and_explicit_dimensions_are_independent(linear_case):
    model, X, y = linear_case
    notebook = visualize_lr(model, X, y, size="notebook", responsive=True)
    explicit = visualize_lr(model, X, y, size="compact", width=930, height=710)

    assert notebook.layout.autosize is True
    assert notebook.layout.width is None
    assert notebook.layout.height == 570
    assert _visual_meta(notebook)["responsive_config"] == {"responsive": True}
    assert explicit.layout.width == 930
    assert explicit.layout.height == 710


def test_prediction_figures_share_the_visual_contract(linear_case, logistic_case):
    linear_model, X_linear, y_linear = linear_case
    logistic_model, X_logistic, y_logistic = logistic_case
    linear_figure = explain_lr_prediction(
        linear_model,
        X_linear,
        y_linear,
        x_query=[[0.4]],
        theme="academic",
        format="compact",
    )
    logistic_figure = explain_logistic_prediction(
        logistic_model,
        X_logistic,
        y_logistic,
        x_query=[[0.4]],
        theme="accessible",
        size="wide",
    )

    assert _visual_meta(linear_figure)["family"] == "linear-prediction"
    assert _visual_meta(logistic_figure)["family"] == "logistic-prediction"
    assert logistic_figure.layout.width == 1400


def test_prediction_values_use_high_contrast_annotation_boxes(linear_case, logistic_case):
    linear_model, X_linear, y_linear = linear_case
    logistic_model, X_logistic, y_logistic = logistic_case
    linear_figure = explain_lr_prediction(
        linear_model,
        X_linear,
        y_linear,
        x_query=[[0.4]],
        theme="academic",
    )
    logistic_figure = explain_logistic_prediction(
        logistic_model,
        X_logistic,
        y_logistic,
        x_query=[[0.4]],
        theme="accessible",
    )

    for figure in (linear_figure, logistic_figure):
        output_button = next(
            button
            for menu in figure.layout.updatemenus
            for button in menu.buttons
            if button.label == "Output"
        )
        prediction_annotation = next(
            annotation
            for annotation in output_button.args[1]["annotations"]
            if annotation.get("showarrow")
        )
        assert prediction_annotation["bgcolor"].startswith("#")
        assert prediction_annotation["borderwidth"] == 1
        assert prediction_annotation["borderpad"] == 5


def test_export_inherits_responsive_metadata(tmp_path, linear_case):
    model, X, y = linear_case
    figure = visualize_lr(model, X, y, responsive=True)
    destination = export_figure(figure, tmp_path / "responsive", include_plotly="cdn")
    html = destination.read_text(encoding="utf-8")

    assert destination.suffix == ".html"
    assert '"responsive": true' in html or '"responsive":true' in html


def test_visual_metadata_is_json_serializable(linear_case):
    model, X, y = linear_case
    figure = visualize_lr(model, X, y, theme="classroom", format="lesson", size="wide")
    payload = json.dumps(_visual_meta(figure), sort_keys=True)

    assert '"schema_version": 1' in payload
    assert '"theme": "classroom"' in payload


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("theme", "unknown", "Unknown theme"),
        ("format", "poster", "format must be"),
        ("size", "tiny", "size must be"),
        ("width", 200, "at least 320"),
        ("responsive", "yes", "boolean"),
    ],
)
def test_visual_options_fail_explicitly(linear_case, argument, value, message):
    model, X, y = linear_case
    with pytest.raises((TypeError, ValueError), match=message):
        visualize_lr(model, X, y, **{argument: value})


def test_neural_architecture_accepts_the_shared_visual_contract():
    torch = pytest.importorskip("torch")
    from mlektic import visualize_nn_architecture

    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Tanh(), torch.nn.Linear(3, 1))
    sample = torch.tensor([[0.2, -0.1]])
    figure = visualize_nn_architecture(
        model,
        sample,
        theme="academic",
        format="compact",
        responsive=True,
    )

    assert _visual_meta(figure)["family"] == "neural"
    assert figure.layout.paper_bgcolor == "#17181c"
    assert figure.layout.autosize is True
