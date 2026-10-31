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


def _annotation_payloads(value):
    if isinstance(value, dict):
        annotations = value.get("annotations")
        if isinstance(annotations, (list, tuple)):
            yield from annotations
        for child in value.values():
            yield from _annotation_payloads(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _annotation_payloads(child)


def test_visual_tokens_are_public_immutable_and_complete():
    assert available_themes() == ("classic", "academic", "classroom", "compact", "accessible")
    academic = get_theme_tokens("academic")
    classroom = get_theme_tokens("classroom")
    assert classroom.title_size == academic.title_size
    assert classroom.body_size == academic.body_size
    assert classroom.typography_scale == academic.typography_scale
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


def test_narrow_tabular_layout_caps_headers_and_removes_fixed_math_boxes(linear_case):
    model, X, y = linear_case
    figure = visualize_lr(
        model,
        X,
        y,
        steps=6,
        detail="academic",
        theme="classroom",
        size="notebook",
    )
    fit = _visual_meta(figure)["narrow_layout_fit"]
    math_annotations = [annotation for annotation in figure.layout.annotations if float(annotation.y or 0) < 0]

    assert fit["applied"] is True
    assert figure.layout.title.font.size <= 22
    assert figure.layout.title.y == 0.96
    assert figure.layout.title.yanchor == "top"
    assert all(annotation.borderwidth == 0 for annotation in math_annotations)
    assert all(annotation.bgcolor is None for annotation in math_annotations)
    assert all(annotation.font.size <= 13 for annotation in math_annotations)


def test_hybrid_slider_reserves_a_non_overlapping_math_band(linear_case):
    model, X, y = linear_case
    for size in ("default", "compact", "notebook", "wide", "classroom"):
        for format_name in ("dashboard", "compact"):
            figure = visualize_lr(
                model,
                X,
                y,
                steps=8,
                max_frames=8,
                animation_mode="hybrid",
                interpolation_frames=4,
                detail="academic",
                theme="academic",
                size=size,
                format=format_name,
            )
            math_annotations = sorted(
                (annotation for annotation in figure.layout.annotations if float(annotation.y or 0) < 0),
                key=lambda annotation: abs(float(annotation.y)),
            )
            plot_height = figure.layout.height - figure.layout.margin.t - figure.layout.margin.b
            centers = [abs(float(annotation.y)) * plot_height for annotation in math_annotations]
            slider_padding = float(figure.layout.sliders[0].pad.t or 0)

            assert centers[0] >= slider_padding + 100, (size, format_name)
            assert all(
                next_center - center >= 50 for center, next_center in zip(centers, centers[1:])
            ), (size, format_name)
            assert centers[2] - centers[1] >= 60, (size, format_name)
            assert centers[-1] <= figure.layout.margin.b - 30, (size, format_name)


def test_narrow_logistic_metric_readouts_are_borderless_and_high_contrast(logistic_case):
    model, X, y = logistic_case
    figure = visualize_logistic(
        model,
        X,
        y,
        steps=8,
        max_frames=6,
        show_loss=True,
        theme="classroom",
        size="notebook",
    )
    metrics = [
        annotation
        for annotation in figure.layout.annotations
        if "<b>" in str(annotation.text) and any(label in str(annotation.text) for label in ("Accuracy", "F1"))
    ]

    assert metrics
    assert all(annotation.borderwidth == 0 for annotation in metrics)
    assert all(annotation.bgcolor is None for annotation in metrics)
    assert all(annotation.font.color == get_theme_tokens("classroom").text for annotation in metrics)


def test_wide_tabular_layout_keeps_borderless_math_panel_composition(linear_case):
    model, X, y = linear_case
    figure = visualize_lr(model, X, y, steps=6, detail="academic", size="wide")
    math_annotations = [annotation for annotation in figure.layout.annotations if float(annotation.y or 0) < 0]

    assert _visual_meta(figure)["narrow_layout_fit"]["applied"] is False
    assert all(annotation.borderwidth == 0 for annotation in math_annotations)
    assert all(annotation.bgcolor is None for annotation in math_annotations)
    assert _visual_meta(figure)["annotation_boxes"]["policy"] == "borderless"
    plot_height = figure.layout.height - figure.layout.margin.t - figure.layout.margin.b
    lowest_center = max(abs(float(annotation.y)) * plot_height for annotation in math_annotations)
    assert lowest_center <= figure.layout.margin.b - 30


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


def test_prediction_values_remain_high_contrast_without_annotation_boxes(linear_case, logistic_case):
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
        assert prediction_annotation.get("bgcolor") is None
        assert prediction_annotation["borderwidth"] == 0
        assert prediction_annotation["borderpad"] == 0
        assert prediction_annotation["font"]["color"] == figure.layout.font.color


@pytest.mark.parametrize("static_option", ["report", "reduced_motion"])
def test_static_logistic_prediction_applies_output_before_removing_controls(
    logistic_case, static_option
):
    model, X, y = logistic_case
    interactive = explain_logistic_prediction(model, X, y, x_query=[[0.4]])
    output_button = next(
        button
        for menu in interactive.layout.updatemenus or ()
        for button in menu.buttons or ()
        if button.label == "Output"
    )
    expected_text = [
        str(annotation.get("text", ""))
        for annotation in output_button.args[1]["annotations"]
    ]
    kwargs = {"format": "report"} if static_option == "report" else {"reduced_motion": True}
    static = explain_logistic_prediction(model, X, y, x_query=[[0.4]], **kwargs)

    assert len(static.layout.updatemenus or ()) == 0
    assert [str(annotation.text or "") for annotation in static.layout.annotations] == expected_text


def test_every_public_family_removes_boxes_from_all_annotation_states(linear_case, logistic_case):
    torch = pytest.importorskip("torch")
    from mlektic import visualize_nn_architecture

    linear_model, X_linear, y_linear = linear_case
    logistic_model, X_logistic, y_logistic = logistic_case
    neural_model = torch.nn.Sequential(torch.nn.Linear(1, 3), torch.nn.Tanh(), torch.nn.Linear(3, 1))
    figures = [
        visualize_lr(linear_model, X_linear, y_linear, steps=4, detail="academic"),
        visualize_logistic(logistic_model, X_logistic, y_logistic, steps=4, show_loss=True),
        explain_lr_prediction(linear_model, X_linear, y_linear, x_query=[[0.4]]),
        explain_logistic_prediction(logistic_model, X_logistic, y_logistic, x_query=[[0.4]]),
        visualize_nn_architecture(neural_model, torch.tensor([[0.2]]), theme="academic"),
    ]

    for figure in figures:
        annotations = list(_annotation_payloads(figure.to_plotly_json()))
        assert annotations
        assert all(annotation.get("bgcolor") is None for annotation in annotations)
        assert all(annotation.get("bordercolor") is None for annotation in annotations)
        assert all(annotation.get("borderwidth", 0) == 0 for annotation in annotations)


def test_export_inherits_responsive_metadata(tmp_path, linear_case):
    model, X, y = linear_case
    figure = visualize_lr(model, X, y, responsive=True)
    destination = export_figure(figure, tmp_path / "responsive", include_plotly="cdn")
    html = destination.read_text(encoding="utf-8")

    assert destination.suffix == ".html"
    assert '"responsive": true' in html or '"responsive":true' in html


def test_visual_metadata_is_json_serializable(linear_case):
    model, X, y = linear_case
    figure = visualize_lr(model, X, y, theme="classroom", format="dashboard", size="wide")
    payload = json.dumps(_visual_meta(figure), sort_keys=True)

    assert '"schema_version": 1' in payload
    assert '"theme": "classroom"' in payload


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("theme", "unknown", "Unknown theme"),
        ("format", "lesson", "format must be"),
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
