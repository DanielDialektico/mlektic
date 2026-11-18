"""Skins are presentation-only themes: they repaint a figure and never move it."""

import re

import numpy as np
import plotly.graph_objects as go
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor

from mlektic import available_themes, get_theme_tokens, visualize_logistic, visualize_lr
from mlektic.visualization.design import _CLASSIC as CLASSIC_TOKENS
from mlektic.visualization.skins import (
    SkinEffects,
    apply_skin,
    available_skins,
    backdrop_image,
    get_skin,
    is_skin,
)
from mlektic.visualization.skins.definitions import SKINS
from mlektic.visualization.skins.effects import _color_numbers
from mlektic.visualization.skins.roles import EFFECT_UID_PREFIX
from mlektic.visualization.theme import _CLASSIC as CLASSIC_PALETTE
from mlektic.visualization.theme import _resolve, get_button_highlight_script

SKIN_NAMES = ("aurora", "editorial", "blueprint")

# Keys that only paint.  Everything else (sizes, positions, anchors, margins,
# ranges, domains, fonts, text) must be identical to the classic figure.
PAINT_KEYS = {
    "color", "bgcolor", "bordercolor", "gridcolor", "zerolinecolor", "linecolor", "tickcolor",
    "activebgcolor", "paper_bgcolor", "plot_bgcolor", "backgroundcolor", "symbol", "dash",
}
_COLOR_MACRO = re.compile(r"\\color\{[^}]*\}\{([^}]*)\}")
_SPAN_COLOR = re.compile(r"color:\s*[^;\"']+")


@pytest.fixture(scope="module")
def sgd_linear():
    rng = np.random.default_rng(3)
    X = rng.uniform(-2, 3, size=(60, 1))
    y = 1.2 + 2.1 * X[:, 0] + rng.normal(0, 0.2, 60)
    model = SGDRegressor(max_iter=40, tol=None, random_state=0, learning_rate="constant", eta0=0.01)
    return model.fit(X, y), X, y


@pytest.fixture(scope="module")
def multiclass():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(90, 1)) * 2
    y = np.digitize(X[:, 0], [-1.0, 1.0])
    return LogisticRegression(max_iter=200).fit(X, y), X, y


def _geometry(value, key=None):
    """Strip paint from a Plotly JSON value, keeping every geometric property."""
    if isinstance(value, dict):
        kept = {k: _geometry(v, k) for k, v in value.items() if k not in PAINT_KEYS | {"meta", "images", "uirevision"}}
        # A container that only held paint (``font: {color}``) carries no geometry.
        return {k: v for k, v in kept.items() if v != {}}
    if isinstance(value, (list, tuple)):
        return [_geometry(v, key) for v in value]
    if isinstance(value, str) and key == "text":
        return _SPAN_COLOR.sub("color:", _COLOR_MACRO.sub(r"\1", value))
    return value


def _original_traces(figure_json):
    return [trace for trace in figure_json["data"] if not str(trace.get("uid", "")).startswith(EFFECT_UID_PREFIX)]


def _frames_without_effects(figure_json):
    effect_indices = {
        i for i, trace in enumerate(figure_json["data"]) if str(trace.get("uid", "")).startswith(EFFECT_UID_PREFIX)
    }
    kept = [i for i in range(len(figure_json["data"])) if i not in effect_indices]
    remap = {old: new for new, old in enumerate(kept)}
    frames = []
    for frame in figure_json.get("frames", []):
        pairs = [(remap[t], d) for t, d in zip(frame["traces"], frame["data"]) if t in remap]
        frames.append({"name": frame.get("name"), "traces": [t for t, _ in pairs], "data": [d for _, d in pairs],
                       "layout": frame.get("layout")})
    return frames


def _assert_same_geometry(classic, skinned):
    a, b = classic.to_plotly_json(), skinned.to_plotly_json()
    layout_a, layout_b = dict(a["layout"]), dict(b["layout"])
    for layout in (layout_a, layout_b):
        layout.pop("template", None)
    assert _geometry(layout_b) == _geometry(layout_a)
    assert _geometry(_original_traces(b)) == _geometry(a["data"])
    assert _geometry(_frames_without_effects(b)) == _geometry(_frames_without_effects(a))


def test_skins_are_registered_as_themes():
    assert available_skins() == SKIN_NAMES
    assert available_themes()[-3:] == SKIN_NAMES
    assert all(is_skin(name) and is_skin(name.upper()) for name in SKIN_NAMES)
    assert not is_skin("classic") and not is_skin(None)
    with pytest.raises(ValueError, match="aurora"):
        visualize_lr(LinearRegression().fit([[0], [1]], [0, 1]), [[0], [1]], [0, 1], theme="aurorа")


def test_builders_receive_classic_geometry_for_every_skin():
    for name in SKIN_NAMES:
        assert get_theme_tokens(name) is CLASSIC_TOKENS
        assert _resolve(name) is CLASSIC_PALETTE


def test_skin_definitions_are_frozen_data():
    skin = get_skin("aurora")
    with pytest.raises((AttributeError, TypeError)):
        skin.palette.model = "#000000"
    assert set(SKINS) == set(SKIN_NAMES)


@pytest.mark.parametrize("name", SKIN_NAMES)
def test_linear_animation_keeps_classic_geometry(sgd_linear, name):
    model, X, y = sgd_linear
    classic = visualize_lr(model, X, y, steps=12, max_frames=12)
    skinned = visualize_lr(model, X, y, steps=12, max_frames=12, theme=name)
    _assert_same_geometry(classic, skinned)


@pytest.mark.parametrize("name", SKIN_NAMES)
def test_multiclass_animation_keeps_classic_geometry(multiclass, name):
    model, X, y = multiclass
    classic = visualize_logistic(model, X, y, steps=10, max_frames=10)
    skinned = visualize_logistic(model, X, y, steps=10, max_frames=10, theme=name)
    _assert_same_geometry(classic, skinned)


def test_aurora_effects_are_reported_and_frames_stay_consistent(sgd_linear):
    model, X, y = sgd_linear
    figure = visualize_lr(model, X, y, steps=12, max_frames=12, theme="aurora")
    report = figure.layout.meta["mlektic_visual"]["skin"]
    assert report["name"] == "aurora"
    assert report["backdrop"] == "radial"
    assert report["effects"]["applied"] == ["trail", "glow", "loss_progress", "equation_delta"]
    for frame in figure.frames:
        assert len(frame.traces) == len(frame.data)
        assert all(0 <= int(t) < len(figure.data) for t in frame.traces)
    roles = [str(t.uid or "") for t in figure.data]
    model_index = roles.index("MODEL_LINE")
    # Trail and glow are drawn under the model line; the comet is drawn on top.
    assert all(uid.startswith(EFFECT_UID_PREFIX) for uid in roles[model_index - 6:model_index])
    assert roles[-1] == f"{EFFECT_UID_PREFIX}COMET_HEAD"


def test_equation_numbers_are_coloured_by_direction():
    text = r"$\hat{y}=(1.5000)x_1+(0.2000)$"
    previous = r"$\hat{y}=(1.2500)x_1+(0.3000)$"
    coloured = _color_numbers(text, previous, "#0f0", "#f00")
    assert coloured == r"$\hat{y}=(\color{#0f0}{1.5000})x_1+(\color{#f00}{0.2000})$"
    assert _color_numbers(text, text, "#0f0", "#f00") == text
    assert _color_numbers(text, r"$(1.0)$", "#0f0", "#f00") == text


def test_effects_skip_autoranged_axes():
    figure = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", uid="MODEL_LINE")],
        frames=[go.Frame(data=[go.Scatter(x=[0, 1], y=[0, 2])], traces=[0], name="1")],
    )
    report = apply_skin(figure, get_skin("aurora"), family="linear")
    assert report["effects"]["skipped"]["trail"] == "axis autoranges"
    assert len(figure.data) == 1


def test_effects_never_touch_neural_figures():
    figure = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", uid="MODEL_LINE", line={"color": "#123456"})],
        layout={"xaxis": {"range": [0, 1]}, "yaxis": {"range": [0, 1]}},
        frames=[go.Frame(data=[go.Scatter(x=[0, 1], y=[0, 2])], traces=[0], name="1")],
    )
    report = apply_skin(figure, get_skin("aurora"), family="neural")
    assert report["effects"]["applied"] == []
    assert len(figure.data) == 1 and figure.data[0].line.color == "#123456"


def test_effects_are_switchable_in_the_definition(sgd_linear):
    model, X, y = sgd_linear
    figure = visualize_lr(model, X, y, steps=8, max_frames=8)
    quiet = get_skin("aurora").__class__(
        name="quiet", description="", palette=get_skin("aurora").palette, effects=SkinEffects()
    )
    traces = len(figure.data)
    report = apply_skin(figure, quiet, family="linear")
    assert report["effects"] == {"applied": [], "skipped": {}}
    assert len(figure.data) == traces


def test_backdrop_covers_the_canvas_and_needs_a_fixed_size():
    figure = go.Figure(layout={"width": 1000, "height": 500, "margin": {"l": 50, "r": 50, "t": 100, "b": 100}})
    image = backdrop_image(get_skin("blueprint").backdrop, figure)
    assert image["layer"] == "below" and image["source"].startswith("data:image/svg+xml;base64,")
    plot_width, plot_height = 900, 300
    assert image["x"] < -50 / plot_width and image["x"] + image["sizex"] > 1 + 50 / plot_width
    assert image["y"] > 1 + 100 / plot_height and image["y"] - image["sizey"] < -100 / plot_height
    figure.layout.autosize = True
    assert backdrop_image(get_skin("blueprint").backdrop, figure) is None


def test_responsive_skin_falls_back_to_solid_background(sgd_linear):
    model, X, y = sgd_linear
    figure = visualize_lr(model, X, y, steps=6, max_frames=6, theme="aurora", responsive=True)
    assert not figure.layout.images
    assert figure.layout.plot_bgcolor == get_skin("aurora").palette.background
    assert figure.layout.meta["mlektic_visual"]["skin"]["backdrop"] is None


def test_button_stylesheet_uses_skin_colours():
    script = get_button_highlight_script(theme="editorial")
    palette = get_skin("editorial").palette
    assert palette.control_background in script and palette.control_text in script
