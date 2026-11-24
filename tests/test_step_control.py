"""The "Step" button: on every animated figure, leading where playback is heavy."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from mlektic import export_figure, visualize_logistic, visualize_lr
from mlektic.visualization.step_control import (
    HEAVY_FRAME_TEX_CHARS,
    SHORT_STEP_LABEL,
    STEP_COMMAND,
    STEP_LABEL,
    is_step_menu,
    without_step_control,
)

RNG = np.random.default_rng(5)


def _menus(figure):
    menus = list(figure.layout.updatemenus)
    step = next(menu for menu in menus if is_step_menu(menu))
    play = next(menu for menu in menus if any(b.label and "Play" in b.label for b in menu.buttons))
    return play, step


def _rows(menu):
    """Vertical and horizontal offsets: both menus hang from the same anchor."""
    return float(menu.pad.b or 0) - float(menu.pad.t or 0), float(menu.pad.l or 0)


@pytest.fixture
def linear_figure():
    X = np.linspace(-2, 2, 30).reshape(-1, 1)
    y = 1.5 * X[:, 0] + RNG.normal(0, 0.1, 30)
    return visualize_lr(LinearRegression().fit(X, y), X, y, steps=6)


@pytest.fixture
def multiclass_figure():
    X = RNG.normal(size=(120, 6))
    y = np.argmax(X @ RNG.normal(size=(6, 6)), axis=1)
    return visualize_logistic(LogisticRegression(max_iter=500).fit(X, y), X, y, steps=6)


def test_every_animated_figure_has_a_step_button(linear_figure):
    play, step = _menus(linear_figure)
    button = step.buttons[0]
    assert button.label in {STEP_LABEL, "⏭  Step", SHORT_STEP_LABEL}
    # Plotly's no-op method; the fluid controller turns the click into one frame forward.
    assert button.method == "skip" and list(button.args) == [STEP_COMMAND]
    report = linear_figure.layout.meta["mlektic_visual"]["step_control"]
    assert report["primary"] is False


def test_play_leads_and_a_step_button_mirrors_it_below(linear_figure):
    play, step = _menus(linear_figure)
    assert linear_figure.layout.meta["mlektic_visual"]["step_control"]["placement"] == "stacked"
    # Same left edge, below Play (less padding above the plot), Play's size and label style.
    assert play.x == pytest.approx(step.x) and play.xanchor == step.xanchor == "left"
    assert _rows(play)[0] > _rows(step)[0]
    assert step.font.size == play.font.size
    assert play.buttons[0].label == "▶  Play" and step.buttons[0].label == "⏭  Step"


def test_a_step_button_beside_pause_is_the_smaller_button():
    import plotly.graph_objects as go

    from mlektic.visualization.step_control import add_step_control

    figure = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1])],
        frames=[go.Frame(name=str(i), data=[go.Scatter(y=[i, 1])]) for i in range(3)],
        layout={
            "height": 400, "width": 800, "margin": {"t": 60, "b": 40, "l": 60, "r": 20},
            # Only one row fits in a 60 px margin.
            "updatemenus": [{"type": "buttons", "x": 0.1, "y": 1.12, "font": {"size": 14},
                             "buttons": [{"label": "▶  Play", "method": "animate", "args": [None]},
                                         {"label": "⏸ Pause", "method": "animate", "args": [[None]]}]}],
        },
    )
    report = add_step_control(figure)
    assert report["placement"].startswith("row")
    step = next(menu for menu in figure.layout.updatemenus if is_step_menu(menu))
    assert step.buttons[0].label == STEP_LABEL and step.font.size < 14


def test_heavy_figures_lead_with_the_step_button(multiclass_figure):
    report = multiclass_figure.layout.meta["mlektic_visual"]["step_control"]
    assert report["frame_tex_chars"] >= HEAVY_FRAME_TEX_CHARS
    assert report["primary"] is True
    play, step = _menus(multiclass_figure)
    assert step.font.size == play.font.size
    step_offset, play_offset = _rows(step), _rows(play)
    # On top of Play, or on its left in a single row.
    assert step_offset[0] > play_offset[0] or step_offset[1] < play_offset[1]


def test_static_figures_get_no_step_button():
    X = np.linspace(-2, 2, 30).reshape(-1, 1)
    y = 1.5 * X[:, 0]
    figure = visualize_lr(LinearRegression().fit(X, y), X, y, steps=6, reduced_motion=True)
    assert not figure.frames
    assert not any(is_step_menu(menu) for menu in figure.layout.updatemenus or ())


def test_renderers_that_run_no_scripts_get_no_step_button(linear_figure, monkeypatch):
    import plotly.io as pio

    monkeypatch.setattr(pio.renderers, "default", "plotly_mimetype")
    bundle = linear_figure._repr_mimebundle_()
    shown = bundle["application/vnd.plotly.v1+json"]["layout"]["updatemenus"]
    assert STEP_COMMAND not in str(shown)  # nothing would drive it there
    monkeypatch.setattr(pio.renderers, "default", "colab")
    assert STEP_COMMAND in linear_figure._repr_mimebundle_()["text/html"]


def test_exports_without_the_controller_drop_the_step_button(linear_figure, tmp_path):
    with_controller = export_figure(linear_figure, tmp_path / "fluid.html", include_plotly="cdn")
    without = export_figure(linear_figure, tmp_path / "plain.html", include_plotly="cdn", fluid_playback=False)
    assert STEP_COMMAND in with_controller.read_text(encoding="utf-8")
    assert STEP_COMMAND not in without.read_text(encoding="utf-8")
    # The figure itself keeps its button.
    assert any(is_step_menu(menu) for menu in linear_figure.layout.updatemenus)
    assert not any(is_step_menu(menu) for menu in without_step_control(linear_figure).layout.updatemenus)
