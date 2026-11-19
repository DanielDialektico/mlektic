import re
import shutil
import subprocess

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import pytest
from sklearn.linear_model import LinearRegression

from mlektic import export_figure, show_optimized, visualize_lr
from mlektic.visualization.playback import _ASSET, fluid_playback_script

MOUNT = re.compile(
    r"window\.__mlekticFluid\.mount\(document\.getElementById\('([^']+)'\), \{autoplay: (true|false)\}\)"
)


@pytest.fixture
def animated_figure():
    X = np.linspace(-2.0, 2.0, 24).reshape(-1, 1)
    y = 1.25 + 2.4 * X[:, 0]
    return visualize_lr(LinearRegression().fit(X, y), X, y, steps=6)


@pytest.fixture
def colab_renderer(monkeypatch):
    monkeypatch.setattr(pio.renderers, "default", "colab")


def _plot_ids(html):
    return re.findall(r'<div id="([^"]+)" class="plotly-graph-div"', html)


def test_controller_asset_is_valid_javascript():
    assert _ASSET.is_file()
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    subprocess.run([node, "--check", str(_ASSET)], check=True)


def test_script_mounts_on_the_plot_placeholder():
    assert fluid_playback_script().endswith(
        "window.__mlekticFluid && window.__mlekticFluid.mount(document.getElementById('{plot_id}'), {autoplay: false});"
    )
    assert fluid_playback_script(autoplay=True).endswith("{autoplay: true});")


def test_export_mounts_controller_on_the_exported_plot(tmp_path, animated_figure):
    html = export_figure(animated_figure, tmp_path / "animated", include_plotly="cdn").read_text(encoding="utf-8")

    mounts = MOUNT.findall(html)
    assert mounts == [(_plot_ids(html)[0], "false")]
    assert "Mlektic fluid playback" in html


def test_export_can_opt_out_and_skips_static_figures(tmp_path, animated_figure):
    opted_out = export_figure(
        animated_figure, tmp_path / "plain", include_plotly="cdn", fluid_playback=False
    ).read_text(encoding="utf-8")
    static = export_figure(
        go.Figure(go.Scatter(x=[0, 1], y=[0, 1])), tmp_path / "static", include_plotly="cdn"
    ).read_text(encoding="utf-8")

    assert "__mlekticFluid" not in opted_out
    assert "__mlekticFluid" not in static
    with pytest.raises(TypeError, match="fluid_playback"):
        export_figure(animated_figure, tmp_path / "bad", fluid_playback="yes")


def test_export_auto_play_is_driven_by_the_controller(tmp_path, animated_figure):
    fluid = export_figure(animated_figure, tmp_path / "fluid", include_plotly="cdn", auto_play=True)
    plain = export_figure(
        animated_figure, tmp_path / "plain", include_plotly="cdn", auto_play=True, fluid_playback=False
    )
    fluid_html = fluid.read_text(encoding="utf-8")
    plain_html = plain.read_text(encoding="utf-8")

    plot_id = _plot_ids(fluid_html)[0]
    assert MOUNT.findall(fluid_html) == [(plot_id, "true")]
    # Plotly's own auto-play call is replaced, not duplicated.
    plotly_autoplay = re.compile(r"Plotly\.animate\(\s*'[^']+',\s*null")
    assert not plotly_autoplay.search(fluid_html)
    assert plotly_autoplay.search(plain_html)


def test_show_optimized_mounts_controller(animated_figure):
    pytest.importorskip("IPython")
    html = show_optimized(animated_figure).data
    plain = show_optimized(animated_figure, fluid_playback=False).data

    assert MOUNT.findall(html) == [(_plot_ids(html)[0], "false")]
    assert "__mlekticFluid" not in plain


def test_notebook_display_paths_mount_controller(animated_figure, colab_renderer, monkeypatch):
    html = animated_figure._repr_mimebundle_()["text/html"]
    assert MOUNT.findall(html) == [(_plot_ids(html)[0], "false")]
    assert MOUNT.search(animated_figure._repr_html_())

    shown = []
    monkeypatch.setattr(pio, "show", lambda fig, *args, **kwargs: shown.append(kwargs))
    animated_figure._ipython_display_()
    scripts = shown[0]["post_script"]
    assert scripts[-1] == fluid_playback_script()
    assert all("mlektic-style-" in script for script in scripts[:-1])


def test_show_keeps_a_caller_post_script(animated_figure, colab_renderer, monkeypatch):
    shown = []
    monkeypatch.setattr(pio, "show", lambda fig, *args, **kwargs: shown.append(kwargs))
    animated_figure.show(post_script="console.log('mine')")

    scripts = shown[0]["post_script"]
    assert scripts[0] == "console.log('mine')"
    assert scripts[-1] == fluid_playback_script()


def test_display_patch_leaves_figure_data_untouched(animated_figure, colab_renderer):
    before = animated_figure.to_json()
    animated_figure._repr_mimebundle_()

    assert animated_figure.to_json() == before
    assert "__mlekticFluid" not in before
