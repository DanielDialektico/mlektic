import json
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


# A notebook page can hold one copy of Plotly.js per output. The copy that
# window.Plotly points to gives a figure drawn by another copy fresh event
# emitters the first time it redraws it. This harness reproduces that: the
# fake Plotly replaces both emitters on its first call for the figure.
_FOREIGN_PLOTLY_HARNESS = r"""
const fs = require("fs");
const { EventEmitter } = require("events");
global.window = global;
global.MathJax = { Hub: { Queue() {}, Typeset() {} } };
global.document = {
  body: { contains: () => true },
  createElementNS: () => ({ innerHTML: "", firstChild: null }),
};
const frames = ["0", "1", "2", "3"].map((name) => ({
  name, data: [{ type: "scatter3d", x: [Number(name)] }], layout: { annotations: [] },
}));
const steps = frames.map((f) => ({ method: "animate", args: [[f.name], {}], label: f.name }));
const buttons = [
  { method: "animate", args: [null, { frame: { duration: 1 }, transition: { duration: 0 } }] },
  { method: "animate", args: [[null], {}] },
];
const clone = (v) => JSON.parse(JSON.stringify(v));
const gd = {
  data: [{ type: "scatter3d" }],
  layout: { sliders: [{ steps, active: 0 }], updatemenus: [{ buttons }] },
  _context: {},
  _transitionData: {
    _frames: frames, _frameHash: Object.fromEntries(frames.map((f) => [f.name, f])), _frameQueue: [],
  },
  addEventListener() {},
};
gd._fullLayout = {
  sliders: clone(gd.layout.sliders), updatemenus: clone(gd.layout.updatemenus), annotations: [],
  width: 600, height: 400, _infolayer: { node: () => ({ querySelector: () => null }) },
};
function emitters() {
  const ev = new EventEmitter(), internal = new EventEmitter();
  gd._ev = ev; gd._internalEv = internal;
  gd.on = ev.on.bind(ev);
  gd._internalOn = internal.on.bind(internal);
  gd.emit = (name, data) => { ev.emit(name, data); internal.emit(name, data); };
}
emitters();
// Plotly's own observer that moves the slider with the animation.
const followed = [];
gd._internalOn("plotly_animatingframe", (e) => followed.push(e.name));
let foreign = true;
function redraw(name) {
  if (foreign) { emitters(); foreign = false; }
  if (name) { gd._fullLayout._currentFrame = name; gd.emit("plotly_animatingframe", { name }); }
  return Promise.resolve();
}
global.Plotly = {
  animate: (_gd, target) => redraw(Array.isArray(target) ? target[0] : null),
  update: () => redraw(null),
};
eval(fs.readFileSync(process.argv[2], "utf8"));
window.__mlekticFluid.mount(gd, { prewarm: false });
const settle = () => new Promise((resolve) => setTimeout(resolve, 30));
function slide(index) {
  gd._fullLayout.sliders[0].active = index;
  gd.emit("plotly_sliderchange", { interaction: true, step: gd._fullLayout.sliders[0].steps[index] });
}
(async () => {
  const result = {};
  slide(1); await settle();
  result.first = gd._fullLayout._currentFrame;
  slide(2); await settle();
  result.afterForeignRedraw = gd._fullLayout._currentFrame;
  followed.length = 0;
  gd.emit("plotly_buttonclicked", { button: gd._fullLayout.updatemenus[0].buttons[0] });
  await settle(); await settle(); await settle();
  result.played = gd._fullLayout._currentFrame;
  result.followed = followed.slice();
  result.listeners = {
    slider: gd._ev.listenerCount("plotly_sliderchange"),
    buttons: gd._ev.listenerCount("plotly_buttonclicked"),
    follower: gd._internalEv.listenerCount("plotly_animatingframe"),
  };
  console.log(JSON.stringify(result));
})();
"""


def test_controls_survive_a_redraw_by_another_plotly_copy(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    harness = tmp_path / "harness.js"
    harness.write_text(_FOREIGN_PLOTLY_HARNESS, encoding="utf-8")
    run = subprocess.run([node, str(harness), str(_ASSET)], check=True, capture_output=True, text=True)
    result = json.loads(run.stdout.strip().splitlines()[-1])
    assert result["first"] == "1"
    # Before the fix the redraw dropped the controller's listeners and the slider stopped here.
    assert result["afterForeignRedraw"] == "2"
    assert result["played"] == "3"
    # Plotly's own slider observer still hears every frame, so the slider follows Play.
    assert result["followed"] == ["0", "1", "2", "3"]
    assert result["listeners"] == {"slider": 1, "buttons": 1, "follower": 1}


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
