"""Fluid playback for animated figures in notebook and HTML output.

Plotly redraws the whole figure, and MathJax re-typesets every equation, each
time a frame changes layout annotations. ``assets/fluid_playback.js`` replays
the equations Plotly already rendered instead, so stepping and playing stay
smooth while every element keeps its exact authored appearance and position.

The script travels with the figure's HTML as a Plotly ``post_script``, which
works in Colab, classic Jupyter, exported files and any other HTML renderer.
Renderers that do not emit HTML (for example the JupyterLab mimetype
extension) simply ignore it and keep Plotly's own playback.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_ASSET = Path(__file__).resolve().parent / "assets" / "fluid_playback.js"


@lru_cache(maxsize=1)
def _controller_source() -> str:
    return _ASSET.read_text(encoding="utf-8")


def fluid_playback_script(*, autoplay: bool = False) -> str:
    """Return the Plotly ``post_script`` that mounts the fluid controller.

    Args:
        autoplay: Start playback once the figure is ready. Use this instead of
            Plotly's ``auto_play`` so the controller drives the animation.
    """
    mount = (
        "window.__mlekticFluid && window.__mlekticFluid.mount("
        "document.getElementById('{plot_id}'), "
        f"{{autoplay: {'true' if autoplay else 'false'}}});"
    )
    return _controller_source() + "\n" + mount


def has_frames(fig) -> bool:
    """Whether *fig* is animated and can use the fluid controller."""
    return bool(getattr(fig, "frames", None))


def _with_post_script(kwargs: dict, script: str) -> dict:
    existing = kwargs.get("post_script")
    if existing is None:
        scripts = [script]
    elif isinstance(existing, (list, tuple)):
        scripts = [*existing, script]
    else:
        scripts = [existing, script]
    return {**kwargs, "post_script": scripts}


def attach_fluid_playback(fig):
    """Patch *fig* so notebook display mounts the fluid controller.

    Three display paths are patched:

    * ``fig`` as the last expression in a cell → ``_ipython_display_()``
    * ``fig.show()`` → ``show(post_script=...)``
    * ``_repr_mimebundle_()`` / ``_repr_html_()`` for other rich displays

    Returns the same *fig* (mutated in-place) for convenience.
    """
    _original_show = fig.show
    _original_bundle = fig._repr_mimebundle_

    def _patched_bundle(include=None, exclude=None, validate=True, **kwargs):
        if has_frames(fig):
            kwargs = _with_post_script(kwargs, fluid_playback_script())
        return _original_bundle(include, exclude, validate, **kwargs)

    def _patched_show(*args, **kwargs):
        if has_frames(fig):
            kwargs = _with_post_script(kwargs, fluid_playback_script())
        return _original_show(*args, **kwargs)

    def _patched_ipython_display():
        import plotly.io as pio

        if pio.renderers.render_on_display and pio.renderers.default:
            kwargs = _with_post_script({}, fluid_playback_script()) if has_frames(fig) else {}
            pio.show(fig, **kwargs)
        else:
            print(repr(fig))

    fig.show = _patched_show
    fig._ipython_display_ = _patched_ipython_display
    fig._repr_mimebundle_ = _patched_bundle
    return fig


__all__ = ["attach_fluid_playback", "fluid_playback_script", "has_frames"]
