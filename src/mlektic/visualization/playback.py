"""Fluid playback for animated figures in notebook and HTML output.

Plotly redraws the whole figure, and MathJax re-typesets every equation, each
time a frame changes layout annotations. ``assets/fluid_playback.js`` replays
the equations Plotly already rendered instead, so stepping and playing stay
smooth while every element keeps its exact authored appearance and position.

The script travels with the figure's HTML as a Plotly ``post_script``, which
works in Colab, classic Jupyter, exported files and any other HTML renderer.
Renderers that do not emit HTML (for example the JupyterLab mimetype
extension) simply ignore it and keep Plotly's own playback; the "Next step"
button, which only the controller can drive, is left out there.
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


def _with_post_script(kwargs: dict, *scripts: str) -> dict:
    scripts = [script for script in scripts if script]
    if not scripts:
        return kwargs
    existing = kwargs.get("post_script")
    if existing is None:
        combined = list(scripts)
    elif isinstance(existing, (list, tuple)):
        combined = [*existing, *scripts]
    else:
        combined = [existing, *scripts]
    return {**kwargs, "post_script": combined}


def figure_scripts(fig, *, fluid: bool = True, autoplay: bool = False) -> list[str]:
    """Every ``post_script`` the figure needs: its button stylesheet and, if animated, playback."""
    from .controls import figure_style_script

    scripts = [figure_style_script(fig)]
    if fluid and has_frames(fig):
        scripts.append(fluid_playback_script(autoplay=autoplay))
    return [script for script in scripts if script]


def renderer_runs_scripts(renderer=None) -> bool:
    """Whether Plotly's renderer (default: the active one) runs a figure's post-scripts.

    HTML renderers (Colab, classic Jupyter, the browser, iframes) accept a
    ``post_script``; mimetype renderers (JupyterLab, VS Code's Plotly renderer)
    draw the figure JSON themselves and run none.
    """
    import plotly.io as pio

    names = [name.strip() for name in str(renderer or pio.renderers.default or "").split("+") if name.strip()]
    try:
        return bool(names) and all(hasattr(pio.renderers[name], "post_script") for name in names)
    except (KeyError, ValueError):
        return False


def _displayable(fig, renderer=None):
    """The figure to hand to a renderer: without the step button where no controller runs."""
    from .step_control import without_step_control

    return fig if renderer_runs_scripts(renderer) else without_step_control(fig)


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
        target = _displayable(fig)
        if target is not fig:
            return target._repr_mimebundle_(include, exclude, validate, **kwargs)
        kwargs = _with_post_script(kwargs, *figure_scripts(fig))
        return _original_bundle(include, exclude, validate, **kwargs)

    def _patched_show(*args, **kwargs):
        import plotly.io as pio

        target = _displayable(fig, kwargs.get("renderer") or (args[0] if args else None))
        if target is not fig:
            return pio.show(target, *args, **kwargs)
        kwargs = _with_post_script(kwargs, *figure_scripts(fig))
        return _original_show(*args, **kwargs)

    def _patched_ipython_display():
        import plotly.io as pio

        if pio.renderers.render_on_display and pio.renderers.default:
            target = _displayable(fig)
            pio.show(target, **_with_post_script({}, *figure_scripts(target)))
        else:
            print(repr(fig))

    fig.show = _patched_show
    fig._ipython_display_ = _patched_ipython_display
    fig._repr_mimebundle_ = _patched_bundle
    return fig


__all__ = ["attach_fluid_playback", "figure_scripts", "fluid_playback_script", "has_frames", "renderer_runs_scripts"]
