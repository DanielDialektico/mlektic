"""Reliable display and HTML export helpers for mathematical figures."""

from __future__ import annotations

from pathlib import Path

from ..visualization.playback import figure_scripts, has_frames


def _resolved_responsive(fig, responsive):
    """Resolve an explicit export choice or inherit the figure contract."""
    if responsive is not None and not isinstance(responsive, bool):
        raise TypeError("responsive must be a boolean value or None.")
    if responsive is not None:
        return responsive
    metadata = fig.layout.meta if isinstance(fig.layout.meta, dict) else {}
    visual = metadata.get("mlektic_visual", {})
    return bool(visual.get("responsive", False))


def _fluid_kwargs(fig, fluid_playback, auto_play):
    """Plotly HTML options that let the fluid controller drive animation."""
    if not isinstance(fluid_playback, bool):
        raise TypeError("fluid_playback must be a boolean value.")
    scripts = figure_scripts(fig, fluid=fluid_playback, autoplay=auto_play)
    fluid = fluid_playback and has_frames(fig)
    # With fluid playback the controller starts playback itself, so Plotly's own
    # loop never competes; the button stylesheet travels with every figure.
    options = {"auto_play": False if fluid else auto_play}
    if scripts:
        options["post_script"] = scripts
    return options


def show_optimized(fig, *, fluid_playback=True):
    """Render a Plotly figure as lightweight notebook HTML.

    This path is useful in Google Colab and other environments where the widget
    renderer is expensive. MathJax is explicitly loaded so equations are not
    left as raw LaTeX.

    Args:
        fig: Plotly figure to display.
        fluid_playback: Replay already-rendered equations when stepping or
            playing animation frames instead of redrawing them. The figure looks
            identical either way; ``False`` keeps Plotly's own playback.
    """
    from IPython.display import HTML

    fig_height = fig.layout.height if fig.layout.height else 600
    wrapper_height = fig_height + 40
    responsive = _resolved_responsive(fig, None)
    html_str = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        include_mathjax="cdn",
        config={"responsive": responsive},
        **_fluid_kwargs(fig, fluid_playback, False),
    )
    html_str = compact_html_escapes(html_str)
    return HTML(f'<div style="height: {wrapper_height}px; width: 100%; overflow: hidden;">{html_str}</div>')


# Plotly writes each "/" of its embedded JSON as a backslash followed by
# "u002f" (six characters). A backslash followed by "/" is the equivalent
# two-character JSON escape.
_SLASH_ESCAPE = chr(92) + "u002f"
_SHORT_SLASH_ESCAPE = chr(92) + "/"


def compact_html_escapes(html: str) -> str:
    """Shorten Plotly's six-character escape of ``/`` to the two-character one.

    Plotly escapes every slash in the embedded JSON so the payload can never
    close its ``<script>`` element. The short escape is standard JSON with the
    same guarantee and a third of the length. Base64 typed arrays are full of
    slashes, so this matters for figures with numeric arrays: a quantized
    heatmap animation shrinks by about a third.
    """
    return html.replace(_SLASH_ESCAPE, _SHORT_SLASH_ESCAPE)


def export_figure(
    fig,
    path,
    *,
    include_plotly="inline",
    include_mathjax="cdn",
    responsive=None,
    auto_play=False,
    fluid_playback=True,
):
    """Export a complete HTML document with explicit dependency semantics.

    Args:
        fig: Plotly figure to export.
        path: Destination path. The ``.html`` suffix is added when omitted.
        include_plotly: ``"inline"`` for a self-contained Plotly runtime or
            ``"cdn"`` for a smaller network-dependent document.
        include_mathjax: ``"cdn"`` to render LaTeX equations or ``False`` to
            omit MathJax intentionally. Plotly does not provide a supported
            self-contained MathJax bundle, so this choice remains explicit.
        responsive: Whether Plotly should resize the figure with its container.
            ``None`` inherits the figure's visual contract; figures without
            that metadata remain fixed-size for backward compatibility.
        auto_play: Whether an animated figure should start automatically.
        fluid_playback: Replay already-rendered equations when stepping or
            playing animation frames instead of redrawing them. The figure looks
            identical either way; ``False`` keeps Plotly's own playback.

    Returns:
        The resolved :class:`pathlib.Path` written to disk.
    """
    if not isinstance(include_plotly, str) or include_plotly not in {"inline", "cdn"}:
        raise ValueError("include_plotly must be 'inline' or 'cdn'.")
    if include_mathjax != "cdn" and include_mathjax is not False:
        raise ValueError("include_mathjax must be 'cdn' or False.")
    responsive = _resolved_responsive(fig, responsive)
    if not isinstance(auto_play, bool):
        raise TypeError("auto_play must be a boolean value.")
    fluid = _fluid_kwargs(fig, fluid_playback, auto_play)

    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or pathlib.Path.")
    destination = Path(path).expanduser()
    if destination.suffix == "":
        destination = destination.with_suffix(".html")
    if destination.suffix.lower() not in {".html", ".htm"}:
        raise ValueError("path must use the .html or .htm suffix.")
    destination.parent.mkdir(parents=True, exist_ok=True)

    html = fig.to_html(
        include_plotlyjs=True if include_plotly == "inline" else "cdn",
        include_mathjax=include_mathjax or False,
        full_html=True,
        config={"responsive": responsive},
        **fluid,
    )
    destination.write_text(compact_html_escapes(html), encoding="utf-8")
    return destination.resolve()


__all__ = ["compact_html_escapes", "export_figure", "show_optimized"]
