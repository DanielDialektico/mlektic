"""Reliable display and HTML export helpers for mathematical figures."""

from __future__ import annotations

from pathlib import Path


def show_optimized(fig):
    """Render a Plotly figure as lightweight notebook HTML.

    This path is useful in Google Colab and other environments where the widget
    renderer is expensive. MathJax is explicitly loaded so equations are not
    left as raw LaTeX.
    """
    from IPython.display import HTML

    fig_height = fig.layout.height if fig.layout.height else 600
    wrapper_height = fig_height + 40
    html_str = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        auto_play=False,
        include_mathjax="cdn",
    )
    return HTML(f'<div style="height: {wrapper_height}px; width: 100%; overflow: hidden;">{html_str}</div>')


def export_figure(
    fig,
    path,
    *,
    include_plotly="inline",
    include_mathjax="cdn",
    responsive=False,
    auto_play=False,
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
            ``False`` preserves the current fixed-size visual default.
        auto_play: Whether an animated figure should start automatically.

    Returns:
        The resolved :class:`pathlib.Path` written to disk.
    """
    if not isinstance(include_plotly, str) or include_plotly not in {"inline", "cdn"}:
        raise ValueError("include_plotly must be 'inline' or 'cdn'.")
    if include_mathjax != "cdn" and include_mathjax is not False:
        raise ValueError("include_mathjax must be 'cdn' or False.")
    if not isinstance(responsive, bool) or not isinstance(auto_play, bool):
        raise TypeError("responsive and auto_play must be boolean values.")

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
        auto_play=auto_play,
        config={"responsive": responsive},
    )
    destination.write_text(html, encoding="utf-8")
    return destination.resolve()


__all__ = ["export_figure", "show_optimized"]
