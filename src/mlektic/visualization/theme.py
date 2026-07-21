"""Theme utilities for Mlektic Plotly visualization.

Supports one standard theme:
  - ``"classic"``  — the original dark-mode theme (default).
"""

from __future__ import annotations

from typing import Any, Dict, List

# ── colour palettes ─────────────────────────────────────────────

_CLASSIC = dict(
    bg="#111111",
    text="white",
    text_secondary="white",
    font_family="Helvetica",
    template="plotly_dark",
    data_marker="#7dd3fc",
    data_marker_border="#0ea5e9",
    data_opacity=0.85,
    model_line="#EF553B",
    model_line_width=4,
    loss_line="#00cc96",
    loss_line_width=3,
    surface_colorscale=None,   # plotly default
    surface_opacity=0.55,
    legend_bg="rgba(220,220,220,0.85)",
    legend_border="rgba(0,0,0,0.6)",
    legend_font_color="black",
    btn_bg="white",
    btn_border="rgba(0,0,0,0.25)",
    btn_font_color="black",
    btn_border_width=1,
    annotation_color="white",
    title_size=24,
    annotation_size=16,
    slider_font_color=None,
    plot_bg=None,
    paper_bg=None,
    grid_color=None,
    transition_duration=0,
)

_THEMES: Dict[str, Dict[str, Any]] = {
    "classic": _CLASSIC,
}


def _resolve(theme: str | None = None) -> Dict[str, Any]:
    """Return the palette dict for *theme*, defaulting to ``"classic"``."""
    key = (theme or "classic").lower()
    if key not in _THEMES:
        key = "classic"
    return _THEMES[key]


# ── public helpers (backward-compatible signatures) ─────────────

def get_base_layout(
    title: str,
    height: int = 600,
    width: int = 1100,
    margin_t: int = 150,
    *,
    theme: str | None = None,
) -> Dict[str, Any]:
    """Get the base layout for all Plotly dark mode figures."""
    p = _resolve(theme)
    layout = dict(
        template=p["template"],
        height=height,
        width=width,
        autosize=False,
        font=dict(family=p["font_family"], color=p["text"]),
        title=dict(
            text=title,
            y=0.96,
            x=0.5,
            xanchor="center",
            font=dict(color=p["text"], size=p["title_size"]),
        ),
        margin=dict(t=margin_t, r=30, l=60, b=70),
    )
    if p["plot_bg"]:
        layout["plot_bgcolor"] = p["plot_bg"]
    if p["paper_bg"]:
        layout["paper_bgcolor"] = p["paper_bg"]
    return layout


def get_legend_props(
    x: float = 0.985,
    y: float = 0.02,
    *,
    theme: str | None = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Get standard legend properties."""
    p = _resolve(theme)
    props = dict(
        x=x,
        y=y,
        xanchor=overrides.pop("xanchor", "right"),
        yanchor=overrides.pop("yanchor", "bottom"),
        bgcolor=p["legend_bg"],
        bordercolor=p["legend_border"],
        borderwidth=1,
        font=dict(size=12, color=p["legend_font_color"]),
    )
    props.update(overrides)
    return props


def _build_play_pause_buttons(
    frame_duration: int = 80,
    *,
    theme: str | None = None,
) -> List[Dict[str, Any]]:
    p = _resolve(theme)
    td = p["transition_duration"]
    return [
        dict(
            label="▶  Play",
            method="animate",
            args=[
                None,
                {
                    "frame": {"duration": frame_duration, "redraw": True},
                    "transition": {"duration": td},
                },
            ],
        ),
        dict(
            label="⏸ Pause",
            method="animate",
            args=[
                [None],
                {
                    "frame": {"duration": 0, "redraw": False},
                    "mode": "immediate",
                },
            ],
        ),
    ]


def get_updatemenus(
    frame_duration: int = 80,
    x: float = 0.10,
    y: float = 1.14,
    *,
    theme: str | None = None,
) -> List[Dict[str, Any]]:
    """Build the play/pause button menu."""
    p = _resolve(theme)
    return [
        dict(
            type="buttons",
            showactive=False,
            direction="left",
            x=x,
            y=y,
            bgcolor=p["btn_bg"],
            bordercolor=p["btn_border"],
            borderwidth=1,
            font=dict(color=p["btn_font_color"], size=14),
            buttons=_build_play_pause_buttons(frame_duration, theme=theme),
        )
    ]


def get_sliders(
    steps_n: int,
    *,
    theme: str | None = None,
) -> List[Dict[str, Any]]:
    """Build the time slider."""
    p = _resolve(theme)
    cv = dict(prefix="Step: ")
    if p["slider_font_color"]:
        cv["font"] = dict(color=p["slider_font_color"])
    return [
        dict(
            active=0,
            currentvalue=cv,
            pad=dict(t=55),
            steps=[
                dict(
                    method="animate",
                    args=[
                        [str(t)],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": True},
                            "transition": {"duration": 0},
                        },
                    ],
                    label=str(t),
                )
                for t in range(steps_n)
            ],
        )
    ]


def create_annotation(
    text: str,
    y: float,
    size: int = 16,
    *,
    theme: str | None = None,
    x: float = 0.5,
    xanchor: str = "center",
    yanchor: str = "bottom",
) -> Dict[str, Any]:
    """Create a standard centered top annotation for formulas."""
    p = _resolve(theme)
    sz = size if theme is None else min(size, p["annotation_size"] + 4)
    return dict(
        x=x,
        y=y,
        xref="paper",
        yref="paper",
        text=text,
        showarrow=False,
        xanchor=xanchor,
        yanchor=yanchor,
        font=dict(color=p["annotation_color"], size=sz),
    )


# ── style helpers for traces ────────────────────────────────────

def data_marker_style(*, theme: str | None = None) -> Dict[str, Any]:
    """Return ``marker`` dict for data scatter traces."""
    p = _resolve(theme)
    m: Dict[str, Any] = dict(size=7, opacity=p["data_opacity"])
    if p["data_marker"]:
        m["color"] = p["data_marker"]
    if p["data_marker_border"]:
        m["line"] = dict(width=1, color=p["data_marker_border"])
    return m


def model_line_style(*, theme: str | None = None) -> Dict[str, Any]:
    """Return ``line`` dict for model / prediction traces."""
    p = _resolve(theme)
    d: Dict[str, Any] = dict(width=p["model_line_width"], simplify=False)
    if p["model_line"]:
        d["color"] = p["model_line"]
    return d


def loss_line_style(*, theme: str | None = None) -> Dict[str, Any]:
    """Return ``line`` dict for loss traces."""
    p = _resolve(theme)
    d: Dict[str, Any] = dict(width=p["loss_line_width"], simplify=False)
    if p["loss_line"]:
        d["color"] = p["loss_line"]
    return d


def surface_style(*, theme: str | None = None) -> Dict[str, Any]:
    """Return kwargs dict for ``go.Surface`` traces."""
    p = _resolve(theme)
    d: Dict[str, Any] = dict(opacity=p["surface_opacity"], showscale=False)
    if p["surface_colorscale"]:
        d["colorscale"] = p["surface_colorscale"]
    return d


def data_3d_marker_style(*, theme: str | None = None) -> Dict[str, Any]:
    """Return ``marker`` dict for 3D scatter data traces."""
    p = _resolve(theme)
    m: Dict[str, Any] = dict(size=4, opacity=p["data_opacity"])
    if p["data_marker"]:
        m["color"] = p["data_marker"]
    if p["data_marker_border"]:
        m["line"] = dict(width=0.5, color=p["data_marker_border"])
    return m


# ── JS injection for stable button styling ───────────────────

def get_button_highlight_script(*, theme: str | None = None) -> str:
    """Return CSS that keeps Play/Pause visually stable during Plotly redraws.

    The historical function name is retained for backward compatibility. Plotly
    rebuilds update-menu SVG nodes while animating 3D traces, so styling every
    current and future button through CSS is more reliable than tracking an
    active node in JavaScript.
    """
    p = _resolve(theme)
    button_bg = p.get("btn_bg", "white")
    text_color = p.get("btn_font_color", "black")

    return (
        "<script>\n"
        "(function() {\n"
        "  if (!document.getElementById('mlektic-stable-button-style')) {\n"
        "    var style = document.createElement('style');\n"
        "    style.id = 'mlektic-stable-button-style';\n"
        "    style.textContent =\n"
        "      '.js-plotly-plot .updatemenu-button rect {' +\n"
        f"      'fill: {button_bg} !important;}}' +\n"
        "      '.js-plotly-plot .updatemenu-button text {' +\n"
        f"      'fill: {text_color} !important;}}';\n"
        "    document.head.appendChild(style);\n"
        "  }\n"
        "})();\n"
        "</script>"
    )


def attach_highlight(fig, *, theme: str | None = None):
    """Patch *fig* so stable button CSS is injected in Jupyter.

    Two display paths are patched:

    * ``fig`` as the last expression in a cell → ``_repr_html_()``
    * ``fig.show()`` → original show + ``IPython.display.HTML(script)``

    Returns the same *fig* (mutated in-place) for convenience.
    """
    _script = get_button_highlight_script(theme=theme)

    # 1) Patch _repr_html_ (used when `fig` is the cell's last expression)
    _original_repr = getattr(fig, "_repr_html_", None)

    def _patched_repr():
        base_html = _original_repr() if _original_repr else ""
        return base_html + _script

    fig._repr_html_ = _patched_repr

    # 2) Patch show() (used when user calls fig.show() explicitly)
    _original_show = fig.show

    def _patched_show(*args, **kwargs):
        _original_show(*args, **kwargs)
        try:
            from IPython.display import HTML, display

            display(HTML(_script))
        except Exception:
            pass  # not in a notebook — ignore silently

    fig.show = _patched_show
    return fig


def configure_animation(fig, frame_duration: int, transition_duration: int | None = None):
    """Configure transitions and redraw frames that modify layout mathematics."""
    if frame_duration < 0:
        raise ValueError("frame_duration must be non-negative.")
    transition = min(round(frame_duration * 0.6), 160) if transition_duration is None else transition_duration
    if transition < 0:
        raise ValueError("transition_duration must be non-negative or None.")
    if frame_duration > 0 and transition >= frame_duration:
        transition = min(round(frame_duration * 0.6), 160)
    has_non_scatter_traces = any(
        getattr(trace, "type", "") in {"surface", "scatter3d", "mesh3d", "volume", "isosurface"}
        for trace in fig.data
    )
    has_layout_updates = any(
        frame.layout is not None and bool(frame.layout.to_plotly_json())
        for frame in fig.frames or ()
    )
    requires_redraw = has_non_scatter_traces or has_layout_updates
    if has_layout_updates:
        fig.update_layout(transition=dict(ordering="traces first"))
    for menu in fig.layout.updatemenus or ():
        for button in menu.buttons or ():
            if button.method != "animate" or not button.args:
                continue
            options = button.args[1] if len(button.args) > 1 else None
            if not isinstance(options, dict):
                continue
            frame = options.setdefault("frame", {})
            if frame.get("duration", 0) > 0:
                frame["duration"] = frame_duration
                frame["redraw"] = requires_redraw
                options["transition"] = {"duration": transition, "easing": "linear"}
    return fig
