"""A "Step" button beside Play/Pause.

Every animated figure gets a button that shows the next state and stops
there, so a student can read each change before moving on. Where every frame
carries many equations (multiclass models with several classes and inputs),
typesetting makes continuous playback slow; there the step button takes
Play's place and Play/Pause move below it, suggesting step-by-step reading
without removing playback. Elsewhere Play stays first: under it, a "Step"
button mirrors Play's label, style, and size; where it has to share Play's row,
a slightly smaller "Step" follows Pause.

Plotly has no command that shows "the next frame", so the button uses
Plotly's ``"skip"`` method (a click does nothing by itself) and the fluid
playback controller (``assets/fluid_playback.js``) advances one frame when it
sees the click. Every display path of mlektic installs that controller;
:func:`without_step_control` removes the button for exports that do not.

Both menus share one left edge in the top margin, just above the plotting
area, separated by pixel padding. Layouts that would cover an annotation (an
equation printed just above the plot) are skipped: the controls then share one
row, or keep the place the builder gave Play. The leading control always comes
first.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from .legend_layout import _annotation_boxes, _figure_size, _overlaps

STEP_COMMAND = "mlektic-step"
STEP_LABEL = "⏭ Step"
SHORT_STEP_LABEL = "Step"  # under Play, mirroring Play's own label
# LaTeX characters a single frame may carry before playback counts as heavy:
# multiclass views of several classes and inputs carry more than this.
HEAVY_FRAME_TEX_CHARS = 900
_GAP_PX = 8.0  # between the two rows, and between the stack and the plot
_CLEARANCE_PX = (14.0, 6.0)  # kept free around the stack (x, y): widths are estimates
_TOP_PX = 6.0  # the stack never starts higher than this below the figure's top edge
_EDGE_PX = 6.0  # nor further left than this
_SLIDER_PX = 80.0  # a slider's current-value label, rail, and tick labels


def _is_play(button: Any) -> bool:
    return button.method == "animate" and bool(button.args) and button.args[0] is None


def _play_menu(fig: Any) -> int | None:
    for index, menu in enumerate(fig.layout.updatemenus or ()):
        if any(_is_play(button) for button in menu.buttons or ()):
            return index
    return None


def is_step_menu(menu: Any) -> bool:
    """Whether an update menu is the step control added by :func:`add_step_control`."""
    return any(
        button.method == "skip" and list(button.args or ())[:1] == [STEP_COMMAND] for button in menu.buttons or ()
    )


def frame_tex_chars(fig: Any) -> int:
    """LaTeX characters in the frame that carries the most equations."""
    heaviest = 0
    for frame in fig.frames or ():
        layout = frame.layout
        if layout is None:
            continue
        annotations = list(layout.annotations or ())
        if layout.scene is not None:
            annotations += list(layout.scene.annotations or ())
        heaviest = max(heaviest, sum(len(str(a.text)) for a in annotations if "$" in str(a.text or "")))
    return heaviest


def _mirrored_label(play: Any) -> str:
    """Play's own label with "Step" in place of "Play" (``▶  Play`` → ``⏭  Step``)."""
    label = next((str(b.label) for b in play.buttons if _is_play(b) and b.label), "Play")
    if "Play" not in label:
        return SHORT_STEP_LABEL
    return label.replace("Play", "Step").replace("▶", "⏭")


def _button_height(font_size: float) -> float:
    # Measured: Plotly never draws a button shorter than 33 px, whatever its font.
    return max(34.0, 2.4 * font_size)


def _menu_width(labels: list[str], font_size: float) -> float:
    # Measured on Plotly's buttons: about half an em per character plus padding.
    return sum(0.52 * font_size * len(label) + 22.0 for label in labels) + 2.0 * max(0, len(labels) - 1)


def _slider_boxes(fig: Any, margin: dict, plot_w: float, plot_h: float) -> list[tuple[float, float, float, float]]:
    """Paper boxes of each slider: its current-value label, rail, and tick labels."""
    boxes = []
    for slider in fig.layout.sliders or ():
        x = float(slider.x if slider.x is not None else 0.0)
        length = float(slider.len if slider.len is not None else 1.0)
        y = float(slider.y if slider.y is not None else 0.0)
        pad_t = float(slider.pad.t or 0) if slider.pad is not None else 0.0
        top_px = margin["t"] + (1.0 - y) * plot_h + pad_t
        if (slider.yanchor or "top") == "bottom":
            top_px -= _SLIDER_PX
        bottom_px = top_px + _SLIDER_PX
        boxes.append((x, x + length, 1.0 - (bottom_px - margin["t"]) / plot_h, 1.0 - (top_px - margin["t"]) / plot_h))
    return boxes


def add_step_control(fig: Any) -> dict[str, Any]:
    """Add the step button to an animated figure and report where it went."""
    play_index = _play_menu(fig)
    if play_index is None or not fig.frames:
        return {}
    menus = list(fig.layout.updatemenus)
    if any(is_step_menu(menu) for menu in menus):
        return {}
    play = menus[play_index]
    tex_chars = frame_tex_chars(fig)
    primary = tex_chars >= HEAVY_FRAME_TEX_CHARS

    width, height, margin = _figure_size(fig)
    plot_w = max(1.0, width - margin["l"] - margin["r"])
    plot_h = max(1.0, height - margin["t"] - margin["b"])

    def paper_y(top_px: float) -> float:
        return 1.0 - (top_px - margin["t"]) / plot_h

    def paper_x(left_px: float) -> float:
        return (left_px - margin["l"]) / plot_w

    play_font = float(play.font.size if play.font is not None and play.font.size else 14)
    step_font = play_font if primary else max(11.0, round(0.86 * play_font))
    play_h, step_h = _button_height(play_font), _button_height(step_font)
    play_w = _menu_width([str(b.label) for b in play.buttons], play_font)
    step_w = _menu_width([STEP_LABEL], step_font)
    # Under Play, a supporting button mirrors it: "Step" in Play's own style and
    # size, so the two read as one column. Beside Pause it keeps the long label.
    under_label = STEP_LABEL if primary else _mirrored_label(play)
    under_w = _menu_width([under_label], play_font)

    # Play's current box, in pixels from the figure's top-left corner.
    anchor_x = float(play.x if play.x is not None else -0.05)
    anchor_px = margin["l"] + anchor_x * plot_w
    xanchor = play.xanchor or "right"
    left_px = {"right": anchor_px - play_w, "center": anchor_px - play_w / 2}.get(xanchor, anchor_px)
    left_px = max(_EDGE_PX, left_px)
    # Both rows hang from the top edge of the plotting area (``y=1``, anchored
    # at their bottom) and are separated with pixel padding. Plotly shrinks the
    # plotting area for the slider (auto-margin), so a position written as a
    # fraction of its height would drift; the top edge and pixel pads do not.
    labels = _annotation_boxes(fig, {}, plot_w, plot_h) + _slider_boxes(fig, margin, plot_w, plot_h)

    def clear(top: float, bottom: float, left: float, right: float) -> bool:
        dx, dy = _CLEARANCE_PX
        box = (paper_x(left - dx), paper_x(right + dx), paper_y(bottom + dy), paper_y(top - dy))
        return top >= _TOP_PX and not any(_overlaps(box, label) for label in labels)

    def hang(menu: Any, *, above_px: float, right_px: float = 0.0) -> None:
        menu.update(
            x=paper_x(left_px), y=1.0, xanchor="left", yanchor="bottom",
            pad={"t": 0, "r": 0, "b": above_px, "l": right_px},
        )

    step_menu = go.layout.Updatemenu(
        type="buttons",
        direction="left",
        showactive=False,
        bgcolor=play.bgcolor,
        bordercolor=play.bordercolor,
        borderwidth=play.borderwidth,
        font={"size": step_font, "color": play.font.color if play.font is not None else None},
        buttons=[{"label": STEP_LABEL, "method": "skip", "args": [STEP_COMMAND], "execute": True}],
    )
    first, second = (step_menu, play) if primary else (play, step_menu)
    first_h, second_h = (step_h, play_h) if primary else (play_h, step_h)
    first_w, second_w = (step_w, play_w) if primary else (play_w, step_w)
    # Widths are estimates: leave room so the second menu never covers the first.
    row_offset = 1.1 * first_w + _GAP_PX
    row_h, row_w = max(play_h, step_h), row_offset + second_w
    # Stacked layouts: the step button is Play's size (it leads, or mirrors Play).
    column_h, column_w = play_h + _GAP_PX + play_h, max(play_w, under_w)
    plot_top = margin["t"]
    # Where the builder put Play (its top edge), for layouts that keep that place.
    anchor_y = float(play.y if play.y is not None else 1.0)
    y_px = plot_top + (1.0 - anchor_y) * plot_h
    play_top = {"bottom": y_px - play_h, "middle": y_px - play_h / 2}.get(play.yanchor or "top", y_px)

    def from_top(menu: Any, *, below_px: float, right_px: float = 0.0) -> None:
        menu.update(
            x=paper_x(left_px), y=paper_y(play_top), xanchor="left", yanchor="top",
            pad={"t": below_px, "r": 0, "b": 0, "l": right_px},
        )

    def stacked_above_plot() -> None:
        hang(second, above_px=_GAP_PX)
        hang(first, above_px=_GAP_PX + play_h + _GAP_PX)

    def row_above_plot() -> None:
        hang(first, above_px=_GAP_PX + (row_h - first_h) / 2)
        hang(second, above_px=_GAP_PX + (row_h - second_h) / 2, right_px=row_offset)

    def row_at_play() -> None:
        from_top(first, below_px=(row_h - first_h) / 2)
        from_top(second, below_px=(row_h - second_h) / 2, right_px=row_offset)

    def stacked_at_play() -> None:
        from_top(first, below_px=0.0)
        from_top(second, below_px=play_h + _GAP_PX)

    # The leading control comes first (on top, or on the left) in every layout.
    # Layouts above the plot are tried first; the first that covers no
    # annotation wins. Layouts that keep Play's place only check the space they
    # add (that place is the builder's), and the step button under Play is the
    # last resort.
    above_column = (plot_top - _GAP_PX - column_h, plot_top - _GAP_PX, left_px, left_px + column_w)
    above_row = (plot_top - _GAP_PX - row_h, plot_top - _GAP_PX, left_px, left_px + row_w)
    under_play = (play_top + play_h + _GAP_PX, play_top + column_h, left_px, left_px + (play_w if primary else under_w))
    right_of_play = (play_top, play_top + row_h, left_px + row_offset, left_px + row_w)
    at_play = [
        ("stacked at play", stacked_at_play, under_play, column_w),
        ("row at play", row_at_play, right_of_play, row_w),
    ]
    if play_top + play_h <= plot_top + 1.0:
        # Play sits in the top margin: the controls may settle just above the plot.
        candidates = [
            ("stacked", stacked_above_plot, above_column, column_w),
            ("row", row_above_plot, above_row, row_w),
            *at_play,
        ]
    else:
        # The builder put Play inside or under the plot (beside its slider, say):
        # Play keeps that place and the step button joins it.
        candidates = at_play[::-1]
    placement, place, _, width_px = next(
        (c for c in candidates if clear(*c[2])),
        at_play[0],
    )
    place()
    if placement.startswith("stacked"):
        step_menu.font.size = play_font
        step_menu.buttons[0].label = under_label
    right_edge = left_px + width_px
    fig.layout.updatemenus = [*menus, step_menu]
    return {
        "primary": primary,
        "placement": placement,
        "frame_tex_chars": tex_chars,
        "controls_right_px": right_edge,
    }


def without_step_control(fig: Any) -> Any:
    """A copy of ``fig`` without the step button, for HTML that has no fluid controller."""
    menus = list(fig.layout.updatemenus or ())
    if not any(is_step_menu(menu) for menu in menus):
        return fig
    # A fresh figure: a deep copy would also carry display patches bound to ``fig``.
    stripped = go.Figure(fig)
    stripped.layout.updatemenus = [menu for menu in menus if not is_step_menu(menu)]
    return stripped


__all__ = [
    "HEAVY_FRAME_TEX_CHARS",
    "STEP_COMMAND",
    "SHORT_STEP_LABEL",
    "STEP_LABEL",
    "add_step_control",
    "frame_tex_chars",
    "is_step_menu",
    "without_step_control",
]
