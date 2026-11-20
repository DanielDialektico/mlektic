"""Colour-only restyling of a built figure.

Every assignment here is a colour, symbol, or dash.  Sizes, positions,
margins, fonts, and text content are never touched, which is what keeps a
skinned figure pixel-aligned with the classic one.
"""

from __future__ import annotations

import re
from typing import Any

from . import roles
from .definitions import Skin, SkinPalette

_NAMED = {"white": (255, 255, 255), "black": (0, 0, 0)}
_RGB = re.compile(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)")
# The classic secondary-text colour used in inline spans (subtitles, captions).
_CLASSIC_MUTED = re.compile(r"(color:\s*)#B8C1CC", re.IGNORECASE)


def _channels(color: Any) -> tuple[float, ...] | None:
    value = str(color).strip().lower()
    if value in _NAMED:
        return _NAMED[value]
    if value.startswith("#") and len(value) in {4, 7}:
        digits = value[1:] if len(value) == 7 else "".join(c * 2 for c in value[1:])
        try:
            return tuple(int(digits[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            return None
    match = _RGB.match(value)
    return tuple(float(match.group(i)) for i in (1, 2, 3)) if match else None


def _text_color(color: Any, palette: SkinPalette) -> str | None:
    """Map a builder's neutral text colour onto the skin; keep chosen hues.

    Grey, white, and black text is "ordinary" text and follows the skin (light
    greys become the main text colour, mid greys the muted one).  A coloured
    text was chosen on purpose, for example to name a class, and is kept.
    """
    if color is None or color == "":
        return palette.text
    channels = _channels(color)
    if channels is None or max(channels) - min(channels) > 28:
        return None
    lightness = sum(channels) / (3 * 255)
    return palette.text if lightness >= 0.6 or lightness <= 0.15 else palette.muted


def _recolor_muted(text: str, palette: SkinPalette) -> str:
    return _CLASSIC_MUTED.sub(lambda m: m.group(1) + palette.muted, text)


def restyle(fig: Any, skin: Skin, *, family: str) -> None:
    """Recolour layout chrome and, outside neural figures, the semantic traces."""
    palette = skin.palette
    _style_layout(fig, palette, transparent_plot=skin.backdrop is not None)
    if skin.font_family:
        # Text that names a family explicitly keeps it; MathJax equations are unaffected.
        fig.layout.font.family = skin.font_family
    _style_annotations(fig, palette)
    if family.startswith("neural"):
        # Neural figures encode values in their colours; only recolour plain text.
        for trace, payloads in _trace_payloads(fig):
            _style_colorbar(trace, palette)
            if roles.trace_role(trace) in {roles.TEXT, roles.EQUATION}:
                for payload in payloads:
                    _style_text_trace(payload, palette)
        return
    class_index = 0
    surface_index = 0
    for trace, payloads in _trace_payloads(fig):
        role = roles.trace_role(trace)
        _style_colorbar(trace, palette)
        class_color = None
        if role == roles.CLASS_CURVE:
            class_color = palette.classes[class_index % len(palette.classes)]
            class_index += 1
        elif role == roles.SURFACE and str(trace.name or "").startswith("Class "):
            # One surface per class, shaded from the scene wall to that class's colour.
            color = palette.classes[surface_index % len(palette.classes)]
            surface_index += 1
            trace.colorscale = [[0.0, palette.scene_wall], [1.0, color]]
            continue
        elif _is_class_coded_scatter(trace):
            _recolor_class_codes(trace, palette)
            continue
        for payload in payloads:
            _style_trace(payload, role, palette, class_color)


def _is_class_coded_scatter(trace: Any) -> bool:
    """3D data points coloured by class index through a colour scale."""
    marker = getattr(trace, "marker", None)
    return (
        str(getattr(trace, "type", "")) == "scatter3d"
        and str(getattr(trace, "name", "") or "").lower() == "data"
        and marker is not None
        and roles.is_array(marker.color)
        and marker.colorscale is not None
    )


def _recolor_class_codes(trace: Any, palette: SkinPalette) -> None:
    """Map class indices 0..K-1 onto the skin's class colours, in steps."""
    import numpy as np

    codes = np.asarray(trace.marker.color, dtype=float)
    count = int(np.nanmax(codes)) + 1 if codes.size else 1
    colors = [palette.classes[k % len(palette.classes)] for k in range(count)]
    if count == 1:
        trace.marker.colorscale = [[0.0, colors[0]], [1.0, colors[0]]]
        return
    scale = []
    for k, color in enumerate(colors):
        scale += [[k / count, color], [(k + 1) / count, color]]
    trace.marker.update(colorscale=scale, cmin=-0.5, cmax=count - 0.5)


def _trace_payloads(fig: Any):
    """Yield each base trace with every object that styles it (itself and its frame updates)."""
    per_trace: list[list[Any]] = [[trace] for trace in fig.data]
    for frame in fig.frames or ():
        targets = list(frame.traces) if frame.traces is not None else list(range(len(frame.data or ())))
        for target, update in zip(targets, frame.data or ()):
            if 0 <= int(target) < len(per_trace):
                per_trace[int(target)].append(update)
    for trace, payloads in zip(fig.data, per_trace):
        yield trace, payloads


def _style_layout(fig: Any, palette: SkinPalette, *, transparent_plot: bool) -> None:
    layout = fig.layout
    layout.paper_bgcolor = palette.background
    layout.plot_bgcolor = "rgba(0,0,0,0)" if transparent_plot else palette.background
    layout.font.color = palette.text
    if layout.title is not None:
        layout.title.font.color = palette.text
        if layout.title.text:
            layout.title.text = _recolor_muted(layout.title.text, palette)
    layout.hoverlabel.bordercolor = palette.axis

    for key, value in layout.to_plotly_json().items():
        if not isinstance(value, dict):
            continue
        if key.startswith(("xaxis", "yaxis")):
            _style_axis(getattr(layout, key), palette)
        elif key.startswith("scene"):
            scene = getattr(layout, key)
            scene.bgcolor = "rgba(0,0,0,0)" if transparent_plot else palette.background
            for axis_name in ("xaxis", "yaxis", "zaxis"):
                axis = getattr(scene, axis_name)
                _style_axis(axis, palette)
                axis.backgroundcolor = palette.scene_wall
        elif key.startswith("legend"):
            legend = getattr(layout, key)
            legend.bgcolor = palette.legend_background
            legend.bordercolor = palette.legend_border
            legend.font.color = palette.legend_text

    for menu in layout.updatemenus or ():
        menu.bgcolor = palette.control_background
        menu.bordercolor = palette.control_border
        menu.font.color = palette.control_text
    for slider in layout.sliders or ():
        slider.bgcolor = palette.slider_track
        slider.activebgcolor = palette.slider_active
        slider.bordercolor = palette.control_border
        slider.tickcolor = palette.axis
        slider.font.color = palette.text
        slider.currentvalue.font.color = palette.text


def _style_axis(axis: Any, palette: SkinPalette) -> None:
    axis.gridcolor = palette.grid
    axis.zerolinecolor = palette.axis
    axis.linecolor = palette.axis
    axis.tickfont.color = palette.text
    axis.title.font.color = palette.text


def _style_annotations(fig: Any, palette: SkinPalette) -> None:
    def recolor(annotations: Any) -> None:
        for annotation in annotations or ():
            color = _text_color(annotation.font.color, palette)
            if color is not None:
                annotation.font.color = color
            if annotation.text:
                annotation.text = _recolor_muted(annotation.text, palette)

    def recolor_dicts(annotations: Any) -> None:
        for annotation in annotations or ():
            if not isinstance(annotation, dict):
                continue
            font = annotation.setdefault("font", {})
            color = _text_color(font.get("color"), palette)
            if color is not None:
                font["color"] = color
            if isinstance(annotation.get("text"), str):
                annotation["text"] = _recolor_muted(annotation["text"], palette)

    recolor(fig.layout.annotations)
    for frame in fig.frames or ():
        if frame.layout is not None:
            recolor(frame.layout.annotations)
    # Explainer buttons (Input, Substitution, Output) carry their own annotation sets.
    for menu in fig.layout.updatemenus or ():
        for button in menu.buttons or ():
            for argument in button.args or ():
                if isinstance(argument, dict):
                    for key, value in argument.items():
                        if key == "annotations" or key.endswith(".annotations"):
                            recolor_dicts(value)


def _style_text_trace(payload: Any, palette: SkinPalette) -> None:
    textfont = getattr(payload, "textfont", None)
    if textfont is not None and not roles.is_array(textfont.color):
        color = _text_color(textfont.color, palette)
        if color is not None:
            textfont.color = color


def _style_colorbar(trace: Any, palette: SkinPalette) -> None:
    """Recolour the text of a trace's colour bar, if it has one."""
    for owner in (trace, getattr(trace, "marker", None), getattr(trace, "line", None)):
        colorbar = getattr(owner, "colorbar", None) if owner is not None else None
        if colorbar is None or not colorbar.to_plotly_json():
            continue
        for font in (colorbar.tickfont, colorbar.title.font):
            color = _text_color(font.color, palette)
            if color is not None:
                font.color = color


def _style_trace(payload: Any, role: str, palette: SkinPalette, class_color: str | None) -> None:
    if role in {roles.TEXT, roles.EQUATION}:
        _style_text_trace(payload, palette)
    elif role == roles.DATA:
        marker = payload.marker
        if marker is None or roles.is_array(marker.color):
            return
        marker.color = palette.data
        if palette.data_symbol:
            marker.symbol = palette.data_symbol
        else:
            marker.line.color = palette.data_border
    elif role == roles.MODEL:
        if getattr(payload, "line", None) is not None:
            payload.line.color = palette.model
    elif role == roles.LOSS:
        if getattr(payload, "line", None) is not None:
            payload.line.color = palette.loss
            if palette.loss_dash:
                payload.line.dash = palette.loss_dash
        marker = getattr(payload, "marker", None)
        if marker is not None and not roles.is_array(marker.color):
            marker.color = palette.loss
    elif role == roles.CLASS_CURVE and class_color is not None:
        if getattr(payload, "line", None) is not None:
            payload.line.color = class_color
