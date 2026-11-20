"""Fit the colours of a neural figure to a skin's background.

Neural builders encode values in colour on the dark neutral canvas of
``NEURAL_COLORS`` (node fills, edge colour scales, cards, captions).  A skin
replaces that canvas, so the encodings are translated rather than repainted:

* neutral greys keep their *relative* position between the dark canvas and the
  light text they were designed for, and are re-expressed between the skin's
  background and text colours.  A dark card stays "slightly off the
  background", a mid grey stays "half way", and near-white text stays text;
* on a light background, saturated hues are darkened to a readable lightness.
  Their hue, and therefore their meaning (sign, layer family, class), is kept.

Only colours change.  Orders within colour scales are preserved, so a value
encoded by a colour stays monotone in the skin.
"""

from __future__ import annotations

import colorsys
import re
from typing import Any, Callable

from .definitions import SkinPalette

# The canvas (#17181c) and text (#f5f7fa) lightness of ``NEURAL_COLORS``.
_DESIGN_BACKGROUND = (0x17 + 0x18 + 0x1C) / (3 * 255)
_DESIGN_TEXT = (0xF5 + 0xF7 + 0xFA) / (3 * 255)
_NEUTRAL_SPREAD = 28
_LIGHT_HUE_MAX_LIGHTNESS = 0.40
_HEX = re.compile(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
_RGB = re.compile(r"^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*)?\)$")
_INLINE = re.compile(r"(color:\s*)(#[0-9a-fA-F]{3,6}|rgba?\([^)]*\))")
_NAMED = {"white": (255.0, 255.0, 255.0), "black": (0.0, 0.0, 0.0)}


def _parse(value: str) -> tuple[tuple[float, float, float], float | None] | None:
    text = value.strip().lower()
    if text in _NAMED:
        return _NAMED[text], None
    match = _HEX.match(text)
    if match:
        digits = match.group(1)
        if len(digits) == 3:
            digits = "".join(c * 2 for c in digits)
        return tuple(float(int(digits[i : i + 2], 16)) for i in (0, 2, 4)), None  # type: ignore[return-value]
    match = _RGB.match(text)
    if match:
        alpha = match.group(4)
        return (
            (float(match.group(1)), float(match.group(2)), float(match.group(3))),
            float(alpha) if alpha is not None else None,
        )
    return None


def _format(channels: tuple[float, float, float], alpha: float | None) -> str:
    r, g, b = (max(0, min(255, round(c))) for c in channels)
    if alpha is None:
        return f"#{r:02x}{g:02x}{b:02x}"
    return f"rgba({r},{g},{b},{alpha:g})"


def _luminance(channels: tuple[float, float, float]) -> float:
    return sum(channels) / (3 * 255)


def color_mapper(palette: SkinPalette) -> Callable[[str], str]:
    """Return a memoised function translating one design colour to the skin."""
    background = _parse(palette.background)
    text = _parse(palette.text)
    if background is None or text is None:  # pragma: no cover - definitions are validated by tests
        return lambda value: value
    bg, fg = background[0], text[0]
    light = _luminance(bg) > 0.5
    cache: dict[str, str] = {}

    def mix(fraction: float) -> tuple[float, float, float]:
        return tuple(b + (f - b) * fraction for b, f in zip(bg, fg))  # type: ignore[return-value]

    def translate(value: str) -> str:
        if value in cache:
            return cache[value]
        parsed = _parse(value)
        result = value
        if parsed is not None:
            channels, alpha = parsed
            if max(channels) - min(channels) <= _NEUTRAL_SPREAD:
                fraction = (_luminance(channels) - _DESIGN_BACKGROUND) / (_DESIGN_TEXT - _DESIGN_BACKGROUND)
                result = _format(mix(min(1.0, max(0.0, fraction))), alpha)
            elif light:
                hue, lightness, saturation = colorsys.rgb_to_hls(*(c / 255 for c in channels))
                if lightness > _LIGHT_HUE_MAX_LIGHTNESS:
                    lightness = _LIGHT_HUE_MAX_LIGHTNESS
                    saturation = min(1.0, saturation * 0.92 + 0.05)
                rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
                result = _format(tuple(c * 255 for c in rgb), alpha)  # type: ignore[arg-type]
        cache[value] = result
        return result

    return translate


def _is_color_key(key: str) -> bool:
    return "color" in key.lower() and key.lower() != "coloraxis"


def _map_value(value: Any, key: str, translate: Callable[[str], str]) -> Any:
    """Return the translated value, or the same object when nothing changes."""
    if isinstance(value, dict):
        changed = {}
        for child_key, child in value.items():
            mapped = _map_value(child, child_key, translate)
            if mapped is not child:
                changed[child_key] = mapped
        if not changed:
            return value
        return {**value, **changed}
    if key.lower().endswith("colorscale") and isinstance(value, (list, tuple)):
        scale = []
        changed = False
        for stop in value:
            if isinstance(stop, (list, tuple)) and len(stop) == 2 and isinstance(stop[1], str):
                mapped = translate(stop[1])
                changed |= mapped != stop[1]
                scale.append([stop[0], mapped])
            else:
                scale.append(stop)
        return scale if changed else value
    if _is_color_key(key):
        if isinstance(value, str):
            mapped = translate(value)
            return mapped if mapped != value else value
        if isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
            mapped_list = [translate(v) for v in value]
            return mapped_list if mapped_list != list(value) else value
        return value
    if key in {"text", "hovertext", "hovertemplate", "texttemplate"}:
        if isinstance(value, str) and "color:" in value:
            mapped = _INLINE.sub(lambda m: m.group(1) + translate(m.group(2)), value)
            return mapped if mapped != value else value
        if isinstance(value, (list, tuple)) and any(isinstance(v, str) and "color:" in v for v in value):
            return [
                _INLINE.sub(lambda m: m.group(1) + translate(m.group(2)), v) if isinstance(v, str) else v for v in value
            ]
        return value
    if isinstance(value, list) and value and isinstance(value[0], dict):
        mapped_items = [_map_value(item, key, translate) for item in value]
        if any(new is not old for new, old in zip(mapped_items, value)):
            return mapped_items
    return value


def _diff(payload: dict[str, Any], translate: Callable[[str], str]) -> dict[str, Any]:
    changed = {}
    for key, value in payload.items():
        if key in {"x", "y", "z", "customdata", "ids", "meta"}:
            continue
        mapped = _map_value(value, key, translate)
        if mapped is not value:
            changed[key] = mapped
    return changed


def _adapt_object(obj: Any, translate: Callable[[str], str]) -> None:
    changed = _diff(obj.to_plotly_json(), translate)
    if changed:
        obj.update(changed, overwrite=True)


def adapt_neural_colors(fig: Any, palette: SkinPalette) -> None:
    """Translate every design colour of a neural figure onto ``palette``."""
    translate = color_mapper(palette)
    for trace in fig.data:
        _adapt_object(trace, translate)
    for frame in fig.frames or ():
        for trace in frame.data or ():
            _adapt_object(trace, translate)
        if frame.layout is not None:
            _adapt_layout_items(frame.layout, translate)
    _adapt_layout_items(fig.layout, translate)
    for menu in fig.layout.updatemenus or ():
        for button in menu.buttons or ():
            arguments = list(button.args or ())
            changed = False
            for index, argument in enumerate(arguments):
                if isinstance(argument, dict):
                    mapped = _map_value(argument, "args", translate)
                    if mapped is not argument:
                        arguments[index] = mapped
                        changed = True
            if changed:
                button.args = arguments


def _adapt_layout_items(layout: Any, translate: Callable[[str], str]) -> None:
    for collection in ("annotations", "shapes"):
        items = getattr(layout, collection, None)
        for item in items or ():
            _adapt_object(item, translate)


__all__ = ["adapt_neural_colors", "color_mapper"]
