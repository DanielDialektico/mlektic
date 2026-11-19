"""Per-figure stylesheet for figure-owned buttons (Play/Pause, Input/Output...).

Plotly paints update-menu buttons with inline styles and, on hover, swaps
their background for a fixed light colour (``#F4FAFF``).  Themes with light
button text then lost the label while the pointer was over the button.

This module derives the button colours from the figure itself and returns a
Plotly ``post_script`` that installs a stylesheet scoped to that one figure, so
several figures with different themes can share a page.  It travels with the
figure's HTML, so it works in notebooks, Colab, and exported files alike.
"""

from __future__ import annotations

import json
from typing import Any

from .skins import get_skin, is_skin


def _skin_hover(fig: Any) -> tuple[str, str] | None:
    meta = fig.layout.meta if isinstance(fig.layout.meta, dict) else {}
    skin = (meta.get("mlektic_visual") or {}).get("skin") or {}
    name = skin.get("name")
    if not is_skin(name):
        return None
    palette = get_skin(name).palette
    if palette.control_hover is None:
        return None
    return palette.control_hover, palette.control_hover_text or palette.control_text


def button_style(fig: Any) -> dict[str, str] | None:
    """Return the colours the figure's buttons should keep, or ``None`` if it has none."""
    menus = list(fig.layout.updatemenus or ())
    if not menus or not any(menu.buttons for menu in menus):
        return None
    menu = menus[0]
    background = menu.bgcolor
    text = menu.font.color if menu.font is not None else None
    if not background or not text:
        return None
    hover, hover_text = _skin_hover(fig) or (
        # Darken or lighten towards the label colour; the label stays readable.
        f"color-mix(in srgb, {text} 18%, {background})",
        text,
    )
    return {
        "background": str(background),
        "text": str(text),
        "border": str(menu.bordercolor or text),
        "hover": str(hover),
        "hover_text": str(hover_text),
    }


def button_style_script(fig: Any) -> str | None:
    """Return a ``post_script`` that keeps the buttons' colours, including on hover."""
    style = button_style(fig)
    if style is None:
        return None
    rules = (
        "{sel} .updatemenu-button rect{{fill:{background}!important;fill-opacity:1!important;"
        "stroke:{border}!important}}"
        "{sel} .updatemenu-button text{{fill:{text}!important}}"
        "{sel} .updatemenu-button:hover rect{{fill:{hover}!important}}"
        "{sel} .updatemenu-button:hover text{{fill:{hover_text}!important}}"
    )
    css = rules.format(sel="[id=\"__ID__\"]", **style)
    return (
        "(function(){var id='{plot_id}';"
        "if(document.getElementById('mlektic-buttons-'+id))return;"
        "var s=document.createElement('style');s.id='mlektic-buttons-'+id;"
        f"s.textContent={json.dumps(css)}.split('__ID__').join(id);"
        "document.head.appendChild(s);})();"
    )


__all__ = ["button_style", "button_style_script"]
