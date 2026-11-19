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


def _skin(fig: Any):
    meta = fig.layout.meta if isinstance(fig.layout.meta, dict) else {}
    name = ((meta.get("mlektic_visual") or {}).get("skin") or {}).get("name")
    return get_skin(name) if is_skin(name) else None


def _skin_frame(fig: Any) -> str | None:
    skin = _skin(fig)
    return skin.frame if skin is not None else None


def _font_loader(fig: Any) -> str:
    """Load the skin's web font, then let Plotly re-measure its text once it is ready."""
    skin = _skin(fig)
    if skin is None or not skin.font_url or not skin.font_family:
        return ""
    primary = skin.font_family.split(",")[0].strip().strip("'\"")
    probe = json.dumps(f'16px "{primary}"')
    return (
        f"var u={json.dumps(skin.font_url)};"
        "if(!document.querySelector('link[href=\"'+u+'\"]')){var l=document.createElement('link');"
        "l.rel='stylesheet';l.href=u;document.head.appendChild(l);}"
        "var gd=document.getElementById(id);"
        f"if(document.fonts&&gd&&window.Plotly){{document.fonts.load({probe}).then(function(f){{"
        f"if(f&&f.length){{Plotly.relayout(gd,{{'font.family':{json.dumps(skin.font_family)}}});}}"
        "}).catch(function(){});}"
    )


def figure_style_script(fig: Any) -> str | None:
    """Return a ``post_script`` with the figure's stylesheet, or ``None`` if it needs none.

    It keeps the buttons' colours, including on hover, draws the skin's frame
    (for example the chalkboard's wooden border) over the figure's edge, and
    loads the skin's typeface.
    """
    css = ""
    style = button_style(fig)
    if style is not None:
        css += (
            "{sel} .updatemenu-button rect{{fill:{background}!important;fill-opacity:1!important;"
            "stroke:{border}!important}}"
            "{sel} .updatemenu-button text{{fill:{text}!important}}"
            "{sel} .updatemenu-button:hover rect{{fill:{hover}!important}}"
            "{sel} .updatemenu-button:hover text{{fill:{hover_text}!important}}"
        ).format(sel="[id=\"__ID__\"]", **style)
    frame = _skin_frame(fig)
    if frame:
        # Drawn above the plot but ignoring the pointer, so every control still works.
        css += (
            "[id=\"__ID__\"]{position:relative}"
            "[id=\"__ID__\"]::after{content:\"\";position:absolute;inset:0;pointer-events:none;"
            f"z-index:1001;box-sizing:border-box;{frame}}}"
        )
    fonts = _font_loader(fig)
    if fonts:
        # Handwriting faces often lack a bold weight; a synthesized one smears them.
        css += "[id=\"__ID__\"] text{font-synthesis:none}"
    if not css and not fonts:
        return None
    return (
        "(function(){var id='{plot_id}';"
        "if(document.getElementById('mlektic-style-'+id))return;"
        "var s=document.createElement('style');s.id='mlektic-style-'+id;"
        f"s.textContent={json.dumps(css)}.split('__ID__').join(id);"
        "document.head.appendChild(s);"
        + fonts
        + "})();"
    )


button_style_script = figure_style_script  # backward-compatible name


__all__ = ["button_style", "button_style_script", "figure_style_script"]
