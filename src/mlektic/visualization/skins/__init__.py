"""Presentation-only themes ("skins") for Mlektic figures.

Layout themes (``classic``, ``academic``, ...) tune typography and spacing and
are handled by :mod:`mlektic.visualization.design`.  Skins change only how a
figure is painted and keep the classic geometry, so every element stays exactly
where the classic dashboard puts it.

The package is split so each layer can change on its own:

* ``definitions`` — pure data: palettes, backdrops, and effect switches;
* ``roles`` — the only contract with the figure builders (what each trace is);
* ``styling``, ``backdrop``, ``effects`` — how a skin is applied.

Builders never import this package; it runs as a final pass over a built figure.
"""

from __future__ import annotations

from typing import Any

from .backdrop import backdrop_image
from .definitions import SKINS, Backdrop, Skin, SkinEffects, SkinPalette
from .effects import apply_effects
from .neural import adapt_neural_colors
from .styling import restyle, skin_hover_colors


def available_skins() -> tuple[str, ...]:
    """Return registered skin names."""
    return tuple(SKINS)


def is_skin(theme: Any) -> bool:
    """Return whether ``theme`` names a registered skin."""
    return isinstance(theme, str) and theme.lower() in SKINS


def get_skin(theme: str) -> Skin:
    """Resolve a skin by name."""
    if not is_skin(theme):
        raise ValueError(f"Unknown skin {theme!r}. Available skins: {', '.join(SKINS)}.")
    return SKINS[theme.lower()]


def apply_skin(fig: Any, skin: Skin, *, family: str) -> dict[str, Any]:
    """Paint ``fig`` with ``skin`` and return a JSON-serializable report."""
    if family.startswith("neural"):
        # Neural encodings were drawn for a dark canvas: translate them first,
        # then let the shared pass recolour chrome and plain text.
        adapt_neural_colors(fig, skin.palette)
    restyle(fig, skin, family=family)
    backdrop = None
    if skin.backdrop is not None:
        image = backdrop_image(skin.backdrop, fig)
        if image is not None:
            fig.layout.images = [
                *(i for i in fig.layout.images or () if i.name != image["name"]),
                image,
            ]
            backdrop = skin.backdrop.kind
        else:
            # Without a fixed canvas the solid fallback colour is used instead.
            fig.layout.plot_bgcolor = skin.palette.background
            for key in fig.layout.to_plotly_json():
                if key.startswith("scene"):
                    getattr(fig.layout, key).bgcolor = skin.palette.background
    effects = (
        {"applied": [], "skipped": {"all": "neural figures keep their own encodings"}}
        if family.startswith("neural")
        else apply_effects(fig, skin)
    )
    return {"name": skin.name, "backdrop": backdrop, "effects": effects}


def button_colors(theme: Any) -> tuple[str, str] | None:
    """Return ``(background, text)`` for Play/Pause buttons, or ``None`` if not a skin."""
    if not is_skin(theme):
        return None
    palette = get_skin(theme).palette
    return palette.control_background, palette.control_text


__all__ = [
    "Backdrop",
    "Skin",
    "SkinEffects",
    "SkinPalette",
    "apply_skin",
    "available_skins",
    "backdrop_image",
    "button_colors",
    "get_skin",
    "is_skin",
    "skin_hover_colors",
]
