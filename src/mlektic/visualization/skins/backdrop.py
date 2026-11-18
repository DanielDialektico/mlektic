"""Canvas backdrops rendered as a Plotly layout image.

A layout image travels inside the figure JSON, so the backdrop appears in
every renderer (Colab, Jupyter, VS Code, exported HTML, static images)
without page CSS.  It sits below the plot and never takes part in layout.
"""

from __future__ import annotations

import base64
from typing import Any

from .definitions import Backdrop

# Plotly's own margin defaults, used when a figure does not set one.
_DEFAULT_MARGIN = {"l": 80, "r": 80, "t": 100, "b": 80}
# Plotly may widen margins at draw time to fit sliders, menus, and legends, so
# the real plot area is only known in the browser.  The image therefore extends
# this fraction of the canvas past every edge; the overflow is clipped.
_BLEED = 0.4


def backdrop_image(backdrop: Backdrop, fig: Any) -> dict[str, Any] | None:
    """Return a layout image covering the whole canvas, or ``None`` if its size is unknown."""
    width = fig.layout.width
    height = fig.layout.height
    if not width or not height or fig.layout.autosize:
        return None
    margin = {**_DEFAULT_MARGIN, **{k: v for k, v in fig.layout.margin.to_plotly_json().items() if k in "lrtb"}}
    plot_width = width - margin["l"] - margin["r"]
    plot_height = height - margin["t"] - margin["b"]
    if plot_width <= 0 or plot_height <= 0:
        return None
    bleed_x, bleed_y = round(_BLEED * width), round(_BLEED * height)
    full_width, full_height = int(width) + 2 * bleed_x, int(height) + 2 * bleed_y
    svg = _svg(backdrop, full_width, full_height, origin=(bleed_x, bleed_y), canvas=(int(width), int(height)))
    return {
        "source": "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii"),
        "xref": "paper",
        "yref": "paper",
        "x": -(margin["l"] + bleed_x) / plot_width,
        "y": 1 + (margin["t"] + bleed_y) / plot_height,
        "sizex": full_width / plot_width,
        "sizey": full_height / plot_height,
        "xanchor": "left",
        "yanchor": "top",
        "sizing": "stretch",
        "layer": "below",
        "name": "mlektic-skin-backdrop",
    }


def _svg(backdrop: Backdrop, width: int, height: int, *, origin: tuple[int, int], canvas: tuple[int, int]) -> str:
    if backdrop.kind == "radial":
        cx = origin[0] + backdrop.center[0] * canvas[0]
        cy = origin[1] + backdrop.center[1] * canvas[1]
        stops = "".join(
            f'<stop offset="{index / max(1, len(backdrop.colors) - 1):.3f}" stop-color="{color}"/>'
            for index, color in enumerate(backdrop.colors)
        )
        paint = (
            f'<defs><radialGradient id="g" gradientUnits="userSpaceOnUse" cx="{cx:.1f}" '
            f'cy="{cy:.1f}" r="{1.15 * max(canvas):.1f}">{stops}</radialGradient></defs>'
            f'<rect width="{width}" height="{height}" fill="url(#g)"/>'
        )
    elif backdrop.kind == "grid":
        base, line = backdrop.colors
        cell = backdrop.cell
        paint = (
            f'<defs><pattern id="p" width="{cell}" height="{cell}" patternUnits="userSpaceOnUse">'
            f'<path d="M {cell} 0 L 0 0 0 {cell}" fill="none" stroke="{line}" stroke-width="1"/></pattern></defs>'
            f'<rect width="{width}" height="{height}" fill="{base}"/>'
            f'<rect width="{width}" height="{height}" fill="url(#p)"/>'
        )
    else:
        raise ValueError(f"Unknown backdrop kind {backdrop.kind!r}; use 'radial' or 'grid'.")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" preserveAspectRatio="none">{paint}</svg>'
    )
