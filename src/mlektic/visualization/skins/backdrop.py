"""Canvas backdrops rendered as a Plotly layout image.

A layout image travels inside the figure JSON, so the backdrop appears in
every renderer (Colab, Jupyter, VS Code, exported HTML, static images)
without page CSS.  It sits below the plot and never takes part in layout.
"""

from __future__ import annotations

import base64
import random
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


def _rgb(color: str) -> tuple[float, float, float]:
    value = color.lstrip("#")
    return tuple(int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _noise_layer(name: str, color: str, opacity: float, *, frequency: float, octaves: int, seed: int,
                 gain: float, bias: float, width: int, height: int) -> str:
    """A rectangle of ``color`` whose alpha follows fractal noise."""
    red, green, blue = _rgb(color)
    return (
        f'<filter id="{name}" x="0" y="0" width="100%" height="100%">'
        f'<feTurbulence type="fractalNoise" baseFrequency="{frequency}" numOctaves="{octaves}" seed="{seed}" '
        'stitchTiles="stitch"/>'
        f'<feColorMatrix type="matrix" values="0 0 0 0 {red:.3f} 0 0 0 0 {green:.3f} 0 0 0 0 {blue:.3f} '
        f'0 0 0 {gain} {bias}"/></filter>'
        f'<rect width="{width}" height="{height}" filter="url(#{name})" opacity="{opacity:.3f}"/>'
    )


def _texture(backdrop: Backdrop, width: int, height: int, *, origin: tuple[int, int], canvas: tuple[int, int]) -> str:
    """Optional surface detail: tone, smudges, laid lines, age spots, grain, and a vignette.

    Every random choice uses a fixed seed, so a figure always renders the same.
    """
    rng = random.Random(11)
    ox, oy = origin
    cw, ch = canvas
    layers = ""
    if backdrop.mottle > 0:
        layers += _noise_layer("m", backdrop.mottle_color, backdrop.mottle, frequency=0.006, octaves=3, seed=3,
                               gain=2.6, bias=-1.0, width=width, height=height)
    if backdrop.smudges > 0:
        layers += '<filter id="s" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="22"/></filter>'
        for _ in range(backdrop.smudges):
            cx = ox + rng.uniform(0.08, 0.92) * cw
            cy = oy + rng.uniform(0.12, 0.88) * ch
            layers += (
                f'<ellipse cx="{cx:.0f}" cy="{cy:.0f}" rx="{rng.uniform(110, 260):.0f}" ry="{rng.uniform(18, 55):.0f}" '
                f'transform="rotate({rng.uniform(-14, 14):.1f} {cx:.0f} {cy:.0f})" fill="{backdrop.smudge_color}" '
                f'opacity="{backdrop.smudge_opacity * rng.uniform(0.6, 1.3):.3f}" filter="url(#s)"/>'
            )
    if backdrop.laid_lines > 0:
        chain = "".join(
            f'<line x1="{x}" y1="0" x2="{x}" y2="{height}"/>' for x in range(ox % 34, width, 34)
        )
        laid = "".join(f'<line x1="0" y1="{y}" x2="{width}" y2="{y}"/>' for y in range(0, height, 4))
        layers += (
            f'<g stroke="{backdrop.laid_color}" stroke-width="1" opacity="{backdrop.laid_lines:.3f}">{chain}</g>'
            f'<g stroke="{backdrop.laid_color}" stroke-width="0.6" opacity="{backdrop.laid_lines * 0.35:.3f}">{laid}</g>'
        )
    if backdrop.spots > 0:
        layers += '<filter id="f" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="2.5"/></filter>'
        for _ in range(backdrop.spots):
            # Age spots gather near the edges and corners, like on real old paper.
            edge = rng.random() < 0.7
            fx = rng.choice((rng.uniform(0.0, 0.14), rng.uniform(0.86, 1.0))) if edge else rng.uniform(0.15, 0.85)
            fy = rng.uniform(0.0, 1.0) if edge else rng.choice((rng.uniform(0.0, 0.15), rng.uniform(0.85, 1.0)))
            radius = rng.uniform(3, 13)
            layers += (
                f'<circle cx="{ox + fx * cw:.0f}" cy="{oy + fy * ch:.0f}" r="{radius:.1f}" fill="{backdrop.spot_color}" '
                f'opacity="{rng.uniform(0.10, 0.24):.3f}" filter="url(#f)"/>'
            )
    if backdrop.grain > 0:
        layers += _noise_layer("n", backdrop.grain_color, backdrop.grain, frequency=0.85, octaves=2, seed=7,
                               gain=1.4, bias=-0.5, width=width, height=height)
    if backdrop.vignette > 0:
        cx = ox + cw / 2
        cy = oy + ch / 2
        layers += (
            f'<defs><radialGradient id="v" gradientUnits="userSpaceOnUse" cx="{cx:.1f}" cy="{cy:.1f}" '
            f'r="{0.75 * max(canvas):.1f}"><stop offset="0.55" stop-color="{backdrop.vignette_color}" stop-opacity="0"/>'
            f'<stop offset="1" stop-color="{backdrop.vignette_color}" stop-opacity="{backdrop.vignette:.3f}"/>'
            '</radialGradient></defs>'
            f'<rect width="{width}" height="{height}" fill="url(#v)"/>'
        )
    return layers


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
    paint += _texture(backdrop, width, height, origin=origin, canvas=canvas)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" preserveAspectRatio="none">{paint}</svg>'
    )
