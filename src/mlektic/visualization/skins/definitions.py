"""Declarative skin definitions.

A skin is a presentation-only theme: colours, a backdrop, and optional motion
effects.  It never carries sizes, positions, margins, or fonts, so a figure
drawn with a skin has exactly the geometry of the classic dashboard.

This module is pure data.  Changing how a skin looks means editing a value
here; nothing in the figure builders or in the styling code needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Backdrop:
    """Background painted behind the whole canvas.

    ``kind`` is ``"radial"`` (a soft glow from ``center``, in canvas fractions)
    or ``"grid"`` (a drafting grid of ``cell`` pixels).  ``colors`` lists the
    gradient stops from the centre outwards, or ``(base, line)`` for a grid.
    """

    kind: str
    colors: tuple[str, ...]
    center: tuple[float, float] = (0.5, 0.0)
    cell: int = 20


@dataclass(frozen=True)
class SkinPalette:
    """Every colour a skin assigns.  Roles are described in ``roles.py``."""

    background: str
    scene_wall: str
    text: str
    muted: str
    grid: str
    axis: str
    data: str
    data_border: str
    model: str
    loss: str
    accent: str
    increase: str
    decrease: str
    classes: tuple[str, ...]
    control_background: str
    control_border: str
    control_text: str
    slider_track: str
    slider_active: str
    legend_background: str
    legend_border: str
    legend_text: str
    data_symbol: str | None = None
    loss_dash: str | None = None


@dataclass(frozen=True)
class SkinEffects:
    """Optional motion effects.  Each one only adds or recolours marks."""

    glow: bool = False
    glow_width: float = 14.0
    glow_opacity: float = 0.18
    trail: int = 0
    trail_opacity: tuple[float, float] = (0.08, 0.36)
    loss_progress: bool = False
    equation_delta: bool = False


@dataclass(frozen=True)
class Skin:
    """A named, presentation-only theme."""

    name: str
    description: str
    palette: SkinPalette
    backdrop: Backdrop | None = None
    effects: SkinEffects = field(default_factory=SkinEffects)


AURORA = Skin(
    name="aurora",
    description="Night-blue gradient with luminous lines and training motion cues.",
    palette=SkinPalette(
        background="#0b1026",
        scene_wall="#141a3f",
        text="#e6ecff",
        muted="#a9b4d6",
        grid="rgba(148,163,255,0.13)",
        axis="rgba(148,163,255,0.30)",
        data="#67e8f9",
        data_border="#0e7490",
        model="#f472b6",
        loss="#a78bfa",
        accent="#fbbf24",
        increase="#34d399",
        decrease="#fb7185",
        classes=("#67e8f9", "#f472b6", "#fbbf24", "#a78bfa", "#34d399", "#fb923c"),
        control_background="rgba(255,255,255,0.06)",
        control_border="#67e8f9",
        control_text="#e0f2fe",
        slider_track="rgba(103,232,249,0.25)",
        slider_active="#f472b6",
        legend_background="rgba(255,255,255,0.06)",
        legend_border="#67e8f9",
        legend_text="#e0f2fe",
    ),
    backdrop=Backdrop(kind="radial", colors=("#2a2470", "#111536", "#070a1a"), center=(0.15, 0.0)),
    effects=SkinEffects(glow=True, trail=5, loss_progress=True, equation_delta=True),
)

EDITORIAL = Skin(
    name="editorial",
    description="Warm paper background with ink colours, for projection and print.",
    palette=SkinPalette(
        background="#f6f1e7",
        scene_wall="#ebe4d4",
        text="#1f2328",
        muted="#6b6f76",
        grid="rgba(31,35,40,0.08)",
        axis="rgba(31,35,40,0.35)",
        data="#2b4c7e",
        data_border="#f6f1e7",
        model="#c8553d",
        loss="#2a9d8f",
        accent="#c8553d",
        increase="#2a9d8f",
        decrease="#c8553d",
        classes=("#2b4c7e", "#c8553d", "#2a9d8f", "#b08900", "#6a4c93", "#5c677d"),
        control_background="#fffdf8",
        control_border="#1f2328",
        control_text="#1f2328",
        slider_track="rgba(31,35,40,0.12)",
        slider_active="#c8553d",
        legend_background="#fffdf8",
        legend_border="#1f2328",
        legend_text="#1f2328",
    ),
)

BLUEPRINT = Skin(
    name="blueprint",
    description="Drafting-table blue with a fine grid, open markers, and dotted loss.",
    palette=SkinPalette(
        background="#0d3866",
        scene_wall="#11426f",
        text="#eaf2ff",
        muted="#b9cde6",
        grid="rgba(255,255,255,0.22)",
        axis="rgba(255,255,255,0.55)",
        data="#ffffff",
        data_border="#ffffff",
        model="#ffd166",
        loss="#9ad1ff",
        accent="#ffd166",
        increase="#9ef0c4",
        decrease="#ffb4a2",
        classes=("#ffffff", "#ffd166", "#9ad1ff", "#9ef0c4", "#ffb4a2", "#d4b8ff"),
        control_background="rgba(255,255,255,0.0)",
        control_border="#eaf2ff",
        control_text="#eaf2ff",
        slider_track="rgba(255,255,255,0.25)",
        slider_active="#ffd166",
        legend_background="rgba(13,56,102,0.85)",
        legend_border="#eaf2ff",
        legend_text="#eaf2ff",
        data_symbol="circle-open",
        loss_dash="dot",
    ),
    backdrop=Backdrop(kind="grid", colors=("#0d3866", "rgba(255,255,255,0.06)"), cell=20),
)

SKINS: dict[str, Skin] = {skin.name: skin for skin in (AURORA, EDITORIAL, BLUEPRINT)}
