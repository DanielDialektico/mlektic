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
    ``grain`` overlays a fine noise texture of ``grain_color`` at that opacity
    (chalk dust, paper fibre); ``vignette`` darkens the edges by that amount.
    """

    kind: str
    colors: tuple[str, ...]
    center: tuple[float, float] = (0.5, 0.0)
    cell: int = 20
    grain: float = 0.0
    grain_color: str = "#ffffff"
    vignette: float = 0.0
    vignette_color: str = "#000000"
    # Uneven tone: large soft blotches of ``mottle_color`` at that opacity.
    mottle: float = 0.0
    mottle_color: str = "#000000"
    # Broad blurred swipes, like a half-erased blackboard.
    smudges: int = 0
    smudge_color: str = "#ffffff"
    smudge_opacity: float = 0.06
    # Small age spots (foxing) on old paper.
    spots: int = 0
    spot_color: str = "#7a5a32"
    # Laid-paper texture: faint chain lines and denser laid lines.
    laid_lines: float = 0.0
    laid_color: str = "#6e5638"


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
    # Button colours while the pointer is over them; None derives them.
    control_hover: str | None = None
    control_hover_text: str | None = None


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
    # Vertical segments from each data point to the model: the errors it still makes.
    residuals: bool = False


@dataclass(frozen=True)
class Skin:
    """A named, presentation-only theme."""

    name: str
    description: str
    palette: SkinPalette
    backdrop: Backdrop | None = None
    effects: SkinEffects = field(default_factory=SkinEffects)
    # CSS for a border drawn over the figure's edge (HTML output only), e.g. a frame.
    frame: str | None = None
    # Typeface for titles, axes, legends, and buttons (equations keep MathJax's).
    # The stylesheet URL is loaded with the figure; the family ends in fallbacks.
    font_family: str | None = None
    font_url: str | None = None


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
        control_hover="rgba(103,232,249,0.24)",
        control_hover_text="#ffffff",
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
        control_hover="#efe6d3",
    ),
    font_family="Lora, Georgia, 'Times New Roman', serif",
    font_url="https://fonts.googleapis.com/css2?family=Lora:wght@400;600&display=swap",
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
        control_hover="rgba(255,255,255,0.18)",
    ),
    backdrop=Backdrop(kind="grid", colors=("#0d3866", "rgba(255,255,255,0.06)"), cell=20),
    font_family="'Architects Daughter', 'Segoe Print', 'Comic Sans MS', cursive",
    font_url="https://fonts.googleapis.com/css2?family=Architects+Daughter&display=swap",
)

DUSK = Skin(
    name="dusk",
    description="Aurora's motion cues in muted slate, dusty rose, and sand.",
    palette=SkinPalette(
        background="#151a24",
        scene_wall="#1d2330",
        text="#e3e6eb",
        muted="#a3abb8",
        grid="rgba(163,171,184,0.12)",
        axis="rgba(163,171,184,0.32)",
        data="#8fb5c4",
        data_border="#4d6f7d",
        model="#cf9585",
        loss="#a29dc4",
        accent="#d6b673",
        increase="#8fbf9f",
        decrease="#cf8a8a",
        classes=("#8fb5c4", "#cf9585", "#d6b673", "#a29dc4", "#8fbf9f", "#c49f7d"),
        control_background="rgba(255,255,255,0.05)",
        control_border="#8fb5c4",
        control_text="#e3e6eb",
        slider_track="rgba(143,181,196,0.25)",
        slider_active="#cf9585",
        legend_background="rgba(255,255,255,0.05)",
        legend_border="#8fb5c4",
        legend_text="#e3e6eb",
        control_hover="rgba(143,181,196,0.24)",
        control_hover_text="#ffffff",
    ),
    backdrop=Backdrop(kind="radial", colors=("#283042", "#1a2030", "#10141c"), center=(0.15, 0.0)),
    effects=SkinEffects(
        glow=True, glow_opacity=0.10, trail=4, trail_opacity=(0.06, 0.26), loss_progress=True, equation_delta=True,
    ),
)

CHALKBOARD = Skin(
    name="chalkboard",
    description="School slate green with white and coloured chalk, drawn by hand.",
    palette=SkinPalette(
        background="#24392f",
        scene_wall="#2b4237",
        text="#f1f1ea",
        muted="#c9d1c4",
        grid="rgba(241,241,234,0.10)",
        axis="rgba(241,241,234,0.45)",
        data="#f1f1ea",
        data_border="#f1f1ea",
        model="#f5e27c",
        loss="#f3a9bb",
        accent="#9fd2ee",
        increase="#b9e3a2",
        decrease="#f3a9bb",
        classes=("#f1f1ea", "#f5e27c", "#9fd2ee", "#f3a9bb", "#b9e3a2", "#f5b97c"),
        control_background="rgba(241,241,234,0.08)",
        control_border="#f1f1ea",
        control_text="#f1f1ea",
        slider_track="rgba(241,241,234,0.25)",
        slider_active="#f5e27c",
        legend_background="rgba(36,57,47,0.85)",
        legend_border="#f1f1ea",
        legend_text="#f1f1ea",
        data_symbol="circle-open",
        control_hover="rgba(241,241,234,0.24)",
        control_hover_text="#ffffff",
    ),
    backdrop=Backdrop(
        kind="radial", colors=("#2e4a3c", "#24392f", "#1a2a22"), center=(0.45, 0.35),
        grain=0.10, grain_color="#f1f1ea", vignette=0.35,
        mottle=0.06, mottle_color="#f1f1ea", smudges=6, smudge_color="#f1f1ea", smudge_opacity=0.07,
    ),
    frame=(
        "border:6px solid #7a5232;border-radius:3px;"
        "box-shadow:inset 0 0 0 1px rgba(0,0,0,0.55),inset 0 0 22px rgba(0,0,0,0.45),"
        "0 0 0 1px #4a3020;"
        "border-image:linear-gradient(135deg,#8a5c36,#5e3d22 40%,#8f6440 60%,#5a3a20) 1"
    ),
    font_family="Gaegu, 'Segoe Print', 'Comic Sans MS', cursive",
    font_url="https://fonts.googleapis.com/css2?family=Gaegu:wght@400;700&display=swap",
)

DAVINCI = Skin(
    name="davinci",
    description="Leonardo's notebook: aged parchment, sepia ink, and red chalk strokes.",
    palette=SkinPalette(
        background="#e9dbb8",
        scene_wall="#e2d0a6",
        text="#3a2a1a",
        muted="#6e5638",
        grid="rgba(58,42,26,0.12)",
        axis="rgba(58,42,26,0.45)",
        data="#3a2a1a",
        data_border="#3a2a1a",
        model="#9a3a24",
        loss="#5c4a33",
        accent="#9a3a24",
        increase="#4f6a3a",
        decrease="#9a3a24",
        classes=("#3a2a1a", "#9a3a24", "#a97b2c", "#4f6a3a", "#4a5d70", "#7a4f6a"),
        control_background="#f3e9cf",
        control_border="#3a2a1a",
        control_text="#3a2a1a",
        slider_track="rgba(58,42,26,0.18)",
        slider_active="#9a3a24",
        legend_background="rgba(243,233,207,0.9)",
        legend_border="#3a2a1a",
        legend_text="#3a2a1a",
        data_symbol="circle-open",
        loss_dash="dash",
        control_hover="#e2cfa3",
    ),
    backdrop=Backdrop(
        kind="radial", colors=("#f1e3c0", "#e4cf9f", "#c7a86f"), center=(0.5, 0.45),
        grain=0.14, grain_color="#6e5638", vignette=0.55, vignette_color="#5a3714",
        mottle=0.16, mottle_color="#8a6232", spots=11, spot_color="#7a4f22", laid_lines=0.07,
    ),
    font_family="Kalam, 'Segoe Print', 'Bradley Hand', 'Comic Sans MS', cursive",
    font_url="https://fonts.googleapis.com/css2?family=Kalam:wght@400;700&display=swap",
)

EMBER = Skin(
    name="ember",
    description="Volcanic dark with fire colours; residual sparks show the model's errors closing.",
    palette=SkinPalette(
        background="#0c0605",
        scene_wall="#1a0b07",
        text="#fde7d3",
        muted="#e0b08e",
        grid="rgba(255,140,60,0.11)",
        axis="rgba(255,140,60,0.35)",
        data="#ffd166",
        data_border="#b4541a",
        model="#ff5a1f",
        loss="#ffaa33",
        accent="#ffe14d",
        increase="#7ee081",
        decrease="#ff5a5f",
        classes=("#ffd166", "#ff5a1f", "#ff9a3c", "#e63946", "#ffe14d", "#c77dff"),
        control_background="rgba(255,90,31,0.12)",
        control_border="#ff7a3c",
        control_text="#ffe8d6",
        slider_track="rgba(255,122,60,0.25)",
        slider_active="#ffd166",
        legend_background="rgba(255,90,31,0.10)",
        legend_border="#ff7a3c",
        legend_text="#ffe8d6",
        control_hover="rgba(255,122,60,0.38)",
        control_hover_text="#ffffff",
    ),
    backdrop=Backdrop(
        kind="radial", colors=("#3d1308", "#1c0905", "#080403"), center=(0.5, 1.05),
        grain=0.05, grain_color="#ff8a3c",
    ),
    effects=SkinEffects(
        glow=True, glow_width=18.0, glow_opacity=0.26, trail=6, loss_progress=True, equation_delta=True,
        residuals=True,
    ),
)

LUMEN = Skin(
    name="lumen",
    description="Classic colours on soft charcoal light, with ember's glow, trail, and residuals.",
    palette=SkinPalette(
        background="#121314",
        scene_wall="#1b1c1f",
        text="#f1f2f4",
        muted="#b8c1cc",
        grid="rgba(255,255,255,0.08)",
        axis="rgba(255,255,255,0.28)",
        data="#7dd3fc",
        data_border="#0ea5e9",
        model="#ef553b",
        loss="#00cc96",
        accent="#f4c26b",
        increase="#5fd3a4",
        decrease="#f2836b",
        classes=("#7dd3fc", "#ef553b", "#00cc96", "#f4c26b", "#ab63fa", "#ff97c3"),
        control_background="rgba(255,255,255,0.07)",
        control_border="rgba(255,255,255,0.55)",
        control_text="#f1f2f4",
        slider_track="rgba(255,255,255,0.18)",
        slider_active="#ef553b",
        legend_background="rgba(255,255,255,0.06)",
        legend_border="rgba(255,255,255,0.35)",
        legend_text="#f1f2f4",
        control_hover="rgba(255,255,255,0.20)",
        control_hover_text="#ffffff",
    ),
    backdrop=Backdrop(
        kind="radial", colors=("#2c2925", "#1a1a1b", "#0e0e0f"), center=(0.5, 1.05),
        grain=0.03, grain_color="#f4c26b", vignette=0.30,
    ),
    effects=SkinEffects(
        glow=True, glow_width=14.0, glow_opacity=0.16, trail=5, trail_opacity=(0.06, 0.30),
        loss_progress=True, equation_delta=True, residuals=True,
    ),
)

SKINS: dict[str, Skin] = {
    skin.name: skin for skin in (AURORA, EDITORIAL, BLUEPRINT, DUSK, CHALKBOARD, DAVINCI, EMBER, LUMEN)
}
