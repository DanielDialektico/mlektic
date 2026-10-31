"""Additive visual-system contracts for Mlektic figures.

The classic dashboard remains the compatibility baseline.  Every other theme,
format, size, and responsive behavior is opt-in and is applied after a figure
has been built from its mathematical state.  This keeps visual composition
independent from history capture and model semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any


@dataclass(frozen=True)
class VisualTokens:
    """Resolved typography, spacing, color, and redundancy tokens."""

    font_family: str
    math_font_family: str
    title_size: int
    subtitle_size: int
    section_size: int
    body_size: int
    equation_size: int
    metric_size: int
    control_size: int
    space_xs: int
    space_sm: int
    space_md: int
    space_lg: int
    radius: int
    line_width: float
    background: str
    panel: str
    text: str
    muted: str
    grid: str
    data: str
    data_border: str
    model: str
    loss: str
    boundary: str
    control_background: str
    control_text: str
    legend_background: str
    legend_text: str
    data_symbol: str = "circle"
    model_dash: str = "solid"
    loss_dash: str = "solid"
    typography_scale: float = 1.0


@dataclass(frozen=True)
class VisualSpec:
    """Validated visual choices resolved for one public figure."""

    theme: str
    format: str
    density: str
    size: str
    width: int | None
    height: int | None
    responsive: bool
    reduced_motion: bool
    tokens: VisualTokens

    def metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable visual contract."""
        return {
            "schema_version": 1,
            "theme": self.theme,
            "format": self.format,
            "density": self.density,
            "size": self.size,
            "requested_width": self.width,
            "requested_height": self.height,
            "responsive": self.responsive,
            "reduced_motion": self.reduced_motion,
            "tokens": asdict(self.tokens),
        }


_CLASSIC = VisualTokens(
    font_family="Helvetica",
    math_font_family="STIX Two Math, serif",
    title_size=24,
    subtitle_size=12,
    section_size=16,
    body_size=13,
    equation_size=16,
    metric_size=13,
    control_size=14,
    space_xs=4,
    space_sm=8,
    space_md=12,
    space_lg=20,
    radius=0,
    line_width=4.0,
    background="#111111",
    panel="#222222",
    text="#ffffff",
    muted="#B8C1CC",
    grid="#2f3e4e",
    data="#7dd3fc",
    data_border="#0ea5e9",
    model="#EF553B",
    loss="#00cc96",
    boundary="#d7d7d7",
    control_background="#ffffff",
    control_text="#000000",
    legend_background="rgba(220,220,220,0.85)",
    legend_text="#000000",
)

_ACADEMIC = VisualTokens(
    font_family="Inter, Arial, sans-serif",
    math_font_family="STIX Two Math, serif",
    title_size=20,
    subtitle_size=12,
    section_size=15,
    body_size=13,
    equation_size=15,
    metric_size=12,
    control_size=12,
    space_xs=4,
    space_sm=8,
    space_md=12,
    space_lg=20,
    radius=6,
    line_width=3.0,
    background="#17181c",
    panel="#202126",
    text="#f5f7fa",
    muted="#aeb4bd",
    grid="#353841",
    data="#77b7ff",
    data_border="#b8d7ff",
    model="#ff7d8e",
    loss="#55d6be",
    boundary="#d7a8ff",
    control_background="#eef1f5",
    control_text="#15171b",
    legend_background="rgba(238,241,245,0.92)",
    legend_text="#15171b",
    typography_scale=0.92,
)

_CLASSROOM = VisualTokens(
    **{
        **asdict(_ACADEMIC),
        # Backward-compatible alias.  Earlier releases enlarged every text
        # surface here, including axes and MathJax annotations.  That made the
        # same composition overflow merely by changing theme.  Canvas size is
        # the supported way to target a projector or a larger screen.
        "typography_scale": _ACADEMIC.typography_scale,
    }
)

_COMPACT = VisualTokens(
    **{
        **asdict(_ACADEMIC),
        "title_size": 18,
        "subtitle_size": 10,
        "section_size": 13,
        "body_size": 11,
        "equation_size": 13,
        "metric_size": 11,
        "control_size": 11,
        "space_sm": 6,
        "space_md": 9,
        "space_lg": 14,
        "line_width": 2.5,
        "typography_scale": 0.82,
    }
)

_ACCESSIBLE = VisualTokens(
    **{
        **asdict(_ACADEMIC),
        "title_size": 22,
        "body_size": 14,
        "equation_size": 16,
        "metric_size": 14,
        "control_size": 14,
        "data": "#56B4E9",
        "data_border": "#D7F0FF",
        "model": "#E69F00",
        "loss": "#009E73",
        "boundary": "#CC79A7",
        "data_symbol": "circle-open",
        "model_dash": "solid",
        "loss_dash": "dot",
        "line_width": 4.0,
        "typography_scale": 1.04,
    }
)

_THEMES = {
    "classic": _CLASSIC,
    "academic": _ACADEMIC,
    "classroom": _CLASSROOM,
    "compact": _COMPACT,
    "accessible": _ACCESSIBLE,
}
_FORMATS = {"dashboard", "compact", "report"}
_DENSITIES = {"essential", "academic", "complete"}
_SIZE_PRESETS: dict[str, tuple[int | None, float]] = {
    "default": (None, 1.0),
    "compact": (820, 0.84),
    "notebook": (1000, 0.95),
    "wide": (1400, 1.08),
    # Retain the public name without maintaining a second, taller layout.
    "classroom": (1400, 1.08),
}


def available_themes() -> tuple[str, ...]:
    """Return registered visual theme names."""
    return tuple(_THEMES)


def get_theme_tokens(theme: str | None = None) -> VisualTokens:
    """Resolve a theme name to immutable visual tokens."""
    if theme is not None and not isinstance(theme, str):
        raise TypeError("theme must be a registered theme name or None.")
    key = (theme or "classic").lower()
    if key not in _THEMES:
        choices = ", ".join(_THEMES)
        raise ValueError(f"Unknown theme {theme!r}. Available themes: {choices}.")
    return _THEMES[key]


def theme_palette(theme: str | None = None) -> dict[str, Any]:
    """Return legacy builder keys derived from the resolved token set."""
    tokens = get_theme_tokens(theme)
    return {
        "bg": tokens.background,
        "text": tokens.text,
        "text_secondary": tokens.muted,
        "font_family": tokens.font_family,
        "template": "plotly_dark",
        "data_marker": tokens.data,
        "data_marker_border": tokens.data_border,
        "data_marker_symbol": tokens.data_symbol,
        "data_opacity": 0.85,
        "model_line": tokens.model,
        "model_line_width": tokens.line_width,
        "model_line_dash": tokens.model_dash,
        "loss_line": tokens.loss,
        "loss_line_width": max(2.5, tokens.line_width - 1.0),
        "loss_line_dash": tokens.loss_dash,
        "panel_bg": tokens.panel,
        "prediction_label_border": tokens.loss,
        "surface_colorscale": None,
        "surface_opacity": 0.55,
        "legend_bg": tokens.legend_background,
        "legend_border": "rgba(0,0,0,0.6)",
        "legend_font_color": tokens.legend_text,
        "btn_bg": tokens.control_background,
        "btn_border": tokens.grid,
        "btn_font_color": tokens.control_text,
        "btn_border_width": 1,
        "control_size": tokens.control_size,
        "annotation_color": tokens.text,
        "title_size": tokens.title_size,
        "annotation_size": tokens.equation_size,
        "slider_font_color": tokens.text if theme not in {None, "classic"} else None,
        "plot_bg": tokens.background if theme not in {None, "classic"} else None,
        "paper_bg": tokens.background if theme not in {None, "classic"} else None,
        "grid_color": tokens.grid if theme not in {None, "classic"} else None,
        "transition_duration": 0,
    }


def resolve_visual_spec(
    *,
    detail: str,
    theme: str | None = None,
    format: str = "dashboard",
    density: str | None = None,
    size: str = "default",
    width: int | None = None,
    height: int | None = None,
    responsive: bool = False,
    reduced_motion: bool = False,
) -> VisualSpec:
    """Validate and resolve every independent visual-system axis."""
    if detail not in _DENSITIES:
        raise ValueError("detail must be 'essential', 'academic', or 'complete'.")
    if density is not None and not isinstance(density, str):
        raise TypeError("density must be 'essential', 'academic', 'complete', or None.")
    resolved_density = detail if density is None else density.lower()
    if resolved_density not in _DENSITIES:
        raise ValueError("density must be 'essential', 'academic', or 'complete'.")
    if density is not None and detail != "essential" and detail != resolved_density:
        raise ValueError("detail and density must match when both are explicitly non-default.")
    if not isinstance(format, str) or format.lower() not in _FORMATS:
        raise ValueError("format must be 'dashboard', 'compact', or 'report'.")
    if not isinstance(size, str) or size.lower() not in _SIZE_PRESETS:
        choices = ", ".join(_SIZE_PRESETS)
        raise ValueError(f"size must be one of: {choices}.")
    _optional_dimension("width", width)
    _optional_dimension("height", height)
    if not isinstance(responsive, bool):
        raise TypeError("responsive must be a boolean value.")
    if not isinstance(reduced_motion, bool):
        raise TypeError("reduced_motion must be a boolean value.")
    theme_key = (theme or "classic").lower() if isinstance(theme, str) or theme is None else theme
    tokens = get_theme_tokens(theme)
    return VisualSpec(
        theme=theme_key,
        format=format.lower(),
        density=resolved_density,
        size=size.lower(),
        width=width,
        height=height,
        responsive=responsive,
        reduced_motion=reduced_motion,
        tokens=tokens,
    )


def apply_visual_system(fig: Any, spec: VisualSpec, *, family: str) -> Any:
    """Apply an already resolved visual specification to a Plotly figure."""
    if spec.theme != "classic":
        _apply_tokens(fig, spec.tokens)
    if spec.theme == "accessible":
        _apply_accessibility_redundancy(fig, spec.tokens)

    if spec.format == "compact":
        _apply_compact_format(fig, spec.tokens)
    elif spec.format == "report":
        _freeze_to_final_state(fig)
        _apply_report_format(fig, spec.tokens)

    if spec.reduced_motion and spec.format != "report":
        _freeze_to_final_state(fig)

    _apply_size(fig, spec)
    tabular_fit = _apply_tabular_narrow_layout(fig, spec, family=family)
    annotation_fit = _remove_annotation_boxes(fig, spec.tokens)
    if spec.responsive:
        fig.layout.autosize = True
        if spec.width is None:
            fig.layout.width = None

    metadata = dict(fig.layout.meta or {}) if isinstance(fig.layout.meta, dict) else {}
    visual = spec.metadata()
    visual.update(
        {
            "family": family,
            "resolved_width": fig.layout.width,
            "resolved_height": fig.layout.height,
            "autosize": bool(fig.layout.autosize),
            "motion_preserved": bool(fig.frames) and not spec.reduced_motion and spec.format != "report",
            "responsive_config": {"responsive": spec.responsive},
            "narrow_layout_fit": tabular_fit,
            "annotation_boxes": annotation_fit,
            "reflow_strategy": (
                "select a dedicated format for structural reflow; responsive scales the resolved composition"
            ),
            "accessibility": {
                "color_is_redundant": spec.theme == "accessible",
                "static_alternative": "format='report' or reduced_motion=True",
            },
        }
    )
    metadata["mlektic_visual"] = visual
    fig.update_layout(meta=metadata, uirevision=f"mlektic-{spec.theme}-{spec.format}")
    return fig


def _remove_annotation_boxes(fig: Any, tokens: VisualTokens) -> dict[str, Any]:
    """Render every explanatory annotation as text without fixed boxes.

    Fixed annotation backgrounds and borders do not grow reliably with
    MathJax or long numeric values.  Apply the policy to the live layout,
    animation frames, sliders, and staged update-menu payloads so switching a
    frame or explanation stage cannot reintroduce an overflowing rectangle.
    Controls and legends keep their own backgrounds because they are not
    explanatory annotations.
    """
    processed: set[int] = set()
    count = 0

    def strip(annotation: Any) -> None:
        nonlocal count
        identity = id(annotation)
        if identity in processed:
            return
        processed.add(identity)
        if isinstance(annotation, dict):
            previous_background = annotation.get("bgcolor")
            previous_border = annotation.get("borderwidth")
            annotation.pop("bgcolor", None)
            annotation.pop("bordercolor", None)
            annotation["borderwidth"] = 0
            annotation["borderpad"] = 0
            font = annotation.setdefault("font", {})
            if previous_background or previous_border or font.get("color") in {
                "black", "#000", "#000000", "white", "#fff", "#ffffff"
            }:
                font["color"] = tokens.text
        else:
            previous_background = getattr(annotation, "bgcolor", None)
            previous_border = getattr(annotation, "borderwidth", None)
            annotation.update(bgcolor=None, bordercolor=None, borderwidth=0, borderpad=0)
            current_color = getattr(getattr(annotation, "font", None), "color", None)
            if previous_background or previous_border or current_color in {
                "black", "#000", "#000000", "white", "#fff", "#ffffff"
            }:
                annotation.font.color = tokens.text
        count += 1

    def walk_payload(value: Any) -> None:
        if isinstance(value, dict):
            annotations = value.get("annotations")
            if isinstance(annotations, (list, tuple)):
                for annotation in annotations:
                    strip(annotation)
            for child in value.values():
                walk_payload(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk_payload(child)

    for annotation in fig.layout.annotations or ():
        strip(annotation)
    for frame in fig.frames or ():
        if frame.layout is not None:
            for annotation in frame.layout.annotations or ():
                strip(annotation)
    for menu in fig.layout.updatemenus or ():
        for button in menu.buttons or ():
            walk_payload(button.args)
    for slider in fig.layout.sliders or ():
        for step in slider.steps or ():
            walk_payload(step.args)
    return {"policy": "borderless", "annotations_processed": count}


def _apply_tabular_narrow_layout(fig: Any, spec: VisualSpec, *, family: str) -> dict[str, Any]:
    """Keep tabular titles and equations readable on compact/notebook canvases."""
    if not family.startswith(("linear", "logistic")):
        return {"applied": False}
    resolved_width = int(fig.layout.width or 1100)
    if resolved_width > 1000:
        return {"applied": False}

    title_cap = 20 if resolved_width <= 850 else 22
    equation_cap = 14 if resolved_width <= 850 else 16
    panel_cap = 12 if resolved_width <= 850 else 13
    if fig.layout.title is not None:
        current_title_size = int(fig.layout.title.font.size or spec.tokens.title_size)
        fig.layout.title.update(
            # Leave enough physical pixels above the first glyph. Plotly's
            # title box can clip a two-line title when top-anchored at 0.99.
            y=0.96,
            yanchor="top",
            font={"size": min(current_title_size, title_cap)},
        )

    metadata = fig.layout.meta if isinstance(fig.layout.meta, dict) else {}
    math_contract = metadata.get("mlektic_math", {}) if isinstance(metadata, dict) else {}
    borderless_math = math_contract.get("detail") in {"academic", "complete"}

    def fit_annotations(annotations: Any) -> None:
        for annotation in annotations or ():
            y_value = float(annotation.y or 0)
            current_size = int(annotation.font.size or spec.tokens.body_size)
            if y_value > 1:
                annotation.font.size = min(current_size, equation_cap)
            elif y_value < 0 and borderless_math:
                annotation.font.size = min(current_size, panel_cap)
                annotation.bgcolor = "rgba(0,0,0,0)"
                annotation.borderwidth = 0
                annotation.borderpad = 0
            if _is_metric_annotation(annotation.text):
                annotation.font.size = min(current_size, panel_cap)
                annotation.font.color = spec.tokens.text
                annotation.bgcolor = "rgba(0,0,0,0)"
                annotation.borderwidth = 0
                annotation.borderpad = 0

    fit_annotations(fig.layout.annotations)
    for frame in fig.frames or ():
        if frame.layout is not None:
            fit_annotations(frame.layout.annotations)

    def fit_equation_trace(trace: Any) -> None:
        if getattr(trace, "uid", None) != "NUMERIC_EQUATION":
            return
        current_size = int(trace.textfont.size or spec.tokens.equation_size)
        trace.textfont.size = min(current_size, equation_cap)

    for trace in fig.data:
        fit_equation_trace(trace)
    for frame in fig.frames or ():
        for trace in frame.data or ():
            fit_equation_trace(trace)

    return {
        "applied": True,
        "resolved_width": resolved_width,
        "title_size_cap": title_cap,
        "equation_size_cap": equation_cap,
        "borderless_math_panel": borderless_math,
    }


def _is_metric_annotation(text: Any) -> bool:
    """Return whether an annotation is one of the tabular metric readouts."""
    normalized = str(text or "").upper()
    if "<B>" not in normalized:
        return False
    return any(
        label in normalized
        for label in ("LOSS", "MSE", "MAE", "R²", "R2", "ACCURACY", "F1", "LOG-LOSS")
    )


def _optional_dimension(name: str, value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, Real) or isinstance(value, bool) or not float(value).is_integer():
        raise TypeError(f"{name} must be a positive integer or None.")
    if int(value) < 320:
        raise ValueError(f"{name} must be at least 320 pixels.")


def _apply_tokens(fig: Any, tokens: VisualTokens) -> None:
    fig.update_layout(
        paper_bgcolor=tokens.background,
        plot_bgcolor=tokens.background,
        font={"family": tokens.font_family, "color": tokens.text, "size": tokens.body_size},
        hoverlabel={"bgcolor": tokens.panel, "font": {"color": tokens.text}},
    )
    if fig.layout.title is not None:
        fig.layout.title.font.update(size=tokens.title_size, color=tokens.text)
    for annotation in fig.layout.annotations or ():
        current_size = annotation.font.size or tokens.body_size
        annotation.font.update(
            size=max(9, round(float(current_size) * tokens.typography_scale)),
            color=tokens.text if annotation.font.color in {None, "white", "#ffffff"} else annotation.font.color,
            family=tokens.font_family,
        )
    for frame in fig.frames or ():
        if frame.layout is not None:
            for annotation in frame.layout.annotations or ():
                current_size = annotation.font.size or tokens.body_size
                annotation.font.update(
                    size=max(9, round(float(current_size) * tokens.typography_scale)),
                    color=(
                        tokens.text
                        if annotation.font.color in {None, "white", "#ffffff"}
                        else annotation.font.color
                    ),
                    family=tokens.font_family,
                )
    _style_axes(fig, tokens)
    _style_controls(fig, tokens)
    for trace in fig.data:
        _style_trace(trace, tokens)
    for frame in fig.frames or ():
        for trace in frame.data or ():
            _style_trace(trace, tokens)


def _style_axes(fig: Any, tokens: VisualTokens) -> None:
    layout_json = fig.layout.to_plotly_json()
    for axis_name, value in layout_json.items():
        if axis_name.startswith(("xaxis", "yaxis")) and isinstance(value, dict):
            getattr(fig.layout, axis_name).update(
                gridcolor=tokens.grid,
                zerolinecolor=tokens.grid,
                tickfont={"size": tokens.body_size, "color": tokens.text},
            )
        if axis_name.startswith("scene") and isinstance(value, dict):
            scene = getattr(fig.layout, axis_name)
            for coordinate in ("xaxis", "yaxis", "zaxis"):
                getattr(scene, coordinate).update(
                    gridcolor=tokens.grid,
                    zerolinecolor=tokens.grid,
                    tickfont={"size": tokens.body_size, "color": tokens.text},
                )


def _style_controls(fig: Any, tokens: VisualTokens) -> None:
    for menu in fig.layout.updatemenus or ():
        # A builder may reserve a precise control size for a dense composition.
        # Themes recolor that control without unexpectedly enlarging it.
        resolved_size = menu.font.size or tokens.control_size
        menu.update(
            bgcolor=tokens.control_background,
            bordercolor=tokens.grid,
            font={"color": tokens.control_text, "size": resolved_size},
        )
    for slider in fig.layout.sliders or ():
        slider.font.update(color=tokens.text, size=tokens.control_size)
        slider.currentvalue.font.update(color=tokens.text, size=tokens.control_size)


def _style_trace(trace: Any, tokens: VisualTokens) -> None:
    name = str(getattr(trace, "name", "") or "").lower()
    uid = str(getattr(trace, "uid", "") or "")
    if uid.startswith(("NN_BLOCK_", "NN_GRAPH_")) or name.startswith("neural graph"):
        return
    mode = str(getattr(trace, "mode", "") or "")
    is_loss = uid == "LOSS_LINE" or "loss" in name or "mse" in name
    is_model = uid == "MODEL_LINE" or any(word in name for word in ("model", "probability curve", "boundary"))
    is_data = name == "data" or (
        "markers" in mode
        and not is_model
        and not is_loss
        and uid not in {"METRIC_VALUES", "NUMERIC_EQUATION"}
        and "prediction" not in name
    )
    if is_data and getattr(trace, "marker", None) is not None:
        trace.marker.update(color=tokens.data, symbol=tokens.data_symbol)
        if getattr(trace.marker, "line", None) is not None:
            trace.marker.line.update(color=tokens.data_border)
    if is_model and getattr(trace, "line", None) is not None:
        trace.line.update(color=tokens.model, width=tokens.line_width, dash=tokens.model_dash)
    if is_loss and getattr(trace, "line", None) is not None:
        trace.line.update(color=tokens.loss, width=max(2.5, tokens.line_width - 1), dash=tokens.loss_dash)


def _apply_accessibility_redundancy(fig: Any, tokens: VisualTokens) -> None:
    marker_symbols = ("circle-open", "square-open", "diamond-open", "triangle-up-open", "x")
    trace_sets = [fig.data, *(frame.data or () for frame in fig.frames or ())]
    for traces in trace_sets:
        class_index = 0
        for trace in traces:
            name = str(getattr(trace, "name", "") or "").lower()
            if name.startswith("class ") and getattr(trace, "marker", None) is not None:
                trace.marker.symbol = marker_symbols[class_index % len(marker_symbols)]
                class_index += 1
            if "boundary" in name and getattr(trace, "line", None) is not None:
                trace.line.update(color=tokens.boundary, dash="dash", width=tokens.line_width)
    for shape in fig.layout.shapes or ():
        if shape.line is not None and shape.type == "line":
            shape.line.width = max(float(shape.line.width or 1), tokens.line_width)


def _apply_compact_format(fig: Any, tokens: VisualTokens) -> None:
    current_height = int(fig.layout.height or 600)
    margin = fig.layout.margin.to_plotly_json()
    has_below_plot = any(float(annotation.y or 0) < 0 for annotation in fig.layout.annotations or ())
    current_top = int(margin.get("t", 100))
    compact_top = max(70, round(current_top * 0.78))
    margin.update(
        t=compact_top,
        r=max(24, round(float(margin.get("r", 30)) * 0.8)),
        l=max(44, round(float(margin.get("l", 60)) * 0.8)),
        b=(
            int(margin.get("b", 70))
            if has_below_plot
            else max(55, round(float(margin.get("b", 70)) * 0.8))
        ),
    )
    if has_below_plot:
        # Mathematical panels below the slider use negative paper coordinates.
        # Preserve the plot-domain pixel height so those coordinates cannot
        # drift upward into slider labels when the header is compacted.
        original_inner_height = max(
            1,
            current_height - current_top - int(margin.get("b", 70)),
        )
        header_shift = (current_top - compact_top) / original_inner_height
        for annotation in fig.layout.annotations or ():
            if float(annotation.y or 0) > 1:
                annotation.y = float(annotation.y) - header_shift
        for frame in fig.frames or ():
            if frame.layout is None:
                continue
            for annotation in frame.layout.annotations or ():
                if float(annotation.y or 0) > 1:
                    annotation.y = float(annotation.y) - header_shift
        fig.layout.height = max(420, current_height - (current_top - compact_top))
    else:
        fig.layout.height = max(420, round(current_height * 0.86))
    fig.layout.margin.update(margin)
    if fig.layout.title is not None:
        fig.layout.title.font.size = min(int(fig.layout.title.font.size or 24), tokens.title_size)
    for legend_name in ("legend", "legend2"):
        legend = getattr(fig.layout, legend_name, None)
        if legend is not None:
            legend.font.update(size=tokens.body_size)


def _apply_report_format(fig: Any, tokens: VisualTokens) -> None:
    fig.layout.updatemenus = ()
    fig.layout.sliders = ()
    fig.update_layout(transition={"duration": 0})
    margin = fig.layout.margin.to_plotly_json()
    height = int(fig.layout.height or 600)
    top = int(margin.get("t", 100))
    bottom = int(margin.get("b", 70))
    removable = max(0, bottom - 55)
    cut = min(80, removable)
    inner_height = max(1, height - top - bottom)
    if cut:
        shift = cut / inner_height
        for annotation in fig.layout.annotations or ():
            if float(annotation.y or 0) < 0:
                annotation.y = float(annotation.y) + shift
        margin["b"] = bottom - cut
        fig.layout.height = max(420, height - cut)
    fig.layout.margin.update(margin)
    if fig.layout.title is not None:
        fig.layout.title.font.size = tokens.title_size


def _freeze_to_final_state(fig: Any) -> None:
    if not fig.frames:
        _apply_static_output_stage(fig)
        fig.layout.updatemenus = ()
        fig.layout.sliders = ()
        return
    final_frame = fig.frames[-1]
    targets = list(final_frame.traces or range(len(final_frame.data or ())))
    for target, update in zip(targets, final_frame.data or ()):
        if 0 <= int(target) < len(fig.data):
            fig.data[int(target)].update(update)
    if final_frame.layout is not None:
        update = final_frame.layout.to_plotly_json()
        if "annotations" in update:
            static_annotations = [
                annotation.to_plotly_json()
                for annotation in fig.layout.annotations or ()
                if float(annotation.y or 0) < 0
            ]
            update["annotations"] = [*update["annotations"], *static_annotations]
        fig.update_layout(**update)
    fig.frames = []
    fig.layout.updatemenus = ()
    fig.layout.sliders = ()


def _apply_static_output_stage(fig: Any) -> bool:
    """Select a prediction explainer's complete Output stage before freezing.

    Prediction explainers use update menus rather than Plotly frames.  Merely
    deleting those controls produced empty N-dimensional panels in report and
    reduced-motion modes.  Apply the exact payload supplied by the public
    Output button first, then let the caller remove the now-redundant menu.
    """
    for menu in fig.layout.updatemenus or ():
        output_button = next(
            (
                button
                for button in menu.buttons or ()
                if str(getattr(button, "label", "")).strip().lower() == "output"
            ),
            None,
        )
        if output_button is None or not output_button.args:
            continue

        args = list(output_button.args)
        method = str(output_button.method or "update").lower()
        if method == "update":
            trace_update = args[0] if args and isinstance(args[0], dict) else {}
            layout_update = args[1] if len(args) > 1 and isinstance(args[1], dict) else {}
            visible = trace_update.get("visible")
            if isinstance(visible, (list, tuple)):
                for trace, is_visible in zip(fig.data, visible):
                    trace.visible = is_visible
            if layout_update:
                _replace_staged_layout_content(fig, layout_update)
            return True
        if method == "relayout" and args and isinstance(args[0], dict):
            _replace_staged_layout_content(fig, args[0])
            return True
    return False


def _replace_staged_layout_content(fig: Any, payload: dict[str, Any]) -> None:
    """Apply stage-owned arrays by replacement rather than Plotly tuple merge."""
    scalar_updates: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "annotations":
            fig.layout.annotations = value
        elif key == "shapes":
            fig.layout.shapes = value
        elif key.endswith(".annotations"):
            container_name = key.removesuffix(".annotations")
            container = getattr(fig.layout, container_name, None)
            if container is not None:
                container.annotations = value
        else:
            scalar_updates[key] = value
    if scalar_updates:
        fig.update_layout(**scalar_updates)


def _apply_size(fig: Any, spec: VisualSpec) -> None:
    preset_width, height_scale = _SIZE_PRESETS[spec.size]
    current_height = int(fig.layout.height or 600)
    if spec.size != "default":
        fig.layout.width = preset_width
        has_below_plot = any(float(annotation.y or 0) < 0 for annotation in fig.layout.annotations or ())
        # Below-plot annotations use paper coordinates resolved against the
        # plot domain.  Scaling total height without scaling the bottom margin
        # moves their centers past the canvas edge.  Their builder has already
        # computed an exact content-aware height, so size presets change width
        # only for these compositions.
        safe_height_scale = 1.0 if has_below_plot else height_scale
        fig.layout.height = max(320, round(current_height * safe_height_scale))
    if spec.width is not None:
        fig.layout.width = int(spec.width)
    if spec.height is not None:
        fig.layout.height = int(spec.height)


__all__ = [
    "VisualSpec",
    "VisualTokens",
    "apply_visual_system",
    "available_themes",
    "get_theme_tokens",
    "resolve_visual_spec",
    "theme_palette",
]
