"""Keep legends off the data they describe.

Builders place each legend at a fixed corner of its subplot. Whether that
corner is empty depends on the data: a binary logistic figure whose class 0
sits on the right puts its points under a bottom-right legend. This pass runs
on every built figure, after its size is known. For each legend that sits
inside a cartesian subplot it estimates the legend's box, collects every point
and line drawn in that subplot in every animation frame, and moves the legend
to the candidate position (corners first, then edge midpoints) that covers no
mark and no annotation (metric cards, threshold labels). When no inside position is free, the least-covered one is kept and the
decision is recorded in ``layout.meta``. Scenes (3-D) and legends placed
outside the plots are left alone.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

import numpy as np

_DEFAULT_MARGIN = {"l": 80, "r": 80, "t": 100, "b": 80}
_INSET = 0.012  # paper units kept between the legend and the subplot frame
_SAMPLES_PER_SEGMENT = 24
_SCAN_STEPS = 21


def _axis_key(name: str | None, letter: str) -> str:
    """``"x2"`` -> ``"xaxis2"``; ``None`` or ``"x"`` -> ``"xaxis"``."""
    suffix = (name or letter)[1:]
    return f"{letter}axis{suffix}"


def _numeric(values: Any) -> np.ndarray | None:
    if values is None or isinstance(values, (str, dict)):
        return None
    try:
        array = np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    return array


def _trace_points(trace: Any, *, x: Any = None, y: Any = None) -> np.ndarray:
    """Points a trace draws, with line segments densified."""
    xs = _numeric(x if x is not None else getattr(trace, "x", None))
    ys = _numeric(y if y is not None else getattr(trace, "y", None))
    if xs is None or ys is None or xs.size == 0 or xs.size != ys.size:
        return np.empty((0, 2))
    points = np.column_stack([xs, ys])
    mode = str(getattr(trace, "mode", "") or "")
    if "lines" in mode and len(points) > 1:
        dense = [points[:1]]
        for start, end in zip(points[:-1], points[1:]):
            if np.isfinite(start).all() and np.isfinite(end).all():
                t = np.linspace(0.0, 1.0, _SAMPLES_PER_SEGMENT)[1:, None]
                dense.append(start + (end - start) * t)
            else:
                dense.append(end[None, :])
        points = np.vstack(dense)
    return points[np.isfinite(points).all(axis=1)]


def _figure_size(fig: Any) -> tuple[float, float, dict[str, float]]:
    layout = fig.layout
    margin = dict(_DEFAULT_MARGIN)
    for key in margin:
        value = getattr(layout.margin, key, None) if layout.margin is not None else None
        if value is not None:
            margin[key] = float(value)
    width = float(layout.width or 1000)
    height = float(layout.height or 600)
    return width, height, margin


def _legend_entries(fig: Any, name: str) -> list[str]:
    entries = []
    groups = set()
    for trace in fig.data:
        if getattr(trace, "showlegend", None) is False or getattr(trace, "visible", True) is False:
            continue
        if (getattr(trace, "legend", None) or "legend") != name:
            continue
        label = str(getattr(trace, "name", "") or "")
        if not label:
            continue
        group = getattr(trace, "legendgroup", None)
        key = (group, label)
        if key in groups:
            continue
        groups.add(key)
        entries.append(re.sub(r"<[^>]+>", "", label))
    return entries


def _legend_box(legend: Any, entries: list[str], width: float, height: float, margin: dict) -> tuple[float, float]:
    """Legend width and height in paper units of the plotting area."""
    font = float(getattr(getattr(legend, "font", None), "size", None) or 12)
    horizontal = str(getattr(legend, "orientation", "v") or "v") == "h"
    longest = max((len(entry) for entry in entries), default=4)
    if horizontal:
        box_w = sum(40 + 0.6 * font * len(entry) for entry in entries) + 10
        box_h = font + 16
    else:
        box_w = 44 + 0.6 * font * longest
        box_h = (font + 8) * len(entries) + 10
    plot_w = max(1.0, width - margin["l"] - margin["r"])
    plot_h = max(1.0, height - margin["t"] - margin["b"])
    return box_w / plot_w, box_h / plot_h


def _subplot_for(fig: Any, x: float, y: float) -> tuple[str, str] | None:
    """The cartesian subplot whose domain contains paper point ``(x, y)``."""
    layout_json = fig.layout.to_plotly_json()
    best = None
    for key, value in layout_json.items():
        if not key.startswith("xaxis") or not isinstance(value, dict):
            continue
        x_domain = value.get("domain", [0, 1])
        anchor = value.get("anchor") or "y" + key[5:]
        y_key = _axis_key(anchor, "y")
        y_domain = (layout_json.get(y_key) or {}).get("domain", [0, 1])
        if x_domain[0] - 1e-6 <= x <= x_domain[1] + 1e-6 and y_domain[0] - 1e-6 <= y <= y_domain[1] + 1e-6:
            area = (x_domain[1] - x_domain[0]) * (y_domain[1] - y_domain[0])
            if best is None or area < best[0]:
                best = (area, key, y_key)
    return None if best is None else (best[1], best[2])


def _marks(fig: Any, x_key: str, y_key: str) -> np.ndarray:
    x_name = "x" + x_key[5:]
    y_name = "y" + y_key[5:]
    collected = []
    owners = []
    for index, trace in enumerate(fig.data):
        if str(getattr(trace, "type", "")) not in {"scatter", "scattergl", "bar"}:
            continue
        if (getattr(trace, "xaxis", None) or "x") != x_name or (getattr(trace, "yaxis", None) or "y") != y_name:
            continue
        if getattr(trace, "visible", True) is False:
            continue
        owners.append(index)
        collected.append(_trace_points(trace))
    owned = set(owners)
    for frame in fig.frames or ():
        targets = list(frame.traces) if frame.traces is not None else list(range(len(frame.data or ())))
        for target, update in zip(targets, frame.data or ()):
            if int(target) in owned:
                base = fig.data[int(target)]
                collected.append(
                    _trace_points(
                        base,
                        x=getattr(update, "x", None) if getattr(update, "x", None) is not None else base.x,
                        y=getattr(update, "y", None) if getattr(update, "y", None) is not None else base.y,
                    )
                )
    collected = [points for points in collected if len(points)]
    return np.vstack(collected) if collected else np.empty((0, 2))


def _axis_range(axis: dict, values: np.ndarray) -> tuple[float, float, bool] | None:
    log = axis.get("type") == "log"
    explicit = axis.get("range")
    if explicit is not None and len(explicit) == 2 and None not in explicit:
        low, high = float(explicit[0]), float(explicit[1])
        if axis.get("autorange") == "reversed":
            low, high = high, low
        return low, high, log
    data = np.log10(values[values > 0]) if log else values
    if data.size == 0:
        return None
    low, high = float(np.min(data)), float(np.max(data))
    pad = 0.06 * (high - low) if high > low else 0.5
    low, high = low - pad, high + pad
    if axis.get("autorange") == "reversed":
        low, high = high, low
    return low, high, log


def _to_paper(values: np.ndarray, axis: dict, bounds: tuple[float, float, bool]) -> np.ndarray:
    low, high, log = bounds
    domain = axis.get("domain", [0, 1])
    data = np.log10(np.where(values > 0, values, np.nan)) if log else values
    return domain[0] + (data - low) / (high - low) * (domain[1] - domain[0])


def _candidates(x_domain, y_domain, box_w, box_h) -> Iterable[tuple[str, float, float, str, str]]:
    left, right = x_domain[0] + _INSET, x_domain[1] - _INSET
    bottom, top = y_domain[0] + _INSET, y_domain[1] - _INSET
    center_x = (x_domain[0] + x_domain[1]) / 2
    center_y = (y_domain[0] + y_domain[1]) / 2
    yield "bottom right", right, bottom, "right", "bottom"
    yield "top right", right, top, "right", "top"
    yield "top left", left, top, "left", "top"
    yield "bottom left", left, bottom, "left", "bottom"
    yield "middle right", right, center_y, "right", "middle"
    yield "middle left", left, center_y, "left", "middle"
    yield "top center", center_x, top, "center", "top"
    yield "bottom center", center_x, bottom, "center", "bottom"


def _box(x, y, xanchor, yanchor, box_w, box_h):
    x0 = {"left": x, "center": x - box_w / 2, "right": x - box_w}.get(xanchor, x - box_w)
    y0 = {"bottom": y, "middle": y - box_h / 2, "top": y - box_h}.get(yanchor, y)
    return x0, x0 + box_w, y0, y0 + box_h


_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+")
_ANNOTATION_WEIGHT = 1_000_000  # covering a label is worse than covering any number of points


def _annotation_extent(annotation: dict) -> tuple[float, float]:
    """Approximate pixel width and height of an annotation's text box."""
    text = str(annotation.get("text") or "")
    font = float((annotation.get("font") or {}).get("size") or 12)
    if "$" in text:
        rows = max(1, text.count("\\\\") + 1)
        body = _LATEX_COMMAND.sub("x", text)
        body = re.sub(r"[{}$&_^ ]", "", body)
        lines = [body[: max(1, len(body) // rows)]] * rows
        line_height = 1.9 * font
    else:
        lines = re.split(r"<br\s*/?>", text, flags=re.I)
        lines = [re.sub(r"<[^>]+>", "", line) for line in lines]
        line_height = 1.3 * font
    pad = 2 * (float(annotation.get("borderpad") or 1) + float(annotation.get("borderwidth") or 0))
    width = 0.6 * font * max((len(line) for line in lines), default=0) + pad
    height = line_height * len(lines) + pad
    return width, height


def _paper_coordinate(value: Any, ref: str, axes: dict) -> float | None:
    """Map an annotation or shape coordinate to paper units."""
    if value is None:
        return None
    if ref == "paper":
        return float(value)
    name, _, domain = ref.partition(" ")
    entry = axes.get(name)
    if entry is None:
        return None
    axis, bounds = entry
    span = axis.get("domain", [0, 1])
    if domain == "domain":
        return span[0] + float(value) * (span[1] - span[0])
    if bounds is None:
        return None
    try:
        return float(_to_paper(np.asarray([float(value)]), axis, bounds)[0])
    except (TypeError, ValueError):
        return None


def _annotation_boxes(fig: Any, axes: dict, plot_w: float, plot_h: float) -> list[tuple[float, float, float, float]]:
    """Paper-unit boxes of every annotation drawn in the layout or any frame."""
    groups = [fig.layout.annotations or ()]
    for frame in fig.frames or ():
        if frame.layout is not None and frame.layout.annotations:
            groups.append(frame.layout.annotations)
    boxes = []
    seen = set()
    for group in groups:
        for item in group:
            annotation = item.to_plotly_json() if hasattr(item, "to_plotly_json") else dict(item)
            if annotation.get("visible") is False or not annotation.get("text"):
                continue
            xref = annotation.get("xref") or "x"
            yref = annotation.get("yref") or "y"
            x = _paper_coordinate(annotation.get("x"), xref, axes)
            y = _paper_coordinate(annotation.get("y"), yref, axes)
            if x is None or y is None:
                continue
            width, height = _annotation_extent(annotation)
            x += float(annotation.get("xshift") or 0) / plot_w
            y += float(annotation.get("yshift") or 0) / plot_h
            box_w, box_h = width / plot_w, height / plot_h
            xanchor = annotation.get("xanchor") or "auto"
            yanchor = annotation.get("yanchor") or "auto"
            xanchor = "center" if xanchor == "auto" else xanchor
            yanchor = "middle" if yanchor == "auto" else yanchor
            box = _box(x, y, xanchor, yanchor, box_w, box_h)
            key = tuple(round(v, 4) for v in box)
            if key not in seen:
                seen.add(key)
                boxes.append(box)
    return boxes


def _shape_marks(fig: Any, x_key: str, y_key: str) -> np.ndarray:
    """Points along line shapes drawn in data coordinates of this subplot."""
    x_name = "x" + x_key[5:]
    y_name = "y" + y_key[5:]
    points = []
    for shape in fig.layout.shapes or ():
        if (shape.xref or "x") != x_name or (shape.yref or "y") != y_name or shape.type not in {"line", None}:
            continue
        try:
            start = np.array([float(shape.x0), float(shape.y0)])
            end = np.array([float(shape.x1), float(shape.y1)])
        except (TypeError, ValueError):
            continue
        t = np.linspace(0.0, 1.0, 4 * _SAMPLES_PER_SEGMENT)[:, None]
        points.append(start + (end - start) * t)
    return np.vstack(points) if points else np.empty((0, 2))


def _overlaps(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def place_legends(fig: Any) -> dict[str, Any]:
    """Move every in-plot legend to a position that covers no data; return a report."""
    if fig.layout.showlegend is False:
        return {}
    width, height, margin = _figure_size(fig)
    layout_json = fig.layout.to_plotly_json()
    report: dict[str, Any] = {}
    for name in ("legend", "legend2", "legend3"):
        legend = getattr(fig.layout, name, None)
        if legend is None:
            continue
        settings = legend.to_plotly_json()
        if name != "legend" and not settings:
            continue
        if settings.get("xref") == "container" or settings.get("yref") == "container":
            continue
        entries = _legend_entries(fig, name)
        if not entries:
            continue
        x = float(settings.get("x", 1.02))
        y = float(settings.get("y", 1.0))
        # Plotly's defaults: ``xanchor`` is "left"; ``yanchor`` "auto" follows y.
        xanchor = settings.get("xanchor") or "left"
        yanchor = settings.get("yanchor") or "auto"
        if xanchor == "auto":
            xanchor = "left" if x <= 1 / 3 else "center" if x < 2 / 3 else "right"
        if yanchor == "auto":
            yanchor = "bottom" if y <= 1 / 3 else "middle" if y < 2 / 3 else "top"
        box_w, box_h = _legend_box(legend, entries, width, height, margin)
        x0, x1, y0, y1 = _box(x, y, xanchor, yanchor, box_w, box_h)
        subplot = _subplot_for(fig, (x0 + x1) / 2, (y0 + y1) / 2)
        if subplot is None:
            continue
        x_key, y_key = subplot
        x_axis = layout_json.get(x_key, {})
        y_axis = layout_json.get(y_key, {})
        marks = _marks(fig, x_key, y_key)
        if marks.size == 0:
            continue
        x_bounds = _axis_range(x_axis, marks[:, 0])
        y_bounds = _axis_range(y_axis, marks[:, 1])
        if x_bounds is None or y_bounds is None:
            continue
        shapes = _shape_marks(fig, x_key, y_key)
        if shapes.size:
            marks = np.vstack([marks, shapes])
        px = _to_paper(marks[:, 0], x_axis, x_bounds)
        py = _to_paper(marks[:, 1], y_axis, y_bounds)
        # Frames repeat most marks; one per 1/1000 of the paper is enough to test overlap.
        finite = np.isfinite(px) & np.isfinite(py)
        grid = np.unique(np.round(np.column_stack([px[finite], py[finite]]), 3), axis=0)
        px, py = grid[:, 0], grid[:, 1]
        pad_x, pad_y = 0.25 * box_h * (height / width), 0.25 * box_h
        plot_w = max(1.0, width - margin["l"] - margin["r"])
        plot_h = max(1.0, height - margin["t"] - margin["b"])
        axes = {}
        for key, value in layout_json.items():
            if isinstance(value, dict) and (key.startswith("xaxis") or key.startswith("yaxis")):
                short = key[0] + key[5:]
                bounds = x_bounds if key == x_key else y_bounds if key == y_key else None
                axes[short] = (value, bounds)
        labels = _annotation_boxes(fig, axes, plot_w, plot_h)

        def covered(candidate_box):
            a0, a1, b0, b1 = candidate_box
            inside = (px >= a0 - pad_x) & (px <= a1 + pad_x) & (py >= b0 - pad_y) & (py <= b1 + pad_y)
            hidden_labels = sum(1 for label in labels if _overlaps(candidate_box, label))
            return int(np.count_nonzero(inside)) + _ANNOTATION_WEIGHT * hidden_labels

        def summary(score):
            labels_hit, marks_hit = divmod(score, _ANNOTATION_WEIGHT)
            return {"covered_marks": marks_hit, "covered_labels": labels_hit}

        current = covered((x0, x1, y0, y1))
        if current == 0:
            report[name] = {"position": "builder default", **summary(0)}
            continue
        x_domain = x_axis.get("domain", [0, 1])
        y_domain = y_axis.get("domain", [0, 1])
        options = []
        for order, (label, cx, cy, ax, ay) in enumerate(_candidates(x_domain, y_domain, box_w, box_h)):
            options.append((covered(_box(cx, cy, ax, ay, box_w, box_h)), order, label, cx, cy, ax, ay))
        count, _, label, cx, cy, ax, ay = min(options)
        if count:
            # No corner or edge is free: search the whole plot, top rows first.
            left, right = x_domain[0] + _INSET, x_domain[1] - _INSET - box_w
            bottom, top = y_domain[0] + _INSET, y_domain[1] - _INSET - box_h
            if right > left and top > bottom:
                scanned = []
                for row, y_start in enumerate(np.linspace(top, bottom, _SCAN_STEPS)):
                    for column, x_start in enumerate(np.linspace(left, right, _SCAN_STEPS)):
                        score = covered((x_start, x_start + box_w, y_start, y_start + box_h))
                        scanned.append((score, row, column, float(x_start), float(y_start)))
                best = min(scanned)
                if best[0] < count:
                    count, label = best[0], "inside"
                    cx, cy, ax, ay = best[3], best[4], "left", "bottom"
        if count >= current:
            report[name] = {"position": "builder default", **summary(current)}
            continue
        legend.update(x=cx, y=cy, xanchor=ax, yanchor=ay)
        previous = summary(current)
        report[name] = {
            "position": label,
            **summary(count),
            "previously_covered_marks": previous["covered_marks"],
            "previously_covered_labels": previous["covered_labels"],
        }
    return report


__all__ = ["place_legends"]
