"""Motion effects for skins.

Effects only add marks (extra traces drawn under or over the originals) or
recolour numbers inside equations.  They never resize, move, or remove an
existing element, and they are skipped on any axis that autoranges, so the
extra marks can never change a computed range.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable

import plotly.graph_objects as go

from . import roles
from .definitions import Skin

_NUMBER = re.compile(r"(?<![\\\w.])[+-]?\d+\.\d+")
_TRAIL_SPAN = 30  # roughly how many trail strides fit in one playback
# Trail and glow are soft, wide decorations; this many points per line is
# visually exact for them and keeps every frame's payload small.
_DECOR_POINTS = 64


def apply_effects(fig: Any, skin: Skin) -> dict[str, Any]:
    """Apply the skin's enabled effects and report which ones ran or why they did not."""
    effects = skin.effects
    report: dict[str, Any] = {"applied": [], "skipped": {}}
    if not any((
        effects.glow, effects.trail, effects.loss_progress, effects.equation_delta, effects.residuals,
    )):
        return report
    if _has_index_bound_controls(fig):
        report["skipped"]["all"] = "controls address traces by index"
        return report

    animated = bool(fig.frames)
    palette = skin.palette

    def run(name: str, enabled: Any, role: str, action: Callable[[int], None], *, needs_frames: bool) -> None:
        if not enabled:
            return
        # Each insertion shifts trace indices, so the target is looked up when the effect runs.
        target = _find(fig, role)
        if target is None:
            report["skipped"][name] = "no matching mark"
        elif needs_frames and not animated:
            report["skipped"][name] = "figure is not animated"
        elif not _fixed_axes(fig, fig.data[target]):
            report["skipped"][name] = "axis autoranges"
        else:
            action(target)
            report["applied"].append(name)

    run("trail", effects.trail, roles.MODEL, lambda i: _trail(fig, i, palette.model, effects), needs_frames=True)
    run("glow", effects.glow, roles.MODEL, lambda i: _glow(fig, i, palette.model, effects), needs_frames=False)
    run("loss_progress", effects.loss_progress, roles.LOSS, lambda i: _loss_progress(fig, i, palette),
        needs_frames=True)
    if effects.residuals:
        model = _find(fig, roles.MODEL)
        data = _find_data(fig, fig.data[model]) if model is not None else None
        if model is None or data is None:
            report["skipped"]["residuals"] = "no data points on the model's axes"
        elif not _fixed_axes(fig, fig.data[model]):
            report["skipped"]["residuals"] = "axis autoranges"
        else:
            _residuals(fig, model, data, palette.accent)
            report["applied"].append("residuals")
    if effects.equation_delta:
        if animated and _equation_delta(fig, palette.increase, palette.decrease):
            report["applied"].append("equation_delta")
        else:
            report["skipped"]["equation_delta"] = "no animated equations"
    return report


# ── trace bookkeeping ────────────────────────────────────────────────


def _find(fig: Any, role: str) -> int | None:
    for index, trace in enumerate(fig.data):
        if roles.trace_role(trace) == role and roles.is_flat(trace) and "lines" in str(trace.mode or "lines"):
            return index
    return None


def _has_index_bound_controls(fig: Any) -> bool:
    """Return whether a button or slider restyles traces by position."""
    controls = [*(fig.layout.updatemenus or ())]
    for menu in controls:
        for button in menu.buttons or ():
            if button.method in {"restyle", "update"}:
                return True
    for slider in fig.layout.sliders or ():
        for step in slider.steps or ():
            if step.method in {"restyle", "update"}:
                return True
    return False


def _fixed_axes(fig: Any, trace: Any, letters: tuple[str, ...] = ("x", "y")) -> bool:
    for letter in letters:
        ref = str(getattr(trace, f"{letter}axis", None) or letter)
        axis = getattr(fig.layout, f"{letter}axis{ref[1:]}", None)
        if axis is None or axis.range is None or axis.autorange not in {None, False}:
            return False
    return True


def _states(fig: Any, index: int) -> list[tuple[Any, Any]]:
    """Return the (x, y) shown by trace ``index`` after each frame."""
    base = fig.data[index]
    current = (base.x, base.y)
    states = []
    for frame in fig.frames:
        targets = list(frame.traces) if frame.traces is not None else list(range(len(frame.data or ())))
        if index in targets:
            update = frame.data[targets.index(index)]
            current = (
                update.x if update.x is not None else current[0],
                update.y if update.y is not None else current[1],
            )
        states.append(current)
    return states


def _insert(fig: Any, position: int, base: list[go.Scatter], per_frame: list[list[go.Scatter]]) -> None:
    """Insert ``base`` traces at ``position`` and append their updates to every frame."""
    count = len(base)
    original = list(fig.data)
    fig.add_traces(base)
    added = list(fig.data[len(original):])
    fig.data = tuple(original[:position] + added + original[position:])
    new_indices = list(range(position, position + count))
    frames = []
    for frame, updates in zip(fig.frames, per_frame or [[] for _ in fig.frames]):
        targets = list(frame.traces) if frame.traces is not None else list(range(len(frame.data or ())))
        shifted = [int(t) + count if int(t) >= position else int(t) for t in targets]
        frames.append(
            go.Frame(
                data=[*(frame.data or ()), *updates],
                traces=shifted + (new_indices if updates else []),
                name=frame.name,
                layout=frame.layout,
                group=frame.group,
                baseframe=frame.baseframe,
            )
        )
    if fig.frames:
        fig.frames = frames


def _quiet(trace: Any, effect: str, **kwargs: Any) -> go.Scatter:
    """An effect mark: no hover, no legend entry, and tagged so it is never restyled."""
    return go.Scatter(
        xaxis=trace.xaxis, yaxis=trace.yaxis, hoverinfo="skip", showlegend=False,
        uid=f"{roles.EFFECT_UID_PREFIX}{effect.upper()}", **kwargs,
    )


# ── effects ──────────────────────────────────────────────────────────


def _trail(fig: Any, index: int, color: str, effects: Any) -> None:
    model = fig.data[index]
    states = _states(fig, index)
    stride = max(1, round(len(fig.frames) / _TRAIL_SPAN))
    count = int(effects.trail)
    low, high = effects.trail_opacity
    opacities = [low + (high - low) * n / max(1, count - 1) for n in range(count)]
    lags = [stride * (count - n) for n in range(count)]  # oldest first

    def trace(state, opacity, n):
        return _quiet(model, f"trail_{n}", x=state[0], y=state[1], mode="lines", line={"color": color, "width": 2},
                      opacity=opacity)

    initial = _thin((model.x, model.y))
    thinned = [_thin(state) for state in states]
    base = [trace(initial, opacity, n) for n, opacity in enumerate(opacities)]
    per_frame = [
        [go.Scatter(x=thinned[max(0, k - lag)][0], y=thinned[max(0, k - lag)][1]) for lag in lags]
        for k in range(len(fig.frames))
    ]
    _insert(fig, index, base, per_frame)


def _glow(fig: Any, index: int, color: str, effects: Any) -> None:
    model = fig.data[index]
    x, y = _thin((model.x, model.y))
    base = [_quiet(model, "glow", x=x, y=y, mode="lines",
                   line={"color": color, "width": effects.glow_width}, opacity=effects.glow_opacity)]
    per_frame = [[go.Scatter(x=x, y=y)] for x, y in map(_thin, _states(fig, index))] if fig.frames else []
    _insert(fig, index, base, per_frame)


def _thin(state: tuple[Any, Any]) -> tuple[list[Any], list[Any]]:
    """Keep at most ``_DECOR_POINTS`` evenly spaced points (ends included), rounded."""
    xs = [] if state[0] is None else list(state[0])
    ys = [] if state[1] is None else list(state[1])
    count = min(len(xs), len(ys))
    if count > _DECOR_POINTS:
        keep = sorted({round(i * (count - 1) / (_DECOR_POINTS - 1)) for i in range(_DECOR_POINTS)})
        xs, ys = [xs[i] for i in keep], [ys[i] for i in keep]

    def short(value: Any) -> Any:
        return float(f"{value:.5g}") if isinstance(value, (int, float)) and _finite(value) else value

    return [short(v) for v in xs], [short(v) for v in ys]


def _find_data(fig: Any, model: Any) -> int | None:
    """The observed points drawn on the same axes as the model line."""
    for index, trace in enumerate(fig.data):
        if (
            roles.trace_role(trace) == roles.DATA
            and roles.is_flat(trace)
            and trace.xaxis == model.xaxis
            and trace.yaxis == model.yaxis
            and trace.x is not None
            and trace.y is not None
        ):
            return index
    return None


_MAX_RESIDUALS = 600  # beyond this the segments become a solid block


def _residual_path(data_x: Any, data_y: Any, state: tuple[Any, Any]) -> tuple[list[Any], list[Any]]:
    """Segments from each point to the model's value at the same x, separated by gaps."""
    import numpy as np

    line_x = np.asarray(state[0], dtype=float)
    line_y = np.asarray(state[1], dtype=float)
    order = np.argsort(line_x)
    fitted = np.interp(data_x, line_x[order], line_y[order])
    xs: list[Any] = []
    ys: list[Any] = []
    for x, y, f in zip(data_x, data_y, fitted):
        xs.extend((float(x), float(x), None))
        ys.extend((float(f"{y:.5g}"), float(f"{f:.5g}"), None))
    return xs, ys


def _residuals(fig: Any, model_index: int, data_index: int, color: str) -> None:
    import numpy as np

    data = fig.data[data_index]
    data_x = np.asarray(data.x, dtype=float)[:_MAX_RESIDUALS]
    data_y = np.asarray(data.y, dtype=float)[:_MAX_RESIDUALS]
    model = fig.data[model_index]
    states = _states(fig, model_index) if fig.frames else []

    def path(state):
        return _residual_path(data_x, data_y, state)

    x0, y0 = path((model.x, model.y))
    base = [_quiet(model, "residuals", x=x0, y=y0, mode="lines", line={"color": color, "width": 1.4}, opacity=0.55)]
    per_frame = [[go.Scatter(x=x, y=y)] for x, y in map(path, states)]
    # Under the data points, so every point stays visible on top of its segment.
    _insert(fig, data_index, base, per_frame)


def _finite(value: Any) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def _loss_progress(fig: Any, index: int, palette: Any) -> None:
    loss = fig.data[index]
    states = _states(fig, index)
    final_x, final_y = states[-1]
    future = [_quiet(loss, "loss_future", x=final_x, y=final_y, mode="lines",
                     line={"color": palette.loss, "width": 2, "dash": "dot"}, opacity=0.3)]
    _insert(fig, index, future, [[] for _ in fig.frames])
    loss_index = index + 1

    def comet(state):
        xs = [] if state[0] is None else list(state[0])
        ys = [] if state[1] is None else list(state[1])
        last = max((i for i, v in enumerate(ys) if _finite(v)), default=None)
        if last is None:
            return [None], [None], [None], [None]
        start = max(0, last - 6)
        return xs[start:last + 1], ys[start:last + 1], [xs[last]], [ys[last]]

    def traces(state, base_style: bool):
        tail_x, tail_y, head_x, head_y = comet(state)
        if not base_style:
            return [go.Scatter(x=tail_x, y=tail_y), go.Scatter(x=head_x, y=head_y), go.Scatter(x=head_x, y=head_y)]
        return [
            _quiet(loss, "comet_tail", x=tail_x, y=tail_y, mode="lines", line={"color": palette.accent, "width": 5}, opacity=0.55),
            _quiet(loss, "comet_halo", x=head_x, y=head_y, mode="markers", marker={"size": 22, "color": palette.accent},
                   opacity=0.25),
            _quiet(loss, "comet_head", x=head_x, y=head_y, mode="markers",
                   marker={"size": 10, "color": palette.accent, "line": {"color": palette.text, "width": 1.5}}),
        ]

    base_state = (fig.data[loss_index].x, fig.data[loss_index].y)
    _insert(fig, len(fig.data), traces(base_state, True), [traces(state, False) for state in states])


def _equation_delta(fig: Any, increase: str, decrease: str) -> bool:
    """Colour each number by the direction it moved since its last change."""
    changed = False
    equation_slots = [i for i, trace in enumerate(fig.data) if roles.trace_role(trace) == roles.EQUATION]
    previous_text: dict[Any, str] = {}
    shown_text: dict[Any, str] = {}
    for index in equation_slots:
        text = fig.data[index].text
        shown_text[("trace", index)] = text[0] if isinstance(text, (list, tuple)) and text else text
    for position, annotation in enumerate(fig.layout.annotations or ()):
        shown_text[("annotation", position)] = annotation.text

    def mark(key: Any, text: Any) -> Any:
        nonlocal changed
        if not isinstance(text, str) or "$" not in text:
            return text
        last = shown_text.get(key)
        if isinstance(last, str) and last != text:
            previous_text[key] = last
        shown_text[key] = text
        reference = previous_text.get(key)
        if reference is None:
            return text
        colored = _color_numbers(text, reference, increase, decrease)
        changed = changed or colored != text
        return colored

    for frame in fig.frames:
        targets = list(frame.traces) if frame.traces is not None else list(range(len(frame.data or ())))
        for target, update in zip(targets, frame.data or ()):
            if int(target) in equation_slots and update.text is not None:
                text = update.text
                if isinstance(text, (list, tuple)):
                    update.text = [mark(("trace", int(target)), t) for t in text[:1]] + list(text[1:])
                else:
                    update.text = mark(("trace", int(target)), text)
        if frame.layout is not None and frame.layout.annotations:
            for position, annotation in enumerate(frame.layout.annotations):
                annotation.text = mark(("annotation", position), annotation.text)
    return changed


def _color_numbers(text: str, reference: str, increase: str, decrease: str) -> str:
    old = _NUMBER.findall(reference)
    new = list(_NUMBER.finditer(text))
    if len(old) != len(new):
        return text
    pieces, last = [], 0
    for match, before in zip(new, old):
        pieces.append(text[last:match.start()])
        value = match.group(0)
        delta = float(value) - float(before)
        if abs(delta) > 0.5 * 10 ** -len(value.split(".")[1]):
            pieces.append(r"\color{%s}{%s}" % (increase if delta > 0 else decrease, value))
        else:
            pieces.append(value)
        last = match.end()
    pieces.append(text[last:])
    return "".join(pieces)
