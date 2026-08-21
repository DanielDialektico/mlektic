"""Exact two-direction loss-slice visualization for PyTorch networks."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import plotly.graph_objects as go

from ...neural.introspection import _require_torch, run_torch_forward
from ._style import NEURAL_COLORS, animation_button_style, neural_layout
from .math_format import parameter_snapshot

CHECKPOINT_TEXT_SIZE = 15


def _checkpoint_annotation(step: int, loss: float, *, final: bool = False) -> Dict[str, Any]:
    """Keep the active checkpoint label in screen space at a constant size."""
    label = "Final checkpoint" if final else "Checkpoint"
    return {
        "x": 0.70,
        "y": 0.80,
        "xref": "paper",
        "yref": "paper",
        "text": f"<b>{label} {step}</b><br>L={loss:.6g}",
        "showarrow": False,
        "xanchor": "left",
        "align": "left",
        "font": {"size": CHECKPOINT_TEXT_SIZE, "color": NEURAL_COLORS["text"]},
        "bgcolor": NEURAL_COLORS["panel"],
        "bordercolor": NEURAL_COLORS["activation"],
        "borderwidth": 1,
        "borderpad": 8,
    }


def _landscape_annotations(
    direction_source: str,
    *,
    step: int,
    loss: float,
    final_step: int,
) -> list[Dict[str, Any]]:
    """Keep the initial surface uncluttered and label only the final state."""
    annotations = [_slice_disclosure_annotation(direction_source)]
    if step == final_step:
        annotations.append(_checkpoint_annotation(step, loss, final=True))
    return annotations


def _slice_disclosure_annotation(direction_source: str) -> Dict[str, Any]:
    """Describe the exact slice without changing across animation frames."""
    return {
        "x": 0.5,
        "y": 1.02,
        "xref": "paper",
        "yref": "paper",
        "text": (
            f"Exact batch evaluation on an affine 2-D slice · directions: {direction_source} · "
            "the original high-dimensional path is projected onto this plane"
        ),
        "showarrow": False,
        "font": {"size": 12, "color": NEURAL_COLORS["muted"]},
    }


def _common_parameter_names(model: Any, history: Dict[str, Any]) -> list[str]:
    captured = history.get("parameters", {})
    names = [name for name, _ in model.named_parameters() if captured.get(name)]
    if not names:
        raise ValueError(
            "The loss slice requires captured parameter tensors. Increase "
            "TorchTrainingRecorder(max_tensor_elements=...) when necessary."
        )
    return names


def _flatten(snapshot: Dict[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    return np.concatenate([np.asarray(snapshot[name], dtype=float).ravel() for name in names])


def _unflatten(
    vector: np.ndarray,
    reference: Dict[str, np.ndarray],
    names: Sequence[str],
) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    offset = 0
    for name in names:
        shape = np.asarray(reference[name]).shape
        size = int(np.prod(shape))
        result[name] = vector[offset : offset + size].reshape(shape)
        offset += size
    return result


def _orthonormal_directions(centered: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    usable = int(np.sum(singular_values > max(float(singular_values[0]) if singular_values.size else 0.0, 1.0) * 1e-10))
    if usable >= 2:
        return right[0], right[1], "history PCA"
    first = right[0] if usable >= 1 else np.zeros(centered.shape[1], dtype=float)
    rng = np.random.default_rng(seed)
    if not np.linalg.norm(first):
        first = rng.normal(size=centered.shape[1])
    first = first / np.linalg.norm(first)
    second = rng.normal(size=centered.shape[1])
    second -= np.dot(second, first) * first
    if np.linalg.norm(second) < 1e-12:
        second = np.roll(first, 1)
        second -= np.dot(second, first) * first
    second = second / max(np.linalg.norm(second), 1e-12)
    return first, second, "history direction + deterministic orthogonal complement"


def _loss_value(
    model: Any,
    inputs: Any,
    targets: Any,
    loss_fn: Any,
    parameters: Dict[str, np.ndarray],
    buffers: Dict[str, np.ndarray] | None,
) -> float:
    torch = _require_torch()
    prediction, _ = run_torch_forward(model, inputs, parameters, buffers)
    target = targets
    if not hasattr(target, "detach"):
        target = torch.as_tensor(target, device=prediction.device)
    else:
        target = target.to(prediction.device)
    value = loss_fn(prediction, target)
    if getattr(value, "numel", lambda: 0)() != 1:
        raise ValueError("loss_fn must return one scalar for the supplied batch.")
    return float(value.detach().cpu().item())


def build_nn_loss_landscape_figure(
    model: Any,
    inputs: Any,
    targets: Any,
    loss_fn: Any,
    history: Dict[str, Any],
    *,
    grid_size: int = 23,
    span: float | None = None,
    max_frames: int | None = 20,
    frame_duration: int = 180,
    seed: int = 17,
    title: str | None = None,
) -> go.Figure:
    """Evaluate an exact affine 2-D loss slice and animate a projected path."""
    if grid_size < 9 or grid_size > 61:
        raise ValueError("grid_size must be between 9 and 61.")
    steps = np.asarray(history.get("steps", []), dtype=int)
    if not steps.size:
        raise ValueError("History has no recorded steps.")
    names = _common_parameter_names(model, history)
    snapshots = [parameter_snapshot(history, index) for index in range(steps.size)]
    vectors = np.vstack([_flatten(snapshot, names) for snapshot in snapshots])
    center = vectors[-1].copy()
    centered = vectors - center
    direction_u, direction_v, direction_source = _orthonormal_directions(centered, seed)
    coordinates_u = centered @ direction_u
    coordinates_v = centered @ direction_v
    observed_radius = max(
        float(np.max(np.abs(coordinates_u))),
        float(np.max(np.abs(coordinates_v))),
        1e-6,
    )
    slice_span = float(span) if span is not None else observed_radius * 1.20
    if slice_span <= 0:
        raise ValueError("span must be positive or None.")
    axis = np.linspace(-slice_span, slice_span, grid_size)
    surface = np.empty((grid_size, grid_size), dtype=float)
    center_snapshot = snapshots[-1]
    buffers = {name: np.asarray(values[-1]) for name, values in history.get("buffers", {}).items() if values}
    for row, value_v in enumerate(axis):
        for column, value_u in enumerate(axis):
            vector = center + value_u * direction_u + value_v * direction_v
            parameters = _unflatten(vector, center_snapshot, names)
            surface[row, column] = _loss_value(model, inputs, targets, loss_fn, parameters, buffers or None)
    projected_loss = np.asarray(
        [
            _loss_value(
                model,
                inputs,
                targets,
                loss_fn,
                _unflatten(center + u * direction_u + v * direction_v, center_snapshot, names),
                buffers or None,
            )
            for u, v in zip(coordinates_u, coordinates_v)
        ],
        dtype=float,
    )
    # Scatter3d markers that sit exactly on a semi-transparent surface can be
    # depth-sorted behind it by the browser. Lift only the checkpoint marker by
    # a small, disclosed visual epsilon; path coordinates and every reported
    # loss remain the exact evaluated values.
    marker_z_offset = max(float(np.ptp(surface)) * 0.018, 1e-8)
    marker_z = projected_loss + marker_z_offset
    if max_frames is None or steps.size <= max_frames:
        selected = np.arange(steps.size, dtype=int)
    else:
        if max_frames < 1:
            raise ValueError("max_frames must be at least 1 or None.")
        selected = np.unique(np.linspace(0, steps.size - 1, max_frames, dtype=int))
    loss_name = loss_fn.__class__.__name__
    figure = go.Figure(
        data=[
            go.Surface(
                x=axis,
                y=axis,
                z=surface,
                colorscale="Viridis",
                opacity=0.86,
                colorbar={
                    "title": loss_name,
                    "thickness": 14,
                    "tickfont": {"size": 11, "color": NEURAL_COLORS["text"]},
                },
                hovertemplate="u=%{x:.4g}<br>v=%{y:.4g}<br>evaluated loss=%{z:.6g}<extra></extra>",
                name="evaluated loss slice",
            ),
            go.Scatter3d(
                x=coordinates_u[: selected[0] + 1],
                y=coordinates_v[: selected[0] + 1],
                z=projected_loss[: selected[0] + 1],
                mode="lines+markers",
                line={"color": NEURAL_COLORS["output"], "width": 7},
                marker={"size": 3, "color": NEURAL_COLORS["text"]},
                hovertemplate="projected path<br>u=%{x:.4g}<br>v=%{y:.4g}<br>loss=%{z:.6g}<extra></extra>",
                name="projected training path",
            ),
            go.Scatter3d(
                x=[coordinates_u[selected[0]]],
                y=[coordinates_v[selected[0]]],
                z=[marker_z[selected[0]]],
                mode="markers",
                marker={
                    "size": 10,
                    "color": NEURAL_COLORS["activation"],
                    "line": {"color": NEURAL_COLORS["text"], "width": 2},
                },
                customdata=[projected_loss[selected[0]]],
                hovertemplate="checkpoint loss=%{customdata:.6g}<extra></extra>",
                name="current projected checkpoint",
            ),
        ]
    )
    figure.frames = [
        go.Frame(
            name=f"step_{index}",
            data=[
                go.Scatter3d(
                    x=coordinates_u[: index + 1],
                    y=coordinates_v[: index + 1],
                    z=projected_loss[: index + 1],
                    mode="lines+markers",
                    line={"color": NEURAL_COLORS["output"], "width": 7},
                    marker={"size": 3, "color": NEURAL_COLORS["text"]},
                ),
                go.Scatter3d(
                    x=[coordinates_u[index]],
                    y=[coordinates_v[index]],
                    z=[marker_z[index]],
                    mode="markers",
                    marker={
                        "size": 10,
                        "color": NEURAL_COLORS["activation"],
                        "line": {"color": NEURAL_COLORS["text"], "width": 2},
                    },
                    customdata=[projected_loss[index]],
                    hovertemplate="checkpoint loss=%{customdata:.6g}<extra></extra>",
                )
            ],
            traces=[1, 2],
            layout=go.Layout(
                annotations=_landscape_annotations(
                    direction_source,
                    step=int(steps[index]),
                    loss=float(projected_loss[index]),
                    final_step=int(steps[-1]),
                )
            ),
        )
        for index in selected
    ]
    if title is None:
        title = "Neural loss surface: exact two-direction slice"
    layout = neural_layout(title, height=760)
    layout.update(
        {
            "margin": {"t": 110, "r": 110, "b": 80, "l": 40},
            "scene": {
                "xaxis_title": "coordinate u · direction 1",
                "yaxis_title": "coordinate v · direction 2",
                "zaxis_title": loss_name,
                "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 0.95}},
            },
            "annotations": _landscape_annotations(
                direction_source,
                step=int(steps[selected[0]]),
                loss=float(projected_loss[selected[0]]),
                final_step=int(steps[-1]),
            ),
            "updatemenus": [
                {
                    "type": "buttons",
                    "direction": "left",
                    "x": 0.0,
                    "y": 1.03,
                    **animation_button_style(),
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {"frame": {"duration": frame_duration, "redraw": True}, "fromcurrent": True},
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}],
                        },
                    ],
                }
            ],
            "sliders": [
                {
                    "active": 0,
                    "currentvalue": {"prefix": "Recorded checkpoint: "},
                    "pad": {"t": 35},
                    "steps": [
                        {
                            "label": str(int(steps[index])),
                            "method": "animate",
                            "args": [
                                [f"step_{index}"],
                                {"mode": "immediate", "frame": {"duration": 0, "redraw": True}},
                            ],
                        }
                        for index in selected
                    ],
                }
            ],
            "meta": {
                "mlektic_neural_loss_slice": {
                    "schema_version": 1,
                    "semantics": "exact affine two-direction batch-loss slice",
                    "direction_source": direction_source,
                    "center": "final recorded checkpoint",
                    "model_mode": "evaluation",
                    "loss": loss_name,
                    "grid_size": grid_size,
                    "span": slice_span,
                    "captured_parameter_tensors": names,
                    "trajectory_is_projected": True,
                    "trajectory_reveal": "progressive through the active checkpoint",
                    "checkpoint_marker_z_offset": marker_z_offset,
                    "checkpoint_marker_z_offset_is_visual_only": True,
                    "final_label_visible_before_final_frame": False,
                    "surface_is_global_loss_landscape": False,
                    "last_recorded_loss": float(projected_loss[-1]),
                    "convergence_claimed": False,
                }
            },
            "showlegend": False,
        }
    )
    figure.update_layout(**layout)
    return figure


__all__ = ["build_nn_loss_landscape_figure"]
