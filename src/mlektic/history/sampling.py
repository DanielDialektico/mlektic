"""Utilities for reducing animation histories before rendering."""

from __future__ import annotations

import numpy as np


def decimate_history(data: dict, max_frames: int | None = 60, frame_step: int | None = 10) -> dict:
    """Return ``data`` sampled down to the configured animation frame budget."""
    if "loss_hist" not in data:
        return data

    steps = len(data["loss_hist"])
    if steps <= 1:
        return data

    indices = _build_frame_indices(steps, max_frames=max_frames, frame_step=frame_step)
    if indices is None:
        return data

    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.shape[0] == steps:
            data[key] = value[indices]
        elif key == "metrics_hist" and isinstance(value, dict):
            _decimate_metric_histories(value, indices, steps)

    return data


def _build_frame_indices(
    steps: int,
    *,
    max_frames: int | None,
    frame_step: int | None,
) -> np.ndarray | None:
    """Build the frame index vector used to thin long histories."""
    if max_frames is not None and steps > max_frames:
        return np.linspace(0, steps - 1, max_frames).astype(int)

    if max_frames is None and frame_step is not None and frame_step > 0:
        indices = np.arange(0, steps, frame_step)
        if indices[-1] != steps - 1:
            indices = np.append(indices, steps - 1)
        return indices

    return None


def _decimate_metric_histories(metrics_hist: dict, indices: np.ndarray, steps: int) -> None:
    """Apply frame indices in-place to all metric arrays with a matching time axis."""
    for metric_name, metric_values in metrics_hist.items():
        if isinstance(metric_values, np.ndarray) and metric_values.shape[0] == steps:
            metrics_hist[metric_name] = metric_values[indices]
