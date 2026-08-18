"""LaTeX formatting helpers shared by neural-network figures."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def display_indices(size: int, limit: int) -> List[int]:
    """Keep both ends of a long dimension so truncation remains representative."""
    if size <= limit:
        return list(range(size))
    left = max(1, limit // 2)
    right = max(1, limit - left)
    return [*range(left), *range(size - right, size)]


def vector_latex(values: Any, dec: int = 3, limit: int = 6) -> str:
    """Format a vector with an explicit ellipsis when values are omitted."""
    flat = np.asarray(values, dtype=float).ravel()
    if flat.size <= limit:
        cells = [f"{value:.{dec}f}" for value in flat]
    else:
        indices = display_indices(flat.size, limit)
        split = len(indices) // 2
        cells = [f"{flat[index]:.{dec}f}" for index in indices[:split]]
        cells.append(r"\cdots")
        cells.extend(f"{flat[index]:.{dec}f}" for index in indices[split:])
    return r"\begin{bmatrix}" + r" & ".join(cells) + r"\end{bmatrix}"


def matrix_latex(values: Any, dec: int = 3, max_rows: int = 4, max_cols: int = 5) -> str:
    """Format a matrix with row and column ellipses while preserving its ends."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 0:
        return f"{float(matrix):.{dec}f}"
    if matrix.ndim == 1:
        return vector_latex(matrix, dec=dec, limit=max_cols)
    if matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    row_indices = display_indices(matrix.shape[0], max_rows)
    col_indices = display_indices(matrix.shape[1], max_cols)
    row_split = len(row_indices) // 2 if matrix.shape[0] > max_rows else None
    col_split = len(col_indices) // 2 if matrix.shape[1] > max_cols else None
    rows: List[str] = []
    for position, row_index in enumerate(row_indices):
        if row_split is not None and position == row_split:
            width = len(col_indices) + (1 if col_split is not None else 0)
            rows.append(r" & ".join([r"\vdots"] * width))
        cells: List[str] = []
        for column_position, column_index in enumerate(col_indices):
            if col_split is not None and column_position == col_split:
                cells.append(r"\cdots")
            cells.append(f"{matrix[row_index, column_index]:.{dec}f}")
        rows.append(r" & ".join(cells))
    return r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"


def parameter_snapshot(history: Dict[str, Any], frame_index: int) -> Dict[str, np.ndarray]:
    """Collect all parameter tensors available for one recorded frame."""
    snapshot: Dict[str, np.ndarray] = {}
    for name, values in history.get("parameters", {}).items():
        if frame_index < len(values):
            snapshot[name] = np.asarray(values[frame_index], dtype=float)
    return snapshot


def gradient_snapshot(history: Dict[str, Any], frame_index: int) -> Dict[str, np.ndarray]:
    """Collect all gradient tensors available for one recorded frame."""
    snapshot: Dict[str, np.ndarray] = {}
    for name, values in history.get("gradients", {}).items():
        if frame_index < len(values):
            snapshot[name] = np.asarray(values[frame_index], dtype=float)
    return snapshot


def buffer_snapshot(history: Dict[str, Any], frame_index: int) -> Dict[str, np.ndarray]:
    """Collect all persistent buffer tensors available for one recorded frame."""
    snapshot: Dict[str, np.ndarray] = {}
    for name, values in history.get("buffers", {}).items():
        if frame_index < len(values):
            snapshot[name] = np.asarray(values[frame_index])
    return snapshot


def compact_parameter_line(
    stages: Sequence[Dict[str, Any]],
    snapshot: Dict[str, np.ndarray],
    *,
    dec: int = 3,
    values_per_layer: int = 3,
    max_layers: int = 4,
) -> str:
    """Build one dynamic line with a few representative weights per dense layer."""
    parts: List[str] = []
    selected_stages = display_indices(len(stages), max_layers)
    split = len(selected_stages) // 2 if len(stages) > max_layers else None
    for position, stage_index in enumerate(selected_stages):
        if split is not None and position == split:
            parts.append(r"\cdots")
        stage = stages[stage_index]
        weights = snapshot.get(stage["weight_name"])
        if weights is None:
            continue
        flat = weights.ravel()
        shown = ",".join(f"{value:.{dec}f}" for value in flat[:values_per_layer])
        suffix = r",\ldots" if flat.size > values_per_layer else ""
        parts.append(rf"\Theta^{{({stage['index']})}}_t=[{shown}{suffix}]")
    return r"\quad{}".join(parts)


__all__ = [
    "compact_parameter_line",
    "buffer_snapshot",
    "display_indices",
    "gradient_snapshot",
    "matrix_latex",
    "parameter_snapshot",
    "vector_latex",
]
