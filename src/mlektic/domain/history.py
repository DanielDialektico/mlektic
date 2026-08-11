"""Typed history payload contracts used across the library."""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray


class HistoryMetadata(TypedDict, total=False):
    """Auditable provenance, temporal, and display semantics for a history."""

    schema_version: int
    source: str
    source_detail: dict[str, Any]
    requested_mode: str
    resolved_mode: str
    requested_steps: int
    training_total_steps: int | None
    captured_steps: int
    displayed_steps: int
    step_indices: NDArray[np.int64]
    displayed_step_indices: NDArray[np.int64]
    state_origins: NDArray[np.str_]
    displayed_state_origins: NDArray[np.str_]
    alpha_values: NDArray[np.float64]
    final_state_matches_estimator: bool | None
    display_space: str
    coefficient_space: str
    smoothing: dict[str, Any]
    decimation: dict[str, Any]
    warnings: list[dict[str, str]]


class LinearHistoryPayload(TypedDict, total=False):
    """Contract for linear-regression history capture payloads."""

    history_kind: str
    history_source: str
    task: str
    loss_hist: NDArray[np.float64]
    loss_raw: NDArray[np.float64]
    loss_display: NDArray[np.float64]
    step_indices: NDArray[np.int64]
    state_origins: NDArray[np.str_]
    alpha_values: NDArray[np.float64]
    metadata: HistoryMetadata
    grid: dict[str, NDArray[np.float64]]
    y_line_hist: NDArray[np.float64] | None
    z_plane_hist: NDArray[np.float64] | None
    w_hist: NDArray[np.float64] | None
    b_hist: NDArray[np.float64] | None
    w_hist_learned: NDArray[np.float64] | None
    b_hist_learned: NDArray[np.float64] | None
    display_space: str
    coefficient_space: str


class LogisticHistoryPayload(TypedDict, total=False):
    """Contract for logistic-regression history capture payloads."""

    history_kind: str
    history_source: str
    task: str
    classes: NDArray[np.float64]
    is_multiclass: bool
    loss_hist: NDArray[np.float64]
    loss_raw: NDArray[np.float64]
    loss_display: NDArray[np.float64]
    step_indices: NDArray[np.int64]
    state_origins: NDArray[np.str_]
    alpha_values: NDArray[np.float64]
    metadata: HistoryMetadata
    grid: dict[str, NDArray[np.float64]]
    p_line_hist: NDArray[np.float64] | None
    p_plane_hist: NDArray[np.float64] | None
    p_curves_hist: NDArray[np.float64] | None
    w_hist: NDArray[np.float64] | None
    b_hist: NDArray[np.float64] | None
    w_hist_learned: NDArray[np.float64] | None
    b_hist_learned: NDArray[np.float64] | None
    display_space: str
    coefficient_space: str
    probability_link: str
    interpolation_target: str
