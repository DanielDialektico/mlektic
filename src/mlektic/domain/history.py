"""Typed history payload contracts used across the library."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


class LinearHistoryPayload(TypedDict, total=False):
    """Contract for linear-regression history capture payloads."""

    history_kind: str
    loss_hist: NDArray[np.float64]
    grid: dict[str, NDArray[np.float64]]
    y_line_hist: NDArray[np.float64] | None
    z_plane_hist: NDArray[np.float64] | None
    w_hist: NDArray[np.float64] | None
    b_hist: NDArray[np.float64] | None
    w_hist_learned: NDArray[np.float64] | None
    b_hist_learned: NDArray[np.float64] | None
    display_space: str


class LogisticHistoryPayload(TypedDict, total=False):
    """Contract for logistic-regression history capture payloads."""

    history_kind: str
    classes: NDArray[np.float64]
    is_multiclass: bool
    loss_hist: NDArray[np.float64]
    grid: dict[str, NDArray[np.float64]]
    p_line_hist: NDArray[np.float64] | None
    p_plane_hist: NDArray[np.float64] | None
    p_curves_hist: NDArray[np.float64] | None
    w_hist: NDArray[np.float64] | None
    b_hist: NDArray[np.float64] | None
    w_hist_learned: NDArray[np.float64] | None
    b_hist_learned: NDArray[np.float64] | None
    display_space: str
