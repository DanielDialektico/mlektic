"""Config classes for the history capture."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LinearHistoryConfig:
    """Configuration for linear history capture."""
    steps: int = 60
    mode: str = "auto"
    smooth: str | None = None
    smooth_beta: float = 0.85
    grid_1d_points: int = 250
    grid_2d_points: int = 40
    baseline: str = "mean"
    display_space: str = "original"
    metrics: dict | None = None
    max_frames: int | None = 60
    frame_step: int | None = 10


@dataclass(frozen=True)
class LogisticHistoryConfig:
    """Configuration for logistic history capture."""
    steps: int = 60
    mode: str = "auto"
    smooth: str | None = None
    smooth_beta: float = 0.85
    grid_1d_points: int = 300
    grid_2d_points: int = 40
    baseline: str = "prior"
    display_space: str = "original"
    metrics: dict | None = None
    max_frames: int | None = 60
    frame_step: int | None = 10
