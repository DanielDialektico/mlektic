"""Validated configuration contracts for tabular history capture."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

_COMMON_MODES = {"auto", "iterative", "final_interp"}
_DISPLAY_SPACES = {"original", "scaled"}
_SMOOTHING_METHODS = {None, "ema"}
_LINEAR_BASELINES = {"mean", "zeros"}
_LOGISTIC_BASELINES = {"prior", "uniform"}
_MULTICLASS_LINKS = {"auto", "softmax", "ovr"}
_LINEAR_METRICS = {"loss", "mse", "r2", "r_2", "mae"}
_LOGISTIC_METRICS = {"loss", "log_loss", "logloss", "accuracy", "f1", "f1_score"}


def _validate_common(config: Any, *, baselines: set[str], metrics: set[str]) -> None:
    """Validate fields shared by the linear and logistic public contracts."""
    _positive_integer("steps", config.steps)
    _positive_integer("grid_1d_points", config.grid_1d_points, minimum=2)
    _positive_integer("grid_2d_points", config.grid_2d_points, minimum=2)
    _optional_positive_integer("max_frames", config.max_frames)
    _optional_positive_integer("frame_step", config.frame_step)

    _one_of("mode", config.mode, _COMMON_MODES)
    _one_of("display_space", config.display_space, _DISPLAY_SPACES)
    _one_of("baseline", config.baseline, baselines)

    smooth = config.smooth
    if smooth == "none":
        warnings.warn(
            "smooth='none' is deprecated; use smooth=None.",
            DeprecationWarning,
            stacklevel=3,
        )
        object.__setattr__(config, "smooth", None)
        smooth = None
    _one_of("smooth", smooth, _SMOOTHING_METHODS)

    if not isinstance(config.smooth_beta, Real) or isinstance(config.smooth_beta, bool):
        raise TypeError("smooth_beta must be a real number in the interval [0, 1).")
    if not 0 <= float(config.smooth_beta) < 1:
        raise ValueError("smooth_beta must be in the interval [0, 1).")

    _validate_metrics(config.metrics, allowed=metrics)


def _positive_integer(name: str, value: Any, *, minimum: int = 1) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer greater than or equal to {minimum}.")
    if value < minimum:
        raise ValueError(f"{name} must be greater than or equal to {minimum}.")


def _optional_positive_integer(name: str, value: Any) -> None:
    if value is not None:
        _positive_integer(name, value)


def _one_of(name: str, value: Any, allowed: set[Any]) -> None:
    choices = ", ".join(repr(item) for item in sorted(allowed, key=lambda item: str(item)))
    try:
        valid = value in allowed
    except TypeError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be one of: {choices}. Received {value!r}.")


def _validate_metrics(metric_config: Any, *, allowed: set[str]) -> None:
    if metric_config is None:
        return
    if isinstance(metric_config, Mapping):
        for label, function in metric_config.items():
            if not str(label).strip():
                raise ValueError("Custom metric labels must not be empty.")
            if not callable(function):
                raise TypeError(f"Custom metric {label!r} must be callable.")
        return
    if isinstance(metric_config, str):
        requested = [metric_config]
    elif isinstance(metric_config, Sequence):
        requested = list(metric_config)
    else:
        raise TypeError("metrics must be None, a metric-name sequence, or a mapping of callables.")

    unknown = [str(name) for name in requested if str(name).lower().replace("-", "_") not in allowed]
    if unknown:
        raise ValueError(
            f"Unknown metric name(s): {', '.join(unknown)}. "
            f"Available built-in metrics are: {', '.join(sorted(allowed))}."
        )


@dataclass(frozen=True)
class LinearHistoryConfig:
    """Validated configuration for linear-regression history capture."""
    steps: int = 60
    mode: str = "auto"
    smooth: str | None = None
    smooth_beta: float = 0.85
    grid_1d_points: int = 250
    grid_2d_points: int = 40
    baseline: str = "mean"
    display_space: str = "original"
    metrics: Any = None
    max_frames: int | None = 60
    frame_step: int | None = 10

    def __post_init__(self) -> None:
        """Reject ambiguous or unsupported configuration values early."""
        _validate_common(self, baselines=_LINEAR_BASELINES, metrics=_LINEAR_METRICS)


@dataclass(frozen=True)
class LogisticHistoryConfig:
    """Validated configuration for logistic-regression history capture."""
    steps: int = 60
    mode: str = "auto"
    smooth: str | None = None
    smooth_beta: float = 0.85
    grid_1d_points: int = 300
    grid_2d_points: int = 40
    baseline: str = "prior"
    display_space: str = "original"
    metrics: Any = None
    max_frames: int | None = 60
    frame_step: int | None = 10
    multiclass_link: str = "auto"

    def __post_init__(self) -> None:
        """Reject ambiguous or unsupported configuration values early."""
        _validate_common(self, baselines=_LOGISTIC_BASELINES, metrics=_LOGISTIC_METRICS)
        _one_of("multiclass_link", self.multiclass_link, _MULTICLASS_LINKS)
