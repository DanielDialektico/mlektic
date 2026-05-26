"""Compatibility facade for linear-regression API.

This module preserves the historical import paths while delegating
implementation to specialized internal packages.
"""

from __future__ import annotations

import plotly.io as pio

from ._internal.common import (
    _as_1d,
    _as_2d,
    _ema_smooth,
    _find_standard_scaler,
    _first_not_none,
    _get_final_estimator,
    _is_iterative,
    _last_step_prefix,
    _make_iterative_replay_estimator,
    _safe_get_scale,
    _transform_up_to_last,
    _try_set_params,
)
from .api.linear import visualize_lr
from .services.linear_history import fit_history
from .visualization.linear.multivar import build_multivar_lr_figure
from .visualization.linear.plane import build_plane_lr_figure
from .visualization.linear.router import build_lr_figure
from .visualization.linear.simple import build_simple_lr_figure

pio.renderers.default = "colab"

__all__ = [
    "fit_history",
    "build_lr_figure",
    "build_simple_lr_figure",
    "build_plane_lr_figure",
    "build_multivar_lr_figure",
    "visualize_lr",
]
