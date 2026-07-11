"""Compatibility facade for linear-regression API.

This module preserves the historical import paths while delegating
implementation to specialized internal packages.
"""

from __future__ import annotations

import plotly.io as pio

from .api.linear import explain_lr_prediction, visualize_lr
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
    "explain_lr_prediction",
]
