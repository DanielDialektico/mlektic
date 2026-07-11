"""Compatibility facade for logistic-regression API.

This module preserves the historical import paths while delegating
implementation to specialized internal packages.
"""

from __future__ import annotations

import plotly.io as pio

from .api.logistic import explain_logistic_prediction, visualize_logistic
from .services.logistic_history import fit_history_logistic
from .visualization.logistic.binary_1d import build_binary_simple_logistic_figure
from .visualization.logistic.binary_2d import build_binary_plane_logistic_figure
from .visualization.logistic.binary_nd import build_binary_multivar_logistic_figure
from .visualization.logistic.multiclass_1d import build_multiclass_1d_logistic_figure
from .visualization.logistic.multiclass_nd import build_multiclass_multivar_logistic_figure
from .visualization.logistic.router import build_logistic_figure

pio.renderers.default = "colab"

__all__ = [
    "build_logistic_figure",
    "fit_history_logistic",
    "visualize_logistic",
    "build_binary_simple_logistic_figure",
    "build_binary_plane_logistic_figure",
    "build_binary_multivar_logistic_figure",
    "build_multiclass_1d_logistic_figure",
    "build_multiclass_multivar_logistic_figure",
    "explain_logistic_prediction",
]
