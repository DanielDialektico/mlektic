"""Mlektic public package exports."""

from .core import (
    build_lr_figure,
    build_multivar_lr_figure,
    build_plane_lr_figure,
    build_simple_lr_figure,
    fit_history,
    visualize_lr,
)
from .logistic import (
    build_binary_multivar_logistic_figure,
    build_binary_plane_logistic_figure,
    build_binary_simple_logistic_figure,
    build_logistic_figure,
    build_multiclass_1d_logistic_figure,
    build_multiclass_multivar_logistic_figure,
    explain_logistic_prediction,
    fit_history_logistic,
    visualize_logistic,
)
from .api.optimize import (
    show_optimized,
)

__all__ = [
    "fit_history",
    "build_lr_figure",
    "build_simple_lr_figure",
    "build_plane_lr_figure",
    "build_multivar_lr_figure",
    "visualize_lr",
    "explain_lr_prediction",
    "build_logistic_figure",
    "fit_history_logistic",
    "visualize_logistic",
    "build_binary_simple_logistic_figure",
    "build_binary_plane_logistic_figure",
    "build_binary_multivar_logistic_figure",
    "build_multiclass_1d_logistic_figure",
    "build_multiclass_multivar_logistic_figure",
    "explain_logistic_prediction",
    "show_optimized",
]
