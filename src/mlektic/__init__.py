"""Mlektic public package exports."""

from .api.neural import (
    TorchTrainingRecorder,
    build_nn_math_report,
    display_nn_math_report,
    explain_nn_prediction,
    export_nn_math_report,
    visualize_nn,
    visualize_nn_architecture,
    visualize_nn_graph,
    visualize_nn_training,
    visualize_nn_weights,
)
from .api.optimize import (
    export_figure,
    show_optimized,
)
from .core import (
    build_lr_figure,
    build_multivar_lr_figure,
    build_plane_lr_figure,
    build_simple_lr_figure,
    explain_lr_prediction,
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
from .visualization.design import VisualSpec, VisualTokens, available_themes, get_theme_tokens

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
    "export_figure",
    "TorchTrainingRecorder",
    "build_nn_math_report",
    "display_nn_math_report",
    "visualize_nn",
    "visualize_nn_architecture",
    "visualize_nn_graph",
    "visualize_nn_training",
    "visualize_nn_weights",
    "explain_nn_prediction",
    "export_nn_math_report",
    "VisualSpec",
    "VisualTokens",
    "available_themes",
    "get_theme_tokens",
]
