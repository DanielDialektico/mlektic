"""Public API functions for mlektic."""

from .neural import (
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
from .optimize import export_figure, show_optimized

__all__ = [
    "TorchTrainingRecorder",
    "build_nn_math_report",
    "display_nn_math_report",
    "explain_nn_prediction",
    "export_nn_math_report",
    "visualize_nn",
    "visualize_nn_architecture",
    "visualize_nn_graph",
    "visualize_nn_training",
    "visualize_nn_weights",
    "export_figure",
    "show_optimized",
]
