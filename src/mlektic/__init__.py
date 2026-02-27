"""
Mlektic: A visual library for understanding the mathematical evolution of machine learning models.
"""

from .core import (
    build_lr_figure,
    build_multivar_lr_figure,
    build_plane_lr_figure,
    build_simple_lr_figure,
    fit_history,
    visualize_lr,
)

__all__ = [
    "fit_history",
    "build_lr_figure",
    "build_simple_lr_figure",
    "build_plane_lr_figure",
    "build_multivar_lr_figure",
    "visualize_lr",
]
