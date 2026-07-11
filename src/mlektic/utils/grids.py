"""Utilities for generating meshgrids."""

import numpy as np


def build_1d_grid(X, grid_size: int = 300):
    """Build a 1D grid between min and max of X[:, 0]."""
    x1 = X[:, 0]
    x_min, x_max = float(x1.min()), float(x1.max())
    return np.linspace(x_min, x_max, int(grid_size))


def build_2d_grid(X, grid_size: int = 30):
    """Build a 2D meshgrid based on X[:, 0] and X[:, 1]."""
    x1 = X[:, 0]
    x2 = X[:, 1]
    x1_grid = np.linspace(float(x1.min()), float(x1.max()), int(grid_size))
    x2_grid = np.linspace(float(x2.min()), float(x2.max()), int(grid_size))
    X1g, X2g = np.meshgrid(x1_grid, x2_grid)
    return x1_grid, x2_grid, X1g, X2g
