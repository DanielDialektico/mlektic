"""Routing logic for linear-regression figure selection."""

from __future__ import annotations

import numpy as np

from ..._internal.common import _first_not_none
from .multivar import build_multivar_lr_figure
from .plane import build_plane_lr_figure
from .simple import build_simple_lr_figure

def build_lr_figure(
    X,
    y,
    w_hist=None,
    b_hist=None,
    *,
    history=None,
    y_line_hist=None,
    x1_grid=None,  # d==1
    z_plane_hist=None,
    X1g=None,
    X2g=None,  # d==2
    loss_hist=None,
    metrics_hist=None,
    show_loss=False,
    history_kind="iterative",
    title=None,
    strict_loss=False,
    dec=4,
    frame_duration=80,
    theme=None,
):
    """
    Route to the appropriate visualization figure based on feature dimensions.

    Depending on the number of features `d` in the dataset `X`, this function delegates
    the plot creation to the respective builder for 1D, 2D, or multivariable data.

    Args:
        X (np.ndarray): The feature matrix of shape (n_samples, d).
        y (np.ndarray): The target vector of shape (n_samples,).
        w_hist (np.ndarray, optional): History of weights (theta). Defaults to None.
        b_hist (np.ndarray, optional): History of biases (intercepts). Defaults to None.
        history (dict, optional): Complete history dictionary returned by `fit_history()`. Defaults to None.
        y_line_hist (np.ndarray, optional): History of prediction lines (for 1D). Defaults to None.
        x1_grid (np.ndarray, optional): X-axis grid for 1D predictions. Defaults to None.
        z_plane_hist (np.ndarray, optional): History of prediction planes (for 2D). Defaults to None.
        X1g (np.ndarray, optional): Grid for first feature in 2D. Defaults to None.
        X2g (np.ndarray, optional): Grid for second feature in 2D. Defaults to None.
        loss_hist (np.ndarray, optional): History of loss values. Defaults to None.
        show_loss (bool, optional): Whether to display the loss chart. Defaults to False.
        history_kind (str, optional): The kind of history collected ("iterative" or "auto"). Defaults to "iterative".
        title (str, optional): The main title of the figure. Defaults to None.
        strict_loss (bool, optional): If True, strictly enforce loss display rules. Defaults to False.
        dec (int, optional): Number of decimal places to show for parameters. Defaults to 4.

    Returns:
        plotly.graph_objects.Figure: The fully constructed Plotly figure.
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    d = int(X.shape[1])

    # ---- history dict (sin OR con arrays) ----
    if history is not None:
        if not isinstance(history, dict):
            raise ValueError("history must be a dict returned by fit_history().")

        history_kind = history.get("history_kind", history_kind)

        loss_hist = _first_not_none(
            history.get("loss_hist", None),
            history.get("losses", None),
            history.get("loss", None),
            loss_hist,
        )
        
        metrics_hist = _first_not_none(history.get("metrics_hist", None), metrics_hist)

        grid = history.get("grid", {}) or {}

        # Prefer theta "para mostrar" (respeta display_space de fit_history)
        w_hist = _first_not_none(history.get("w_hist", None), w_hist)
        b_hist = _first_not_none(history.get("b_hist", None), b_hist)

        if d == 1:
            y_line_hist = _first_not_none(history.get("y_line_hist", None), y_line_hist)
            x1_grid = _first_not_none(grid.get("x1_grid", None), x1_grid)

        elif d == 2:
            z_plane_hist = _first_not_none(history.get("z_plane_hist", None), z_plane_hist)
            X1g = _first_not_none(grid.get("X1g", None), X1g)
            X2g = _first_not_none(grid.get("X2g", None), X2g)

    # ---- routing ----
    if d == 1:
        x1 = X[:, 0]
        if title is None:
            title = "Linear Regression (Simple, 1 variable)"
        return build_simple_lr_figure(
            x1,
            y,
            w_hist=w_hist,
            b_hist=b_hist,
            y_line_hist=y_line_hist,
            x1_grid=x1_grid,
            loss_hist=loss_hist,
            metrics_hist=metrics_hist,
            show_loss=show_loss,
            history_kind=history_kind,
            title=title,
            strict_loss=strict_loss,
            dec=dec,
            frame_duration=frame_duration,
            theme=theme,
        )

    if d == 2:
        x1 = X[:, 0]
        x2 = X[:, 1]
        if title is None:
            title = "Linear Regression (2 variables)"
        return build_plane_lr_figure(
            x1,
            x2,
            y,
            w_hist=w_hist,
            b_hist=b_hist,
            z_plane_hist=z_plane_hist,
            X1g=X1g,
            X2g=X2g,
            loss_hist=loss_hist,
            metrics_hist=metrics_hist,
            show_loss=show_loss,
            history_kind=history_kind,
            title=title,
            strict_loss=strict_loss,
            dec=dec,
            frame_duration=frame_duration,
            theme=theme,
        )

    if d > 2:
        # Para d>2 esta vista ES theta-based (no hay "pred-grid" equivalente aquí)
        if w_hist is None or b_hist is None:
            raise ValueError("For d>2, this visualization expects w_hist and b_hist (parameter-display-based).")
        if title is None:
            title = f"Multivariable Linear Regression Model ({d} variables)"
        return build_multivar_lr_figure(
            X,
            y,
            w_hist,
            b_hist=b_hist,
            loss_hist=loss_hist,
            metrics_hist=metrics_hist,
            show_loss=show_loss,
            history_kind=history_kind,
            title=title,
            strict_loss=strict_loss,
            dec=dec,
            frame_duration=frame_duration,
            theme=theme,
        )

    raise ValueError(f"Unexpected d={d}.")


__all__ = ["build_lr_figure"]
