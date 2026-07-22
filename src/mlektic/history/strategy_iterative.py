"""Iterative strategy for history capture."""

import numpy as np
from sklearn.metrics import mean_squared_error

from ..adapters.base import BaseModelAdapter
from ..utils.math import _binary_log_loss_from_p, _multiclass_cross_entropy
from .base import HistoryCaptureStrategy


class IterativeCapture(HistoryCaptureStrategy):
    """Captures history by iteratively fitting the model step by step."""

    def capture_linear(self, adapter: BaseModelAdapter, X: np.ndarray, y: np.ndarray, config) -> dict:
        """Capture linear-regression history through incremental replay."""
        _, d = X.shape
        steps = config.steps

        # We need a new adapter instance configured for replay
        replay_adapter = adapter.clone_for_replay()
        replay_adapter.fit(X, y)  # Init state

        grid = {}
        y_line_hist = z_plane_hist = None
        Xg_pred = None

        if d == 1:
            x1_grid = np.linspace(float(X[:, 0].min()), float(X[:, 0].max()), config.grid_1d_points)
            grid["x1_grid"] = x1_grid
            y_line_hist = np.zeros((steps, x1_grid.size), dtype=float)
            Xg_pred = x1_grid.reshape(-1, 1)
        elif d == 2:
            x1_grid = np.linspace(float(X[:, 0].min()), float(X[:, 0].max()), config.grid_2d_points)
            x2_grid = np.linspace(float(X[:, 1].min()), float(X[:, 1].max()), config.grid_2d_points)
            X1g, X2g = np.meshgrid(x1_grid, x2_grid)
            grid["x1_grid"] = x1_grid
            grid["x2_grid"] = x2_grid
            grid["X1g"] = X1g
            grid["X2g"] = X2g
            z_plane_hist = np.zeros((steps, X1g.shape[0], X1g.shape[1]), dtype=float)
            Xg_pred = np.column_stack([X1g.ravel(), X2g.ravel()])

        loss_hist = np.zeros(steps, dtype=float)
        w_hist = np.zeros((steps, d), dtype=float)
        b_hist = np.zeros(steps, dtype=float)

        # Step 0
        y_pred0 = replay_adapter.predict(X)
        loss_hist[0] = float(mean_squared_error(y, y_pred0))
        g0 = replay_adapter.predict(Xg_pred) if Xg_pred is not None else None
        if g0 is not None:
            if d == 1:
                y_line_hist[0] = g0
            elif d == 2:
                z_plane_hist[0] = g0.reshape(X1g.shape)

        w0, b0 = replay_adapter.extract_linear_theta(d_expected=d)
        if w0 is not None:
            w_hist[0] = w0
            b_hist[0] = b0

        Xt = replay_adapter.transform_X(X) if replay_adapter.is_pipeline else X

        for t in range(1, steps):
            try:
                replay_adapter.partial_fit(Xt, y)
            except Exception:
                replay_adapter.fit(X, y)  # Fallback

            y_pred = replay_adapter.predict(X)
            loss_hist[t] = float(mean_squared_error(y, y_pred))

            gt = replay_adapter.predict(Xg_pred) if Xg_pred is not None else None
            if gt is not None:
                if d == 1:
                    y_line_hist[t] = gt
                elif d == 2:
                    z_plane_hist[t] = gt.reshape(X1g.shape)

            wt, bt = replay_adapter.extract_linear_theta(d_expected=d)
            if wt is not None:
                w_hist[t] = wt
                b_hist[t] = bt

        return {
            "history_kind": "iterative",
            "loss_hist": loss_hist,
            "grid": grid,
            "y_line_hist": y_line_hist,
            "z_plane_hist": z_plane_hist,
            "w_hist_learned": w_hist if np.any(w_hist) else None,
            "b_hist_learned": b_hist if np.any(b_hist) else None,
            "scaler_params": replay_adapter.get_scaler_params(),
        }

    def capture_logistic(self, adapter: BaseModelAdapter, X: np.ndarray, y: np.ndarray, config) -> dict:
        """Capture logistic-regression history through incremental replay."""
        _, d = X.shape
        steps = config.steps

        replay_adapter = adapter.clone_for_replay()
        replay_adapter.fit(X, y)
        classes = replay_adapter.classes if replay_adapter.classes is not None else np.unique(y)
        K = len(classes)
        is_multiclass = K > 2
        probability_link = (
            replay_adapter.resolve_multiclass_link(X, config.multiclass_link)
            if is_multiclass
            else "sigmoid"
        )

        grid = {}
        p_line_hist = p_plane_hist = p_curves_hist = p_surfaces_hist = None
        Xg_pred = None

        if d == 1:
            x1_grid = np.linspace(float(X[:, 0].min()), float(X[:, 0].max()), config.grid_1d_points)
            grid["x1_grid"] = x1_grid
            Xg_pred = x1_grid.reshape(-1, 1)
            if not is_multiclass:
                p_line_hist = np.zeros((steps, x1_grid.size), dtype=float)
            else:
                p_curves_hist = np.zeros((steps, x1_grid.size, K), dtype=float)

        elif d == 2:
            x1_grid = np.linspace(float(X[:, 0].min()), float(X[:, 0].max()), config.grid_2d_points)
            x2_grid = np.linspace(float(X[:, 1].min()), float(X[:, 1].max()), config.grid_2d_points)
            X1g, X2g = np.meshgrid(x1_grid, x2_grid)
            grid.update({"x1_grid": x1_grid, "x2_grid": x2_grid, "X1g": X1g, "X2g": X2g})
            Xg_pred = np.column_stack([X1g.ravel(), X2g.ravel()])
            if not is_multiclass:
                p_plane_hist = np.zeros((steps, X1g.shape[0], X1g.shape[1]), dtype=float)
            else:
                p_surfaces_hist = np.zeros((steps, X1g.shape[0], X1g.shape[1], K), dtype=float)

        loss_hist = np.zeros(steps, dtype=float)
        w_hist = b_hist = None

        # Initialize history stores based on multiclass or binary
        theta0 = replay_adapter.extract_logistic_theta(d_expected=d)
        if theta0:
            if theta0["task"] == "binary":
                w_hist = np.zeros((steps, d), dtype=float)
                b_hist = np.zeros(steps, dtype=float)
                w_hist[0] = theta0["w"]
                b_hist[0] = theta0["b"]
            else:
                w_hist = np.zeros((steps, d, K), dtype=float)
                b_hist = np.zeros((steps, K), dtype=float)
                w_hist[0] = theta0["W"]
                b_hist[0] = theta0["b"]

        def _record_step(t, adapter_inst):
            nonlocal w_hist, b_hist
            Pt = adapter_inst.predict_proba(X, classes)
            if not is_multiclass:
                loss_hist[t] = _binary_log_loss_from_p(Pt[:, 1], (y == classes[1]).astype(float))
            else:
                loss_hist[t] = _multiclass_cross_entropy(Pt, y, classes)

            if Xg_pred is not None:
                gt = adapter_inst.predict_proba(Xg_pred, classes)
                if d == 1 and not is_multiclass:
                    p_line_hist[t] = gt[:, 1]
                elif d == 1 and is_multiclass:
                    p_curves_hist[t] = gt
                elif d == 2 and not is_multiclass:
                    p_plane_hist[t] = gt[:, 1].reshape(X1g.shape)
                elif d == 2 and is_multiclass:
                    p_surfaces_hist[t] = gt.reshape(X1g.shape[0], X1g.shape[1], K)

            thetat = adapter_inst.extract_logistic_theta(d_expected=d)
            if thetat and w_hist is not None:
                if thetat["task"] == "binary":
                    w_hist[t] = thetat["w"]
                    b_hist[t] = thetat["b"]
                else:
                    w_hist[t] = thetat["W"]
                    b_hist[t] = thetat["b"]

        _record_step(0, replay_adapter)
        Xt = replay_adapter.transform_X(X) if replay_adapter.is_pipeline else X

        for t in range(1, steps):
            try:
                replay_adapter.partial_fit(Xt, y)
            except Exception:
                replay_adapter.fit(X, y)
            _record_step(t, replay_adapter)

        return {
            "history_kind": "iterative",
            "classes": classes,
            "is_multiclass": is_multiclass,
            "probability_link": probability_link,
            "loss_hist": loss_hist,
            "grid": grid,
            "p_line_hist": p_line_hist,
            "p_plane_hist": p_plane_hist,
            "p_curves_hist": p_curves_hist,
            "p_surfaces_hist": p_surfaces_hist,
            "w_hist_learned": w_hist,
            "b_hist_learned": b_hist,
            "scaler_params": replay_adapter.get_scaler_params(),
        }
