"""Interpolation strategy for history capture."""

import numpy as np
from sklearn.metrics import mean_squared_error

from ..adapters.base import BaseModelAdapter
from ..utils.math import _binary_log_loss_from_p, _multiclass_cross_entropy
from .base import HistoryCaptureStrategy


class InterpolationCapture(HistoryCaptureStrategy):
    """Captures history by interpolating between a baseline and the final model."""

    def capture_linear(self, adapter: BaseModelAdapter, X: np.ndarray, y: np.ndarray, config) -> dict:
        """Capture linear-regression history by interpolating to final predictions."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        _, d = X.shape
        steps = config.steps

        # Grid Setup
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

        # Baseline
        if config.baseline == "zeros":
            y0 = np.zeros_like(y, dtype=float)
            g0 = np.zeros(Xg_pred.shape[0]) if Xg_pred is not None else None
        else:
            mean_y = float(np.mean(y))
            y0 = np.full_like(y, mean_y, dtype=float)
            g0 = np.full(Xg_pred.shape[0], mean_y) if Xg_pred is not None else None

        # Final Predictions
        yF = adapter.predict(X)
        gF = adapter.predict(Xg_pred) if Xg_pred is not None else None

        # Interpolate
        loss_hist = np.zeros(steps, dtype=float)
        for t in range(steps):
            alpha = t / (steps - 1) if steps > 1 else 1.0

            y_pred = (1 - alpha) * y0 + alpha * yF
            loss_hist[t] = float(mean_squared_error(y, y_pred))

            if Xg_pred is not None:
                gt = (1 - alpha) * g0 + alpha * gF
                if d == 1:
                    y_line_hist[t] = gt
                if d == 2:
                    z_plane_hist[t] = gt.reshape(X1g.shape)

        wF, bF = adapter.extract_linear_theta(d_expected=d)
        w_hist = b_hist = None
        if wF is not None and bF is not None:
            if config.baseline == "zeros":
                w0 = np.zeros_like(wF)
                b0 = 0.0
            else:
                w0 = np.zeros_like(wF)
                b0 = float(np.mean(y))

            w_hist = np.zeros((steps, d), dtype=float)
            b_hist = np.zeros(steps, dtype=float)
            for t in range(steps):
                alpha = t / (steps - 1) if steps > 1 else 1.0
                w_hist[t] = (1 - alpha) * w0 + alpha * wF
                b_hist[t] = (1 - alpha) * b0 + alpha * bF

        return {
            "history_kind": "final_interp",
            "loss_hist": loss_hist,
            "grid": grid,
            "y_line_hist": y_line_hist,
            "z_plane_hist": z_plane_hist,
            "w_hist_learned": w_hist,
            "b_hist_learned": b_hist,
            "scaler_params": adapter.get_scaler_params(),
        }

    def capture_logistic(self, adapter: BaseModelAdapter, X: np.ndarray, y: np.ndarray, config) -> dict:
        """Capture logistic-regression history by interpolating to final probabilities."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n, d = X.shape
        steps = config.steps
        classes = adapter.classes if adapter.classes is not None else np.unique(y)
        K = len(classes)
        is_multiclass = K > 2
        probability_link = (
            adapter.resolve_multiclass_link(X, config.multiclass_link)
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

        def _get_baseline_probs(n_points):
            if config.baseline == "uniform":
                if not is_multiclass:
                    return np.column_stack([np.full(n_points, 0.5), np.full(n_points, 0.5)])
                return np.full((n_points, K), 1.0 / K)

            if not is_multiclass:
                p1 = float(np.mean(y == classes[1]))
                return np.column_stack([np.full(n_points, 1.0 - p1), np.full(n_points, p1)])

            priors = np.array([(y == c).mean() for c in classes], dtype=float)
            return np.tile(priors.reshape(1, -1), (n_points, 1))

        P0 = _get_baseline_probs(n)
        PF = adapter.predict_proba(X, classes)

        g0 = _get_baseline_probs(Xg_pred.shape[0]) if Xg_pred is not None else None
        if Xg_pred is not None:
            gF = adapter.predict_proba(Xg_pred, classes)

        loss_hist = np.zeros(steps, dtype=float)

        for t in range(steps):
            alpha = t / (steps - 1) if steps > 1 else 1.0
            Pt = (1 - alpha) * P0 + alpha * PF

            if not is_multiclass:
                loss_hist[t] = _binary_log_loss_from_p(Pt[:, 1], (y == classes[1]).astype(float))
            else:
                loss_hist[t] = _multiclass_cross_entropy(Pt, y, classes)

            if Xg_pred is not None:
                gt = (1 - alpha) * g0 + alpha * gF
                if d == 1 and not is_multiclass:
                    p_line_hist[t] = gt[:, 1]
                elif d == 1 and is_multiclass:
                    p_curves_hist[t] = gt
                elif d == 2 and not is_multiclass:
                    p_plane_hist[t] = gt[:, 1].reshape(X1g.shape)
                elif d == 2 and is_multiclass:
                    p_surfaces_hist[t] = gt.reshape(X1g.shape[0], X1g.shape[1], K)

        thetaF = adapter.extract_logistic_theta(d_expected=d)
        w_hist = b_hist = None
        if thetaF:
            if thetaF["task"] == "binary":
                wF = thetaF["w"]
                bF = float(thetaF["b"])
                w0 = np.zeros_like(wF)

                if config.baseline == "uniform":
                    b0 = 0.0
                else:
                    p1 = float(np.mean(y == classes[1]))
                    p1 = max(min(p1, 1 - 1e-15), 1e-15)
                    b0 = np.log(p1 / (1 - p1))

                w_hist = np.zeros((steps, d), dtype=float)
                b_hist = np.zeros(steps, dtype=float)
                for t in range(steps):
                    alpha = t / (steps - 1) if steps > 1 else 1.0
                    w_hist[t] = (1 - alpha) * w0 + alpha * wF
                    b_hist[t] = (1 - alpha) * b0 + alpha * bF
            else:
                WF = thetaF["W"]
                bF_arr = thetaF["b"]
                W0 = np.zeros_like(WF)

                if config.baseline == "uniform":
                    b0_arr = np.zeros_like(bF_arr)
                else:
                    priors = np.array([(y == c).mean() for c in classes], dtype=float)
                    priors = np.maximum(priors, 1e-15)
                    b0_arr = np.log(priors)
                    b0_arr = b0_arr - np.mean(b0_arr)

                w_hist = np.zeros((steps, d, K), dtype=float)
                b_hist = np.zeros((steps, K), dtype=float)
                for t in range(steps):
                    alpha = t / (steps - 1) if steps > 1 else 1.0
                    w_hist[t] = (1 - alpha) * W0 + alpha * WF
                    b_hist[t] = (1 - alpha) * b0_arr + alpha * bF_arr

        return {
            "history_kind": "final_interp",
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
            "scaler_params": adapter.get_scaler_params(),
        }
