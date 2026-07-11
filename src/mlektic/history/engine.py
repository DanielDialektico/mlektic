"""History capture engine and facade."""

import numpy as np

from ..adapters.sklearn import SklearnAdapter
from ..utils.math import _ema_smooth
from .base import (
    _scale_linear_theta,
    _scale_logistic_binary_theta,
    _scale_logistic_multiclass_theta,
)
from .metrics import build_linear_metrics, build_logistic_metrics
from .sampling import decimate_history
from .strategy_interp import InterpolationCapture
from .strategy_iterative import IterativeCapture


class HistoryEngine:
    """Engine that orchestrates history capture."""

    def __init__(self, estimator):
        """Create a history engine for an estimator."""
        self.adapter = SklearnAdapter(estimator)

    def capture_linear(self, X, y, config) -> dict:
        """Capture linear-regression history."""
        mode = self._resolve_mode(config.mode)
        strategy = IterativeCapture() if mode == "iterative" else InterpolationCapture()
        data = strategy.capture_linear(self.adapter, X, y, config)
        self._apply_theta_scaling(data, config, is_linear=True, is_multiclass=False)
        data["display_space"] = config.display_space

        w_hist = data.get("w_hist")
        b_hist = data.get("b_hist")
        if w_hist is not None and b_hist is not None:
            X_eval = self._resolve_eval_X(X, data, config)
            y_pred_hist = X_eval @ w_hist.T + b_hist
            data["metrics_hist"] = build_linear_metrics(y, y_pred_hist, data["loss_hist"], config.metrics)

        data = self._decimate(data, config)
        self._apply_smoothing(data, config, is_linear=True)
        return data

    def capture_logistic(self, X, y, config) -> dict:
        """Capture logistic regression history."""
        mode = self._resolve_mode(config.mode)
        strategy = IterativeCapture() if mode == "iterative" else InterpolationCapture()
        data = strategy.capture_logistic(self.adapter, X, y, config)
        self._apply_theta_scaling(data, config, is_linear=False, is_multiclass=data["is_multiclass"])
        data["display_space"] = config.display_space

        w_hist = data.get("w_hist")
        b_hist = data.get("b_hist")
        if w_hist is not None and b_hist is not None:
            X_eval = self._resolve_eval_X(X, data, config)
            data["metrics_hist"] = build_logistic_metrics(
                y,
                X_eval,
                w_hist,
                b_hist,
                data["loss_hist"],
                data["classes"],
                config.metrics,
                is_multiclass=data.get("is_multiclass", False),
            )

        data = self._decimate(data, config)
        self._apply_smoothing(data, config, is_linear=False)
        return data

    def _resolve_mode(self, requested_mode: str) -> str:
        if requested_mode == "auto":
            return "iterative" if self.adapter.is_iterative else "final_interp"
        return requested_mode

    def _resolve_eval_X(self, X, data: dict, config):
        """Return X in the same feature space as the displayed coefficients."""
        X_eval = np.asarray(X, dtype=float)
        if config.display_space != "scaled" or "scaler_params" not in data:
            return X_eval

        mu, scale = data.get("scaler_params", (None, None))
        if mu is not None:
            X_eval = X_eval - mu
        if scale is not None:
            X_eval = X_eval / (scale + 1e-12)
        return X_eval

    def _decimate(self, data: dict, config) -> dict:
        """Apply animation frame reduction according to the capture config."""
        return decimate_history(
            data,
            max_frames=getattr(config, "max_frames", 60),
            frame_step=getattr(config, "frame_step", 10),
        )

    def _apply_smoothing(self, data: dict, config, is_linear: bool):
        if config.smooth != "ema":
            return

        beta = config.smooth_beta
        data["loss_hist"] = _ema_smooth(data["loss_hist"], beta)

        if is_linear:
            if data.get("y_line_hist") is not None:
                h = data["y_line_hist"]
                for j in range(h.shape[1]):
                    h[:, j] = _ema_smooth(h[:, j], beta)
            if data.get("z_plane_hist") is not None:
                h = data["z_plane_hist"]
                Z = h.reshape(h.shape[0], -1)
                for j in range(Z.shape[1]):
                    Z[:, j] = _ema_smooth(Z[:, j], beta)
                data["z_plane_hist"] = Z.reshape(h.shape)
        else:
            if data.get("p_line_hist") is not None:
                h = data["p_line_hist"]
                for j in range(h.shape[1]):
                    h[:, j] = _ema_smooth(h[:, j], beta)
            if data.get("p_plane_hist") is not None:
                h = data["p_plane_hist"]
                Z = h.reshape(h.shape[0], -1)
                for j in range(Z.shape[1]):
                    Z[:, j] = _ema_smooth(Z[:, j], beta)
                data["p_plane_hist"] = Z.reshape(h.shape)
            if data.get("p_curves_hist") is not None:
                h = data["p_curves_hist"]
                P = h.reshape(h.shape[0], -1)
                for j in range(P.shape[1]):
                    P[:, j] = _ema_smooth(P[:, j], beta)
                data["p_curves_hist"] = P.reshape(h.shape)
            if data.get("p_surfaces_hist") is not None:
                h = data["p_surfaces_hist"]
                P = h.reshape(h.shape[0], -1)
                for j in range(P.shape[1]):
                    P[:, j] = _ema_smooth(P[:, j], beta)
                data["p_surfaces_hist"] = P.reshape(h.shape)

    def _apply_theta_scaling(self, data: dict, config, is_linear: bool, is_multiclass: bool):
        w_learned = data.get("w_hist_learned")
        b_learned = data.get("b_hist_learned")

        if w_learned is None or b_learned is None:
            data["w_hist"] = None
            data["b_hist"] = None
            return

        # Default behavior is just point w_hist to learned
        if config.display_space != "original" or "scaler_params" not in data:
            data["w_hist"] = w_learned
            data["b_hist"] = b_learned
            return

        scaler_params = data.get("scaler_params", (None, None))

        w_show = np.zeros_like(w_learned)
        b_show = np.zeros_like(b_learned)

        steps = w_learned.shape[0]
        for t in range(steps):
            if is_linear:
                w, b = _scale_linear_theta(w_learned[t], b_learned[t], scaler_params)
            else:
                if is_multiclass:
                    w, b = _scale_logistic_multiclass_theta(w_learned[t], b_learned[t], scaler_params)
                else:
                    w, b = _scale_logistic_binary_theta(w_learned[t], b_learned[t], scaler_params)
            w_show[t] = w
            b_show[t] = b

        data["w_hist"] = w_show
        data["b_hist"] = b_show
