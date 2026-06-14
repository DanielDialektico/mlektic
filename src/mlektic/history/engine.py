"""History capture engine and facade."""

import numpy as np

from ..adapters.base import BaseModelAdapter
from ..adapters.sklearn import SklearnAdapter
from ..utils.math import _ema_smooth
from .strategy_interp import InterpolationCapture
from .strategy_iterative import IterativeCapture
from .base import (
    _scale_linear_theta,
    _scale_logistic_binary_theta,
    _scale_logistic_multiclass_theta
)

class HistoryEngine:
    """Engine that orchestrates history capture."""

    def __init__(self, estimator):
        # Determine adapter
        # Right now we only have SklearnAdapter
        self.adapter = SklearnAdapter(estimator)

    def capture_linear(self, X, y, config) -> dict:
        """Capture linear linear history."""
        mode = self._resolve_mode(config.mode)
        
        if mode == "iterative":
            strategy = IterativeCapture()
        else:
            strategy = InterpolationCapture()
            
        data = strategy.capture_linear(self.adapter, X, y, config)
        self._apply_smoothing(data, config, is_linear=True)
        self._apply_theta_scaling(data, config, is_linear=True, is_multiclass=False)
        data["display_space"] = config.display_space

        # Calculate MSE and R2 for all steps
        w_hist = data["w_hist"]
        b_hist = data["b_hist"]
        if w_hist is not None and b_hist is not None:
            # We must use the appropriate X depending on display_space
            X_eval = X
            if config.display_space == "scaled" and "scaler_params" in data:
                scaler_params = data["scaler_params"]
                if scaler_params[0] is not None:
                    mu, scale = scaler_params
                    X_eval = X.copy()
                    if mu is not None:
                        X_eval = X_eval - mu
                    if scale is not None:
                        X_eval = X_eval / (scale + 1e-12)
            
            # Predict for all steps: (n, d) @ (d, steps) + (steps,) -> (n, steps)
            y_pred_hist = X_eval @ w_hist.T + b_hist
            mse_hist = np.mean((y.reshape(-1, 1) - y_pred_hist)**2, axis=0)
            
            y_mean = np.mean(y)
            ss_tot = np.sum((y - y_mean)**2)
            if ss_tot > 1e-12:
                ss_res = np.sum((y.reshape(-1, 1) - y_pred_hist)**2, axis=0)
                r2_hist = 1.0 - (ss_res / ss_tot)
            else:
                r2_hist = np.zeros_like(mse_hist)
                
            metrics_hist = {
                "Loss": data["loss_hist"],
                "MSE": mse_hist,
                "R²": r2_hist
            }
            
            if config.metrics:
                for name, fn in config.metrics.items():
                    metrics_hist[name] = np.array([fn(y, y_pred_hist[:, t]) for t in range(y_pred_hist.shape[1])])
            
            # keep up to 5 metrics maximum
            data["metrics_hist"] = {k: metrics_hist[k] for k in list(metrics_hist)[:5]}

        return data

    def capture_logistic(self, X, y, config) -> dict:
        """Capture logistic regression history."""
        mode = self._resolve_mode(config.mode)
        
        if mode == "iterative":
            strategy = IterativeCapture()
        else:
            strategy = InterpolationCapture()
            
        data = strategy.capture_logistic(self.adapter, X, y, config)
        self._apply_smoothing(data, config, is_linear=False)
        self._apply_theta_scaling(data, config, is_linear=False, is_multiclass=data["is_multiclass"])
        data["display_space"] = config.display_space

        w_hist = data["w_hist"]
        b_hist = data["b_hist"]
        if w_hist is not None and b_hist is not None:
            X_eval = X
            if config.display_space == "scaled" and "scaler_params" in data:
                scaler_params = data["scaler_params"]
                if scaler_params[0] is not None:
                    mu, scale = scaler_params
                    X_eval = X.copy()
                    if mu is not None:
                        X_eval = X_eval - mu
                    if scale is not None:
                        X_eval = X_eval / (scale + 1e-12)
            
            is_multiclass = data.get("is_multiclass", False)
            steps = w_hist.shape[0]
            acc_hist = np.zeros(steps)
            f1_hist = np.zeros(steps)
            
            from sklearn.metrics import accuracy_score, f1_score
            if is_multiclass:
                for t in range(steps):
                    z_t = X_eval @ w_hist[t] + b_hist[t]
                    y_pred = np.argmax(z_t, axis=1)
                    acc_hist[t] = accuracy_score(y, y_pred)
                    f1_hist[t] = f1_score(y, y_pred, average='macro', zero_division=0)
            else:
                if w_hist.ndim == 1:
                    w_t = w_hist.reshape(-1, 1)
                else:
                    w_t = w_hist
                    
                z_hist = X_eval @ w_t.T + b_hist
                from ..utils.math import _sigmoid
                p_hist = _sigmoid(z_hist)
                y_pred_hist = (p_hist >= 0.5).astype(int)
                
                for t in range(steps):
                    acc_hist[t] = accuracy_score(y, y_pred_hist[:, t])
                    f1_hist[t] = f1_score(y, y_pred_hist[:, t], average='binary', zero_division=0)
                    
            metrics_hist = {
                "Log-loss": data["loss_hist"],
                "Accuracy": acc_hist,
                "F1 Score": f1_hist
            }
            
            if config.metrics:
                for name, fn in config.metrics.items():
                    if is_multiclass:
                        metrics_hist[name] = np.array([fn(y, np.argmax(X_eval @ w_hist[t] + b_hist[t], axis=1)) for t in range(steps)])
                    else:
                        metrics_hist[name] = np.array([fn(y, y_pred_hist[:, t]) for t in range(steps)])
            
            data["metrics_hist"] = {k: metrics_hist[k] for k in list(metrics_hist)[:5]}

        return data

    def _resolve_mode(self, requested_mode: str) -> str:
        if requested_mode == "auto":
            return "iterative" if self.adapter.is_iterative else "final_interp"
        return requested_mode

    def _apply_smoothing(self, data: dict, config, is_linear: bool):
        if config.smooth != "ema":
            return
            
        beta = config.smooth_beta
        data["loss_hist"] = _ema_smooth(data["loss_hist"], beta)
        
        if is_linear:
            if data.get("y_line_hist") is not None:
                h = data["y_line_hist"]
                for j in range(h.shape[1]): h[:, j] = _ema_smooth(h[:, j], beta)
            if data.get("z_plane_hist") is not None:
                h = data["z_plane_hist"]
                Z = h.reshape(h.shape[0], -1)
                for j in range(Z.shape[1]): Z[:, j] = _ema_smooth(Z[:, j], beta)
                data["z_plane_hist"] = Z.reshape(h.shape)
        else:
            if data.get("p_line_hist") is not None:
                h = data["p_line_hist"]
                for j in range(h.shape[1]): h[:, j] = _ema_smooth(h[:, j], beta)
            if data.get("p_plane_hist") is not None:
                h = data["p_plane_hist"]
                Z = h.reshape(h.shape[0], -1)
                for j in range(Z.shape[1]): Z[:, j] = _ema_smooth(Z[:, j], beta)
                data["p_plane_hist"] = Z.reshape(h.shape)
            if data.get("p_curves_hist") is not None:
                h = data["p_curves_hist"]
                P = h.reshape(h.shape[0], -1)
                for j in range(P.shape[1]): P[:, j] = _ema_smooth(P[:, j], beta)
                data["p_curves_hist"] = P.reshape(h.shape)
            if data.get("p_surfaces_hist") is not None:
                h = data["p_surfaces_hist"]
                P = h.reshape(h.shape[0], -1)
                for j in range(P.shape[1]): P[:, j] = _ema_smooth(P[:, j], beta)
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
