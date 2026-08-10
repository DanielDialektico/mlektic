"""History capture engine and facade."""

from __future__ import annotations

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
        X, y = self._validate_xy(X, y, task="linear")
        mode = self._resolve_mode(config.mode)
        strategy = IterativeCapture() if mode == "iterative" else InterpolationCapture()
        data = strategy.capture_linear(self.adapter, X, y, config)
        self._apply_theta_scaling(data, config, is_linear=True, is_multiclass=False)
        data["display_space"] = config.display_space

        self._initialize_loss_contract(data)

        w_hist = data.get("w_hist")
        b_hist = data.get("b_hist")
        if w_hist is not None and b_hist is not None:
            X_eval = self._resolve_eval_X(X, data, config)
            y_pred_hist = X_eval @ w_hist.T + b_hist
            data["metrics_hist"] = build_linear_metrics(y, y_pred_hist, data["loss_raw"], config.metrics)

        self._attach_metadata(data, config, task="linear")
        data = self._decimate(data, config)
        self._apply_smoothing(data, config)
        self._finalize_metadata(data)
        return data

    def capture_logistic(self, X, y, config) -> dict:
        """Capture logistic regression history."""
        X, y = self._validate_xy(X, y, task="logistic")
        mode = self._resolve_mode(config.mode)
        strategy = IterativeCapture() if mode == "iterative" else InterpolationCapture()
        data = strategy.capture_logistic(self.adapter, X, y, config)
        self._apply_theta_scaling(data, config, is_linear=False, is_multiclass=data["is_multiclass"])
        data["display_space"] = config.display_space

        self._initialize_loss_contract(data)

        w_hist = data.get("w_hist")
        b_hist = data.get("b_hist")
        if w_hist is not None and b_hist is not None:
            X_eval = self._resolve_eval_X(X, data, config)
            data["metrics_hist"] = build_logistic_metrics(
                y,
                X_eval,
                w_hist,
                b_hist,
                data["loss_raw"],
                data["classes"],
                config.metrics,
                is_multiclass=data.get("is_multiclass", False),
            )

        self._attach_metadata(data, config, task="logistic")
        data = self._decimate(data, config)
        self._apply_smoothing(data, config)
        self._finalize_metadata(data)
        return data

    def _resolve_mode(self, requested_mode: str) -> str:
        if requested_mode == "auto":
            return "iterative" if self.adapter.is_iterative else "final_interp"
        if requested_mode == "iterative" and not self.adapter.is_iterative:
            estimator_name = self.adapter.final_estimator.__class__.__name__
            raise ValueError(
                f"mode='iterative' requires an estimator with partial_fit; {estimator_name} does not provide it. "
                "Use mode='final_interp' for a transparent synthetic interpolation."
            )
        return requested_mode

    @staticmethod
    def _validate_xy(X, y, *, task: str) -> tuple[np.ndarray, np.ndarray]:
        """Normalize and validate training data before any history is constructed."""
        try:
            X_array = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as error:
            raise TypeError("X must be a finite numeric feature matrix.") from error
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        if X_array.ndim != 2 or X_array.shape[0] == 0 or X_array.shape[1] == 0:
            raise ValueError("X must be a non-empty two-dimensional feature matrix.")
        if not np.all(np.isfinite(X_array)):
            raise ValueError("X must contain only finite values.")

        y_array = np.asarray(y)
        if y_array.ndim == 2 and y_array.shape[1] == 1:
            y_array = y_array.ravel()
        if y_array.ndim != 1:
            raise ValueError("y must be one-dimensional; multi-output histories are not supported.")
        if y_array.shape[0] != X_array.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if task == "linear":
            try:
                y_array = np.asarray(y_array, dtype=float)
            except (TypeError, ValueError) as error:
                raise TypeError("Linear-regression targets must be finite numeric values.") from error
            if not np.all(np.isfinite(y_array)):
                raise ValueError("Linear-regression targets must contain only finite values.")
        elif any(value is None or (isinstance(value, float) and np.isnan(value)) for value in y_array):
            raise ValueError("Logistic-regression labels must not contain missing values.")
        return X_array, y_array

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

    def _apply_smoothing(self, data: dict, config):
        raw = np.asarray(data["loss_raw"], dtype=float)
        display = _ema_smooth(raw, config.smooth_beta) if config.smooth == "ema" else raw.copy()
        data["loss_display"] = display
        data["loss_hist"] = display  # Backward-compatible alias used by figure builders.

        loss_label = "Loss" if data.get("task") == "linear" else "Log-loss"
        if loss_label in data.get("metrics_hist", {}):
            data["metrics_hist"][loss_label] = display.copy()

    @staticmethod
    def _initialize_loss_contract(data: dict) -> None:
        """Keep empirical values immutable and reserve ``loss_hist`` as a display alias."""
        raw = np.asarray(data["loss_hist"], dtype=float).copy()
        data["loss_raw"] = raw
        data["loss_display"] = raw.copy()
        data["loss_hist"] = data["loss_display"]

    def _attach_metadata(self, data: dict, config, *, task: str) -> None:
        """Attach an auditable provenance and timeline contract before decimation."""
        source = data.get("history_source", "interpolated")
        step_indices = np.asarray(data.get("step_indices", np.arange(len(data["loss_raw"]))))
        final_matches = self._final_state_matches_estimator(data, task=task)
        warnings_list = []
        if source == "replayed":
            warnings_list.append(
                {
                    "code": "replay_not_original_training",
                    "message": (
                        "This history was reconstructed by fitting a cloned estimator; "
                        "it is not a recording of the original fit call."
                    ),
                }
            )
            if final_matches is False:
                warnings_list.append(
                    {
                        "code": "replay_final_differs",
                        "message": "The final replay checkpoint does not match the fitted estimator parameters.",
                    }
                )
        else:
            warnings_list.append(
                {
                    "code": "synthetic_interpolation",
                    "message": (
                        "States are mathematical interpolations between a baseline and the fitted model; "
                        "they are not optimizer updates."
                    ),
                }
            )

        data["task"] = task
        data["metadata"] = {
            "schema_version": 1,
            "source": source,
            "source_detail": self._source_detail(source, data, config),
            "requested_mode": config.mode,
            "resolved_mode": "iterative" if source == "replayed" else "final_interp",
            "requested_steps": int(config.steps),
            "training_total_steps": self._training_total_steps(),
            "captured_steps": int(step_indices.size),
            "displayed_steps": int(step_indices.size),
            "step_indices": step_indices.copy(),
            "displayed_step_indices": step_indices.copy(),
            "final_state_matches_estimator": final_matches,
            "display_space": config.display_space,
            "smoothing": {"method": config.smooth, "beta": float(config.smooth_beta)},
            "decimation": {"max_frames": config.max_frames, "frame_step": config.frame_step},
            "warnings": warnings_list,
        }

    def _source_detail(self, source: str, data: dict, config) -> dict:
        """Describe the effective mechanism without exposing non-serializable objects."""
        estimator = self.adapter.final_estimator
        if source == "replayed":
            return dict(data.get("source_detail", {"estimator": estimator.__class__.__name__}))
        return {
            "estimator": estimator.__class__.__name__,
            "path": "baseline_to_fitted_model",
            "baseline": config.baseline,
        }

    def _training_total_steps(self) -> int | None:
        """Return the estimator-reported iteration count when it is available."""
        value = getattr(self.adapter.final_estimator, "n_iter_", None)
        if value is None:
            return None
        values = np.asarray(value)
        if values.size == 0:
            return None
        try:
            return int(np.max(values.astype(int)))
        except (TypeError, ValueError):
            return None

    def _final_state_matches_estimator(self, data: dict, *, task: str) -> bool | None:
        """Compare the last constructed parameter state with the fitted estimator."""
        w_hist = data.get("w_hist_learned")
        b_hist = data.get("b_hist_learned")
        if w_hist is None or b_hist is None or len(w_hist) == 0:
            return None
        if task == "linear":
            w_final, b_final = self.adapter.extract_linear_theta(d_expected=np.asarray(w_hist).shape[-1])
            if w_final is None:
                return None
            return bool(
                np.allclose(np.asarray(w_hist)[-1], w_final, rtol=1e-7, atol=1e-9)
                and np.allclose(np.asarray(b_hist)[-1], b_final, rtol=1e-7, atol=1e-9)
            )

        theta = self.adapter.extract_logistic_theta(d_expected=np.asarray(w_hist).shape[1])
        if theta is None:
            return None
        w_final = theta["w"] if theta["task"] == "binary" else theta["W"]
        b_final = theta["b"]
        return bool(
            np.allclose(np.asarray(w_hist)[-1], w_final, rtol=1e-7, atol=1e-9)
            and np.allclose(np.asarray(b_hist)[-1], b_final, rtol=1e-7, atol=1e-9)
        )

    @staticmethod
    def _finalize_metadata(data: dict) -> None:
        """Record the retained temporal coordinates after frame decimation."""
        displayed = np.asarray(data.get("step_indices", []))
        metadata = data["metadata"]
        metadata["displayed_steps"] = int(displayed.size)
        metadata["displayed_step_indices"] = displayed.copy()
        if "alpha_values" in data:
            metadata["alpha_values"] = np.asarray(data["alpha_values"], dtype=float).copy()

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
