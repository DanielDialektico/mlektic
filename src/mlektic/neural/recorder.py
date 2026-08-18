"""Training recorder for PyTorch models used in educational visualizations."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from .hyperparameters import scheduler_configuration
from .introspection import _leaf_modules, _require_torch
from .metrics import infer_performance_metrics


class TorchTrainingRecorder:
    """Capture compact, frame-aligned training data from a PyTorch model.

    Call :meth:`record` after ``optimizer.step()`` and before ``zero_grad()`` to
    retain both the updated parameters and the gradient that produced them.
    """

    def __init__(
        self,
        model: Any,
        *,
        optimizer: Any | None = None,
        loss_fn: Any | None = None,
        scheduler: Any | None = None,
        record_every: int = 1,
        capture_weights: bool = True,
        capture_gradients: bool = True,
        capture_activations: bool = True,
        capture_buffers: bool = True,
        capture_optimizer_state: bool = False,
        max_tensor_elements: int = 4096,
        max_activation_elements: int = 512,
    ) -> None:
        """Create a recorder attached to *model* without changing its behavior."""
        if record_every < 1:
            raise ValueError("record_every must be at least 1.")
        if max_tensor_elements < 1:
            raise ValueError("max_tensor_elements must be at least 1.")
        if max_activation_elements < 1:
            raise ValueError("max_activation_elements must be at least 1.")
        _require_torch()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.record_every = record_every
        self.capture_weights = capture_weights
        self.capture_gradients = capture_gradients
        self.capture_activations = capture_activations
        self.capture_buffers = capture_buffers
        self.capture_optimizer_state = capture_optimizer_state
        self.max_tensor_elements = max_tensor_elements
        self.max_activation_elements = max_activation_elements
        self.steps: List[int] = []
        self.loss: List[float] = []
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.parameters: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.gradients: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.buffers: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.parameter_norms: Dict[str, List[float]] = defaultdict(list)
        self.gradient_norms: Dict[str, List[float]] = defaultdict(list)
        self.buffer_norms: Dict[str, List[float]] = defaultdict(list)
        self.optimizer_groups: List[List[Dict[str, Any]]] = []
        self.optimizer_state_norms: List[Dict[str, Dict[str, float]]] = []
        self.frame_semantics: List[Dict[str, str]] = []
        self.activations: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: {"mean": [], "std": [], "min": [], "max": []}
        )
        self.activation_vectors: Dict[str, List[np.ndarray]] = defaultdict(list)
        self._hooks = []
        self._latest_activations: Dict[str, Dict[str, Any]] = {}
        if capture_activations:
            self._register_activation_hooks()

    def _register_activation_hooks(self) -> None:
        for name, module in _leaf_modules(self.model):
            self._hooks.append(module.register_forward_hook(self._activation_hook(name)))

    def _activation_hook(self, name: str):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if tensor is None or not hasattr(tensor, "detach"):
                return
            values = tensor.detach().float().cpu().numpy()
            self._latest_activations[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            if values.ndim >= 2:
                reduction_axes = (0, *range(2, values.ndim))
                vector = np.mean(values, axis=reduction_axes)
            else:
                vector = values
            vector = np.asarray(vector, dtype=float).ravel()
            if vector.size <= self.max_activation_elements:
                self._latest_activations[name]["vector"] = vector.copy()

        return hook

    def record(
        self,
        step: int,
        *,
        loss: Any | None = None,
        metrics: Dict[str, Any] | None = None,
        predictions: Any | None = None,
        targets: Any | None = None,
        task: str = "auto",
        capture_phase: str = "post_step",
        observation_phase: str | None = None,
    ) -> bool:
        """Save one frame and optionally infer three performance metrics.

        Args:
            step: Training-step index. Frames outside ``record_every`` are skipped.
            loss: Scalar loss for the current step.
            metrics: Explicit scalar metrics to store or override inferred values.
            predictions: Model outputs used with ``targets`` for metric inference.
            targets: Ground-truth values paired with ``predictions``.
            task: ``"classification"``, ``"regression"`` or ``"auto"``.
            capture_phase: Point at which parameters and buffers are captured:
                ``"post_forward"``, ``"post_backward"``, or ``"post_step"``.
            observation_phase: Phase associated with loss, predictions, targets,
                and the latest activations. Defaults to ``"pre_step"`` when
                parameters are captured after an optimizer step.

        Returns:
            ``True`` when a frame was captured and ``False`` when it was skipped.

        Raises:
            ValueError: If only one of ``predictions`` and ``targets`` is given.
        """
        if step % self.record_every:
            return False
        if (predictions is None) != (targets is None):
            raise ValueError("predictions and targets must be provided together.")
        valid_phases = {"post_forward", "post_backward", "post_step"}
        if capture_phase not in valid_phases:
            raise ValueError("capture_phase must be 'post_forward', 'post_backward', or 'post_step'.")
        if observation_phase is None:
            observation_phase = "pre_step" if capture_phase == "post_step" else capture_phase
        self.steps.append(int(step))
        self.loss.append(self._scalar(loss))
        self.frame_semantics.append(
            {
                "capture_phase": capture_phase,
                "parameter_phase": capture_phase,
                "buffer_phase": capture_phase,
                "observation_phase": observation_phase,
                "gradient_phase": "post_backward" if capture_phase == "post_step" else capture_phase,
            }
        )
        provided = infer_performance_metrics(predictions, targets, task=task) if predictions is not None else {}
        provided.update(metrics or {})
        metric_names = list(self.metrics)
        metric_names.extend(name for name in provided if name not in self.metrics)
        for name in metric_names:
            values = self.metrics[name]
            while len(values) < len(self.steps) - 1:
                values.append(float("nan"))
            values.append(self._scalar(provided.get(name)))

        for name, parameter in self.model.named_parameters():
            values = parameter.detach().float().cpu().numpy().copy()
            self.parameter_norms[name].append(float(np.linalg.norm(values)))
            if self.capture_weights and values.size <= self.max_tensor_elements:
                self.parameters[name].append(values)
            gradient = parameter.grad
            if gradient is None:
                gradient_values = None
                self.gradient_norms[name].append(float("nan"))
            else:
                gradient_values = gradient.detach().float().cpu().numpy().copy()
                self.gradient_norms[name].append(float(np.linalg.norm(gradient_values)))
            if self.capture_gradients and values.size <= self.max_tensor_elements:
                aligned_gradient = np.zeros_like(values) if gradient_values is None else gradient_values
                self.gradients[name].append(aligned_gradient)

        for name, buffer in self.model.named_buffers():
            values = buffer.detach().float().cpu().numpy().copy()
            self.buffer_norms[name].append(float(np.linalg.norm(values)))
            if self.capture_buffers and values.size <= self.max_tensor_elements:
                self.buffers[name].append(values)

        self.optimizer_groups.append(self._optimizer_groups())
        self.optimizer_state_norms.append(self._optimizer_state_snapshot())

        if self.capture_activations:
            activation_names = set(self.activations) | set(self._latest_activations)
            for name in activation_names:
                summary = self._latest_activations.get(name, {})
                for statistic in ("mean", "std", "min", "max"):
                    values = self.activations[name][statistic]
                    while len(values) < len(self.steps) - 1:
                        values.append(float("nan"))
                    values.append(float(summary.get(statistic, float("nan"))))
                vector = summary.get("vector")
                vectors = self.activation_vectors[name]
                while len(vectors) < len(self.steps) - 1:
                    vectors.append(np.asarray([], dtype=float))
                vectors.append(
                    np.asarray(vector, dtype=float).copy() if vector is not None else np.asarray([], dtype=float)
                )
            self._latest_activations.clear()
        return True

    def _optimizer_groups(self) -> List[Dict[str, Any]]:
        """Capture effective serializable values for every optimizer group."""
        if self.optimizer is None:
            return []
        groups = []
        for group in getattr(self.optimizer, "param_groups", []):
            groups.append(
                {
                    name: value
                    for name, value in group.items()
                    if name != "params" and (value is None or isinstance(value, (bool, int, float, str, tuple, list)))
                }
            )
        return groups

    def _optimizer_state_snapshot(self) -> Dict[str, Dict[str, float]]:
        """Capture compact optimizer-state norms without retaining full tensors."""
        if self.optimizer is None or not self.capture_optimizer_state:
            return {}
        names = {id(parameter): name for name, parameter in self.model.named_parameters()}
        snapshot: Dict[str, Dict[str, float]] = {}
        for parameter, state in getattr(self.optimizer, "state", {}).items():
            parameter_name = names.get(id(parameter), f"parameter_{len(snapshot)}")
            values: Dict[str, float] = {}
            for name, value in state.items():
                if hasattr(value, "detach"):
                    array = value.detach().float().cpu().numpy()
                    values[name] = float(np.linalg.norm(array))
                elif isinstance(value, (bool, int, float)):
                    values[name] = float(value)
            snapshot[parameter_name] = values
        return snapshot

    @staticmethod
    def _scalar(value: Any | None) -> float:
        if value is None:
            return float("nan")
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        return float(value)

    def to_history(self) -> Dict[str, Any]:
        """Return an immutable-by-convention payload consumed by Plotly builders."""
        if not self.steps:
            raise ValueError("No frames were captured. Call record() during training first.")
        return {
            "steps": np.asarray(self.steps, dtype=int),
            "loss": np.asarray(self.loss, dtype=float),
            "metrics": {name: np.asarray(values, dtype=float) for name, values in self.metrics.items()},
            "parameters": {name: list(values) for name, values in self.parameters.items()},
            "gradients": {name: list(values) for name, values in self.gradients.items()},
            "buffers": {name: list(values) for name, values in self.buffers.items()},
            "parameter_norms": {name: np.asarray(values, dtype=float) for name, values in self.parameter_norms.items()},
            "gradient_norms": {name: np.asarray(values, dtype=float) for name, values in self.gradient_norms.items()},
            "buffer_norms": {name: np.asarray(values, dtype=float) for name, values in self.buffer_norms.items()},
            "activations": {
                name: {statistic: np.asarray(values, dtype=float) for statistic, values in summary.items()}
                for name, summary in self.activations.items()
            },
            "activation_vectors": {name: list(values) for name, values in self.activation_vectors.items()},
            "optimizer_groups": list(self.optimizer_groups),
            "optimizer_state_norms": list(self.optimizer_state_norms),
            "frame_semantics": list(self.frame_semantics),
            "history_schema_version": 2,
            "training_config": self._training_config(),
        }

    def _training_config(self) -> Dict[str, Any]:
        """Return lightweight optimizer, loss, and capture configuration metadata."""
        config: Dict[str, Any] = {
            "model": self.model.__class__.__name__,
            "record_every": self.record_every,
        }
        if self.optimizer is not None:
            config["optimizer"] = self.optimizer.__class__.__name__
            defaults = getattr(self.optimizer, "defaults", {})
            config["optimizer_hyperparameters"] = {
                name: value
                for name, value in defaults.items()
                if value is None or isinstance(value, (bool, int, float, str, tuple, list))
            }
            config["parameter_groups"] = self._optimizer_groups()
        if self.loss_fn is not None:
            config["loss"] = self.loss_fn.__class__.__name__
            config["loss_hyperparameters"] = {
                name: value
                for name, value in vars(self.loss_fn).items()
                if not name.startswith("_")
                and (value is None or isinstance(value, (bool, int, float, str, tuple, list)))
            }
        if self.scheduler is not None:
            config["scheduler"] = self.scheduler.__class__.__name__
            config["scheduler_hyperparameters"] = {
                name: value
                for name, value in scheduler_configuration(self.scheduler).items()
                if value is None or isinstance(value, (bool, int, float, str, tuple, list))
            }
        config["capture"] = {
            "weights": self.capture_weights,
            "gradients": self.capture_gradients,
            "activations": self.capture_activations,
            "buffers": self.capture_buffers,
            "optimizer_state": self.capture_optimizer_state,
            "max_tensor_elements": self.max_tensor_elements,
            "max_activation_elements": self.max_activation_elements,
        }
        return config

    def close(self) -> None:
        """Remove forward hooks when the recorder is no longer needed."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __enter__(self) -> "TorchTrainingRecorder":
        """Support use as a context manager."""
        return self

    def __exit__(self, *_exc_info) -> None:
        """Clean up activation hooks on context exit."""
        self.close()
