"""Training recorder for PyTorch models used in educational visualizations."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from .introspection import _leaf_modules, _require_torch


class TorchTrainingRecorder:
    """Capture compact, frame-aligned training data from a PyTorch model.

    Call :meth:`record` after ``loss.backward()`` and before ``optimizer.step()``
    to retain the gradient responsible for each parameter update.
    """

    def __init__(
        self,
        model: Any,
        *,
        record_every: int = 1,
        capture_weights: bool = True,
        capture_gradients: bool = True,
        capture_activations: bool = True,
        max_tensor_elements: int = 4096,
    ) -> None:
        """Create a recorder attached to *model* without changing its behavior."""
        if record_every < 1:
            raise ValueError("record_every must be at least 1.")
        if max_tensor_elements < 1:
            raise ValueError("max_tensor_elements must be at least 1.")
        _require_torch()
        self.model = model
        self.record_every = record_every
        self.capture_weights = capture_weights
        self.capture_gradients = capture_gradients
        self.capture_activations = capture_activations
        self.max_tensor_elements = max_tensor_elements
        self.steps: List[int] = []
        self.loss: List[float] = []
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.parameters: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.gradients: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.parameter_norms: Dict[str, List[float]] = defaultdict(list)
        self.gradient_norms: Dict[str, List[float]] = defaultdict(list)
        self.activations: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: {"mean": [], "std": [], "min": [], "max": []}
        )
        self._hooks = []
        self._latest_activations: Dict[str, Dict[str, float]] = {}
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

        return hook

    def record(self, step: int, *, loss: Any | None = None, metrics: Dict[str, Any] | None = None) -> bool:
        """Save one frame and return whether it was retained by ``record_every``."""
        if step % self.record_every:
            return False
        self.steps.append(int(step))
        self.loss.append(self._scalar(loss))
        provided = metrics or {}
        for name in set(self.metrics) | set(provided):
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
            if (
                self.capture_gradients
                and gradient_values is not None
                and gradient_values.size <= self.max_tensor_elements
            ):
                self.gradients[name].append(gradient_values)

        if self.capture_activations:
            for name, summary in self._latest_activations.items():
                for statistic, value in summary.items():
                    self.activations[name][statistic].append(value)
        return True

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
            "parameter_norms": {
                name: np.asarray(values, dtype=float) for name, values in self.parameter_norms.items()
            },
            "gradient_norms": {
                name: np.asarray(values, dtype=float) for name, values in self.gradient_norms.items()
            },
            "activations": {
                name: {statistic: np.asarray(values, dtype=float) for statistic, values in summary.items()}
                for name, summary in self.activations.items()
            },
        }

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
