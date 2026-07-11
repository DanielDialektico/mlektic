"""Small, lazy PyTorch helpers used by neural-network visualizations."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


def _require_torch():
    """Import PyTorch only when a neural-network feature is used."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch support requires the optional dependency. Install it with "
            '`pip install "mlektic[torch]"`.'
        ) from exc
    return torch


def _leaf_modules(model: Any) -> Iterable[Tuple[str, Any]]:
    """Yield named leaf modules, preserving PyTorch execution order where possible."""
    for name, module in model.named_modules():
        if name and not list(module.children()):
            yield name, module


def _shape_of(value: Any) -> Tuple[int, ...] | None:
    """Return a tensor shape when *value* is a tensor-like output."""
    if isinstance(value, (tuple, list)):
        value = value[0] if value else None
    return tuple(value.shape) if hasattr(value, "shape") else None


def _as_model_input(model: Any, input_sample: Any):
    """Convert one array-like sample to the device and dtype of *model*."""
    torch = _require_torch()
    parameter = next(model.parameters(), None)
    kwargs: Dict[str, Any] = {}
    if parameter is not None:
        kwargs["device"] = parameter.device
        if parameter.is_floating_point():
            kwargs["dtype"] = parameter.dtype
    sample = torch.as_tensor(input_sample, **kwargs)
    return sample.unsqueeze(0) if sample.ndim == 1 else sample


def _module_formula(module: Any) -> str:
    """Return the mathematical rule most useful to a learner for a module."""
    name = module.__class__.__name__
    formulas = {
        "Linear": r"z = Wa + b",
        "ReLU": r"a = \max(0, z)",
        "Sigmoid": r"a = \sigma(z)",
        "Tanh": r"a = \tanh(z)",
        "GELU": r"a = \operatorname{GELU}(z)",
        "LeakyReLU": r"a = \max(z, \alpha z)",
        "Softmax": r"a_i = \frac{e^{z_i}}{\sum_j e^{z_j}}",
        "Dropout": r"a = \operatorname{Dropout}(z)",
        "Flatten": r"a = \operatorname{vec}(z)",
    }
    return formulas.get(name, name)


def describe_torch_model(model: Any, input_sample: Any | None = None) -> List[Dict[str, Any]]:
    """Describe leaf modules, parameter counts, and optional observed output shapes."""
    torch = _require_torch()
    layers: List[Dict[str, Any]] = []
    shapes: Dict[str, Tuple[int, ...] | None] = {}
    hooks = []

    if input_sample is not None:
        sample = _as_model_input(model, input_sample)

        def capture(name: str):
            def hook(_module, _inputs, output):
                shapes[name] = _shape_of(output)

            return hook

        hooks = [module.register_forward_hook(capture(name)) for name, module in _leaf_modules(model)]
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad():
                model(sample)
        finally:
            model.train(was_training)
            for hook in hooks:
                hook.remove()

    for index, (name, module) in enumerate(_leaf_modules(model)):
        own_parameters = list(module.parameters(recurse=False))
        parameter_count = sum(parameter.numel() for parameter in own_parameters)
        units = None
        if hasattr(module, "out_features"):
            units = int(module.out_features)
        elif hasattr(module, "out_channels"):
            units = int(module.out_channels)
        layers.append(
            {
                "index": index,
                "name": name,
                "type": module.__class__.__name__,
                "units": units,
                "parameters": parameter_count,
                "output_shape": shapes.get(name),
                "formula": _module_formula(module),
            }
        )
    if not layers:
        raise ValueError("The PyTorch model has no leaf modules to visualize.")
    return layers


def run_torch_forward(model: Any, input_sample: Any) -> Tuple[Any, "OrderedDict[str, Dict[str, np.ndarray]]"]:
    """Run one inference while retaining inputs and outputs of leaf modules."""
    torch = _require_torch()
    sample = _as_model_input(model, input_sample)
    records: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()

    def capture(name: str):
        def hook(_module, inputs, output):
            input_tensor = inputs[0] if inputs else None
            output_tensor = output[0] if isinstance(output, (tuple, list)) else output
            if input_tensor is not None and hasattr(input_tensor, "detach") and hasattr(output_tensor, "detach"):
                records[name] = {
                    "input": input_tensor.detach().cpu().numpy(),
                    "output": output_tensor.detach().cpu().numpy(),
                }

        return hook

    hooks = [module.register_forward_hook(capture(name)) for name, module in _leaf_modules(model)]
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            output = model(sample)
    finally:
        model.train(was_training)
        for hook in hooks:
            hook.remove()
    return output, records
