"""Small, lazy PyTorch helpers used by neural-network visualizations."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from .taxonomy import module_formula, module_hyperparameters, module_role


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
    sample = torch.as_tensor(input_sample, **kwargs)
    if parameter is not None and parameter.is_floating_point() and sample.is_floating_point():
        sample = sample.to(dtype=parameter.dtype)
    if sample.ndim == 1:
        return sample.unsqueeze(0)
    first_leaf = next(_leaf_modules(model), (None, None))[1]
    convolution_rank = {
        "Conv1d": 1,
        "ConvTranspose1d": 1,
        "Conv2d": 2,
        "ConvTranspose2d": 2,
        "Conv3d": 3,
        "ConvTranspose3d": 3,
    }.get(first_leaf.__class__.__name__ if first_leaf is not None else "")
    if convolution_rank is not None and sample.ndim == convolution_rank + 1:
        return sample.unsqueeze(0)
    return sample


def describe_torch_model(model: Any, input_sample: Any | None = None) -> List[Dict[str, Any]]:
    """Describe leaf modules, parameter counts, and optional observed output shapes."""
    torch = _require_torch()
    layers: List[Dict[str, Any]] = []
    input_shapes: Dict[str, Tuple[int, ...] | None] = {}
    shapes: Dict[str, Tuple[int, ...] | None] = {}
    hooks = []

    if input_sample is not None:
        sample = _as_model_input(model, input_sample)

        def capture(name: str):
            def hook(_module, inputs, output):
                input_shapes[name] = _shape_of(inputs)
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

    mathematical_index = 0
    for index, (name, module) in enumerate(_leaf_modules(model)):
        role = module_role(module.__class__.__name__)
        if role == "learnable":
            mathematical_index += 1
        formula_index = max(mathematical_index, 1)
        own_parameters = list(module.parameters(recurse=False))
        parameter_shapes = {
            parameter_name: tuple(parameter.shape)
            for parameter_name, parameter in module.named_parameters(recurse=False)
        }
        parameter_count = sum(parameter.numel() for parameter in own_parameters)
        units = None
        if hasattr(module, "out_features"):
            units = int(module.out_features)
        elif hasattr(module, "out_channels"):
            units = int(module.out_channels)
        layers.append(
            {
                "index": index,
                "math_index": formula_index,
                "name": name,
                "type": module.__class__.__name__,
                "units": units,
                "parameters": parameter_count,
                "trainable_parameters": sum(
                    parameter.numel() for parameter in own_parameters if parameter.requires_grad
                ),
                "parameter_shapes": parameter_shapes,
                "input_shape": input_shapes.get(name),
                "output_shape": shapes.get(name),
                "formula": module_formula(module, formula_index),
                "hyperparameters": module_hyperparameters(module),
                "role": role,
            }
        )
    if not layers:
        raise ValueError("The PyTorch model has no leaf modules to visualize.")
    return layers


def run_torch_forward(
    model: Any,
    input_sample: Any,
    parameter_values: Dict[str, np.ndarray] | None = None,
    buffer_values: Dict[str, np.ndarray] | None = None,
) -> Tuple[Any, "OrderedDict[str, Dict[str, np.ndarray]]"]:
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
    originals: Dict[str, Any] = {}
    original_buffers: Dict[str, Any] = {}
    if parameter_values:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name not in parameter_values:
                    continue
                originals[name] = parameter.detach().clone()
                replacement = torch.as_tensor(parameter_values[name], device=parameter.device, dtype=parameter.dtype)
                if replacement.shape == parameter.shape:
                    parameter.copy_(replacement)
    if buffer_values:
        with torch.no_grad():
            for name, buffer in model.named_buffers():
                if name not in buffer_values:
                    continue
                original_buffers[name] = buffer.detach().clone()
                replacement = torch.as_tensor(buffer_values[name], device=buffer.device, dtype=buffer.dtype)
                if replacement.shape == buffer.shape:
                    buffer.copy_(replacement)
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            output = model(sample)
    finally:
        model.train(was_training)
        if originals:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in originals:
                        parameter.copy_(originals[name])
        if original_buffers:
            with torch.no_grad():
                for name, buffer in model.named_buffers():
                    if name in original_buffers:
                        buffer.copy_(original_buffers[name])
        for hook in hooks:
            hook.remove()
    return output, records
