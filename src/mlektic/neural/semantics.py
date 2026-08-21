"""Extensible semantic descriptors for PyTorch modules and tensor operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from .taxonomy import module_formula, module_hyperparameters, module_role

FormulaFactory = Callable[[Any, int], str]


@dataclass(frozen=True)
class NeuralSemantic:
    """Pedagogical meaning attached to a captured graph operation."""

    role: str
    label: str
    formula: str
    math_status: str


@dataclass(frozen=True)
class NeuralDescriptor:
    """Reusable descriptor registered for a module type or operation target."""

    role: str
    label: str
    formula: str | FormulaFactory
    math_status: str = "specialized"

    def resolve(self, subject: Any, layer_index: int) -> NeuralSemantic:
        """Resolve a possibly dynamic formula for one captured node."""
        formula = self.formula(subject, layer_index) if callable(self.formula) else self.formula
        return NeuralSemantic(self.role, self.label, formula, self.math_status)


class NeuralDescriptorRegistry:
    """Registry that lets core and third parties add neural semantics safely."""

    def __init__(self) -> None:
        """Create an empty descriptor registry."""
        self._modules: Dict[str, NeuralDescriptor] = {}
        self._operations: Dict[str, NeuralDescriptor] = {}

    def register_module(self, module_type: str, descriptor: NeuralDescriptor, *, replace: bool = False) -> None:
        """Register a descriptor by class name without importing PyTorch eagerly."""
        if module_type in self._modules and not replace:
            raise ValueError(f"A neural descriptor is already registered for module '{module_type}'.")
        self._modules[module_type] = descriptor

    def register_operation(self, target: str, descriptor: NeuralDescriptor, *, replace: bool = False) -> None:
        """Register a descriptor for an FX function or method target."""
        if target in self._operations and not replace:
            raise ValueError(f"A neural descriptor is already registered for operation '{target}'.")
        self._operations[target] = descriptor

    def describe_module(self, module: Any, layer_index: int) -> NeuralSemantic:
        """Return specialized semantics or an explicit generic fallback."""
        module_type = module.__class__.__name__
        descriptor = self._modules.get(module_type)
        if descriptor is not None:
            return descriptor.resolve(module, layer_index)
        return NeuralSemantic(
            role=module_role(module_type),
            label=module_type,
            formula=module_formula(module, layer_index),
            math_status="generic",
        )

    def describe_operation(self, target: str, layer_index: int) -> NeuralSemantic:
        """Return semantics for a tensor operation using tolerant target matching."""
        descriptor = self._operations.get(target)
        if descriptor is None:
            descriptor = next(
                (value for key, value in self._operations.items() if target.endswith(key)),
                None,
            )
        if descriptor is not None:
            return descriptor.resolve(target, layer_index)
        return NeuralSemantic(
            role="operation",
            label=target.rsplit(".", 1)[-1],
            formula=rf"\mathbf{{a}}^{{({layer_index})}}=\mathcal{{O}}_{{{layer_index}}}(\cdot)",
            math_status="generic",
        )


def _module_formula(subject: Any, layer_index: int) -> str:
    return module_formula(subject, layer_index)


def _build_default_registry() -> NeuralDescriptorRegistry:
    registry = NeuralDescriptorRegistry()
    specialized_modules = {
        "Linear": "learnable",
        "Conv1d": "learnable",
        "Conv2d": "learnable",
        "Conv3d": "learnable",
        "ConvTranspose1d": "learnable",
        "ConvTranspose2d": "learnable",
        "ConvTranspose3d": "learnable",
        "BatchNorm1d": "normalization",
        "BatchNorm2d": "normalization",
        "BatchNorm3d": "normalization",
        "LayerNorm": "normalization",
        "GroupNorm": "normalization",
        "InstanceNorm1d": "normalization",
        "InstanceNorm2d": "normalization",
        "InstanceNorm3d": "normalization",
        "Embedding": "embedding",
        "RNN": "recurrent",
        "RNNCell": "recurrent",
        "LSTM": "recurrent",
        "LSTMCell": "recurrent",
        "GRU": "recurrent",
        "GRUCell": "recurrent",
        "MultiheadAttention": "attention",
        "TransformerEncoderLayer": "attention",
        "TransformerDecoderLayer": "attention",
        "ReLU": "activation",
        "Sigmoid": "activation",
        "Tanh": "activation",
        "GELU": "activation",
        "LeakyReLU": "activation",
        "Softmax": "activation",
        "Dropout": "regularization",
        "Dropout1d": "regularization",
        "Dropout2d": "regularization",
        "Dropout3d": "regularization",
        "Flatten": "reshape",
        "Unflatten": "reshape",
        "MaxPool1d": "pooling",
        "MaxPool2d": "pooling",
        "MaxPool3d": "pooling",
        "AvgPool1d": "pooling",
        "AvgPool2d": "pooling",
        "AvgPool3d": "pooling",
        "AdaptiveAvgPool1d": "pooling",
        "AdaptiveAvgPool2d": "pooling",
        "AdaptiveAvgPool3d": "pooling",
    }
    for module_type, role in specialized_modules.items():
        registry.register_module(
            module_type,
            NeuralDescriptor(role=role, label=module_type, formula=_module_formula),
        )
    for module_type in {
        "ELU",
        "Hardshrink",
        "Hardsigmoid",
        "Hardswish",
        "LogSigmoid",
        "Mish",
        "PReLU",
        "ReLU6",
        "SELU",
        "SiLU",
        "Softplus",
        "Softsign",
    }:
        registry.register_module(
            module_type,
            NeuralDescriptor(
                role="activation",
                label=module_type,
                formula=_module_formula,
                math_status="generic",
            ),
        )

    operations = {
        "add": NeuralDescriptor("merge", "Add", r"\mathbf{y}=\mathbf{x}_1+\mathbf{x}_2"),
        "sub": NeuralDescriptor("merge", "Subtract", r"\mathbf{y}=\mathbf{x}_1-\mathbf{x}_2"),
        "cat": NeuralDescriptor(
            "merge",
            "Concatenate",
            r"\mathbf{y}=\operatorname{concat}(\mathbf{x}_1,\ldots,\mathbf{x}_m)",
        ),
        "stack": NeuralDescriptor(
            "merge",
            "Stack",
            r"\mathbf{y}=\operatorname{stack}(\mathbf{x}_1,\ldots,\mathbf{x}_m)",
        ),
        "mul": NeuralDescriptor("operation", "Multiply", r"\mathbf{y}=\mathbf{x}_1\odot\mathbf{x}_2"),
        "truediv": NeuralDescriptor("operation", "Divide", r"\mathbf{y}=\mathbf{x}_1\oslash\mathbf{x}_2"),
        "matmul": NeuralDescriptor("operation", "Matrix multiply", r"\mathbf{y}=\mathbf{A}\mathbf{x}"),
        "linear": NeuralDescriptor(
            "learnable",
            "Linear",
            r"\mathbf{z}=\Theta\mathbf{x}+\boldsymbol{\theta}_0",
        ),
        "relu": NeuralDescriptor("activation", "ReLU", r"\mathbf{y}=\max(0,\mathbf{x})"),
        "sigmoid": NeuralDescriptor(
            "activation",
            "Sigmoid",
            r"\mathbf{y}=\sigma(\mathbf{x})=\frac{1}{1+e^{-\mathbf{x}}}",
        ),
        "tanh": NeuralDescriptor("activation", "Tanh", r"\mathbf{y}=\tanh(\mathbf{x})"),
        "softmax": NeuralDescriptor(
            "activation",
            "Softmax",
            r"y_i=\frac{e^{x_i}}{\sum_j e^{x_j}}",
        ),
        "flatten": NeuralDescriptor("reshape", "Flatten", r"\mathbf{y}=\operatorname{vec}(\mathbf{x})"),
        "reshape": NeuralDescriptor("reshape", "Reshape", r"\mathbf{y}=\operatorname{reshape}(\mathbf{x})"),
        "view": NeuralDescriptor("reshape", "View", r"\mathbf{y}=\operatorname{reshape}(\mathbf{x})"),
        "permute": NeuralDescriptor("reshape", "Permute", r"\mathbf{y}=\operatorname{permute}(\mathbf{x})"),
        "transpose": NeuralDescriptor("reshape", "Transpose", r"\mathbf{y}=\mathbf{x}^{\mathsf{T}}"),
        "mean": NeuralDescriptor("reduction", "Mean", r"\mathbf{y}=\operatorname{mean}(\mathbf{x})"),
        "sum": NeuralDescriptor("reduction", "Sum", r"\mathbf{y}=\sum_i x_i"),
        "getitem": NeuralDescriptor("reshape", "Select", r"\mathbf{y}=\operatorname{select}(\mathbf{x})"),
    }
    for target, descriptor in operations.items():
        registry.register_operation(target, descriptor)
    return registry


NEURAL_DESCRIPTORS = _build_default_registry()


def register_neural_descriptor(
    key: str,
    *,
    role: str,
    label: str,
    formula: str | FormulaFactory,
    kind: str = "module",
    math_status: str = "specialized",
    replace: bool = False,
) -> None:
    """Register a custom module or operation descriptor in the public registry."""
    descriptor = NeuralDescriptor(role, label, formula, math_status)
    if kind == "module":
        NEURAL_DESCRIPTORS.register_module(key, descriptor, replace=replace)
    elif kind == "operation":
        NEURAL_DESCRIPTORS.register_operation(key, descriptor, replace=replace)
    else:
        raise ValueError("kind must be 'module' or 'operation'.")


def semantic_hyperparameters(module: Any) -> Dict[str, Any]:
    """Expose the shared stable hyperparameter extraction contract."""
    return module_hyperparameters(module)


__all__ = [
    "NEURAL_DESCRIPTORS",
    "NeuralDescriptor",
    "NeuralDescriptorRegistry",
    "NeuralSemantic",
    "register_neural_descriptor",
    "semantic_hyperparameters",
]
