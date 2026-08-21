"""PyTorch-specific inspection and training-history helpers."""

from .capture import capture_neural_graph
from .graph_ir import CaptureProvenance, NeuralEdge, NeuralGraph, NeuralNode, ParameterSpec, TensorSpec
from .hyperparameters import (
    HyperparameterComponent,
    HyperparameterItem,
    describe_hyperparameter_contract,
)
from .introspection import describe_torch_model, run_torch_forward
from .recorder import TorchTrainingRecorder
from .report import build_nn_math_report, display_nn_math_report, export_nn_math_report
from .semantics import NeuralDescriptor, NeuralDescriptorRegistry, register_neural_descriptor

__all__ = [
    "TorchTrainingRecorder",
    "CaptureProvenance",
    "HyperparameterComponent",
    "HyperparameterItem",
    "NeuralDescriptor",
    "NeuralDescriptorRegistry",
    "NeuralEdge",
    "NeuralGraph",
    "NeuralNode",
    "ParameterSpec",
    "TensorSpec",
    "build_nn_math_report",
    "capture_neural_graph",
    "describe_torch_model",
    "describe_hyperparameter_contract",
    "display_nn_math_report",
    "export_nn_math_report",
    "run_torch_forward",
    "register_neural_descriptor",
]
