"""Framework-neutral intermediate representation for neural execution graphs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class TensorSpec:
    """Compact, serializable description of one tensor flowing through a graph."""

    shape: Tuple[Any, ...] | None = None
    dtype: str | None = None
    device: str | None = None
    requires_grad: bool | None = None


@dataclass(frozen=True)
class ParameterSpec:
    """Metadata for one parameter or persistent buffer owned by a graph node."""

    name: str
    shape: Tuple[int, ...]
    dtype: str
    trainable: bool
    kind: str = "parameter"


@dataclass(frozen=True)
class NeuralNode:
    """One executed module, tensor operation, input, output, or summary block."""

    id: str
    name: str
    op: str
    target: str
    role: str
    label: str
    formula: str
    math_status: str
    module_path: str | None = None
    module_type: str | None = None
    call_index: int = 1
    inputs: Tuple[TensorSpec, ...] = ()
    outputs: Tuple[TensorSpec, ...] = ()
    parameters: Tuple[ParameterSpec, ...] = ()
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NeuralEdge:
    """Directed tensor-flow relationship between two graph nodes."""

    source: str
    target: str
    source_port: int = 0
    target_port: int = 0
    kind: str = "data"
    tensor: TensorSpec | None = None
    label: str | None = None


@dataclass(frozen=True)
class CaptureProvenance:
    """Declare how faithfully a graph represents the observed PyTorch program."""

    backend: str
    kind: str
    model_type: str
    torch_version: str
    exact_for_input: bool
    includes_functional_ops: bool
    includes_dynamic_control_flow: bool
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class NeuralGraph:
    """Versioned neural graph consumed by renderers and pedagogical reports."""

    nodes: Tuple[NeuralNode, ...]
    edges: Tuple[NeuralEdge, ...]
    input_nodes: Tuple[str, ...]
    output_nodes: Tuple[str, ...]
    provenance: CaptureProvenance
    warnings: Tuple[str, ...] = ()
    schema_version: int = 1

    def node_map(self) -> Dict[str, NeuralNode]:
        """Return nodes keyed by stable graph identifier."""
        return {node.id: node for node in self.nodes}

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible representation for reports and plugins."""
        return asdict(self)


__all__ = [
    "CaptureProvenance",
    "NeuralEdge",
    "NeuralGraph",
    "NeuralNode",
    "ParameterSpec",
    "TensorSpec",
]
