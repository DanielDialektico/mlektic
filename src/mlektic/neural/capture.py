"""Capture adapters that translate PyTorch execution into :mod:`graph_ir`."""

from __future__ import annotations

import inspect
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np

from .graph_ir import (
    CaptureProvenance,
    NeuralEdge,
    NeuralGraph,
    NeuralNode,
    ParameterSpec,
    TensorSpec,
)
from .introspection import _require_torch
from .semantics import NEURAL_DESCRIPTORS, semantic_hyperparameters


def _walk_values(value: Any, path: str = "value") -> Iterable[Tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk_values(item, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _walk_values(item, f"{path}.{index}")
    else:
        yield path, value


def _model_device_dtype(model: Any) -> Tuple[Any | None, Any | None]:
    parameter = next(model.parameters(), None)
    if parameter is None:
        return None, None
    return parameter.device, parameter.dtype if parameter.is_floating_point() else None


def _prepare_value(value: Any, *, device: Any | None, floating_dtype: Any | None) -> Any:
    torch = _require_torch()
    if isinstance(value, Mapping):
        return value.__class__(
            (key, _prepare_value(item, device=device, floating_dtype=floating_dtype))
            for key, item in value.items()
        )
    if isinstance(value, tuple):
        return tuple(_prepare_value(item, device=device, floating_dtype=floating_dtype) for item in value)
    if isinstance(value, list):
        return [_prepare_value(item, device=device, floating_dtype=floating_dtype) for item in value]
    if isinstance(value, np.ndarray):
        value = torch.as_tensor(value)
    if isinstance(value, torch.Tensor):
        kwargs: Dict[str, Any] = {}
        if device is not None:
            kwargs["device"] = device
        if value.is_floating_point() and floating_dtype is not None:
            kwargs["dtype"] = floating_dtype
        return value.to(**kwargs) if kwargs else value
    return value


def _prepare_call(model: Any, inputs: Any, input_kwargs: Mapping[str, Any] | None) -> Tuple[tuple, dict]:
    device, floating_dtype = _model_device_dtype(model)
    positional = inputs if isinstance(inputs, tuple) else (inputs,)
    args = tuple(
        _prepare_value(value, device=device, floating_dtype=floating_dtype)
        for value in positional
    )
    kwargs = {
        key: _prepare_value(value, device=device, floating_dtype=floating_dtype)
        for key, value in dict(input_kwargs or {}).items()
    }
    return args, kwargs


def _tensor_spec(value: Any) -> TensorSpec | None:
    torch = _require_torch()
    if not isinstance(value, torch.Tensor):
        return None
    return TensorSpec(
        shape=tuple(int(dimension) for dimension in value.shape),
        dtype=str(value.dtype).replace("torch.", ""),
        device=str(value.device),
        requires_grad=bool(value.requires_grad),
    )


def _tensor_specs(value: Any) -> Tuple[TensorSpec, ...]:
    return tuple(
        spec
        for _path, item in _walk_values(value)
        if (spec := _tensor_spec(item)) is not None
    )


def _parameter_specs(module: Any) -> Tuple[ParameterSpec, ...]:
    specs: List[ParameterSpec] = []
    for name, parameter in module.named_parameters(recurse=False):
        specs.append(
            ParameterSpec(
                name=name,
                shape=tuple(int(value) for value in parameter.shape),
                dtype=str(parameter.dtype).replace("torch.", ""),
                trainable=bool(parameter.requires_grad),
            )
        )
    for name, buffer in module.named_buffers(recurse=False):
        specs.append(
            ParameterSpec(
                name=name,
                shape=tuple(int(value) for value in buffer.shape),
                dtype=str(buffer.dtype).replace("torch.", ""),
                trainable=False,
                kind="buffer",
            )
        )
    return tuple(specs)


def _target_name(target: Any) -> str:
    if isinstance(target, str):
        return target
    module = getattr(target, "__module__", "")
    name = getattr(target, "__qualname__", getattr(target, "__name__", str(target)))
    return f"{module}.{name}" if module else name


def _meta_specs(meta: Any) -> Tuple[TensorSpec, ...]:
    if meta is None:
        return ()
    if hasattr(meta, "shape") and hasattr(meta, "dtype"):
        shape = tuple(int(value) if isinstance(value, int) else str(value) for value in meta.shape)
        return (
            TensorSpec(
                shape=shape,
                dtype=str(meta.dtype).replace("torch.", ""),
                device=None,
                requires_grad=bool(getattr(meta, "requires_grad", False)),
            ),
        )
    if isinstance(meta, Mapping):
        return tuple(spec for value in meta.values() for spec in _meta_specs(value))
    if isinstance(meta, (tuple, list)):
        return tuple(spec for value in meta for spec in _meta_specs(value))
    return ()


def _fx_dependencies(value: Any) -> Iterable[Any]:
    torch = _require_torch()
    node_type = torch.fx.Node
    if isinstance(value, node_type):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _fx_dependencies(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _fx_dependencies(item)


def _capture_fx(model: Any, args: tuple, kwargs: dict) -> NeuralGraph:
    torch = _require_torch()
    from torch.fx import symbolic_trace
    from torch.fx.passes.shape_prop import ShapeProp

    was_training = model.training
    try:
        model.eval()
        traced = symbolic_trace(model)
        signature = inspect.signature(traced.forward)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        propagation_values: List[Any] = []
        for name, parameter in signature.parameters.items():
            value = bound.arguments[name]
            if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                propagation_values.extend(value)
            elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
                propagation_values.extend(value.values())
            else:
                propagation_values.append(value)
        ShapeProp(traced).propagate(*propagation_values)
    finally:
        model.train(was_training)

    nodes: List[NeuralNode] = []
    edges: List[NeuralEdge] = []
    input_nodes: List[str] = []
    output_nodes: List[str] = []
    mathematical_index = 0
    call_counts: Dict[str, int] = defaultdict(int)

    for fx_node in traced.graph.nodes:
        target = _target_name(fx_node.target)
        outputs = _meta_specs(fx_node.meta.get("tensor_meta"))
        if fx_node.op == "placeholder":
            semantic = ("input", str(fx_node.target), r"\mathbf{x}", "exact")
            input_nodes.append(fx_node.name)
            module = None
        elif fx_node.op == "output":
            semantic = ("output", "Output", r"\hat{\mathbf{y}}=f_\theta(\mathbf{x})", "exact")
            output_nodes.append(fx_node.name)
            module = None
        elif fx_node.op == "get_attr":
            semantic = ("parameter", str(fx_node.target), r"\theta\in\operatorname{state}(f_\theta)", "exact")
            module = None
        elif fx_node.op == "call_module":
            module = traced.get_submodule(str(fx_node.target))
            mathematical_index += 1
            resolved = NEURAL_DESCRIPTORS.describe_module(module, mathematical_index)
            semantic = (resolved.role, resolved.label, resolved.formula, resolved.math_status)
        else:
            module = None
            mathematical_index += 1
            resolved = NEURAL_DESCRIPTORS.describe_operation(target, mathematical_index)
            semantic = (resolved.role, resolved.label, resolved.formula, resolved.math_status)

        call_key = str(fx_node.target)
        call_counts[call_key] += 1
        dependencies = [*_fx_dependencies(fx_node.args), *_fx_dependencies(fx_node.kwargs)]
        input_specs = tuple(
            spec
            for dependency in dependencies
            for spec in _meta_specs(dependency.meta.get("tensor_meta"))
        )
        parameters = _parameter_specs(module) if module is not None else ()
        hyperparameters = semantic_hyperparameters(module) if module is not None else {}
        nodes.append(
            NeuralNode(
                id=fx_node.name,
                name=target,
                op=fx_node.op,
                target=target,
                role=semantic[0],
                label=semantic[1],
                formula=semantic[2],
                math_status=semantic[3],
                module_path=str(fx_node.target) if fx_node.op == "call_module" else None,
                module_type=module.__class__.__name__ if module is not None else None,
                call_index=call_counts[call_key],
                inputs=input_specs,
                outputs=outputs,
                parameters=parameters,
                hyperparameters=hyperparameters,
                metadata={"fx_name": fx_node.name},
            )
        )
        for target_port, dependency in enumerate(dependencies):
            dependency_specs = _meta_specs(dependency.meta.get("tensor_meta"))
            edges.append(
                NeuralEdge(
                    source=dependency.name,
                    target=fx_node.name,
                    target_port=target_port,
                    tensor=dependency_specs[0] if dependency_specs else None,
                )
            )

    return NeuralGraph(
        nodes=tuple(nodes),
        edges=tuple(edges),
        input_nodes=tuple(input_nodes),
        output_nodes=tuple(output_nodes),
        provenance=CaptureProvenance(
            backend="torch.fx",
            kind="symbolic-static",
            model_type=model.__class__.__name__,
            torch_version=str(torch.__version__),
            exact_for_input=True,
            includes_functional_ops=True,
            includes_dynamic_control_flow=False,
            notes=("Static Python control flow is specialized into the captured graph.",),
        ),
    )


def _leaf_or_root_modules(model: Any) -> List[Tuple[str, Any]]:
    if _is_compound_primitive(model):
        return [("<root>", model)]
    leaves = [
        (name, module)
        for name, module in model.named_modules()
        if name and not list(module.children())
    ]
    return leaves or [("<root>", model)]


def _is_compound_primitive(model: Any) -> bool:
    """Return whether a PyTorch primitive should remain one semantic block.

    A few public ``torch.nn`` layers own implementation-detail children.  For
    example, ``MultiheadAttention`` exposes ``out_proj`` even though learners
    should see the attention operation as one layer.  User-defined composite
    models are deliberately excluded so FX can retain their branches and
    functional operations.
    """
    return model.__class__.__module__.startswith("torch.nn") and model.__class__.__name__ in {
        "MultiheadAttention",
        "TransformerEncoderLayer",
        "TransformerDecoderLayer",
        "TransformerEncoder",
        "TransformerDecoder",
        "Transformer",
    }


def _capture_hooks(model: Any, args: tuple, kwargs: dict, fx_error: Exception | None = None) -> NeuralGraph:
    torch = _require_torch()
    nodes: List[NeuralNode] = []
    edges: List[NeuralEdge] = []
    input_nodes: List[str] = []
    producer_by_tensor: Dict[int, Tuple[str, int]] = {}
    call_counts: Dict[str, int] = defaultdict(int)
    implicit_count = 0

    for index, (path, value) in enumerate((*_walk_values(args, "args"), *_walk_values(kwargs, "kwargs"))):
        spec = _tensor_spec(value)
        if spec is None:
            continue
        node_id = f"input_{index}"
        input_nodes.append(node_id)
        nodes.append(
            NeuralNode(
                id=node_id,
                name=path,
                op="input",
                target=path,
                role="input",
                label=path,
                formula=r"\mathbf{x}",
                math_status="exact",
                outputs=(spec,),
            )
        )
        producer_by_tensor[id(value)] = (node_id, 0)

    hooks = []

    def capture(path: str):
        def hook(module: Any, module_args: tuple, module_kwargs: dict, output: Any) -> None:
            nonlocal implicit_count
            call_counts[path] += 1
            call_index = call_counts[path]
            node_id = f"module_{len(nodes)}"
            semantic = NEURAL_DESCRIPTORS.describe_module(module, len(nodes) + 1)
            input_values = [
                value
                for _name, value in (*_walk_values(module_args, "args"), *_walk_values(module_kwargs, "kwargs"))
                if _tensor_spec(value) is not None
            ]
            output_values = [
                value for _name, value in _walk_values(output, "output") if _tensor_spec(value) is not None
            ]
            for target_port, input_value in enumerate(input_values):
                producer = producer_by_tensor.get(id(input_value))
                if producer is None:
                    implicit_count += 1
                    implicit_id = f"implicit_{implicit_count}"
                    spec = _tensor_spec(input_value)
                    nodes.append(
                        NeuralNode(
                            id=implicit_id,
                            name="uncaptured tensor operation",
                            op="implicit",
                            target="uncaptured",
                            role="operation",
                            label="Uncaptured operation",
                            formula=r"\mathbf{y}=\mathcal{O}(\mathbf{x})",
                            math_status="unavailable",
                            outputs=(spec,) if spec is not None else (),
                        )
                    )
                    producer = (implicit_id, 0)
                edges.append(
                    NeuralEdge(
                        source=producer[0],
                        target=node_id,
                        source_port=producer[1],
                        target_port=target_port,
                        tensor=_tensor_spec(input_value),
                    )
                )
            nodes.append(
                NeuralNode(
                    id=node_id,
                    name=path,
                    op="call_module",
                    target=path,
                    role=semantic.role,
                    label=semantic.label,
                    formula=semantic.formula,
                    math_status=semantic.math_status,
                    module_path=None if path == "<root>" else path,
                    module_type=module.__class__.__name__,
                    call_index=call_index,
                    inputs=tuple(spec for value in input_values if (spec := _tensor_spec(value)) is not None),
                    outputs=tuple(spec for value in output_values if (spec := _tensor_spec(value)) is not None),
                    parameters=_parameter_specs(module),
                    hyperparameters=semantic_hyperparameters(module),
                )
            )
            for output_port, output_value in enumerate(output_values):
                producer_by_tensor[id(output_value)] = (node_id, output_port)

        return hook

    for path, module in _leaf_or_root_modules(model):
        hooks.append(module.register_forward_hook(capture(path), with_kwargs=True))

    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            output = model(*args, **kwargs)
    finally:
        model.train(was_training)
        for registered_hook in hooks:
            registered_hook.remove()

    output_id = "output"
    output_specs = _tensor_specs(output)
    nodes.append(
        NeuralNode(
            id=output_id,
            name="output",
            op="output",
            target="output",
            role="output",
            label="Output",
            formula=r"\hat{\mathbf{y}}=f_\theta(\mathbf{x})",
            math_status="exact",
            inputs=output_specs,
            outputs=output_specs,
        )
    )
    for target_port, (_path, value) in enumerate(_walk_values(output, "output")):
        spec = _tensor_spec(value)
        if spec is None:
            continue
        producer = producer_by_tensor.get(id(value))
        if producer is not None:
            edges.append(
                NeuralEdge(
                    source=producer[0],
                    target=output_id,
                    source_port=producer[1],
                    target_port=target_port,
                    tensor=spec,
                )
            )
    notes = [
        "The eager fallback records executed module calls but cannot guarantee capture of functional tensor operations."
    ]
    if fx_error is not None:
        notes.append(f"torch.fx fallback reason: {type(fx_error).__name__}: {fx_error}")
    return NeuralGraph(
        nodes=tuple(nodes),
        edges=tuple(edges),
        input_nodes=tuple(input_nodes),
        output_nodes=(output_id,),
        provenance=CaptureProvenance(
            backend="eager-hooks",
            kind="observed-module-path",
            model_type=model.__class__.__name__,
            torch_version=str(torch.__version__),
            exact_for_input=False,
            includes_functional_ops=False,
            includes_dynamic_control_flow=True,
            notes=tuple(notes),
        ),
        warnings=("Functional operations may appear as explicit uncaptured-operation nodes.",),
    )


def capture_neural_graph(
    model: Any,
    inputs: Any,
    *,
    input_kwargs: Mapping[str, Any] | None = None,
    backend: str = "auto",
) -> NeuralGraph:
    """Capture a versioned graph while preserving input dtype and model state.

    ``backend='auto'`` prefers ``torch.fx`` because it captures functional
    operations and branches. Models with data-dependent Python control flow or
    unsupported tracing behavior fall back to observed eager module calls.
    Tuples are interpreted as positional model arguments; keyword model inputs
    belong in ``input_kwargs``.
    """
    if backend not in {"auto", "fx", "hooks"}:
        raise ValueError("backend must be 'auto', 'fx', or 'hooks'.")
    _require_torch()
    args, kwargs = _prepare_call(model, inputs, input_kwargs)
    is_torch_primitive = model.__class__.__module__.startswith("torch.nn")
    if backend == "hooks" or (
        backend == "auto"
        and (
            (is_torch_primitive and not list(model.children()))
            or _is_compound_primitive(model)
        )
    ):
        return _capture_hooks(model, args, kwargs)
    try:
        return _capture_fx(model, args, kwargs)
    except Exception as exc:
        if backend == "fx":
            raise RuntimeError(f"torch.fx could not capture {model.__class__.__name__}: {exc}") from exc
        return _capture_hooks(model, args, kwargs, fx_error=exc)


__all__ = ["capture_neural_graph"]
