"""Mathematical taxonomy and configuration helpers for PyTorch modules."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

ACTIVATION_TYPES = {
    "ELU",
    "GELU",
    "Hardshrink",
    "Hardsigmoid",
    "Hardswish",
    "LeakyReLU",
    "LogSigmoid",
    "Mish",
    "PReLU",
    "ReLU",
    "ReLU6",
    "SELU",
    "Sigmoid",
    "SiLU",
    "Softmax",
    "Softplus",
    "Softsign",
    "Tanh",
}

LEARNABLE_TYPES = {
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "Embedding",
    "LayerNorm",
    "Linear",
}

RESHAPE_TYPES = {
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AvgPool1d",
    "AvgPool2d",
    "Flatten",
    "MaxPool1d",
    "MaxPool2d",
    "Unflatten",
}


def shape_tex(shape: Sequence[int] | None, *, drop_batch: bool = True) -> str:
    """Format a tensor shape as a compact LaTeX product."""
    if not shape:
        return "?"
    dims = tuple(shape[1:]) if drop_batch and len(shape) > 1 else tuple(shape)
    return r" \times ".join(str(value) for value in dims) if dims else "1"


def module_role(module_type: str) -> str:
    """Return the visual and pedagogical role of a module type."""
    if module_type in ACTIVATION_TYPES:
        return "activation"
    if module_type in LEARNABLE_TYPES or module_type.startswith("Conv"):
        return "learnable"
    if module_type in RESHAPE_TYPES or "Pool" in module_type:
        return "reshape"
    if module_type in {"Dropout", "Dropout1d", "Dropout2d", "Dropout3d"}:
        return "regularization"
    return "operation"


def module_formula(module: Any, layer_index: int | None = None) -> str:
    """Return a rigorous but concise mathematical definition for a module."""
    name = module.__class__.__name__
    index = r"\ell" if layer_index is None else str(layer_index)
    previous = r"\ell-1" if layer_index is None else str(layer_index - 1)
    formulas = {
        "Linear": (
            rf"\mathbf{{z}}^{{({index})}}=W^{{({index})}}"
            rf"\mathbf{{a}}^{{({previous})}}+\mathbf{{b}}^{{({index})}}"
        ),
        "ReLU": rf"\mathbf{{a}}^{{({index})}}=\max(0,\mathbf{{z}}^{{({index})}})",
        "Sigmoid": (
            rf"\mathbf{{a}}^{{({index})}}=\sigma(\mathbf{{z}}^{{({index})}})"
            rf"=\frac{{1}}{{1+e^{{-\mathbf{{z}}^{{({index})}}}}}}"
        ),
        "Tanh": rf"\mathbf{{a}}^{{({index})}}=\tanh(\mathbf{{z}}^{{({index})}})",
        "GELU": rf"\mathbf{{a}}^{{({index})}}=\operatorname{{GELU}}(\mathbf{{z}}^{{({index})}})",
        "LeakyReLU": rf"a_i^{{({index})}}=\max(z_i^{{({index})}},\alpha z_i^{{({index})}})",
        "Softmax": rf"a_i^{{({index})}}=\frac{{e^{{z_i^{{({index})}}}}}}{{\sum_j e^{{z_j^{{({index})}}}}}}",
        "Dropout": (
            rf"\mathbf{{a}}^{{({index})}}\leftarrow"
            rf"\frac{{\mathbf{{m}}\odot\mathbf{{a}}^{{({index})}}}}{{1-p}},"
            rf"\quad m_i\sim\operatorname{{Bernoulli}}(1-p)"
        ),
        "Flatten": rf"\mathbf{{a}}^{{({index})}}=\operatorname{{vec}}(\mathbf{{z}}^{{({index})}})",
        "BatchNorm1d": (
            rf"a_i^{{({index})}}=\gamma_i"
            rf"\frac{{z_i^{{({index})}}-\mu_i}}{{\sqrt{{\sigma_i^2+\varepsilon}}}}+\beta_i"
        ),
        "BatchNorm2d": (
            rf"a_c^{{({index})}}=\gamma_c"
            rf"\frac{{z_c^{{({index})}}-\mu_c}}{{\sqrt{{\sigma_c^2+\varepsilon}}}}+\beta_c"
        ),
        "LayerNorm": (
            rf"\mathbf{{a}}^{{({index})}}=\boldsymbol{{\gamma}}\odot"
            rf"\frac{{\mathbf{{z}}^{{({index})}}-\mu}}{{\sqrt{{\sigma^2+\varepsilon}}}}"
            rf"+\boldsymbol{{\beta}}"
        ),
        "Conv1d": (
            rf"z_{{c,t}}^{{({index})}}=\sum_{{k,r}}K_{{c,k,r}}^{{({index})}}"
            rf"a_{{k,t+r}}^{{({previous})}}+b_c^{{({index})}}"
        ),
        "Conv2d": (
            rf"z_{{c,i,j}}^{{({index})}}=\sum_{{k,r,s}}K_{{c,k,r,s}}^{{({index})}}"
            rf"a_{{k,i+r,j+s}}^{{({previous})}}+b_c^{{({index})}}"
        ),
        "MaxPool1d": rf"a_{{c,t}}^{{({index})}}=\max_{{r\in\mathcal{{K}}}}z_{{c,t+r}}^{{({index})}}",
        "MaxPool2d": rf"a_{{c,i,j}}^{{({index})}}=\max_{{(r,s)\in\mathcal{{K}}}}z_{{c,i+r,j+s}}^{{({index})}}",
    }
    return formulas.get(name, rf"\mathbf{{a}}^{{({index})}}=\mathcal{{M}}_{{{index}}}(\mathbf{{a}}^{{({previous})}})")


def parameter_definition(parameter_name: str, shape: Sequence[int], layer_index: int) -> str:
    """Explain the role and dimensionality of one learnable tensor."""
    dims = shape_tex(shape, drop_batch=False)
    if parameter_name == "weight" and len(shape) == 2:
        return rf"W^{{({layer_index})}}\in\mathbb{{R}}^{{{dims}}}\;\text{{maps inputs to output neurons}}"
    if parameter_name == "weight" and len(shape) >= 3:
        return rf"K^{{({layer_index})}}\in\mathbb{{R}}^{{{dims}}}\;\text{{contains convolution kernels}}"
    if parameter_name == "bias":
        return rf"\mathbf{{b}}^{{({layer_index})}}\in\mathbb{{R}}^{{{dims}}}\;\text{{shifts the pre-activation}}"
    return rf"\theta_{{\mathrm{{{parameter_name}}}}}^{{({layer_index})}}\in\mathbb{{R}}^{{{dims}}}"


def module_hyperparameters(module: Any) -> Dict[str, Any]:
    """Extract stable public configuration values without serializing tensors."""
    preferred = (
        "in_features",
        "out_features",
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        "output_size",
        "normalized_shape",
        "num_features",
        "num_embeddings",
        "embedding_dim",
        "p",
        "eps",
        "momentum",
        "affine",
        "elementwise_affine",
        "bias",
        "dim",
        "start_dim",
        "end_dim",
        "inplace",
        "ceil_mode",
    )
    result: Dict[str, Any] = {}
    for name in preferred:
        if not hasattr(module, name):
            continue
        value = getattr(module, name)
        if name == "bias" and value is not None and hasattr(value, "shape"):
            result[name] = True
        elif value is None or isinstance(value, (bool, int, float, str, tuple, list)):
            result[name] = value
    return result


def format_hyperparameters(values: Dict[str, Any], *, limit: int = 5) -> str:
    """Render configuration values in a compact human-readable form."""
    items = list(values.items())
    shown = [f"{name}={value}" for name, value in items[:limit]]
    if len(items) > limit:
        shown.append("...")
    return ", ".join(shown) if shown else "no configurable hyperparameters"


def activation_symbol(module_type: str) -> str:
    """Return a short function symbol for composed-network notation."""
    symbols = {
        "ELU": r"\operatorname{ELU}",
        "GELU": r"\operatorname{GELU}",
        "LeakyReLU": r"\operatorname{LReLU}",
        "ReLU": r"\operatorname{ReLU}",
        "Sigmoid": r"\sigma",
        "SiLU": r"\operatorname{SiLU}",
        "Softmax": r"\operatorname{softmax}",
        "Tanh": r"\tanh",
    }
    return symbols.get(module_type, r"\operatorname{id}")


def dense_stages(model: Any) -> List[Dict[str, Any]]:
    """Group each dense transformation with its immediately following activation."""
    modules = list(_named_leaf_modules(model))
    stages: List[Dict[str, Any]] = []
    index = 0
    while index < len(modules):
        name, module = modules[index]
        if module.__class__.__name__ != "Linear":
            index += 1
            continue
        activation_name = None
        activation_type = None
        if index + 1 < len(modules) and modules[index + 1][1].__class__.__name__ in ACTIVATION_TYPES:
            activation_name, activation = modules[index + 1]
            activation_type = activation.__class__.__name__
        stages.append(
            {
                "index": len(stages) + 1,
                "name": name,
                "weight_name": f"{name}.weight",
                "bias_name": f"{name}.bias" if module.bias is not None else None,
                "in_features": int(module.in_features),
                "out_features": int(module.out_features),
                "activation_name": activation_name,
                "activation_type": activation_type,
            }
        )
        index += 2 if activation_name is not None else 1
    return stages


def composed_dense_function(stages: Sequence[Dict[str, Any]]) -> str:
    """Build one-line nested-function notation for a dense network."""
    if not stages:
        return r"\hat{\mathbf{y}}=\mathcal{N}_{\theta}(\mathbf{x})"
    expression = r"\mathbf{x}"
    for stage in stages:
        layer = stage["index"]
        affine = rf"W^{{({layer})}}{expression}+\mathbf{{b}}^{{({layer})}}"
        symbol = activation_symbol(stage.get("activation_type") or "")
        expression = rf"{symbol}\!\left({affine}\right)" if symbol != r"\operatorname{id}" else affine
    return rf"\hat{{\mathbf{{y}}}}=f_\theta(\mathbf{{x}})={expression}"


def select_with_ellipsis(items: Sequence[Any], limit: int) -> List[Any | None]:
    """Select both ends of a long sequence and use ``None`` as an ellipsis."""
    if len(items) <= limit:
        return list(items)
    before = max(1, limit // 2)
    after = max(1, limit - before - 1)
    return [*items[:before], None, *items[-after:]]


def _named_leaf_modules(model: Any) -> Iterable[tuple[str, Any]]:
    for name, module in model.named_modules():
        if name and not list(module.children()):
            yield name, module


__all__ = [
    "ACTIVATION_TYPES",
    "activation_symbol",
    "composed_dense_function",
    "dense_stages",
    "format_hyperparameters",
    "module_formula",
    "module_hyperparameters",
    "module_role",
    "parameter_definition",
    "select_with_ellipsis",
    "shape_tex",
]
