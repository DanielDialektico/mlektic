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
    "GroupNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "LayerNorm",
    "Linear",
    "MultiheadAttention",
    "RNN",
    "RNNCell",
    "LSTM",
    "LSTMCell",
    "GRU",
    "GRUCell",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
}

RESHAPE_TYPES = {
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "Flatten",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
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
    if module_type in {"MultiheadAttention", "TransformerEncoderLayer", "TransformerDecoderLayer"}:
        return "attention"
    if module_type in {"RNN", "RNNCell", "LSTM", "LSTMCell", "GRU", "GRUCell"}:
        return "recurrent"
    if module_type == "Embedding":
        return "embedding"
    if "Norm" in module_type:
        return "normalization"
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
    if name.startswith("Dropout"):
        return (
            rf"\mathbf{{a}}^{{({index})}}\leftarrow"
            rf"\frac{{\mathbf{{m}}\odot\mathbf{{a}}^{{({index})}}}}{{1-p}},"
            rf"\quad m_i\sim\operatorname{{Bernoulli}}(1-p)"
        )
    if name.startswith("BatchNorm"):
        return (
            rf"a_i^{{({index})}}=\gamma_i"
            rf"\frac{{z_i^{{({index})}}-\mu_i}}{{\sqrt{{\sigma_i^2+\varepsilon}}}}+\beta_i"
        )
    if name.startswith("InstanceNorm"):
        return (
            rf"a_{{n,c}}^{{({index})}}=\gamma_c"
            rf"\frac{{z_{{n,c}}^{{({index})}}-\mu_{{n,c}}}}{{\sqrt{{\sigma_{{n,c}}^2+\varepsilon}}}}+\beta_c"
        )
    if name == "GroupNorm":
        return (
            rf"a_i^{{({index})}}=\gamma_i"
            rf"\frac{{z_i^{{({index})}}-\mu_{{g(i)}}}}{{\sqrt{{\sigma_{{g(i)}}^2+\varepsilon}}}}+\beta_i"
        )
    if name == "Embedding":
        return rf"\mathbf{{a}}^{{({index})}}_t=E[x_t],\quad E\in\mathbb{{R}}^{{V\times d}}"
    if name in {"RNN", "RNNCell"}:
        return r"\mathbf{h}_t=\phi(W_{ih}\mathbf{x}_t+W_{hh}\mathbf{h}_{t-1}+\mathbf{b})"
    if name in {"GRU", "GRUCell"}:
        return (
            r"\mathbf{z}_t=\sigma(W_z\mathbf{x}_t+U_z\mathbf{h}_{t-1}),\quad"
            r"\mathbf{h}_t=(1-\mathbf{z}_t)\odot\mathbf{n}_t+\mathbf{z}_t\odot\mathbf{h}_{t-1}"
        )
    if name in {"LSTM", "LSTMCell"}:
        return (
            r"(\mathbf{i}_t,\mathbf{f}_t,\mathbf{g}_t,\mathbf{o}_t)="
            r"(\sigma,\sigma,\tanh,\sigma)(W\mathbf{x}_t+U\mathbf{h}_{t-1}+\mathbf{b}),\quad"
            r"\mathbf{c}_t=\mathbf{f}_t\odot\mathbf{c}_{t-1}+\mathbf{i}_t\odot\mathbf{g}_t"
        )
    if name == "MultiheadAttention":
        return (
            r"\operatorname{MHA}(Q,K,V)=\operatorname{Concat}(head_1,\ldots,head_h)W^O,\quad "
            r"head_i=\operatorname{softmax}(QW_i^Q(KW_i^K)^\mathsf{T}/\sqrt{d_k})VW_i^V"
        )
    if name.startswith("TransformerEncoder"):
        return (
            r"\mathbf{y}=\operatorname{Norm}(\mathbf{x}+"
            r"\operatorname{FFN}(\operatorname{Norm}(\mathbf{x}+\operatorname{MHA}(\mathbf{x}))))"
        )
    if name.startswith("TransformerDecoder"):
        return r"\mathbf{y}=\operatorname{DecoderBlock}(\mathbf{x},\mathbf{memory})"
    if name.startswith("ConvTranspose"):
        return (
            rf"\mathbf{{z}}^{{({index})}}=\Theta^{{({index})}}\star^\mathsf{{T}}"
            rf"\mathbf{{a}}^{{({previous})}}+\boldsymbol{{\theta}}_0^{{({index})}}"
        )
    if name == "Conv3d":
        return (
            rf"z_{{c,d,i,j}}^{{({index})}}=\sum_{{k,q,r,s}}\theta_{{c,k,q,r,s}}^{{({index})}}"
            rf"a_{{k,d+q,i+r,j+s}}^{{({previous})}}+\theta_{{0,c}}^{{({index})}}"
        )
    if name.startswith("AvgPool") or name.startswith("AdaptiveAvgPool"):
        return rf"a_u^{{({index})}}=\frac{{1}}{{|\mathcal{{K}}_u|}}\sum_{{v\in\mathcal{{K}}_u}}z_v^{{({index})}}"
    if name == "MaxPool3d":
        return rf"a_{{c,d,i,j}}^{{({index})}}=\max_{{(q,r,s)\in\mathcal{{K}}}}z_{{c,d+q,i+r,j+s}}^{{({index})}}"
    if name == "Unflatten":
        return rf"\mathbf{{a}}^{{({index})}}=\operatorname{{unvec}}(\mathbf{{z}}^{{({index})}})"
    formulas = {
        "Linear": (
            rf"\mathbf{{z}}^{{({index})}}=\Theta^{{({index})}}"
            rf"\mathbf{{a}}^{{({previous})}}+\boldsymbol{{\theta}}_0^{{({index})}}"
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
        "Flatten": rf"\mathbf{{a}}^{{({index})}}=\operatorname{{vec}}(\mathbf{{z}}^{{({index})}})",
        "LayerNorm": (
            rf"\mathbf{{a}}^{{({index})}}=\boldsymbol{{\gamma}}\odot"
            rf"\frac{{\mathbf{{z}}^{{({index})}}-\mu}}{{\sqrt{{\sigma^2+\varepsilon}}}}"
            rf"+\boldsymbol{{\beta}}"
        ),
        "Conv1d": (
            rf"z_{{c,t}}^{{({index})}}=\sum_{{k,r}}\theta_{{c,k,r}}^{{({index})}}"
            rf"a_{{k,t+r}}^{{({previous})}}+\theta_{{0,c}}^{{({index})}}"
        ),
        "Conv2d": (
            rf"z_{{c,i,j}}^{{({index})}}=\sum_{{k,r,s}}\theta_{{c,k,r,s}}^{{({index})}}"
            rf"a_{{k,i+r,j+s}}^{{({previous})}}+\theta_{{0,c}}^{{({index})}}"
        ),
        "MaxPool1d": rf"a_{{c,t}}^{{({index})}}=\max_{{r\in\mathcal{{K}}}}z_{{c,t+r}}^{{({index})}}",
        "MaxPool2d": rf"a_{{c,i,j}}^{{({index})}}=\max_{{(r,s)\in\mathcal{{K}}}}z_{{c,i+r,j+s}}^{{({index})}}",
    }
    return formulas.get(name, rf"\mathbf{{a}}^{{({index})}}=\mathcal{{M}}_{{{index}}}(\mathbf{{a}}^{{({previous})}})")


def parameter_definition(parameter_name: str, shape: Sequence[int], layer_index: int) -> str:
    """Explain the role and dimensionality of one learnable tensor."""
    dims = shape_tex(shape, drop_batch=False)
    if parameter_name == "weight" and len(shape) == 2:
        return rf"\Theta^{{({layer_index})}}\in\mathbb{{R}}^{{{dims}}}\;\text{{maps inputs to output neurons}}"
    if parameter_name == "weight" and len(shape) >= 3:
        return rf"\Theta^{{({layer_index})}}\in\mathbb{{R}}^{{{dims}}}\;\text{{contains convolution kernels}}"
    if parameter_name == "bias":
        return (
            rf"\boldsymbol{{\theta}}_0^{{({layer_index})}}\in\mathbb{{R}}^{{{dims}}}"
            r"\;\text{shifts the pre-activation}"
        )
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
        "input_size",
        "hidden_size",
        "num_layers",
        "nonlinearity",
        "batch_first",
        "dropout",
        "bidirectional",
        "proj_size",
        "embed_dim",
        "num_heads",
        "head_dim",
        "kdim",
        "vdim",
        "add_bias_kv",
        "add_zero_attn",
        "dim_feedforward",
        "norm_first",
        "p",
        "negative_slope",
        "alpha",
        "approximate",
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
        "padding_mode",
        "output_padding",
        "return_indices",
        "count_include_pad",
        "divisor_override",
        "track_running_stats",
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
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
    attention = getattr(module, "self_attn", None)
    if attention is not None:
        for name in ("embed_dim", "num_heads", "head_dim", "batch_first", "dropout"):
            value = getattr(attention, name, None)
            if value is None or isinstance(value, (bool, int, float, str, tuple, list)):
                result.setdefault(name, value)
    feedforward = getattr(module, "linear1", None)
    if feedforward is not None and hasattr(feedforward, "out_features"):
        result.setdefault("dim_feedforward", int(feedforward.out_features))
    dropout_layer = getattr(module, "dropout1", None)
    if dropout_layer is not None and hasattr(dropout_layer, "p"):
        result.setdefault("dropout", float(dropout_layer.p))
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
        affine = rf"\Theta^{{({layer})}}{expression}+\boldsymbol{{\theta}}_0^{{({layer})}}"
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
