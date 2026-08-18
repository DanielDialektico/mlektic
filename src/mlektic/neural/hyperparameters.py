"""PyTorch-aligned hyperparameter semantics for educational figures.

The contract in this module is deliberately instance based: it describes the
effective configuration of the supplied modules, parameter groups, objective,
and scheduler.  It is not a static catalogue that pretends every PyTorch class
is active in one training run.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .introspection import _leaf_modules, describe_torch_model


@dataclass(frozen=True)
class HyperparameterItem:
    """One effective PyTorch argument and its mathematical interpretation."""

    name: str
    value: Any
    mathematics: str
    definition: str
    mathematical: bool = True
    definition_status: str = "specialized"


@dataclass(frozen=True)
class HyperparameterComponent:
    """One module or training component with complete detected configuration."""

    scope: str
    label: str
    type_name: str
    operation: str
    items: Sequence[HyperparameterItem]
    source_url: str


_IGNORED_CONSTRUCTOR_ARGUMENTS = {
    "self",
    "params",
    "optimizer",
    "device",
    "dtype",
    "factory_kwargs",
    "size_average",
    "reduce",
}


def display_value(value: Any) -> str:
    """Return a stable, bounded representation without serializing tensors."""
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        shape = tuple(int(item) for item in value.shape)
        return f"Tensor(shape={shape}, dtype={value.dtype})"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def serializable_configuration(subject: Any, *, extra_skip: Iterable[str] = ()) -> Dict[str, Any]:
    """Read effective constructor arguments exposed by a PyTorch object.

    Deprecated aliases are excluded because PyTorch resolves them into the
    effective public setting (for example ``reduction`` on loss modules).
    Tensor-valued settings remain tensor objects here and are summarized only
    at presentation time.
    """
    skip = _IGNORED_CONSTRUCTOR_ARGUMENTS | set(extra_skip)
    try:
        parameters = inspect.signature(subject.__class__.__init__).parameters
    except (TypeError, ValueError):
        parameters = {}
    configuration: Dict[str, Any] = {}
    for name, parameter in parameters.items():
        if name in skip or parameter.kind in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}:
            continue
        if not hasattr(subject, name):
            continue
        value = getattr(subject, name)
        if name == "bias" and hasattr(subject, "_parameters"):
            value = value is not None
        configuration[name] = value
    return configuration


def scheduler_configuration(scheduler: Any) -> Dict[str, Any]:
    """Read constructor-level scheduler configuration, excluding mutable caches."""
    return serializable_configuration(scheduler, extra_skip={"verbose"})


def _latex_value(value: Any) -> str:
    if value is None:
        return r"\mathrm{None}"
    if isinstance(value, bool):
        return rf"\mathrm{{{value}}}"
    if isinstance(value, str):
        escaped = value.replace("_", r"\_")
        return rf"\mathtt{{{escaped}}}"
    if isinstance(value, (tuple, list)):
        return r"\left(" + ",".join(_latex_value(item) for item in value) + r"\right)"
    if hasattr(value, "shape"):
        shape = tuple(int(item) for item in value.shape)
        return rf"\mathrm{{Tensor}}\,{_latex_value(shape)}"
    return str(value)


def _reduction_formula(value: Any) -> str:
    normalized = str(value).lower()
    if normalized == "sum":
        return r"\mathcal{L}=\sum_{n=1}^{N}\ell_n"
    if normalized == "none":
        return r"\mathcal{L}=(\ell_1,\ldots,\ell_N)"
    return r"\mathcal{L}=\frac{1}{N}\sum_{n=1}^{N}\ell_n"


def _definition(scope: str, type_name: str, name: str, value: Any) -> HyperparameterItem:
    """Resolve a concise PyTorch argument definition used by the figure."""
    value_tex = _latex_value(value)
    module_definitions = {
        "in_features": (r"W\in\mathbb{R}^{d_{out}\times d_{in}}", "Input-vector dimension."),
        "out_features": (r"z\in\mathbb{R}^{d_{out}}", "Number of produced coordinates."),
        "in_channels": (r"x\in\mathbb{R}^{N\times C_{in}\times\cdots}", "Input channel count."),
        "out_channels": (r"y\in\mathbb{R}^{N\times C_{out}\times\cdots}", "Produced channel count."),
        "kernel_size": (r"k=(k_1,\ldots,k_D)", "Spatial extent of each convolution or pooling window."),
        "stride": (r"o=\left\lfloor\frac{i+2p-d(k-1)-1}{s}+1\right\rfloor", "Step between adjacent windows."),
        "padding": (r"x\mapsto\operatorname{pad}_{p}(x)", "Implicit boundary padding."),
        "dilation": (r"k_{eff}=d(k-1)+1", "Spacing between kernel elements."),
        "groups": (r"C_{in}/g\;\longrightarrow\;C_{out}/g", "Number of blocked channel-connection groups."),
        "output_padding": (r"o\mapsto o+p_{out}", "Resolves transposed-convolution output-shape ambiguity."),
        "padding_mode": (r"x\mapsto\operatorname{pad}_{mode}(x)", "Rule used to construct padded boundary values."),
        "output_size": (r"y\in\mathbb{R}^{\mathrm{output\_size}}", "Requested adaptive output extent."),
        "bias": (r"z=Wx+b\;\text{if enabled, else }z=Wx", "Whether an additive learnable offset is present."),
        "num_features": (r"\gamma,\beta\in\mathbb{R}^{C}", "Feature or channel count normalized independently."),
        "normalized_shape": (
            r"\mu,\sigma^2\;\text{over the final normalized dimensions}",
            "Dimensions used to compute normalization statistics.",
        ),
        "eps": (r"\hat{x}=\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}}", "Positive numerical-stability term."),
        "affine": (r"y=\gamma\hat{x}+\beta", "Enables learnable scale and shift."),
        "elementwise_affine": (r"y=\gamma\odot\hat{x}+\beta", "Enables elementwise learnable scale and shift."),
        "track_running_stats": (
            r"(\hat\mu,\hat\sigma^2)\;\text{stored for evaluation}",
            "Whether running statistics are retained.",
        ),
        "p": (
            r"m_i\sim\operatorname{Bernoulli}(1-p),\quad y_i=\frac{m_i x_i}{1-p}",
            "Training-time dropout probability.",
        ),
        "inplace": (
            r"f(x)\;\text{is unchanged; output storage may alias input}",
            "Execution/storage option, not a different mathematical function.",
        ),
        "dim": (
            r"y_i=\frac{e^{x_i}}{\sum_{j\in\mathrm{dim}}e^{x_j}}",
            "Tensor dimension over which the operation is applied.",
        ),
        "start_dim": (r"y=\operatorname{flatten}(x;d_{start},d_{end})", "First dimension included in flattening."),
        "end_dim": (r"y=\operatorname{flatten}(x;d_{start},d_{end})", "Last dimension included in flattening."),
        "negative_slope": (r"\operatorname{LReLU}(x)=\max(x,\alpha x)", "Slope used for negative inputs."),
        "alpha": (
            r"\operatorname{ELU}(x)=x\;\text{if }x>0,\;\alpha(e^x-1)\;\text{otherwise}",
            "Negative-region ELU scale.",
        ),
        "approximate": (
            r"\operatorname{GELU}(x)=x\Phi(x)\;\text{or its selected approximation}",
            "Exact or tanh-based GELU evaluation.",
        ),
        "input_size": (r"x_t\in\mathbb{R}^{d_{in}}", "Per-time-step input dimension."),
        "hidden_size": (r"h_t\in\mathbb{R}^{d_h}", "Recurrent hidden-state dimension."),
        "num_layers": (r"h_t^{(\ell)},\quad \ell=1,\ldots,L", "Number of stacked recurrent layers."),
        "nonlinearity": (r"h_t=\phi(W_{ih}x_t+W_{hh}h_{t-1}+b)", "Recurrent activation function."),
        "batch_first": (
            r"x\in\mathbb{R}^{N\times T\times D}\;\text{when enabled}",
            "Places batch before sequence in input/output shapes.",
        ),
        "dropout": (
            r"h^{(\ell)}\mapsto m\odot h^{(\ell)}/(1-p)",
            "Dropout applied between stacked recurrent/transformer layers.",
        ),
        "bidirectional": (r"h_t=[\overrightarrow{h_t};\overleftarrow{h_t}]", "Adds a reverse temporal direction."),
        "proj_size": (
            r"h_t\mapsto W_{hr}h_t\in\mathbb{R}^{d_{proj}}",
            "Optional recurrent hidden-state projection size.",
        ),
        "embed_dim": (r"Q,K,V\in\mathbb{R}^{\cdots\times d_{model}}", "Total attention embedding dimension."),
        "num_heads": (r"d_{head}=d_{model}/h", "Number of parallel attention heads."),
        "head_dim": (r"d_{head}=d_{model}/h", "Per-head attention dimension."),
        "kdim": (r"K\in\mathbb{R}^{\cdots\times d_k}", "Key feature dimension."),
        "vdim": (r"V\in\mathbb{R}^{\cdots\times d_v}", "Value feature dimension."),
        "add_bias_kv": (r"K\leftarrow[K;b_k],\quad V\leftarrow[V;b_v]", "Appends learned key/value bias vectors."),
        "add_zero_attn": (r"K\leftarrow[K;0],\quad V\leftarrow[V;0]", "Appends a zero key/value token."),
        "dim_feedforward": (
            r"\operatorname{FFN}(x)=W_2\phi(W_1x+b_1)+b_2",
            "Hidden width of the transformer feed-forward block.",
        ),
        "norm_first": (
            r"x\mapsto x+F(\operatorname{Norm}(x))\;\text{when enabled}",
            "Selects pre-norm instead of post-norm ordering.",
        ),
        "num_embeddings": (r"E\in\mathbb{R}^{V\times d_e}", "Number of rows in the embedding table."),
        "embedding_dim": (r"E\in\mathbb{R}^{V\times d_e}", "Embedding-vector dimension."),
        "padding_idx": (
            r"E_{i_{pad}}\;\text{does not receive a gradient update}",
            "Embedding row kept fixed as padding.",
        ),
        "max_norm": (r"\|E_i\|_p\leq M", "Optional maximum norm enforced on accessed embeddings."),
        "norm_type": (r"\|E_i\|_p=(\sum_j|E_{ij}|^p)^{1/p}", "Norm exponent used by embedding renormalization."),
        "scale_grad_by_freq": (
            r"\nabla E_i\leftarrow\nabla E_i/f_i",
            "Scales each embedding gradient by inverse mini-batch frequency.",
        ),
        "sparse": (r"\nabla E\;\text{contains only accessed rows}", "Uses a sparse embedding-weight gradient."),
        "ceil_mode": (
            r"o=\left\lceil\frac{i+2p-d(k-1)-1}{s}+1\right\rceil",
            "Uses ceiling rather than floor for pooling output size.",
        ),
        "return_indices": (
            r"y=(\operatorname{pool}(x),\operatorname{argmax}(x))",
            "Returns pooling indices with values.",
        ),
        "count_include_pad": (
            r"\operatorname{avg}=\frac{\sum x}{\#\text{window including pad}}",
            "Includes padded zeros in average-pooling divisor.",
        ),
        "divisor_override": (
            r"\operatorname{avg}=\frac{\sum x}{d_{override}}",
            "Overrides the average-pooling divisor.",
        ),
    }
    optimizer_definitions = {
        "lr": (r"\theta_t=\theta_{t-1}-\gamma u_t", "Learning-rate step scale."),
        "momentum": (r"v_t=\mu v_{t-1}+(1-\tau)g_t", "Momentum coefficient."),
        "dampening": (r"v_t=\mu v_{t-1}+(1-\tau)g_t", "Dampening applied to the current gradient in SGD momentum."),
        "weight_decay": (
            r"g_t\leftarrow g_t+\lambda\theta_{t-1}",
            "L2 gradient contribution unless the optimizer decouples it.",
        ),
        "nesterov": (r"u_t=g_t+\mu v_t", "Uses the Nesterov look-ahead direction."),
        "betas": (
            r"m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2",
            "First- and second-moment decay coefficients.",
        ),
        "eps": (r"u_t=\widehat m_t/(\sqrt{\widehat v_t}+\varepsilon)", "Optimizer denominator stability term."),
        "amsgrad": (r"\widehat v_t^{max}=\max(\widehat v_{t-1}^{max},v_t)", "Uses the AMSGrad maximum second moment."),
        "maximize": (r"g_t\leftarrow-g_t", "Maximizes rather than minimizes the objective."),
        "decoupled_weight_decay": (
            r"\theta_t\leftarrow(1-\gamma\lambda)\theta_{t-1}-\gamma u_t",
            "Applies AdamW-style decoupled weight decay.",
        ),
        "foreach": (r"\Delta\theta\;\text{is mathematically unchanged}", "Selects a tensor-list implementation."),
        "fused": (
            r"\Delta\theta\;\text{is mathematically unchanged}",
            "Selects a fused implementation when supported.",
        ),
        "capturable": (
            r"\Delta\theta\;\text{is mathematically unchanged}",
            "Makes the step safe for graph capture on supported devices.",
        ),
        "differentiable": (
            r"\nabla\,\operatorname{step}(\theta)\;\text{is retained}",
            "Runs the optimizer step with autograd tracking.",
        ),
    }
    loss_definitions = {
        "reduction": (_reduction_formula(value), "Reduction applied to unreduced sample losses."),
        "weight": (r"\ell_n\leftarrow w_{y_n}\ell_n", "Optional sample/class rescaling tensor."),
        "pos_weight": (r"\ell^+\leftarrow p_c\ell^+", "Positive-class term rescaling for binary logits."),
        "ignore_index": (
            r"\ell_n=0\quad\text{when }y_n=i_{ignore}",
            "Target index omitted from loss and mean denominator.",
        ),
        "label_smoothing": (
            r"q=(1-\varepsilon)e_y+\frac{\varepsilon}{C}\mathbf{1}",
            "Mixes the target with a uniform class distribution.",
        ),
        "beta": (
            r"\ell(r)=\frac{r^2}{2\beta}\;\text{if }|r|<\beta,\;|r|-\frac{\beta}{2}\;\text{otherwise}",
            "Smooth-L1 quadratic-to-linear transition.",
        ),
        "margin": (r"\ell=\max(0,m-y\,s)", "Required separation margin."),
    }
    scheduler_definitions = {
        "step_size": (r"\gamma_t=\gamma_0\eta^{\lfloor t/s\rfloor}", "Number of epochs between StepLR decays."),
        "gamma": (r"\gamma_{new}=\eta\gamma_{old}", "Multiplicative learning-rate decay factor."),
        "last_epoch": (r"t_{last}\;\text{indexes scheduler state}", "Last processed epoch index."),
        "factor": (r"\gamma_{new}=f\gamma_{old}", "Plateau-triggered learning-rate multiplier."),
        "patience": (
            r"\#\{\text{allowed non-improving epochs}\}=P",
            "Plateau tolerance before reducing learning rate.",
        ),
        "threshold": (r"|\Delta metric|>\delta", "Minimum metric change considered significant."),
        "threshold_mode": (r"\delta\;\text{is relative or absolute}", "Rule used to compare metric improvements."),
        "cooldown": (
            r"t\in[t_{reduce},t_{reduce}+C]\Rightarrow\text{no new reduction}",
            "Cooldown epochs after a reduction.",
        ),
        "min_lr": (r"\gamma_t\geq\gamma_{min}", "Lower learning-rate bound."),
        "max_lr": (r"\gamma_t\leq\gamma_{max}", "Upper learning-rate bound."),
        "total_steps": (r"T=\text{total optimization steps in the schedule}", "Total schedule length."),
        "epochs": (r"T=E\,S\;\text{when total_steps is inferred}", "Epoch count used to infer schedule length."),
        "steps_per_epoch": (r"T=E\,S", "Batch steps per epoch used to infer schedule length."),
        "pct_start": (r"T_{rise}=pT", "Fraction of a one-cycle schedule spent increasing learning rate."),
        "anneal_strategy": (r"\gamma_t=\operatorname{anneal}(t)", "Linear or cosine interpolation rule."),
    }

    if scope == "optimizer":
        formulas = optimizer_definitions
    elif scope == "objective":
        formulas = loss_definitions
    elif scope == "scheduler":
        formulas = scheduler_definitions
    else:
        formulas = module_definitions
        if name == "momentum" and "BatchNorm" in type_name:
            formulas = dict(formulas)
            formulas["momentum"] = (
                r"\widehat{x}_{new}=(1-m)\widehat{x}+m x_t",
                "Running-statistic update coefficient; unlike optimizer momentum.",
            )
        elif name == "bias" and "BatchNorm" in type_name:
            formulas = dict(formulas)
            formulas["bias"] = (
                r"y=\gamma\widehat{x}+\beta\;\text{when enabled}",
                "Whether the learnable BatchNorm shift beta is present.",
            )
    if scope == "optimizer" and name == "decoupled_weight_decay":
        if bool(value):
            formulas = dict(formulas)
            formulas[name] = (
                r"\theta_t\leftarrow(1-\gamma\lambda)\theta_{t-1}-\gamma u_t",
                "Applies AdamW-style decoupled weight decay.",
            )
        else:
            formulas = dict(formulas)
            formulas[name] = (
                r"g_t\leftarrow g_t+\lambda\theta_{t-1}",
                "Uses coupled L2 weight decay when lambda is nonzero.",
            )
    if scope == "optimizer" and name == "initial_lr":
        formulas = dict(formulas)
        formulas[name] = (
            r"\gamma_0=\mathtt{initial\_lr}",
            "Base learning rate retained for scheduler evaluation.",
        )
    escaped_name = name.replace("_", r"\_")
    definition_status = "specialized" if name in formulas else "generic"
    mathematics, definition = formulas.get(
        name,
        (
            rf"\mathtt{{{escaped_name}}}={value_tex}",
            "Effective public PyTorch configuration value.",
        ),
    )
    nonmathematical = name in {"inplace", "foreach", "fused", "capturable", "differentiable"}
    return HyperparameterItem(
        name,
        value,
        mathematics,
        definition,
        not nonmathematical,
        definition_status,
    )


def _optimizer_operation(name: str) -> str:
    if name == "SGD":
        return (
            r"g_t=\nabla_\theta\mathcal{L}_t,\quad v_t=\mu v_{t-1}+(1-\tau)g_t,\quad \theta_t=\theta_{t-1}-\gamma v_t"
        )
    if name in {"Adam", "AdamW", "NAdam", "RAdam", "SparseAdam"}:
        return (
            r"m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad "
            r"v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,\quad "
            r"\theta_t=\theta_{t-1}-\gamma\frac{\widehat m_t}"
            r"{\sqrt{\widehat v_t}+\varepsilon}"
        )
    return r"\theta_t=\operatorname{optimizer}(\theta_{t-1},\nabla_\theta\mathcal{L}_t;\mathcal{H})"


def _loss_operation(name: str) -> str:
    formulas = {
        "MSELoss": r"\ell_n=(y_n-\widehat y_n)^2",
        "L1Loss": r"\ell_n=|y_n-\widehat y_n|",
        "BCELoss": r"\ell_n=-[y_n\log p_n+(1-y_n)\log(1-p_n)]",
        "BCEWithLogitsLoss": r"\ell_n=-[y_n\log\sigma(z_n)+(1-y_n)\log(1-\sigma(z_n))]",
        "CrossEntropyLoss": r"\ell_n=-w_{y_n}\log\frac{e^{x_{n,y_n}}}{\sum_{c=1}^{C}e^{x_{n,c}}}",
        "NLLLoss": r"\ell_n=-w_{y_n}x_{n,y_n}",
        "SmoothL1Loss": r"\ell_n=\operatorname{smooth}_{\beta}(y_n-\widehat y_n)",
    }
    return formulas.get(name, r"\mathcal{L}=\operatorname{objective}(\widehat y,y;\mathcal{H})")


def _scheduler_operation(name: str) -> str:
    formulas = {
        "StepLR": r"\gamma_t=\gamma_0\eta^{\lfloor t/s\rfloor}",
        "ExponentialLR": r"\gamma_t=\gamma_0\eta^t",
        "CosineAnnealingLR": r"\gamma_t=\gamma_{min}+\frac{1}{2}(\gamma_{max}-\gamma_{min})(1+\cos(\pi t/T_{max}))",
        "ReduceLROnPlateau": r"\gamma_{new}=f\gamma_{old}\quad\text{after }P\text{ non-improving epochs}",
        "OneCycleLR": r"\gamma_t:\gamma_{initial}\nearrow\gamma_{max}\searrow\gamma_{final}",
    }
    return formulas.get(name, r"\gamma_t=\operatorname{schedule}(t;\mathcal{H})")


def _items(scope: str, type_name: str, configuration: Mapping[str, Any]) -> List[HyperparameterItem]:
    return [_definition(scope, type_name, name, value) for name, value in configuration.items()]


def describe_hyperparameter_contract(
    model: Any,
    *,
    history: Mapping[str, Any] | None = None,
    optimizer: Any | None = None,
    loss_fn: Any | None = None,
    scheduler: Any | None = None,
) -> List[HyperparameterComponent]:
    """Describe every effective hyperparameter detected for the supplied run."""
    components: List[HyperparameterComponent] = []
    leaf_modules = dict(_leaf_modules(model))
    for layer in describe_torch_model(model):
        type_name = str(layer["type"])
        module = leaf_modules.get(str(layer["name"]))
        # Constructor introspection is the completeness baseline.  The
        # taxonomy contributes stable derived attributes for PyTorch classes
        # whose Python signature does not expose every effective setting.
        configuration = serializable_configuration(module) if module is not None else {}
        for name, value in dict(layer["hyperparameters"]).items():
            # Some taxonomy values (for example attention head_dim) are useful
            # derived effective configuration.  A generic tensor named
            # ``bias`` is not an independent constructor argument, so it must
            # not be invented for modules such as BatchNorm.
            if name == "bias" and name not in configuration:
                continue
            configuration.setdefault(name, value)
        components.append(
            HyperparameterComponent(
                "module",
                f"{layer['name']} · {type_name}",
                type_name,
                str(layer["formula"]),
                _items("module", type_name, configuration),
                f"https://docs.pytorch.org/docs/stable/generated/torch.nn.{type_name}.html",
            )
        )

    training = dict((history or {}).get("training_config", {}))
    optimizer_name = optimizer.__class__.__name__ if optimizer is not None else training.get("optimizer")
    if optimizer_name:
        if optimizer is not None:
            groups = [
                {name: value for name, value in group.items() if name != "params"} for group in optimizer.param_groups
            ]
        else:
            recorded_groups = list((history or {}).get("optimizer_groups", []))
            groups = list(recorded_groups[-1]) if recorded_groups else list(training.get("parameter_groups", []))
            if not groups:
                groups = [dict(training.get("optimizer_hyperparameters", {}))]
        for group_index, configuration in enumerate(groups):
            components.append(
                HyperparameterComponent(
                    "optimizer",
                    f"{optimizer_name} · parameter group {group_index}",
                    str(optimizer_name),
                    _optimizer_operation(str(optimizer_name)),
                    _items("optimizer", str(optimizer_name), configuration),
                    f"https://docs.pytorch.org/docs/stable/generated/torch.optim.{optimizer_name}.html",
                )
            )

    loss_name = loss_fn.__class__.__name__ if loss_fn is not None else training.get("loss")
    if loss_name:
        configuration = (
            serializable_configuration(loss_fn)
            if loss_fn is not None
            else dict(training.get("loss_hyperparameters", {}))
        )
        components.append(
            HyperparameterComponent(
                "objective",
                str(loss_name),
                str(loss_name),
                _loss_operation(str(loss_name)),
                _items("objective", str(loss_name), configuration),
                f"https://docs.pytorch.org/docs/stable/generated/torch.nn.{loss_name}.html",
            )
        )

    scheduler_name = scheduler.__class__.__name__ if scheduler is not None else training.get("scheduler")
    if scheduler_name:
        configuration = (
            scheduler_configuration(scheduler)
            if scheduler is not None
            else dict(training.get("scheduler_hyperparameters", {}))
        )
        components.append(
            HyperparameterComponent(
                "scheduler",
                str(scheduler_name),
                str(scheduler_name),
                _scheduler_operation(str(scheduler_name)),
                _items("scheduler", str(scheduler_name), configuration),
                (f"https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.{scheduler_name}.html"),
            )
        )
    return components


__all__ = [
    "HyperparameterComponent",
    "HyperparameterItem",
    "describe_hyperparameter_contract",
    "display_value",
    "scheduler_configuration",
    "serializable_configuration",
]
