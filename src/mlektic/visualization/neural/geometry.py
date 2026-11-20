r"""Shared replay, notation, and playback helpers for the geometric neural figures.

The decision-surface, representation, training-step, gradient-flow, and Hinton
figures all read a dense chain

.. math::

   \mathbf a^{(0)}=\mathbf x,\quad
   \mathbf z^{(\ell)}=\Theta^{(\ell)}\mathbf a^{(\ell-1)}+\boldsymbol\theta_0^{(\ell)},\quad
   \mathbf a^{(\ell)}=\phi_\ell(\mathbf z^{(\ell)})

with the course notation. Every value they draw comes from an exact forward
pass of the real model with recorded parameters (``run_torch_forward``).

Efficiency rules shared by these builders:

* static marks live in ``fig.data`` only; a frame lists just the traces it
  changes through ``traces=[...]``;
* raster fields are quantized to integers in ``[0, 255]`` against one global
  range, so frames stay small without base64 payloads that some renderers do
  not decode inside frames;
* positions that move are tweened by Plotly transitions instead of extra frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from ...neural.introspection import _leaf_modules, _require_torch, run_torch_forward
from ._style import NEURAL_COLORS, animation_button_style
from .math_format import buffer_snapshot, parameter_snapshot

# Elementwise activations a dense replay can name exactly.
ACTIVATION_TEX = {
    "Tanh": r"\tanh",
    "Sigmoid": r"\sigma",
    "ReLU": r"\operatorname{ReLU}",
    "LeakyReLU": r"\operatorname{LeakyReLU}",
    "ELU": r"\operatorname{ELU}",
    "SELU": r"\operatorname{SELU}",
    "GELU": r"\operatorname{GELU}",
    "SiLU": r"\operatorname{SiLU}",
    "Softmax": r"\operatorname{softmax}",
    "LogSoftmax": r"\log\operatorname{softmax}",
    "Identity": r"\operatorname{id}",
}
# The same names as Plotly rich text, for marks that change in every frame
# (text traces animate without MathJax).
ACTIVATION_TEXT = {
    "Tanh": "tanh",
    "Sigmoid": "σ",
    "ReLU": "ReLU",
    "LeakyReLU": "LeakyReLU",
    "ELU": "ELU",
    "SELU": "SELU",
    "GELU": "GELU",
    "SiLU": "SiLU",
    "Softmax": "softmax",
    "LogSoftmax": "log softmax",
    "Identity": "id",
}
_TRANSPARENT = {"Dropout", "Identity", "AlphaDropout", "Flatten"}
# Discrete class colours: binary tasks reuse the signed pair of the graph.
CLASS_COLORS = (
    NEURAL_COLORS["negative"],
    NEURAL_COLORS["positive"],
    NEURAL_COLORS["activation"],
    NEURAL_COLORS["linear"],
    NEURAL_COLORS["regularization"],
    NEURAL_COLORS["weight_max"],
)
CLASS_SYMBOLS = ("circle", "diamond", "square", "triangle-up", "cross", "star")
UNIT_COLORS = (
    NEURAL_COLORS["linear"],
    NEURAL_COLORS["activation"],
    NEURAL_COLORS["regularization"],
    "#9be15d",
    "#ff9fd6",
    "#7ee8fa",
)
SIGNED_SCALE = [[0.0, NEURAL_COLORS["negative"]], [0.5, "#2a2c33"], [1.0, NEURAL_COLORS["positive"]]]
UNIT_SCALE = [[0.0, NEURAL_COLORS["weight_min"]], [0.5, "#2a2c33"], [1.0, NEURAL_COLORS["weight_max"]]]
GRADIENT_SCALE = [[0.0, NEURAL_COLORS["update_negative"]], [0.5, "#3a3d46"], [1.0, NEURAL_COLORS["update_positive"]]]
MAGNITUDE_SCALE = [
    [0.0, "#17181c"],
    [0.3, "#2b2f6b"],
    [0.55, "#6a3d9a"],
    [0.8, "#e0727a"],
    [1.0, "#f6d27a"],
]


@dataclass(frozen=True)
class DenseLayer:
    """One dense layer ``z = Θ a + θ0`` and its elementwise activation."""

    index: int
    linear: str
    activation: str | None
    activation_type: str
    in_features: int
    out_features: int
    weight: str
    bias: str | None

    @property
    def activation_tex(self) -> str:
        """LaTeX name of the activation."""
        return ACTIVATION_TEX.get(self.activation_type, r"\phi_{%d}" % self.index)

    @property
    def activation_text(self) -> str:
        """Plotly rich-text name of the activation."""
        return ACTIVATION_TEXT.get(self.activation_type, f"φ{self.index}")


def dense_chain(model: Any, *, purpose: str) -> List[DenseLayer]:
    """Return the model as a dense chain, or explain why it is not one."""
    leaves = list(_leaf_modules(model))
    layers: List[DenseLayer] = []
    pending: Dict[str, Any] | None = None
    for name, module in leaves:
        kind = module.__class__.__name__
        if kind == "Linear":
            if pending is not None:
                layers.append(_finish(pending, None, "Identity"))
            pending = {"name": name, "module": module}
        elif kind in _TRANSPARENT:
            continue
        elif kind in ACTIVATION_TEX and pending is not None:
            layers.append(_finish(pending, name, kind))
            pending = None
        else:
            raise ValueError(
                f"{purpose} replays a dense chain of Linear layers and elementwise activations; "
                f"module {name!r} ({kind}) is not part of such a chain. Use architecture_mode='blocks' "
                "to inspect the executed graph instead."
            )
    if pending is not None:
        layers.append(_finish(pending, None, "Identity"))
    if not layers:
        raise ValueError(f"{purpose} requires at least one torch.nn.Linear layer.")
    return [
        DenseLayer(index=position + 1, **{k: v for k, v in vars(layer).items() if k != "index"})
        for position, layer in enumerate(layers)
    ]


def _finish(pending: Dict[str, Any], activation: str | None, kind: str) -> DenseLayer:
    module = pending["module"]
    return DenseLayer(
        index=0,
        linear=pending["name"],
        activation=activation,
        activation_type=kind,
        in_features=int(module.in_features),
        out_features=int(module.out_features),
        weight=f"{pending['name']}.weight",
        bias=f"{pending['name']}.bias" if module.bias is not None else None,
    )


def forward_states(
    model: Any,
    chain: Sequence[DenseLayer],
    inputs: Any,
    parameters: Dict[str, np.ndarray] | None,
    buffers: Dict[str, np.ndarray] | None = None,
) -> Dict[str, List[np.ndarray]]:
    """Exact ``z^(ℓ)`` and ``a^(ℓ)`` for every layer, from one real forward pass."""
    output, records = run_torch_forward(model, inputs, parameters, buffers)
    z_values, a_values = [], []
    for layer in chain:
        z = np.asarray(records[layer.linear]["output"], dtype=float)
        a = np.asarray(records[layer.activation]["output"], dtype=float) if layer.activation else z
        z_values.append(z.reshape(z.shape[0], -1))
        a_values.append(a.reshape(a.shape[0], -1))
    final = np.asarray(output.detach().cpu().numpy(), dtype=float)
    return {"z": z_values, "a": a_values, "output": final.reshape(final.shape[0], -1)}


def snapshots(history: Dict[str, Any], index: int) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray] | None]:
    """Parameters and buffers recorded at one checkpoint."""
    buffers = buffer_snapshot(history, index)
    return parameter_snapshot(history, index), (buffers or None)


def recorded_steps(history: Dict[str, Any]) -> np.ndarray:
    """Recorded step numbers, or an error for an empty history."""
    steps = np.asarray(history.get("steps", []), dtype=int)
    if not steps.size:
        raise ValueError("History has no recorded steps.")
    return steps


def select_frames(count: int, max_frames: int | None) -> np.ndarray:
    """Evenly spaced checkpoints that always keep the first and the last one."""
    if max_frames is None or count <= max_frames:
        return np.arange(count, dtype=int)
    if max_frames < 2:
        raise ValueError("max_frames must be at least 2 or None.")
    return np.unique(np.linspace(0, count - 1, max_frames).round().astype(int))


def as_numpy(values: Any) -> np.ndarray:
    """Tensor or array-like to a NumPy array."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def quantize(values: np.ndarray, low: float, high: float) -> List[List[int]]:
    """Integers in ``[0, 255]`` on a fixed range, as nested lists (frame-safe)."""
    span = high - low if high > low else 1.0
    scaled = np.clip(np.rint((np.asarray(values, dtype=float) - low) / span * 255.0), 0, 255).astype(int)
    return scaled.tolist()


def quantized_colorbar(low: float, high: float, *, title: str, ticks: int = 5, dec: int = 2) -> Dict[str, Any]:
    """A colour bar whose labels show real values for a quantized field."""
    values = np.linspace(low, high, ticks)
    return {
        "title": {"text": title, "side": "right"},
        "tickvals": [float(v) for v in np.linspace(0, 255, ticks)],
        "ticktext": [format_number(v, dec) for v in values],
        "thickness": 12,
        "len": 0.55,
        "outlinewidth": 0,
        "tickfont": {"size": 10, "color": NEURAL_COLORS["muted"]},
    }


def format_number(value: float, dec: int = 4) -> str:
    """Rounded value without a negative zero (course convention)."""
    text = f"{float(value):.{dec}f}"
    if text.startswith("-") and float(text) == 0:
        text = text[1:]
    return text


def rounded(values: Any, dec: int = 4) -> List[Any]:
    """Frame-safe nested lists with bounded precision (``None`` for NaN)."""
    array = np.round(np.asarray(values, dtype=float), dec)
    return (
        [None if np.isnan(v) else float(v) for v in array.ravel()]
        if array.ndim == 1
        else [[None if np.isnan(v) else float(v) for v in row] for row in array]
    )


def playback_controls(
    names: Sequence[str],
    labels: Sequence[str],
    *,
    frame_duration: int,
    transition: int = 0,
    redraw: bool = True,
    prefix: str = "Recorded step: ",
    y: float = -0.04,
    easing: str = "cubic-in-out",
) -> Dict[str, Any]:
    """Play/Pause and slider in the visual language of the neural figures."""
    animate = {
        "frame": {"duration": frame_duration, "redraw": redraw},
        "transition": {"duration": transition, "easing": easing},
        "fromcurrent": True,
        "mode": "immediate",
    }
    step_args = {
        "mode": "immediate",
        "frame": {"duration": 0, "redraw": redraw},
        "transition": {"duration": transition, "easing": easing},
    }
    return {
        "updatemenus": [
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": y,
                "xanchor": "left",
                "yanchor": "top",
                "pad": {"t": 0, "r": 8},
                **animation_button_style(),
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [None, animate]},
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}],
                    },
                ],
            }
        ],
        "sliders": [
            {
                "active": 0,
                "x": 0.13,
                "len": 0.86,
                "y": y + 0.005,
                "yanchor": "top",
                "pad": {"t": 0},
                "currentvalue": {"prefix": prefix, "font": {"size": 12, "color": NEURAL_COLORS["muted"]}},
                "font": {"size": 10, "color": NEURAL_COLORS["muted"]},
                "transition": {"duration": transition, "easing": easing},
                "steps": [
                    {"label": label, "method": "animate", "args": [[name], step_args]}
                    for name, label in zip(names, labels)
                ],
            }
        ],
    }


def recorded_metric(history: Dict[str, Any], name: str) -> np.ndarray | None:
    """One recorded metric series, if present."""
    values = history.get("metrics", {}).get(name)
    if values is None or not len(values):
        return None
    return np.asarray(values, dtype=float)


def output_semantics(chain: Sequence[DenseLayer], history: Dict[str, Any], targets: np.ndarray | None) -> str:
    """``probability``, ``logit``, ``multiclass``, or ``regression``."""
    last = chain[-1]
    loss = str(history.get("training_config", {}).get("loss", ""))
    if last.out_features > 1:
        if last.activation_type in {"Softmax", "LogSoftmax"} or "CrossEntropy" in loss or "NLL" in loss:
            return "multiclass"
        return "regression"
    if last.activation_type == "Sigmoid":
        return "probability"
    if "BCEWithLogits" in loss or "SoftMargin" in loss:
        return "logit"
    if targets is not None:
        unique = np.unique(np.asarray(targets).ravel())
        if unique.size == 2 and set(unique.tolist()) <= {0, 1, 0.0, 1.0} and "MSE" not in loss and "L1" not in loss:
            return "logit"
    return "regression"


def require_torch_tensor(values: Any, model: Any):
    """Values as a tensor with the model's dtype and device."""
    torch = _require_torch()
    parameter = next(model.parameters(), None)
    tensor = torch.as_tensor(as_numpy(values))
    if parameter is not None and tensor.is_floating_point():
        tensor = tensor.to(dtype=parameter.dtype, device=parameter.device)
    return tensor


def loss_label(history: Dict[str, Any], loss_fn: Any | None = None) -> str:
    """Plain name of the training objective; BCE is called Log-loss (course)."""
    name = (
        loss_fn.__class__.__name__ if loss_fn is not None else str(history.get("training_config", {}).get("loss", ""))
    )
    if name in {"BCELoss", "BCEWithLogitsLoss"}:
        return f"Log-loss ({name})"
    return name or "loss"


__all__ = [
    "ACTIVATION_TEX",
    "ACTIVATION_TEXT",
    "CLASS_COLORS",
    "CLASS_SYMBOLS",
    "DenseLayer",
    "GRADIENT_SCALE",
    "MAGNITUDE_SCALE",
    "SIGNED_SCALE",
    "UNIT_COLORS",
    "UNIT_SCALE",
    "as_numpy",
    "dense_chain",
    "format_number",
    "forward_states",
    "loss_label",
    "output_semantics",
    "playback_controls",
    "quantize",
    "quantized_colorbar",
    "recorded_metric",
    "recorded_steps",
    "rounded",
    "select_frames",
    "snapshots",
]
