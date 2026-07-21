"""Shared visual language for neural-network figures."""

from __future__ import annotations

from typing import Any, Dict

NEURAL_COLORS = {
    "background": "#17181c",
    "panel": "#202126",
    "text": "#f5f7fa",
    "muted": "#aeb4bd",
    "grid": "#353841",
    "input": "#55d6be",
    "linear": "#77b7ff",
    "activation": "#ffb86b",
    "regularization": "#d7a8ff",
    "output": "#ff7d8e",
    "positive": "#55d6be",
    "negative": "#ff7d8e",
    "weight_min": "#5b8ff9",
    "weight_mid": "#4b4e57",
    "weight_max": "#f6bd60",
    "backprop": "#8f2942",
    "button_bg": "#eef1f5",
    "button_text": "#15171b",
}


def animation_button_style() -> Dict[str, Any]:
    """Keep Plotly animation labels readable in normal and hover states."""
    return {
        "bgcolor": NEURAL_COLORS["button_bg"],
        "bordercolor": NEURAL_COLORS["grid"],
        "borderwidth": 1,
        "font": {"color": NEURAL_COLORS["button_text"], "size": 12},
        "showactive": False,
    }


def neural_layout(title: str, *, height: int = 680) -> Dict[str, Any]:
    """Return a restrained, notebook-friendly dark layout."""
    return {
        "template": "plotly_dark",
        "height": height,
        "paper_bgcolor": NEURAL_COLORS["background"],
        "plot_bgcolor": NEURAL_COLORS["background"],
        "font": {"family": "Inter, Arial, sans-serif", "color": NEURAL_COLORS["text"]},
        "title": {"text": title, "x": 0.02, "xanchor": "left", "font": {"size": 22}},
        "margin": {"t": 90, "r": 40, "b": 70, "l": 60},
        "hoverlabel": {"bgcolor": NEURAL_COLORS["panel"], "font": {"color": NEURAL_COLORS["text"]}},
    }


def layer_color(layer_type: str, is_output: bool = False) -> str:
    """Map layer families to stable semantic colors."""
    if is_output:
        return NEURAL_COLORS["output"]
    if layer_type == "Linear" or "Conv" in layer_type:
        return NEURAL_COLORS["linear"]
    if layer_type in {"ReLU", "Sigmoid", "Tanh", "GELU", "LeakyReLU", "Softmax"}:
        return NEURAL_COLORS["activation"]
    if layer_type in {"Dropout", "BatchNorm1d", "BatchNorm2d", "LayerNorm"}:
        return NEURAL_COLORS["regularization"]
    return NEURAL_COLORS["muted"]
