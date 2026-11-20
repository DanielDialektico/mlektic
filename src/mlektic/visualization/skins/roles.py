"""Semantic roles of figure marks.

This is the single contract between the figure builders and the skins.
Builders describe *what* a trace is (through its ``uid`` or ``name``); skins
decide *how* each role looks.  A builder can be rewritten freely as long as
it keeps tagging its traces, and a skin can be redesigned without reading any
builder code.
"""

from __future__ import annotations

import re
from typing import Any

DATA = "data"
MODEL = "model"
SURFACE = "surface"
LOSS = "loss"
CLASS_CURVE = "class_curve"
EQUATION = "equation"
TEXT = "text"
ENCODED = "encoded"
EFFECT = "effect"
OTHER = "other"

# Marks added by skin effects carry this uid prefix so no pass ever restyles them.
EFFECT_UID_PREFIX = "SKIN_"
_CLASS_CURVE = re.compile(r"^p\(class\b", re.IGNORECASE)
_LOSS_WORDS = ("loss", "mse", "entropy")


def is_array(value: Any) -> bool:
    """Return whether a style value is per-point data rather than one colour."""
    return value is not None and not isinstance(value, str) and hasattr(value, "__len__")


def trace_role(trace: Any) -> str:
    """Return the semantic role of a Plotly trace."""
    uid = str(getattr(trace, "uid", "") or "")
    name = str(getattr(trace, "name", "") or "")
    lowered = name.lower()
    kind = str(getattr(trace, "type", "") or "")
    mode = str(getattr(trace, "mode", "") or "")

    if uid.startswith(EFFECT_UID_PREFIX):
        return EFFECT
    if uid.startswith(("NN_", "NN_BLOCK_", "NN_GRAPH_")) or lowered.startswith("neural graph"):
        return ENCODED
    if uid == "NUMERIC_EQUATION":
        return EQUATION
    if uid == "METRIC_VALUES" or mode == "text":
        return TEXT
    if uid in {"MODEL_SURFACE", "MODEL_PLANE"} or kind in {"surface", "mesh3d"}:
        return SURFACE
    if uid == "MODEL_LINE":
        return MODEL
    if uid == "LOSS_LINE" or any(word in lowered for word in _LOSS_WORDS):
        return LOSS
    if _CLASS_CURVE.match(name):
        return CLASS_CURVE
    marker = getattr(trace, "marker", None)
    if lowered == "data" and "markers" in mode and not is_array(getattr(marker, "color", None)):
        return DATA
    if lowered == "model" and "lines" in mode:
        return MODEL
    if marker is not None and is_array(getattr(marker, "color", None)):
        return ENCODED
    return OTHER


def is_flat(trace: Any) -> bool:
    """Return whether the trace is drawn on a 2D cartesian (non-WebGL) subplot."""
    return str(getattr(trace, "type", "") or "") == "scatter"
