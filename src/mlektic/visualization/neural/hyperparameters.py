"""Dedicated mathematical hyperparameter contract figure."""

from __future__ import annotations

from textwrap import wrap
from typing import Any, Dict, Mapping, Sequence

import plotly.graph_objects as go

from ...neural.hyperparameters import (
    HyperparameterComponent,
    describe_hyperparameter_contract,
    display_value,
)
from ._style import NEURAL_COLORS, neural_layout

_SCOPE_COLORS = {
    "module": NEURAL_COLORS["linear"],
    "optimizer": NEURAL_COLORS["positive"],
    "objective": NEURAL_COLORS["output"],
    "scheduler": NEURAL_COLORS["activation"],
}


def _component_height(component: HyperparameterComponent) -> int:
    """Allocate a non-overlapping pixel corridor for one component."""
    return 88 + 40 * max(1, len(component.items))


def _wrapped_text(text: str, *, width: int = 48) -> str:
    """Wrap prose inside its reserved column without clipping the panel."""
    return "<br>".join(wrap(text, width=width, break_long_words=False))


def build_nn_hyperparameter_figure(
    model: Any,
    *,
    history: Mapping[str, Any] | None = None,
    optimizer: Any | None = None,
    loss_fn: Any | None = None,
    scheduler: Any | None = None,
    title: str | None = None,
) -> go.Figure:
    """Show the complete effective PyTorch configuration supplied by a run.

    Rows are never truncated.  Execution-only options remain visible and are
    explicitly distinguished from arguments that alter the mathematical map.
    """
    components = describe_hyperparameter_contract(
        model,
        history=history,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
    )
    content_height = 55 + sum(_component_height(component) + 16 for component in components)
    canvas_height = max(720, content_height + 120)
    figure = go.Figure(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, float(content_height)],
            mode="markers",
            marker={"opacity": 0.0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    annotations: list[Dict[str, Any]] = []
    shapes: list[Dict[str, Any]] = []
    cursor = content_height - 18
    for component in components:
        panel_height = _component_height(component)
        top = cursor
        bottom = top - panel_height
        color = _SCOPE_COLORS.get(component.scope, NEURAL_COLORS["muted"])
        shapes.append(
            {
                "type": "rect",
                "xref": "paper",
                "yref": "y",
                "x0": 0.02,
                "x1": 0.98,
                "y0": bottom,
                "y1": top,
                "line": {"color": NEURAL_COLORS["grid"], "width": 1},
                "fillcolor": NEURAL_COLORS["panel"],
                "layer": "below",
            }
        )
        shapes.append(
            {
                "type": "rect",
                "xref": "paper",
                "yref": "y",
                "x0": 0.02,
                "x1": 0.026,
                "y0": bottom,
                "y1": top,
                "line": {"width": 0},
                "fillcolor": color,
                "layer": "below",
            }
        )
        annotations.extend(
            [
                {
                    "x": 0.04,
                    "y": top - 20,
                    "xref": "paper",
                    "yref": "y",
                    "text": f"<b>{component.scope.upper()}</b> · {component.label}",
                    "showarrow": False,
                    "xanchor": "left",
                    "yanchor": "middle",
                    "font": {"size": 15, "color": color},
                },
                {
                    "x": 0.33,
                    "y": top - 20,
                    "xref": "paper",
                    "yref": "y",
                    "text": f"${component.operation}$",
                    "showarrow": False,
                    "xanchor": "left",
                    "yanchor": "middle",
                    "font": {"size": 14, "color": NEURAL_COLORS["text"]},
                },
            ]
        )
        items: Sequence[Any] = component.items or (None,)
        for item_index, item in enumerate(items):
            row_y = top - 65 - 40 * item_index
            if item is None:
                label = "No configurable hyperparameters"
                mathematics = r"\mathcal{H}=\varnothing"
                definition = "The PyTorch module exposes no configurable mathematical arguments."
                mathematical = True
            else:
                label = f"<b>{item.name}</b> = {display_value(item.value)}"
                mathematics = item.mathematics
                definition = item.definition
                mathematical = item.mathematical
            annotations.extend(
                [
                    {
                        "x": 0.05,
                        "y": row_y,
                        "xref": "paper",
                        "yref": "y",
                        "text": label,
                        "showarrow": False,
                        "xanchor": "left",
                        "yanchor": "middle",
                        "font": {"size": 13, "color": NEURAL_COLORS["text"]},
                    },
                    {
                        "x": 0.34,
                        "y": row_y,
                        "xref": "paper",
                        "yref": "y",
                        "text": f"${mathematics}$",
                        "showarrow": False,
                        "xanchor": "left",
                        "yanchor": "middle",
                        "font": {
                            "size": 14,
                            "color": (NEURAL_COLORS["text"] if mathematical else NEURAL_COLORS["muted"]),
                        },
                    },
                    {
                        "x": 0.69,
                        "y": row_y,
                        "xref": "paper",
                        "yref": "y",
                        "text": _wrapped_text(definition),
                        "showarrow": False,
                        "xanchor": "left",
                        "yanchor": "middle",
                        "align": "left",
                        "font": {"size": 12, "color": NEURAL_COLORS["muted"]},
                    },
                ]
            )
        cursor = bottom - 16

    annotations.append(
        {
            "x": 0.5,
            "y": 13,
            "xref": "paper",
            "yref": "y",
            "text": (
                "Effective values read from the supplied PyTorch objects or recorder history · "
                "definitions follow the official PyTorch argument semantics"
            ),
            "showarrow": False,
            "xanchor": "center",
            "font": {"size": 11, "color": NEURAL_COLORS["muted"]},
        }
    )
    if title is None:
        title = "PyTorch hyperparameters: effective values and mathematical definitions"
    layout = neural_layout(title, height=canvas_height)
    layout["title"].update({"x": 0.5, "xanchor": "center"})
    layout["margin"] = {"t": 95, "r": 28, "b": 35, "l": 28}
    layout["meta"] = {
        "mlektic_neural_hyperparameters": {
            "coverage": "all detected effective configuration values; no displayed rows are truncated",
            "component_count": len(components),
            "hyperparameter_count": sum(len(component.items) for component in components),
            "component_labels": [component.label for component in components],
            "source_urls": [component.source_url for component in components],
            "components": [
                {
                    "scope": component.scope,
                    "label": component.label,
                    "type": component.type_name,
                    "operation": component.operation,
                    "source_url": component.source_url,
                    "hyperparameters": [
                        {
                            "name": item.name,
                            "value": display_value(item.value),
                            "mathematical": item.mathematical,
                            "definition_status": item.definition_status,
                        }
                        for item in component.items
                    ],
                }
                for component in components
            ],
            "execution_options_are_marked_nonmathematical": True,
            "instance_based_not_global_catalogue": True,
            "specialized_definition_count": sum(
                item.definition_status == "specialized" for component in components for item in component.items
            ),
            "generic_definition_count": sum(
                item.definition_status == "generic" for component in components for item in component.items
            ),
            "content_min_height": canvas_height,
        }
    }
    figure.update_layout(
        **layout,
        annotations=annotations,
        shapes=shapes,
        showlegend=False,
    )
    figure.update_xaxes(visible=False, fixedrange=True, range=[0.0, 1.0])
    figure.update_yaxes(visible=False, fixedrange=True, range=[0.0, float(content_height)])
    return figure


__all__ = ["build_nn_hyperparameter_figure"]
