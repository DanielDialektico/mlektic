"""Lightweight standalone mathematical report for PyTorch networks."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .introspection import describe_torch_model
from .taxonomy import (
    composed_dense_function,
    dense_stages,
    format_hyperparameters,
    parameter_definition,
    shape_tex,
)


def _table(rows: list[tuple[str, str]]) -> str:
    body = "".join(f"<tr><th>{escape(label)}</th><td>{value}</td></tr>" for label, value in rows)
    return f"<table><tbody>{body}</tbody></table>"


def _history_section(history: Dict[str, Any] | None) -> str:
    if not history:
        return ""
    steps = np.asarray(history.get("steps", []), dtype=int)
    loss = np.asarray(history.get("loss", []), dtype=float)
    config = history.get("training_config", {})
    rows = [
        ("Recorded frames", str(steps.size)),
        ("Step range", f"{steps[0]} to {steps[-1]}" if steps.size else "not recorded"),
        ("Optimizer", escape(str(config.get("optimizer", "not supplied")))),
        (
            "Optimizer configuration",
            escape(format_hyperparameters(config.get("optimizer_hyperparameters", {}), limit=20)),
        ),
        ("Loss", escape(str(config.get("loss", "not supplied")))),
    ]
    if loss.size and np.isfinite(loss).any():
        finite = loss[np.isfinite(loss)]
        rows.append(("Loss evolution", rf"\( {finite[0]:.6f}\rightarrow {finite[-1]:.6f} \)"))
    metrics = history.get("metrics", {})
    for name, values in metrics.items():
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        if finite.size:
            rows.append((f"Metric: {name}", rf"\( {finite[0]:.5f}\rightarrow {finite[-1]:.5f} \)"))
    norm_rows = []
    for name, values in history.get("parameter_norms", {}).items():
        array = np.asarray(values, dtype=float)
        if array.size:
            norm_rows.append(
                rf"<tr><td><code>{escape(name)}</code></td><td>\({array[0]:.5f}\)</td>"
                rf"<td>\({array[-1]:.5f}\)</td><td>\({array[-1] - array[0]:+.5f}\)</td></tr>"
            )
    norm_table = ""
    if norm_rows:
        norm_table = (
            "<h3>Parameter movement</h3><table><thead><tr><th>Tensor</th><th>Initial norm</th>"
            "<th>Final norm</th><th>Change</th></tr></thead><tbody>"
            + "".join(norm_rows)
            + "</tbody></table>"
        )
    return "<section><h2>Training configuration and evolution</h2>" + _table(rows) + norm_table + "</section>"


def _layer_section(layer: Dict[str, Any]) -> str:
    parameter_rows = []
    for parameter_name, shape in layer["parameter_shapes"].items():
        definition = parameter_definition(
            parameter_name,
            shape,
            layer.get("math_index", layer["index"] + 1),
        )
        parameter_rows.append(
            f"<tr><td><code>{escape(parameter_name)}</code></td>"
            rf"<td>\({shape_tex(shape, drop_batch=False)}\)</td><td>\({definition}\)</td></tr>"
        )
    parameters = (
        "<table><thead><tr><th>Tensor</th><th>Shape</th><th>Mathematical role</th></tr></thead><tbody>"
        + "".join(parameter_rows)
        + "</tbody></table>"
        if parameter_rows
        else "<p class='muted'>This layer has no learnable tensors.</p>"
    )
    input_dimension = shape_tex(layer.get("input_shape"))
    output_dimension = shape_tex(layer.get("output_shape"))
    metadata = _table(
        [
            ("Tensor map", rf"\(\mathbb{{R}}^{{{input_dimension}}}\to\mathbb{{R}}^{{{output_dimension}}}\)"),
            ("Parameters", f"{layer['parameters']:,} total; {layer['trainable_parameters']:,} trainable"),
            ("Hyperparameters", escape(format_hyperparameters(layer["hyperparameters"], limit=30))),
        ]
    )
    return (
        f"<section class='layer'><div class='layer-heading'><span class='index'>{layer['index'] + 1}</span>"
        f"<div><h2>{escape(layer['name'])} <span>{escape(layer['type'])}</span></h2>"
        f"<p class='role'>{escape(layer['role'])}</p></div></div>"
        rf"<div class='formula'>\[{layer['formula']}\]</div>{metadata}{parameters}</section>"
    )


def build_nn_math_report(
    model: Any,
    input_sample: Any,
    *,
    history: Dict[str, Any] | None = None,
    title: str = "Mlektic neural-network mathematical report",
) -> str:
    """Return a responsive standalone HTML report with complete layer taxonomy."""
    layers = describe_torch_model(model, input_sample)
    total_parameters = sum(layer["parameters"] for layer in layers)
    trainable_parameters = sum(layer["trainable_parameters"] for layer in layers)
    function = composed_dense_function(dense_stages(model))
    summary = _table(
        [
            ("Model", escape(model.__class__.__name__)),
            ("Leaf layers", str(len(layers))),
            ("Parameters", f"{total_parameters:,} total; {trainable_parameters:,} trainable"),
            ("Input", rf"\(\mathbb{{R}}^{{{shape_tex(layers[0].get('input_shape'))}}}\)"),
            ("Output", rf"\(\mathbb{{R}}^{{{shape_tex(layers[-1].get('output_shape'))}}}\)"),
        ]
    )
    layer_sections = "".join(_layer_section(layer) for layer in layers)
    history_section = _history_section(history)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
:root {{ color-scheme: dark; --bg:#17181c; --surface:#202126; --text:#f5f7fa; --muted:#aeb4bd;
  --line:#353841; --learn:#77b7ff; --positive:#55d6be; --negative:#ff7d8e; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text);
  font-family:Inter,Segoe UI,Arial,sans-serif; line-height:1.55; }}
main {{ width:min(1080px, calc(100% - 32px)); margin:0 auto; padding:40px 0 72px; }}
header {{ border-bottom:1px solid var(--line); padding-bottom:28px; margin-bottom:28px; }}
h1 {{ margin:0 0 8px; font-size:clamp(25px,4vw,38px); font-weight:500; }}
h2 {{ margin:0; font-size:20px; font-weight:500; }}
h2 span {{ color:var(--learn); font-size:14px; margin-left:8px; }}
h3 {{ font-size:16px; font-weight:500; margin-top:28px; }}
p {{ margin:6px 0; }} .muted,.role {{ color:var(--muted); }}
.formula {{ overflow-x:auto; padding:14px 0 18px; color:var(--text); }}
section {{ padding:26px 0; border-bottom:1px solid var(--line); }}
.layer-heading {{ display:flex; gap:14px; align-items:center; }}
.index {{ display:grid; place-items:center; width:36px; height:36px;
  border:1px solid var(--positive); color:var(--positive); }}
table {{ width:100%; border-collapse:collapse; margin-top:14px; }}
th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
th {{ color:var(--muted); font-weight:500; width:24%; }} code {{ color:var(--positive); }}
@media (max-width:600px) {{
  main {{ width:min(100% - 20px,1080px); padding-top:24px; }}
  th,td {{ display:block; width:100%; }}
  th {{ border-bottom:0; padding-bottom:2px; }}
  td {{ padding-top:2px; }}
}}
</style>
<script>window.MathJax={{tex:{{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]}}}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head>
<body><main>
<header><h1>{escape(title)}</h1>
<p class="muted">Layer taxonomy, dimensions, definitions, configuration, and training movement.</p>
<div class="formula">\\[{function}\\]</div>{summary}</header>
{history_section}
{layer_sections}
</main></body></html>"""


def export_nn_math_report(
    model: Any,
    input_sample: Any,
    *,
    history: Dict[str, Any] | None = None,
    path: str | Path = "mlektic_nn_math_report.html",
    title: str = "Mlektic neural-network mathematical report",
) -> Path:
    """Write the complete mathematical report and return its resolved path."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        build_nn_math_report(model, input_sample, history=history, title=title),
        encoding="utf-8",
    )
    return destination


def display_nn_math_report(
    model: Any,
    input_sample: Any,
    *,
    history: Dict[str, Any] | None = None,
    title: str = "Mlektic neural-network mathematical report",
):
    """Return an IPython HTML object suitable for Jupyter and Colab display."""
    try:
        from IPython.display import HTML
    except ImportError as exc:
        raise ImportError("Notebook display requires IPython. Use export_nn_math_report() instead.") from exc
    return HTML(build_nn_math_report(model, input_sample, history=history, title=title))


__all__ = ["build_nn_math_report", "display_nn_math_report", "export_nn_math_report"]
