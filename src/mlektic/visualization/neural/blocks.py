"""Hierarchical block renderer for captured :class:`NeuralGraph` objects."""

from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Sequence, Tuple

import plotly.graph_objects as go

from ...neural.graph_ir import NeuralEdge, NeuralGraph, NeuralNode, TensorSpec
from ...neural.taxonomy import format_hyperparameters
from ._style import NEURAL_COLORS, neural_layout

ROLE_COLORS = {
    "input": NEURAL_COLORS["input"],
    "output": NEURAL_COLORS["output"],
    "learnable": NEURAL_COLORS["linear"],
    "activation": NEURAL_COLORS["activation"],
    "regularization": NEURAL_COLORS["regularization"],
    "normalization": "#c6a0f6",
    "pooling": "#8bd5ca",
    "reshape": "#91d7e3",
    "merge": "#eed49f",
    "attention": "#f5a97f",
    "recurrent": "#c6a0f6",
    "embedding": "#8aadf4",
    "reduction": "#a6da95",
    "parameter": NEURAL_COLORS["muted"],
    "operation": NEURAL_COLORS["muted"],
    "summary": NEURAL_COLORS["muted"],
}

HOVER_FORMULAS = {
    "Input": "x: model input tensor",
    "Output": "ŷ = fθ(x): model output tensor",
    "Linear": "z = Θx + θ₀",
    "ReLU": "y = max(0, x)",
    "Sigmoid": "y = σ(x) = 1 / (1 + exp(−x))",
    "Tanh": "y = tanh(x)",
    "GELU": "y = GELU(x)",
    "LeakyReLU": "yᵢ = max(xᵢ, αxᵢ)",
    "Softmax": "yᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)",
    "Add": "y = x₁ + x₂",
    "Subtract": "y = x₁ − x₂",
    "Multiply": "y = x₁ ⊙ x₂",
    "Divide": "y = x₁ ⊘ x₂",
    "Concatenate": "y = concatenate(x₁, …, xₘ)",
    "Stack": "y = stack(x₁, …, xₘ)",
    "Matrix multiply": "y = Ax",
    "Flatten": "y = vectorize(x)",
    "Unflatten": "y = unvectorize(x)",
    "Reshape": "y = reshape(x)",
    "View": "y = reshape(x)",
    "Permute": "y = permute(x)",
    "Transpose": "y = xᵀ",
    "Mean": "y = mean(x)",
    "Sum": "y = Σᵢxᵢ",
    "Select": "y = selected elements of x",
    "Embedding": "aₜ = E[xₜ], where E is the embedding table",
    "RNN": "hₜ = φ(Wᵢₕxₜ + Wₕₕhₜ₋₁ + b)",
    "RNNCell": "hₜ = φ(Wᵢₕxₜ + Wₕₕhₜ₋₁ + b)",
    "GRU": "zₜ = σ(Wzxₜ + Uzhₜ₋₁); hₜ mixes candidate and previous states",
    "GRUCell": "zₜ = σ(Wzxₜ + Uzhₜ₋₁); hₜ mixes candidate and previous states",
    "LSTM": "(iₜ, fₜ, gₜ, oₜ) = gates(xₜ, hₜ₋₁); cₜ = fₜ⊙cₜ₋₁ + iₜ⊙gₜ",
    "LSTMCell": "(iₜ, fₜ, gₜ, oₜ) = gates(xₜ, hₜ₋₁); cₜ = fₜ⊙cₜ₋₁ + iₜ⊙gₜ",
    "MultiheadAttention": ("MHA(Q,K,V) = Concat(head₁, …, headₕ)Wᴼ;<br>headᵢ = softmax(QWᵢQ(KWᵢK)ᵀ / √dₖ)VWᵢV"),
    "TransformerEncoderLayer": "y = normalization and residual composition of self-attention and FFN",
    "TransformerDecoderLayer": "y = decoder block(x, memory)",
}


def _shape_text(specs: Sequence[TensorSpec]) -> str:
    if not specs:
        return "?"
    values = []
    for spec in specs:
        shape = " × ".join(str(value) for value in spec.shape) if spec.shape is not None else "?"
        values.append(shape)
    return ", ".join(values)


def _display_specs(node: NeuralNode) -> Sequence[TensorSpec]:
    """Return the tensor specifications that should be printed for a node."""
    if node.outputs:
        return node.outputs
    if node.role == "output":
        return node.inputs
    return ()


def _plain_formula(node: NeuralNode) -> str:
    """Return readable hover mathematics without exposing raw LaTeX syntax."""
    known = HOVER_FORMULAS.get(node.label)
    if known is not None:
        return known
    text = node.formula
    replacements = {
        r"\ldots": "…",
        r"\cdot": "·",
        r"\times": "×",
        r"\odot": "⊙",
        r"\oslash": "⊘",
        r"\quad": " ",
        r"\left": "",
        r"\right": "",
        r"\mathsf{T}": "T",
        r"\Theta": "Θ",
        r"\theta": "θ",
        r"\sigma": "σ",
        r"\gamma": "γ",
        r"\beta": "β",
        r"\varepsilon": "ε",
        r"\ell": "ℓ",
        r"\sum": "Σ",
        r"\sqrt": "sqrt",
        r"\in": "∈",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    for command in ("operatorname", "mathbf", "boldsymbol", "mathrm", "mathcal", "mathbb", "text"):
        pattern = re.compile(rf"\\{command}\{{([^{{}}]*)\}}")
        while pattern.search(text):
            text = pattern.sub(r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text or f"{node.label}: tensor transformation"


def _visible_nodes(graph: NeuralGraph, max_nodes: int) -> Tuple[List[NeuralNode], List[NeuralEdge]]:
    if max_nodes < 5:
        raise ValueError("max_nodes must be at least 5.")
    nodes = list(graph.nodes)
    if len(nodes) <= max_nodes:
        return nodes, list(graph.edges)
    fixed_ids = {*graph.input_nodes, *graph.output_nodes}
    candidates = [node for node in nodes if node.id not in fixed_ids]
    available = max(1, max_nodes - len(fixed_ids) - 1)
    left = max(1, available // 2)
    right = max(0, available - left)
    selected_candidates = [*candidates[:left], *(candidates[-right:] if right else [])]
    selected_ids = {*fixed_ids, *(node.id for node in selected_candidates)}
    summary_id = "collapsed_operations"
    collapsed_count = len(nodes) - len(selected_ids)
    summary = NeuralNode(
        id=summary_id,
        name="collapsed operations",
        op="summary",
        target="summary",
        role="summary",
        label=f"{collapsed_count} collapsed operations",
        formula=r"\vdots",
        math_status="summary",
    )
    visible = [node for node in nodes if node.id in selected_ids]
    insertion = max(1, len(visible) // 2)
    visible.insert(insertion, summary)
    mapped_edges = set()
    edges: List[NeuralEdge] = []
    for edge in graph.edges:
        source = edge.source if edge.source in selected_ids else summary_id
        target = edge.target if edge.target in selected_ids else summary_id
        if source == target:
            continue
        key = (source, target, edge.kind)
        if key in mapped_edges:
            continue
        mapped_edges.add(key)
        edges.append(NeuralEdge(source, target, kind=edge.kind, tensor=edge.tensor))
    return visible, edges


def _depths(nodes: Sequence[NeuralNode], edges: Sequence[NeuralEdge]) -> Dict[str, int]:
    node_ids = {node.id for node in nodes}
    incoming: Dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    outgoing: Dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        if edge.source not in node_ids or edge.target not in node_ids:
            continue
        incoming[edge.target].add(edge.source)
        outgoing[edge.source].add(edge.target)
    queue = deque(node_id for node_id, sources in incoming.items() if not sources)
    depths = {node_id: 0 for node_id in queue}
    processed = set()
    while queue:
        source = queue.popleft()
        processed.add(source)
        for target in outgoing[source]:
            depths[target] = max(depths.get(target, 0), depths[source] + 1)
            incoming[target].discard(source)
            if not incoming[target]:
                queue.append(target)
    for node in nodes:
        if node.id not in processed:
            depths[node.id] = max(depths.values(), default=0) + 1
    return depths


def _positions(nodes: Sequence[NeuralNode], edges: Sequence[NeuralEdge]) -> Dict[str, Tuple[float, float]]:
    depths = _depths(nodes, edges)
    groups: Dict[int, List[NeuralNode]] = defaultdict(list)
    for node in nodes:
        groups[depths[node.id]].append(node)
    maximum_depth = max(groups, default=0)
    positions: Dict[str, Tuple[float, float]] = {}
    for depth, group in sorted(groups.items()):
        x = 0.08 + 0.84 * depth / max(maximum_depth, 1)
        if len(group) == 1:
            y_values = [0.52]
        else:
            y_values = [0.82 - index * 0.60 / (len(group) - 1) for index in range(len(group))]
        for node, y in zip(group, y_values):
            positions[node.id] = (x, y)
    return positions


def _node_hover(node: NeuralNode) -> str:
    parameters = "<br>".join(f"{parameter.kind} {parameter.name}: {parameter.shape}" for parameter in node.parameters)
    hyperparameter_items = [f"{name}={value}" for name, value in node.hyperparameters.items()]
    hyperparameters = "<br>".join(
        ", ".join(hyperparameter_items[index : index + 4]) for index in range(0, len(hyperparameter_items), 4)
    ) or format_hyperparameters({}, limit=30)
    return (
        f"<b>{node.name} · {node.label}</b><br>"
        f"role: {node.role}<br>call: {node.call_index}<br>"
        f"input: {_shape_text(node.inputs)}<br>output: {_shape_text(_display_specs(node))}<br>"
        f"mathematics: {node.math_status}<br>formula: {_plain_formula(node)}<br>"
        f"parameters: {parameters or 'none'}<br>configuration: {hyperparameters}"
    )


def _node_caption(node: NeuralNode) -> str:
    if node.op == "call_module":
        return node.module_path or node.module_type or node.name
    if node.role in {"input", "output"}:
        return node.name
    if node.call_index > 1:
        return f"tensor operation · call {node.call_index}"
    return "tensor operation"


def _edge_traces(edges: Iterable[NeuralEdge], positions: Dict[str, Tuple[float, float]]) -> List[go.Scatter]:
    traces = []
    for edge in edges:
        if edge.source not in positions or edge.target not in positions:
            continue
        x0, y0 = positions[edge.source]
        x1, y1 = positions[edge.target]
        shape = _shape_text((edge.tensor,)) if edge.tensor is not None else "?"
        if x1 - x0 > 0.25:
            x_values = [x0, (x0 + x1) / 2.0, x1]
            y_values = [y0, min(0.92, max(y0, y1) + 0.18), y1]
        else:
            x_values = [x0, x1]
            y_values = [y0, y1]
        traces.append(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                line={"color": NEURAL_COLORS["grid"], "width": 1.5, "shape": "spline"},
                customdata=[shape, shape],
                hovertemplate="tensor %{customdata}<extra></extra>",
                showlegend=False,
            )
        )
    return traces


def build_nn_block_figure(
    graph: NeuralGraph,
    *,
    title: str | None = None,
    max_nodes: int = 48,
    show_formulas: bool = True,
) -> go.Figure:
    """Render a captured execution graph as extensible semantic blocks."""
    nodes, edges = _visible_nodes(graph, max_nodes)
    positions = _positions(nodes, edges)
    column_sizes = [
        len([node for node in nodes if positions[node.id][0] == x_position])
        for x_position in {position[0] for position in positions.values()}
    ]
    inline_formulas = show_formulas and len(nodes) <= 12 and max(column_sizes, default=1) <= 3
    figure = go.Figure(data=_edge_traces(edges, positions))
    for role in dict.fromkeys(node.role for node in nodes):
        role_nodes = [node for node in nodes if node.role == role]
        node_text = []
        for node in role_nodes:
            if node.role == "summary":
                collapsed = node.label.split(" collapsed", 1)[0]
                node_text.append(f"<b>{collapsed} ops</b><br>summarized")
            else:
                node_text.append(f"<b>{node.label}</b><br>{_node_caption(node)}<br>{_shape_text(_display_specs(node))}")
        figure.add_trace(
            go.Scatter(
                x=[positions[node.id][0] for node in role_nodes],
                y=[positions[node.id][1] for node in role_nodes],
                mode="markers+text",
                text=node_text,
                textposition="bottom center",
                textfont={"size": 12, "color": NEURAL_COLORS["text"]},
                marker={
                    "size": 42 if role == "summary" else 34 if role not in {"input", "output", "merge"} else 38,
                    "symbol": "diamond" if role in {"merge", "output"} else "circle" if role == "input" else "square",
                    "color": ROLE_COLORS.get(role, NEURAL_COLORS["muted"]),
                    "line": {"color": NEURAL_COLORS["text"], "width": 1},
                },
                customdata=[_node_hover(node) for node in role_nodes],
                hovertemplate="%{customdata}<extra></extra>",
                name=role.replace("_", " ").title(),
                uid=f"NN_BLOCK_{role.upper()}",
            )
        )
    annotations = [
        {
            "x": 0.5,
            "y": 1.10,
            "xref": "paper",
            "yref": "paper",
            "text": (
                f"Capture: <b>{graph.provenance.backend}</b> · {graph.provenance.kind} · "
                f"functional operations: {'yes' if graph.provenance.includes_functional_ops else 'partial'}"
            ),
            "showarrow": False,
            "font": {"size": 12, "color": NEURAL_COLORS["muted"]},
        }
    ]
    if inline_formulas:
        formula_nodes = [node for node in nodes if node.role not in {"input", "output", "parameter", "summary"}]
        for formula_index, node in enumerate(formula_nodes):
            x, y = positions[node.id]
            above_node = formula_index % 2 == 0
            annotations.append(
                {
                    "x": x,
                    "y": y,
                    "xref": "x",
                    "yref": "y",
                    "text": f"${node.formula}$",
                    "showarrow": False,
                    "yanchor": "bottom" if above_node else "top",
                    "yshift": 58 if above_node else -84,
                    "font": {"size": 13, "color": NEURAL_COLORS["muted"]},
                }
            )
    elif show_formulas:
        annotations.append(
            {
                "x": 0.5,
                "y": 1.055,
                "xref": "paper",
                "yref": "paper",
                "text": "Inline formulas hidden to prevent overlap · available in node hover",
                "showarrow": False,
                "font": {"size": 11, "color": NEURAL_COLORS["muted"]},
            }
        )
    if title is None:
        title = "Neural execution graph"
    layout = neural_layout(
        title,
        height=max(650, 440 + 48 * max(0, max(column_sizes, default=1) - 1)),
    )
    layout["margin"] = {"t": 115, "r": 45, "b": 75, "l": 45}
    layout["meta"] = {
        "mlektic_neural_graph": {
            "schema_version": graph.schema_version,
            "capture": {
                "backend": graph.provenance.backend,
                "kind": graph.provenance.kind,
                "model_type": graph.provenance.model_type,
                "torch_version": graph.provenance.torch_version,
                "exact_for_input": graph.provenance.exact_for_input,
                "includes_functional_ops": graph.provenance.includes_functional_ops,
                "includes_dynamic_control_flow": graph.provenance.includes_dynamic_control_flow,
                "notes": list(graph.provenance.notes),
            },
            "warnings": list(graph.warnings),
            "captured_nodes": len(graph.nodes),
            "captured_edges": len(graph.edges),
            "rendered_nodes": len(nodes),
            "max_nodes": max_nodes,
            "visually_collapsed": len(graph.nodes) > max_nodes,
            "show_formulas": show_formulas,
            "inline_formulas": inline_formulas,
            "formula_layout": "alternating above and below adjacent execution blocks",
        }
    }
    figure.update_layout(**layout, annotations=annotations, legend={"orientation": "h", "y": -0.08})
    figure.update_xaxes(visible=False, range=[0, 1])
    figure.update_yaxes(visible=False, range=[0, 1])
    return figure


__all__ = ["build_nn_block_figure"]
