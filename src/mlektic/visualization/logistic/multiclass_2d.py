"""Multiclass logistic-regression (2D) figure builder."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..theme import (
    get_base_layout,
    get_legend_props,
    get_updatemenus,
    get_sliders,
    create_annotation,
    loss_line_style,
)

def _row1_formula_latex_2d(K):
    return rf"$$\mathbf{{z}}=\Theta^\top\mathbf{{x}},\quad \mathbf{{x}}=\begin{{bmatrix}}x_1\\x_2\\1\end{{bmatrix}}\in\mathbb{{R}}^{{3}},\quad \Theta\in\mathbb{{R}}^{{3\times {K}}}$$"

def _row3_formula_latex_2d(K):
    return rf"$$\hat{{\mathbf{{p}}}}=\mathrm{{softmax}}(\mathbf{{z}}),\quad \mathrm{{softmax}}(\mathbf{{z}})_k=\dfrac{{e^{{z_k}}}}{{\sum_{{j=1}}^{{{K}}}e^{{z_j}}}},\;\;z_k(\mathbf{{x}})=\theta_{{1,k}}x_1+\theta_{{2,k}}x_2+\theta_{{0,k}}$$"

def _theta_matrix_latex_math_style_2d(w_hist, b_hist, t, max_elems, dec):
    Theta = np.vstack([w_hist[t, 0], w_hist[t, 1], b_hist[t]])
    K_local = Theta.shape[1]
    def fmt(v): return rf"{v:.{dec}f}"
    if K_local <= max_elems:
        row1 = " & ".join(fmt(Theta[0, j]) for j in range(K_local))
        row2 = " & ".join(fmt(Theta[1, j]) for j in range(K_local))
        row3 = " & ".join(fmt(Theta[2, j]) for j in range(K_local))
        cols_spec = "c" * K_local
        return r"$$" + r"\Theta=\left[\begin{array}{" + cols_spec + r"}" + row1 + r"\\" + row2 + r"\\" + row3 + r"\end{array}\right]" + r"$$"
    head = (max_elems - 1) // 2
    tail = (max_elems - 1) - head
    head_idx = list(range(head))
    tail_idx = list(range(K_local - tail, K_local))
    row1_items = [fmt(Theta[0, j]) for j in head_idx] + [r"\cdots"] + [fmt(Theta[0, j]) for j in tail_idx]
    row2_items = [fmt(Theta[1, j]) for j in head_idx] + [r"\cdots"] + [fmt(Theta[1, j]) for j in tail_idx]
    row3_items = [fmt(Theta[2, j]) for j in head_idx] + [r"\cdots"] + [fmt(Theta[2, j]) for j in tail_idx]
    row1 = " & ".join(row1_items); row2 = " & ".join(row2_items); row3 = " & ".join(row3_items)
    cols_spec = "c" * max_elems
    return r"$$" + r"\Theta=\left[\begin{array}{" + cols_spec + r"}" + row1 + r"\\" + row2 + r"\\" + row3 + r"\end{array}\right]" + r"$$"

def _z_numeric_expr_bivar(Theta, class_idx, dec):
    def num(v): return f"{v:+.{dec}f}"
    return rf"\left({num(Theta[0, class_idx])}\right)x_1 + \left({num(Theta[1, class_idx])}\right)x_2 + \left({num(Theta[2, class_idx])}\right)"

def _denom_three_terms_tex_2d(Theta, K_local, dec):
    z1 = _z_numeric_expr_bivar(Theta, 0, dec=dec)
    if K_local == 1: return rf"e^{{{z1}}}"
    z2 = _z_numeric_expr_bivar(Theta, 1, dec=dec)
    if K_local == 2: return rf"e^{{{z1}}} + e^{{{z2}}}"
    zK = _z_numeric_expr_bivar(Theta, K_local - 1, dec=dec)
    return rf"e^{{{z1}}} + e^{{{z2}}} + \cdots + e^{{{zK}}}"

def _final_prob_example_latex_2d(w_hist, b_hist, t, example_class, dec):
    Theta = np.vstack([w_hist[t, 0], w_hist[t, 1], b_hist[t]])
    K_local = Theta.shape[1]
    k = max(0, min(int(example_class), K_local - 1))
    z_k = _z_numeric_expr_bivar(Theta, k, dec=dec)
    return r"$$" + r"\begin{aligned}" + rf"\hat{{p}}(y=1\mid \mathbf{{x}}) &= \frac{{e^{{z_1(\mathbf{{x}})}}}}{{\sum_{{j=1}}^{{{K_local}}} e^{{z_j(\mathbf{{x}})}}}} \\[6pt]" + rf"&= \frac{{e^{{{z_k}}}}}{{{_denom_three_terms_tex_2d(Theta, K_local, dec=dec)}}}" + r"\end{aligned}" + r"$$"

def _vertical_dots_latex_2d():
    return r"$$\vdots$$"

def _last_class_tail_latex_2d(w_hist, b_hist, t, dec):
    Theta = np.vstack([w_hist[t, 0], w_hist[t, 1], b_hist[t]])
    K_local = Theta.shape[1]
    z_last = _z_numeric_expr_bivar(Theta, K_local - 1, dec=dec)
    return r"$$" + r"\begin{aligned}" + rf"\hat{{p}}(y={K_local}\mid \mathbf{{x}}) &= \frac{{e^{{{z_last}}}}}{{{_denom_three_terms_tex_2d(Theta, K_local, dec=dec)}}}" + r"\end{aligned}" + r"$$"

def build_multiclass_2d_logistic_figure(
    x1,
    x2,
    y,
    w_hist,
    b_hist,
    *,
    p_surfaces_hist=None,
    X1g=None,
    X2g=None,
    loss_hist=None,
    metrics_hist=None,
    show_loss=False,
    history_kind="iterative",
    title=None,
    strict_loss=False,
    dec=3,
    example_class=0,
    frame_duration=80,
    max_theta_cols=8,
    theme=None,
):
    if show_loss and history_kind != "iterative":
        if strict_loss:
            raise ValueError("show_loss=True is only allowed for iterative histories.")
        show_loss = False
        loss_hist = None

    x1 = np.asarray(x1).ravel()
    x2 = np.asarray(x2).ravel()
    y = np.asarray(y).ravel()
    
    if p_surfaces_hist is None or X1g is None or X2g is None:
        raise ValueError("p_surfaces_hist, X1g, X2g are required for multiclass 2D.")

    p_surfaces_hist = np.asarray(p_surfaces_hist, dtype=float)
    X1g = np.asarray(X1g, dtype=float)
    X2g = np.asarray(X2g, dtype=float)
    
    steps_n, h, w, K = p_surfaces_hist.shape

    if title is None:
        title = f"Multiclass Logistic Regression (K={K}, d=2)"

    if show_loss:
        if loss_hist is None:
            raise ValueError("show_loss=True requires loss_hist.")
        loss_hist = np.asarray(loss_hist, dtype=float).ravel()
        if loss_hist.size != steps_n:
            raise ValueError("loss_hist must match steps.")

    step_axis = np.arange(steps_n)
    margin_t = 120

    if show_loss:
        cols = 3
        column_widths = [0.60, 0.24, 0.16]
        specs = [[{"type": "xy"}, {"type": "scene"}, {"type": "xy"}]]
        X_TEXT = 0.28
        X_VDOTS = 0.31
    else:
        cols = 2
        column_widths = [0.60, 0.40]
        specs = [[{"type": "xy"}, {"type": "scene"}]]
        X_TEXT = 0.30
        X_VDOTS = 0.34

    x1_min, x1_max = float(np.min(X1g)), float(np.max(X1g))
    x2_min, x2_max = float(np.min(X2g)), float(np.max(X2g))
    
    def _pad(lo, hi, frac=0.10):
        span = (hi - lo) + 1e-9
        return [lo - frac * span, hi + frac * span]

    x1_range = _pad(x1_min, x1_max)
    x2_range = _pad(x2_min, x2_max)
    y_range = [-0.05, 1.05]

    CAMERA = dict(eye=dict(x=1.55, y=1.55, z=1.15))

    if show_loss:
        lmin, lmax = float(loss_hist.min()), float(loss_hist.max())
        lpad = 0.10 * (lmax - lmin + 1e-9)

    fig = make_subplots(
        rows=1, cols=cols,
        column_widths=column_widths,
        horizontal_spacing=0.06,
        specs=specs,
    )

    def metrics_annotations(t):
        ann = []
        if metrics_hist is not None:
            for i, (name, hist) in enumerate(metrics_hist.items()):
                val = hist[t]
                y_pos = 0.72 - (i * 0.10)
                fmt = ".6f" if name.lower() == "log-loss" or name.lower() == "loss" else ".4f"
                ann.append(dict(
                    x=0.98, y=y_pos, xref="paper", yref="paper", 
                    text=f"<b>{name}</b><br>{val:{fmt}}", showarrow=False, 
                    xanchor="right", yanchor="top", font=dict(size=11, color="black"), 
                    bgcolor="white", bordercolor="black", borderwidth=1, borderpad=4
                ))
        return ann

    def make_annotations(t):
        base_ann = [
            dict(x=X_TEXT, y=0.98, xref="paper", yref="paper", text=_row1_formula_latex_2d(K), showarrow=False, xanchor="center", yanchor="top", font=dict(size=18, color="white")),
            dict(x=X_TEXT, y=0.72, xref="paper", yref="paper", text=_theta_matrix_latex_math_style_2d(w_hist, b_hist, t, max_theta_cols, dec), showarrow=False, xanchor="center", yanchor="middle", font=dict(size=20, color="white")),
            dict(x=X_TEXT, y=0.51, xref="paper", yref="paper", text=_row3_formula_latex_2d(K), showarrow=False, xanchor="center", yanchor="top", font=dict(size=18, color="white")),
            dict(x=X_TEXT, y=0.30, xref="paper", yref="paper", text=_final_prob_example_latex_2d(w_hist, b_hist, t, example_class, dec), showarrow=False, xanchor="center", yanchor="middle", font=dict(size=13, color="white")),
            dict(x=X_VDOTS, y=0.17, xref="paper", yref="paper", text=_vertical_dots_latex_2d(), showarrow=False, xanchor="center", yanchor="middle", font=dict(size=22, color="white")),
            dict(x=X_TEXT, y=0.06, xref="paper", yref="paper", text=_last_class_tail_latex_2d(w_hist, b_hist, t, dec), showarrow=False, xanchor="center", yanchor="middle", font=dict(size=13, color="white")),
        ]
        
        prob_title_x = 0.72 if show_loss else 0.80
        base_ann.append(
            dict(x=prob_title_x, y=0.88, xref="paper", yref="paper", text="<b>Probability</b>", showarrow=False, xanchor="center", yanchor="bottom", font=dict(size=14, color="white"))
        )

        if show_loss:
            base_ann.append(
                dict(x=0.92, y=0.77, xref="paper", yref="paper", text="<b>Cross-entropy</b>", showarrow=False, xanchor="center", yanchor="bottom", font=dict(size=12, color="white"))
            )
            
        return base_ann + metrics_annotations(t)

    # colorscales for surfaces
    colorscales = ["Blues", "Reds", "Greens", "Oranges", "Purples", "Greys", "YlGnBu", "YlOrRd"]
    
    # Add Data
    scatter = go.Scatter3d(
        x=x1,
        y=x2,
        z=np.zeros_like(x1) - 0.02,
        mode="markers",
        name="Data",
        marker=dict(size=4, color=y, colorscale="Jet", opacity=0.8, showscale=False),
        hovertemplate="<b>Data</b><br>x1: %{x}<br>x2: %{y}<br>Class: %{marker.color}<extra></extra>",
        showlegend=False,
    )
    if show_loss:
        fig.add_trace(scatter, row=1, col=2)
    else:
        fig.add_trace(scatter, row=1, col=2)
        
    for k in range(K):
        cs = colorscales[k % len(colorscales)]
        surf = go.Surface(
            x=X1g,
            y=X2g,
            z=p_surfaces_hist[0, :, :, k],
            name=f"Class {k}",
            colorscale=cs,
            opacity=0.65,
            showscale=False,
            showlegend=True,
            legendgroup=f"class_{k}",
        )
        if show_loss:
            fig.add_trace(surf, row=1, col=2)
        else:
            fig.add_trace(surf, row=1, col=2)

    if show_loss:
        step_axis_list = step_axis.tolist()
        loss_hist_list = loss_hist.tolist()
        fig.add_trace(
            go.Scatter(
                x=[step if i == 0 else None for i, step in enumerate(step_axis_list)],
                y=[val if i == 0 else None for i, val in enumerate(loss_hist_list)],
                mode="lines",
                name="Log-loss",
                line=loss_line_style(theme=theme),
                legendgroup="loss",
                showlegend=True,
            ),
            row=1,
            col=3,
        )

    frames = []
    for t in range(steps_n):
        frame_data = [
            go.Scatter3d(x=x1, y=x2, z=np.zeros_like(x1) - 0.02)
        ]
        
        for k in range(K):
            frame_data.append(
                go.Surface(
                    x=X1g,
                    y=X2g,
                    z=p_surfaces_hist[t, :, :, k],
                )
            )
            
        if show_loss:
            frame_data.append(
                go.Scatter(
                    x=[step if i <= t else None for i, step in enumerate(step_axis_list)],
                    y=[val if i <= t else None for i, val in enumerate(loss_hist_list)],
                )
            )
            traces = list(range(K + 2))
        else:
            traces = list(range(K + 1))
            
        frames.append(
            go.Frame(
                name=str(t),
                data=frame_data,
                traces=traces,
                layout=go.Layout(
                    annotations=make_annotations(t),
                ),
            )
        )
    fig.frames = frames

    layout_kwargs = get_base_layout(title=title, margin_t=margin_t, theme=theme)
    fig.update_layout(
        **layout_kwargs,
        annotations=make_annotations(0),
        legend=dict(orientation="v", **get_legend_props(x=1.02, y=0.85, yanchor="top", xanchor="left", theme=theme)),
        legend2=dict(orientation="v", **get_legend_props(x=1.02, y=0.30, yanchor="top", xanchor="left", theme=theme)),
        scene=dict(
            xaxis=dict(title="x₁", range=x1_range),
            yaxis=dict(title="x₂", range=x2_range),
            zaxis=dict(title="p(y|x)", range=y_range),
            aspectmode="cube",
            camera=CAMERA,
        ),
        sliders=get_sliders(steps_n, theme=theme),
        updatemenus=get_updatemenus(frame_duration, theme=theme),
    )
    
    # Ocultar ejes del primer subplot para dejar espacio puro al texto
    fig.update_xaxes(visible=False, row=1, col=1, range=[0, 1])
    fig.update_yaxes(visible=False, row=1, col=1, range=[0, 1])

    if show_loss:
        fig.data[-1].update(legend="legend2")
        fig.update_xaxes(title="Step", range=[0, steps_n - 1], row=1, col=3)
        fig.update_yaxes(title="Log-loss", range=[lmin - lpad, lmax + lpad], domain=[0.25, 0.75], row=1, col=3)

    return fig

__all__ = ["build_multiclass_2d_logistic_figure"]
