"""Multiclass logistic-regression (1D) figure builder."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..theme import (
    get_base_layout,
    get_legend_props,
    get_updatemenus,
    get_sliders,
)
from ...utils.math import _softmax

def _row1_formula_latex(K):
    return rf"$$\mathbf{{z}}=\Theta^\top\mathbf{{x}},\quad \mathbf{{x}}=\begin{{bmatrix}}x\\1\end{{bmatrix}}\in\mathbb{{R}}^{{2}},\quad \Theta\in\mathbb{{R}}^{{2\times {K}}}$$"

def _row3_formula_latex(K):
    return rf"$$\hat{{\mathbf{{p}}}}=\mathrm{{softmax}}(\mathbf{{z}}),\quad \mathrm{{softmax}}(\mathbf{{z}})_k=\dfrac{{e^{{z_k}}}}{{\sum_{{j=1}}^{{{K}}}e^{{z_j}}}},\;\;k=1,\dots,{K},\quad z_k(x)=\theta_{{1,k}}x+\theta_{{0,k}}$$"

def _theta_matrix_latex_math_style(w_hist, b_hist, t, max_elems, dec):
    Theta = np.vstack([w_hist[t, 0], b_hist[t]])  # (2,K)
    K_local = Theta.shape[1]

    def fmt(v):
        return rf"{v:.{dec}f}"

    if K_local <= max_elems:
        row1 = " & ".join(fmt(Theta[0, j]) for j in range(K_local))
        row2 = " & ".join(fmt(Theta[1, j]) for j in range(K_local))
        cols_spec = "c" * K_local
        return (
            r"$$"
            r"\Theta=\left[\begin{array}{" + cols_spec + r"}" + row1 + r"\\" + row2 + r"\end{array}\right]"
            r"$$"
        )

    head = (max_elems - 1) // 2
    tail = (max_elems - 1) - head
    head_idx = list(range(head))
    tail_idx = list(range(K_local - tail, K_local))

    row1_items = [fmt(Theta[0, j]) for j in head_idx] + [r"\cdots"] + [fmt(Theta[0, j]) for j in tail_idx]
    row2_items = [fmt(Theta[1, j]) for j in head_idx] + [r"\cdots"] + [fmt(Theta[1, j]) for j in tail_idx]

    row1 = " & ".join(row1_items)
    row2 = " & ".join(row2_items)
    cols_spec = "c" * max_elems

    return (
        r"$$"
        r"\Theta=\left[\begin{array}{" + cols_spec + r"}" + row1 + r"\\" + row2 + r"\end{array}\right]"
        r"$$"
    )

def _z_numeric_expr_univar(Theta, class_idx, dec):
    def num(v):
        return f"{v:+.{dec}f}"

    theta_1k = num(Theta[0, class_idx])
    theta_0k = num(Theta[1, class_idx])
    return rf"\left({theta_1k}\right)x + \left({theta_0k}\right)"

def _denom_three_terms_tex(Theta, K_local, dec):
    z1 = _z_numeric_expr_univar(Theta, 0, dec=dec)
    if K_local == 1:
        return rf"e^{{{z1}}}"

    z2 = _z_numeric_expr_univar(Theta, 1, dec=dec)
    if K_local == 2:
        return rf"e^{{{z1}}} + e^{{{z2}}}"

    zK = _z_numeric_expr_univar(Theta, K_local - 1, dec=dec)
    return rf"e^{{{z1}}} + e^{{{z2}}} + \cdots + e^{{{zK}}}"

def _final_prob_example_latex(w_hist, b_hist, t, example_class, dec):
    Theta = np.vstack([w_hist[t, 0], b_hist[t]])
    K_local = Theta.shape[1]

    k = int(example_class)
    k = max(0, min(k, K_local - 1))

    z_k = _z_numeric_expr_univar(Theta, k, dec=dec)
    num_tex = rf"e^{{{z_k}}}"
    denom_tex = _denom_three_terms_tex(Theta, K_local, dec=dec)

    return (
        r"$$"
        r"\begin{aligned}"
        + rf"\hat{{p}}(y=1\mid x) &= \frac{{e^{{z_1(x)}}}}{{\sum_{{j=1}}^{{{K_local}}} e^{{z_j(x)}}}} \\[6pt]"
        + rf"&= \frac{{{num_tex}}}{{{denom_tex}}}"
        r"\end{aligned}"
        r"$$"
    )

def _vertical_dots_latex():
    return r"$$\vdots$$"

def _last_class_tail_latex(w_hist, b_hist, t, dec):
    Theta = np.vstack([w_hist[t, 0], b_hist[t]])
    K_local = Theta.shape[1]
    last_idx = K_local - 1

    z_last = _z_numeric_expr_univar(Theta, last_idx, dec=dec)
    num_tex = rf"e^{{{z_last}}}"
    denom_tex = _denom_three_terms_tex(Theta, K_local, dec=dec)

    return (
        r"$$"
        r"\begin{aligned}" + rf"\hat{{p}}(y={K_local}\mid x) &= \frac{{{num_tex}}}{{{denom_tex}}}"
        r"\end{aligned}"
        r"$$"
    )

def build_multiclass_1d_logistic_figure(
    x1,
    y,
    w_hist,
    b_hist,
    *,
    p_curves_hist=None,
    x1_grid=None,
    loss_hist=None,
    metrics_hist=None,
    show_loss=True,
    history_kind="iterative",
    title=None,
    strict_loss=False,
    dec=3,
    example_class=0,
    max_theta_cols=8,
    frame_duration=80,
    theme=None,
):
    """Internal method to build build_multiclass_1d_logistic_figure."""
    if show_loss and history_kind != "iterative":
        if strict_loss:
            raise ValueError("show_loss=True is only allowed for iterative histories.")
        show_loss = False
        loss_hist = None

    x1 = np.asarray(x1).ravel()
    y = np.asarray(y).ravel()

    w_hist = np.asarray(w_hist, dtype=float)  # (T, 1, K)
    b_hist = np.asarray(b_hist, dtype=float)  # (T, K)

    if w_hist.ndim != 3 or w_hist.shape[1] != 1:
        raise ValueError("For multiclass 1D, w_hist must have shape (steps, 1, K).")
    if b_hist.ndim != 2:
        raise ValueError("For multiclass 1D, b_hist must have shape (steps, K).")
    if w_hist.shape[0] != b_hist.shape[0] or w_hist.shape[2] != b_hist.shape[1]:
        raise ValueError("w_hist and b_hist shapes are inconsistent.")

    steps_n = int(w_hist.shape[0])
    K = int(w_hist.shape[2])

    if title is None:
        title = f"Multiclass Logistic Regression (K={K}, d=1)"

    if p_curves_hist is not None:
        p_curves_hist = np.asarray(p_curves_hist, dtype=float)
        if x1_grid is None:
            raise ValueError("If p_curves_hist is provided, x1_grid must be provided.")
        x1_grid = np.asarray(x1_grid, dtype=float).ravel()

        if p_curves_hist.ndim != 3:
            raise ValueError("p_curves_hist must have shape (steps, grid_points, K).")
        if p_curves_hist.shape != (steps_n, x1_grid.size, K):
            raise ValueError("p_curves_hist shape mismatch with steps/x1_grid/K.")

        def p_curves(t):
            return p_curves_hist[t]

    else:
        x_min, x_max = float(x1.min()), float(x1.max())
        x1_grid = np.linspace(x_min, x_max, 350)

        def p_curves(t):
            Zg = x1_grid.reshape(-1, 1) @ w_hist[t] + b_hist[t].reshape(1, -1)
            return _softmax(Zg)

    if show_loss:
        if loss_hist is None:
            raise ValueError("show_loss=True requires loss_hist.")
        loss_hist = np.asarray(loss_hist, dtype=float).ravel()
        if loss_hist.size != steps_n:
            raise ValueError("loss_hist must match steps.")
        cols = 3
        column_widths = [0.60, 0.22, 0.18]
        specs = [[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]]
        X_TEXT = 0.20
        X_VDOTS = 0.28
    else:
        cols = 2
        column_widths = [0.72, 0.28]
        specs = [[{"type": "xy"}, {"type": "xy"}]]
        X_TEXT = 0.24
        X_VDOTS = 0.32

    ep = np.arange(steps_n)
    ep_list = ep.tolist()
    if show_loss:
        loss_hist_list = loss_hist.tolist()

    fig = make_subplots(
        rows=1, cols=cols,
        column_widths=column_widths,
        horizontal_spacing=0.06,
        specs=specs,
    )

    Pg0 = p_curves(0)
    for k in range(K):
        fig.add_trace(
            go.Scatter(
                x=x1_grid,
                y=Pg0[:, k],
                mode="lines",
                name=f"p(class {k})",
                line=dict(width=4),
                legendgroup="curves" if show_loss else None,
            ),
            row=1, col=2,
        )

    if show_loss:
        fig.add_trace(
            go.Scatter(
                x=[step if i == 0 else None for i, step in enumerate(ep_list)],
                y=[val if i == 0 else None for i, val in enumerate(loss_hist_list)],
                mode="lines",
                name="Cross-entropy",
                line=dict(width=3),
                legendgroup="loss",
                showlegend=True,
            ),
            row=1, col=3,
        )

    def metrics_annotations(t):
        ann = []
        if metrics_hist is not None:
            for i, (name, hist) in enumerate(metrics_hist.items()):
                val = hist[t]
                y_pos = 0.83 - (i * 0.14) if show_loss else 0.83 - (i * 0.14)
                fmt = ".6f" if name.lower() == "log-loss" or name.lower() == "loss" else ".4f"
                ann.append(dict(
                    x=0.98 if show_loss else 0.86, y=y_pos, xref="paper", yref="paper", 
                    text=f"<b>{name}</b><br>{val:{fmt}}", showarrow=False, 
                    xanchor="right" if show_loss else "center", yanchor="top" if show_loss else "bottom", font=dict(size=14, color="black"), 
                    bgcolor="white", bordercolor="black", borderwidth=1, borderpad=6
                ))
        return ann

    def make_annotations(t):
        base_ann = [
            dict(x=X_TEXT, y=0.96, xref="paper", yref="paper", text=_row1_formula_latex(K), showarrow=False, xanchor="center", yanchor="top", font=dict(size=18, color="white")),
            dict(x=X_TEXT, y=0.80, xref="paper", yref="paper", text=_theta_matrix_latex_math_style(w_hist, b_hist, t, max_theta_cols, dec), showarrow=False, xanchor="center", yanchor="middle", font=dict(size=20, color="white")),
            dict(x=X_TEXT, y=0.64, xref="paper", yref="paper", text=_row3_formula_latex(K), showarrow=False, xanchor="center", yanchor="top", font=dict(size=18, color="white")),
            dict(x=X_TEXT, y=0.36, xref="paper", yref="paper", text=_final_prob_example_latex(w_hist, b_hist, t, example_class, dec), showarrow=False, xanchor="center", yanchor="middle", font=dict(size=16, color="white")),
            dict(x=X_VDOTS, y=0.195, xref="paper", yref="paper", text=_vertical_dots_latex(), showarrow=False, xanchor="center", yanchor="middle", font=dict(size=22, color="white")),
            dict(x=X_TEXT, y=0.08, xref="paper", yref="paper", text=_last_class_tail_latex(w_hist, b_hist, t, dec), showarrow=False, xanchor="center", yanchor="middle", font=dict(size=16, color="white")),
        ]
        
        prob_title_x = 0.685 if show_loss else 0.86
        base_ann.append(
            dict(x=prob_title_x, y=0.88, xref="paper", yref="paper", text="<b>Probability</b>", showarrow=False, xanchor="center", yanchor="bottom", font=dict(size=14, color="white"))
        )

        if show_loss:
            base_ann.append(
                dict(x=0.92, y=0.88, xref="paper", yref="paper", text="<b>Cross-entropy</b>", showarrow=False, xanchor="center", yanchor="bottom", font=dict(size=14, color="white"))
            )
            
        return base_ann + metrics_annotations(t)

    frames = []
    for t in range(steps_n):
        Pg = p_curves(t)
        curve_updates = [go.Scatter(x=x1_grid, y=Pg[:, k]) for k in range(K)]
        
        if show_loss:
            loss_update = go.Scatter(
                x=[step if i <= t else None for i, step in enumerate(ep_list)],
                y=[val if i <= t else None for i, val in enumerate(loss_hist_list)]
            )
            frame_data = curve_updates + [loss_update]
            traces = list(range(0, K + 1))
        else:
            frame_data = curve_updates
            traces = list(range(0, K))

        frames.append(
            go.Frame(
                name=str(t),
                data=frame_data,
                traces=traces,
                layout=go.Layout(annotations=make_annotations(t)),
            )
        )
    fig.frames = frames

    if show_loss:
        loss_min, loss_max = float(loss_hist.min()), float(loss_hist.max())
        loss_pad = 0.08 * ((loss_max - loss_min) + 1e-9)

    layout_kwargs = get_base_layout(title=title, margin_t=110, theme=theme)
    if show_loss:
        fig.update_layout(
            **layout_kwargs,
            legend=dict(orientation="v", **get_legend_props(x=1.02, y=0.85, yanchor="top", xanchor="left", theme=theme)),
            legend2=dict(orientation="v", **get_legend_props(x=1.02, y=0.30, yanchor="top", xanchor="left", theme=theme)),
            sliders=get_sliders(steps_n, theme=theme),
            updatemenus=get_updatemenus(frame_duration, theme=theme),
            annotations=make_annotations(0),
        )
        fig.data[K].update(legend="legend2")
    else:
        fig.update_layout(
            **layout_kwargs,
            legend=dict(orientation="v", **get_legend_props(x=1.02, y=0.85, yanchor="top", xanchor="left", theme=theme)),
            sliders=get_sliders(steps_n, theme=theme),
            updatemenus=get_updatemenus(frame_duration, theme=theme),
            annotations=make_annotations(0),
        )

    fig.update_xaxes(visible=False, row=1, col=1, range=[0, 1])
    fig.update_yaxes(visible=False, row=1, col=1, range=[0, 1])
    fig.update_xaxes(title=r"$x$", row=1, col=2)
    fig.update_yaxes(range=[-0.02, 1.02], domain=[0.15, 0.85], row=1, col=2)
    
    if show_loss:
        fig.update_xaxes(title="Step", range=[0, steps_n - 1], row=1, col=3)
        fig.update_yaxes(range=[loss_min - loss_pad, loss_max + loss_pad], domain=[0.15, 0.85], row=1, col=3)
        
    return fig

__all__ = ["build_multiclass_1d_logistic_figure"]
