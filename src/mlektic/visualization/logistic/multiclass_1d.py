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

def build_multiclass_1d_logistic_figure(
    x1,
    y,
    w_hist,
    b_hist,
    *,
    p_curves_hist=None,
    x1_grid=None,
    loss_hist=None,
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

    ep = np.arange(steps_n)

    def model_formula_latex():
        return r"$$\mathbf{z}=\Theta^\top\mathbf{x},\qquad \hat{\mathbf{p}}=\mathrm{softmax}(\mathbf{z})$$"

    def x_definition_latex():
        return r"$$\mathbf{x}=\begin{bmatrix}x\\1\end{bmatrix}\in\mathbb{R}^{2}$$"

    def softmax_def_latex():
        return rf"$$\mathrm{{softmax}}(\mathbf{{z}})_k=\dfrac{{e^{{z_k}}}}{{\sum_{{j=1}}^{{{K}}}e^{{z_j}}}},\;\;k=1,\dots,{K}$$"

    def theta_definition_latex():
        return rf"$$\Theta\in\mathbb{{R}}^{{2\times {K}}},\quad z_k(x)=\theta_{{1,k}}x+\theta_{{0,k}}$$"

    def theta_matrix_latex_math_style(t, max_elems=max_theta_cols, dec=dec):
        Theta = np.vstack([w_hist[t, 0], b_hist[t]])  # (2,K)
        K_local = Theta.shape[1]

        def fmt(v):
            return rf"{v:+.{dec}f}"

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

    def z_numeric_expr_univar(Theta, class_idx, dec=dec):
        def num(v):
            return f"{v:+.{dec}f}"

        theta_1k = num(Theta[0, class_idx])
        theta_0k = num(Theta[1, class_idx])
        return rf"\left({theta_1k}\right)x + \left({theta_0k}\right)"

    def denom_three_terms_tex(Theta, K_local, dec=dec):
        z1 = z_numeric_expr_univar(Theta, 0, dec=dec)
        if K_local == 1:
            return rf"e^{{{z1}}}"

        z2 = z_numeric_expr_univar(Theta, 1, dec=dec)
        if K_local == 2:
            return rf"e^{{{z1}}} + e^{{{z2}}}"

        zK = z_numeric_expr_univar(Theta, K_local - 1, dec=dec)
        return rf"e^{{{z1}}} + e^{{{z2}}} + \cdots + e^{{{zK}}}"

    def final_prob_example_latex(t, class_k=example_class, dec=dec):
        Theta = np.vstack([w_hist[t, 0], b_hist[t]])
        K_local = Theta.shape[1]

        k = int(class_k)
        k = max(0, min(k, K_local - 1))

        z_k = z_numeric_expr_univar(Theta, k, dec=dec)
        num_tex = rf"e^{{{z_k}}}"
        denom_tex = denom_three_terms_tex(Theta, K_local, dec=dec)

        return (
            r"$$"
            r"\begin{aligned}"
            + rf"\hat{{p}}(y=1\mid x) &= \frac{{e^{{z_1(x)}}}}{{\sum_{{j=1}}^{{{K_local}}} e^{{z_j(x)}}}} \\[6pt]"
            + rf"&= \frac{{{num_tex}}}{{{denom_tex}}}"
            r"\end{aligned}"
            r"$$"
        )

    def vertical_dots_latex():
        return r"$$\vdots$$"

    def last_class_tail_latex(t, dec=dec):
        Theta = np.vstack([w_hist[t, 0], b_hist[t]])
        K_local = Theta.shape[1]
        last_idx = K_local - 1

        z_last = z_numeric_expr_univar(Theta, last_idx, dec=dec)
        num_tex = rf"e^{{{z_last}}}"
        denom_tex = denom_three_terms_tex(Theta, K_local, dec=dec)

        return (
            r"$$"
            r"\begin{aligned}" + rf"\hat{{p}}(y={K_local}\mid x) &= \frac{{{num_tex}}}{{{denom_tex}}}"
            r"\end{aligned}"
            r"$$"
        )

    if show_loss:
        fig = make_subplots(
            rows=1,
            cols=3,
            column_widths=[0.22, 0.18, 0.28],
            horizontal_spacing=0.06,
            specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
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
                    legendgroup="curves",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[loss_hist[0]],
                mode="lines",
                name="Cross-entropy",
                line=dict(width=3),
                legendgroup="loss",
                showlegend=True,
            ),
            row=1,
            col=2,
        )

        X_TEXT = 0.78
        X_VDOTS = 0.82

        def make_annotations(t):
            return [
                dict(
                    x=X_TEXT,
                    y=0.955,
                    xref="paper",
                    yref="paper",
                    text=model_formula_latex(),
                    showarrow=False,
                    xanchor="center",
                    yanchor="top",
                    font=dict(size=20, color="white"),
                ),
                dict(
                    x=X_TEXT,
                    y=0.895,
                    xref="paper",
                    yref="paper",
                    text=x_definition_latex(),
                    showarrow=False,
                    xanchor="center",
                    yanchor="top",
                    font=dict(size=18, color="white"),
                ),
                dict(
                    x=X_TEXT,
                    y=0.805,
                    xref="paper",
                    yref="paper",
                    text=softmax_def_latex(),
                    showarrow=False,
                    xanchor="center",
                    yanchor="top",
                    font=dict(size=18, color="white"),
                ),
                dict(
                    x=X_TEXT,
                    y=0.700,
                    xref="paper",
                    yref="paper",
                    text=theta_definition_latex(),
                    showarrow=False,
                    xanchor="center",
                    yanchor="top",
                    font=dict(size=18, color="white"),
                ),
                dict(
                    x=X_TEXT,
                    y=0.575,
                    xref="paper",
                    yref="paper",
                    text=theta_matrix_latex_math_style(t),
                    showarrow=False,
                    xanchor="center",
                    yanchor="middle",
                    font=dict(size=20, color="white"),
                ),
                dict(
                    x=X_TEXT,
                    y=0.340,
                    xref="paper",
                    yref="paper",
                    text=final_prob_example_latex(t),
                    showarrow=False,
                    xanchor="center",
                    yanchor="middle",
                    font=dict(size=16, color="white"),
                ),
                dict(
                    x=X_VDOTS,
                    y=0.200,
                    xref="paper",
                    yref="paper",
                    text=vertical_dots_latex(),
                    showarrow=False,
                    xanchor="center",
                    yanchor="middle",
                    font=dict(size=22, color="white"),
                ),
                dict(
                    x=X_TEXT,
                    y=0.095,
                    xref="paper",
                    yref="paper",
                    text=last_class_tail_latex(t),
                    showarrow=False,
                    xanchor="center",
                    yanchor="middle",
                    font=dict(size=16, color="white"),
                ),
            ]

        frames = []
        for t in range(steps_n):
            Pg = p_curves(t)
            curve_updates = [go.Scatter(x=x1_grid, y=Pg[:, k]) for k in range(K)]
            loss_update = go.Scatter(x=ep[: t + 1], y=loss_hist[: t + 1])

            frames.append(
                go.Frame(
                    name=str(t),
                    data=curve_updates + [loss_update],
                    traces=list(range(0, K + 1)),
                    layout=go.Layout(annotations=make_annotations(t)),
                )
            )
        fig.frames = frames

        loss_min, loss_max = float(loss_hist.min()), float(loss_hist.max())
        loss_pad = 0.08 * ((loss_max - loss_min) + 1e-9)

        fig.update_layout(
            **get_base_layout(title=title, margin_t=110, theme=theme),
            legend=dict(orientation="v", **get_legend_props(x=0.28, y=0.9, yanchor="top", theme=theme)),
            legend2=dict(orientation="v", **get_legend_props(x=0.58, y=0.9, yanchor="top", theme=theme)),
            sliders=get_sliders(steps_n, theme=theme),
            updatemenus=get_updatemenus(frame_duration, theme=theme),
            annotations=make_annotations(0),
        )

        fig.data[K].update(legend="legend2")
        fig.update_xaxes(title=r"$x$", row=1, col=1)
        fig.update_yaxes(title="Probability", range=[-0.02, 1.02], row=1, col=1)
        fig.update_xaxes(title="Step", range=[0, steps_n - 1], row=1, col=2)
        fig.update_yaxes(title="Cross-entropy", range=[loss_min - loss_pad, loss_max + loss_pad], row=1, col=2)
        fig.update_xaxes(visible=False, row=1, col=3, range=[0, 1])
        fig.update_yaxes(visible=False, row=1, col=3, range=[0, 1])
        return fig

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.58, 0.42],
        horizontal_spacing=0.06,
        specs=[[{"type": "xy"}, {"type": "xy"}]],
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
            ),
            row=1,
            col=1,
        )

    X_TEXT = 0.78
    X_VDOTS = 0.82

    def make_annotations_no_loss(t):
        return [
            dict(
                x=X_TEXT,
                y=0.955,
                xref="paper",
                yref="paper",
                text=model_formula_latex(),
                showarrow=False,
                xanchor="center",
                yanchor="top",
                font=dict(size=20, color="white"),
            ),
            dict(
                x=X_TEXT,
                y=0.885,
                xref="paper",
                yref="paper",
                text=x_definition_latex(),
                showarrow=False,
                xanchor="center",
                yanchor="top",
                font=dict(size=18, color="white"),
            ),
            dict(
                x=X_TEXT,
                y=0.785,
                xref="paper",
                yref="paper",
                text=softmax_def_latex(),
                showarrow=False,
                xanchor="center",
                yanchor="top",
                font=dict(size=18, color="white"),
            ),
            dict(
                x=X_TEXT,
                y=0.665,
                xref="paper",
                yref="paper",
                text=theta_definition_latex(),
                showarrow=False,
                xanchor="center",
                yanchor="top",
                font=dict(size=18, color="white"),
            ),
            dict(
                x=X_TEXT,
                y=0.53,
                xref="paper",
                yref="paper",
                text=theta_matrix_latex_math_style(t),
                showarrow=False,
                xanchor="center",
                yanchor="middle",
                font=dict(size=20, color="white"),
            ),
            dict(
                x=X_TEXT,
                y=0.29,
                xref="paper",
                yref="paper",
                text=final_prob_example_latex(t),
                showarrow=False,
                xanchor="center",
                yanchor="middle",
                font=dict(size=16, color="white"),
            ),
            dict(
                x=X_VDOTS,
                y=0.17,
                xref="paper",
                yref="paper",
                text=vertical_dots_latex(),
                showarrow=False,
                xanchor="center",
                yanchor="middle",
                font=dict(size=22, color="white"),
            ),
            dict(
                x=X_TEXT,
                y=0.08,
                xref="paper",
                yref="paper",
                text=last_class_tail_latex(t),
                showarrow=False,
                xanchor="center",
                yanchor="middle",
                font=dict(size=16, color="white"),
            ),
        ]

    frames = []
    for t in range(steps_n):
        Pg = p_curves(t)
        frame_data = [go.Scatter(x=x1_grid, y=Pg[:, k]) for k in range(K)]
        frames.append(
            go.Frame(
                name=str(t),
                data=frame_data,
                traces=list(range(0, K)),
                layout=go.Layout(annotations=make_annotations_no_loss(t)),
            )
        )
    fig.frames = frames

    fig.update_layout(
        **get_base_layout(title=title, margin_t=110, theme=theme),
        legend=dict(orientation="v", **get_legend_props(x=0.55, y=0.9, yanchor="top", theme=theme)),
        sliders=get_sliders(steps_n, theme=theme),
        updatemenus=get_updatemenus(frame_duration, theme=theme),
        annotations=make_annotations_no_loss(0),
    )

    fig.update_xaxes(title=r"$x$", row=1, col=1)
    fig.update_yaxes(title="Probability", range=[-0.02, 1.02], row=1, col=1)
    fig.update_xaxes(visible=False, row=1, col=2, range=[0, 1])
    fig.update_yaxes(visible=False, row=1, col=2, range=[0, 1])
    return fig


__all__ = ["build_multiclass_1d_logistic_figure"]
