"""Multiclass logistic-regression (multivariate) figure builder."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...utils.probability import multiclass_link_latex
from ..theme import (
    get_base_layout,
    get_sliders,
    get_updatemenus,
)
from ._math_layout import (
    MULTICLASS_ELLIPSIS_FONT_SIZE,
    MULTICLASS_PROBABILITY_FONT_SIZE,
    MULTICLASS_PROBABILITY_ROW_GAP,
    compact_probability_fraction_latex,
)


def build_multiclass_multivar_logistic_figure(
    X,
    y,
    w_hist,
    b_hist,
    *,
    loss_hist=None,
    metrics_hist=None,
    show_loss=True,
    classes=None,
    show_class_labels=False,
    history_kind="iterative",
    title=None,
    strict_loss=False,
    dec=3,
    example_class=0,
    max_features_in_z=3,
    max_theta_cols=6,
    frame_duration=80,
    probability_link="softmax",
    theme=None,
):
    """Internal method to build build_multiclass_multivar_logistic_figure."""
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    w_hist = np.asarray(w_hist, dtype=float)  # (T, d, K)
    b_hist = np.asarray(b_hist, dtype=float)  # (T, K)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if w_hist.ndim != 3:
        raise ValueError("For multiclass multivar, w_hist must have shape (steps, d, K).")
    if b_hist.ndim != 2:
        raise ValueError("For multiclass multivar, b_hist must have shape (steps, K).")
    if w_hist.shape[0] != b_hist.shape[0] or w_hist.shape[2] != b_hist.shape[1]:
        raise ValueError("w_hist and b_hist shapes are inconsistent.")

    steps_n = int(w_hist.shape[0])
    d = int(w_hist.shape[1])
    K = int(w_hist.shape[2])

    if X.shape[1] != d:
        raise ValueError(f"X has d={X.shape[1]} but w_hist has d={d}.")

    if title is None:
        title = f"Multiclass Logistic Regression (K={K}, d={d})"

    if show_loss:
        if loss_hist is None:
            raise ValueError("show_loss=True requires loss_hist.")
        loss_hist = np.asarray(loss_hist, dtype=float).ravel()
        if loss_hist.size != steps_n:
            raise ValueError("loss_hist must match steps.")

    ep = np.arange(steps_n)
    ep_list = ep.tolist()

    if show_loss:
        loss_hist_list = loss_hist.tolist()
        loss_min, loss_max = float(loss_hist.min()), float(loss_hist.max())
        loss_pad = 0.08 * ((loss_max - loss_min) + 1e-9)

    def row1_formula_latex():
        return rf"$$\mathbf{{z}}=\Theta^\top\mathbf{{x}}+\boldsymbol{{\theta}}_0,\quad \mathbf{{x}}\in\mathbb{{R}}^{{{d}}},\quad \Theta\in\mathbb{{R}}^{{{d}\times {K}}},\quad \boldsymbol{{\theta}}_0\in\mathbb{{R}}^{{{K}}}$$"

    def row3_formula_latex():
        definition = multiclass_link_latex(probability_link, K)
        return rf"$$z_k(\mathbf{{x}})=\theta_{{0,k}}+\sum_{{j=1}}^{{{d}}}\theta_{{j,k}}x_j,\quad {definition},\quad k=1,\ldots,{K}$$"

    def x_vector_latex_capped(d_local, max_rows=7, max_cols=4):
        entries = [rf"x_{{{j}}}" for j in range(1, d_local + 1)]
        D = d_local
        capacity = max_rows * max_cols

        def vdots_row():
            return " & ".join([r"\vdots"] * max_cols)

        lines = []

        if D <= capacity:
            visible_cols = min(max_cols, D)
            visible_rows = max(1, (D + visible_cols - 1) // visible_cols)
            visible_capacity = visible_rows * visible_cols
            padded = entries + [r"\;"] * (visible_capacity - D)
            M = np.array(padded, dtype=object).reshape(visible_rows, visible_cols)
            for row in range(visible_rows):
                lines.append(" & ".join(M[row, col] for col in range(visible_cols)))
        else:
            head_rows = max(2, max_rows // 2 - 1)
            tail_rows = max_rows - head_rows - 1

            head_count = head_rows * max_cols
            tail_count = tail_rows * max_cols

            head_items = entries[:head_count]
            tail_items = entries[-tail_count:]

            H = np.array(head_items, dtype=object).reshape(head_rows, max_cols)
            for r in range(head_rows):
                lines.append(" & ".join(H[r, c] for c in range(max_cols)))

            lines.append(vdots_row())

            T = np.array(tail_items, dtype=object).reshape(tail_rows, max_cols)
            for r in range(tail_rows):
                lines.append(" & ".join(T[r, c] for c in range(max_cols)))

        body = r" \\ ".join(lines)
        return rf"$$\mathbf{{x}}=\begin{{bmatrix}} {body} \end{{bmatrix}}$$"

    def Theta_matrix_latex_capped(t, max_rows=7, max_cols=6, dec=dec):
        Theta = w_hist[t]
        R, C = Theta.shape

        def fmt(v):
            return rf"{v:.{dec}f}"

        if R <= max_rows:
            row_slots = list(range(R))
        else:
            head_r = (max_rows - 1) // 2
            tail_r = max_rows - head_r - 1
            row_slots = list(range(head_r)) + [None] + list(range(R - tail_r, R))

        if C <= max_cols:
            col_slots = list(range(C))
        else:
            head_c = (max_cols - 1) // 2
            tail_c = max_cols - head_c - 1
            col_slots = list(range(head_c)) + [None] + list(range(C - tail_c, C))

        lines = []
        for r in row_slots:
            items = []
            for c in col_slots:
                if r is None and c is None:
                    items.append(r"\ddots")
                elif r is None:
                    items.append(r"\vdots")
                elif c is None:
                    items.append(r"\cdots")
                else:
                    items.append(fmt(Theta[r, c]))
            lines.append(" & ".join(items))

        body = r" \\ ".join(lines)
        cols_spec = "c" * len(col_slots)

        bias_slots = []
        for c in col_slots:
            bias_slots.append(r"\cdots" if c is None else fmt(b_hist[t, c]))
        bias_body = " & ".join(bias_slots)
        return (
            r"$$\begin{aligned}\Theta_t&=\left[\begin{array}{"
            + cols_spec
            + r"}"
            + body
            + rf"\end{{array}}\right]\in\mathbb{{R}}^{{{d}\times {K}}}\\"
            + r"\boldsymbol{\theta}_{0,t}&=\begin{bmatrix}"
            + bias_body
            + rf"\end{{bmatrix}}\in\mathbb{{R}}^{{{K}}}\end{{aligned}}$$"
        )

    def linked_term(expression):
        if probability_link == "ovr":
            return rf"\sigma\!\left({expression}\right)"
        return rf"e^{{{expression}}}"

    def z_numeric_expr(Theta, class_idx, d_local, max_feat=max_features_in_z, dec=dec):
        def num(v):
            return f"{v:.{dec}f}"

        feat_count = min(d_local, max_feat)
        terms = [rf"\left({num(Theta[j, class_idx])}\right)x_{{{j + 1}}}" for j in range(feat_count)]
        if d_local > max_feat:
            terms.append(r"\cdots")
        terms.append(rf"\left({num(Theta[d_local, class_idx])}\right)")
        return r" + ".join(terms)

    def final_prob_top_latex(t, class_k=example_class, max_feat=max_features_in_z, dec=dec, compact=False):
        Theta = np.vstack([w_hist[t], b_hist[t].reshape(1, -1)])
        D, K_local = Theta.shape
        d_local = D - 1
        k = int(class_k)
        k = max(0, min(k, K_local - 1))
        z_k = z_numeric_expr(Theta, k, d_local, max_feat=max_feat, dec=dec)
        if compact:
            probability = compact_probability_fraction_latex(k + 1, K_local, probability_link)
        else:
            num_tex = linked_term(z_k)
            z_first = z_numeric_expr(Theta, 0, d_local, max_feat=max_feat, dec=dec)
            z_last_expr = z_numeric_expr(Theta, K_local - 1, d_local, max_feat=max_feat, dec=dec)
            if K_local == 1:
                denom_tex = linked_term(z_first)
            else:
                denom_tex = rf"{linked_term(z_first)} + \cdots + {linked_term(z_last_expr)}"
            probability = rf"\frac{{{num_tex}}}{{{denom_tex}}}"
        return (
            r"$$"
            r"\begin{aligned}"
            + rf"z_{{{k + 1}}}(\mathbf{{x}})&={z_k}"
            + MULTICLASS_PROBABILITY_ROW_GAP
            + rf"\hat{{p}}(Y=c_{{{k + 1}}}\mid\mathbf{{x}})&={probability}"
            r"\end{aligned}"
            r"$$"
        )

    def vertical_dots_latex():
        return r"$$\vdots$$"

    def final_prob_bottom_latex(t, max_feat=max_features_in_z, dec=dec, compact=False):
        Theta = np.vstack([w_hist[t], b_hist[t].reshape(1, -1)])
        D, K_local = Theta.shape
        d_local = D - 1
        if compact:
            probability = compact_probability_fraction_latex(K_local, K_local, probability_link)
        else:
            z_first = z_numeric_expr(Theta, 0, d_local, max_feat=max_feat, dec=dec)
            z_last_expr = z_numeric_expr(Theta, K_local - 1, d_local, max_feat=max_feat, dec=dec)
            num_tex_last = linked_term(z_last_expr)
            if K_local == 1:
                denom_tex = linked_term(z_first)
            else:
                denom_tex = rf"{linked_term(z_first)} + \cdots + {linked_term(z_last_expr)}"
            probability = rf"\frac{{{num_tex_last}}}{{{denom_tex}}}"
        return (
            r"$$"
            r"\begin{aligned}"
            + rf"\hat{{p}}(Y=c_{{{K_local}}}\mid\mathbf{{x}})&={probability}"
            r"\end{aligned}"
            r"$$"
        )

    if show_loss:
        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.75, 0.25],
            horizontal_spacing=0.06,
            specs=[[{"type": "xy"}, {"type": "xy"}]],
        )
    else:
        fig = make_subplots(rows=1, cols=1, specs=[[{"type": "xy"}]])

    fig.add_trace(
        go.Scatter(
            x=[step if i == 0 else None for i, step in enumerate(ep_list)] if show_loss else [],
            y=[val if i == 0 else None for i, val in enumerate(loss_hist_list)] if show_loss else [],
            mode="lines",
            name="Cross-entropy",
            line=dict(width=3)
        ),
        row=1, col=2 if show_loss else 1
    )

    def make_annotations(t):
        dense_layout = d > 12
        math_center = 0.275 if show_loss else 0.5
        theta_x = 0.24 if show_loss else 0.35
        dots_x = 0.13 if show_loss else 0.31
        matrix_y = 0.72 if dense_layout else 0.80
        row3_y = 0.40 if dense_layout else 0.48
        probability_y = 0.28 if dense_layout else 0.36
        dots_y = 0.07 if dense_layout else 0.08
        bottom_y = -0.10 if dense_layout else -0.12
        ann = [
            dict(
                x=math_center,
                y=1.04,
                xref="paper",
                yref="paper",
                text=row1_formula_latex(),
                showarrow=False,
                xanchor="center",
                yanchor="top",
                font=dict(size=18, color="white"),
            ),
            dict(
                x=0.001,
                y=matrix_y,
                xref="paper",
                yref="paper",
                text=x_vector_latex_capped(d, max_rows=7, max_cols=4),
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(size=16, color="white"),
            ),
            dict(
                x=theta_x,
                y=matrix_y,
                xref="paper",
                yref="paper",
                text=Theta_matrix_latex_capped(t, max_rows=7, max_cols=max_theta_cols, dec=dec),
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(size=16, color="white"),
            ),
            dict(
                x=0.01,
                y=row3_y,
                xref="paper",
                yref="paper",
                text=row3_formula_latex(),
                showarrow=False,
                xanchor="left",
                yanchor="top",
                font=dict(size=14, color="white"),
            ),
            dict(
                x=0.01,
                y=probability_y,
                xref="paper",
                yref="paper",
                text=final_prob_top_latex(t, compact=show_loss),
                showarrow=False,
                xanchor="left",
                yanchor="top",
                font=dict(size=MULTICLASS_PROBABILITY_FONT_SIZE, color="white"),
            ),
            dict(
                x=dots_x,
                y=dots_y,
                xref="paper",
                yref="paper",
                text=vertical_dots_latex(),
                showarrow=False,
                xanchor="center",
                yanchor="middle",
                font=dict(size=MULTICLASS_ELLIPSIS_FONT_SIZE, color="white"),
            ),
            dict(
                x=0.01,
                y=bottom_y,
                xref="paper",
                yref="paper",
                text=final_prob_bottom_latex(t, max_feat=max_features_in_z, dec=dec, compact=show_loss),
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font=dict(size=MULTICLASS_PROBABILITY_FONT_SIZE, color="white"),
            ),
        ]

        if show_loss:
            if metrics_hist is not None:
                for i, (name, hist) in enumerate(metrics_hist.items()):
                    val = hist[t]
                    y_pos = 0.83 - (i * 0.14)
                    fmt = ".6f" if name.lower() == "cross-entropy" or name.lower() == "loss" else ".4f"
                    ann.append(
                        dict(
                            x=0.98,
                            y=y_pos,
                            xref="paper",
                            yref="paper",
                            text=f"<b>{name}</b><br>{val:{fmt}}",
                            showarrow=False,
                            xanchor="right",
                            yanchor="top",
                            font=dict(size=11, color="black"),
                            bgcolor="white",
                            bordercolor="black",
                            borderwidth=1,
                            borderpad=4,
                        )
                    )
            else:
                ann.append(
                    dict(
                        x=0.98,
                        y=0.95,
                        xref="paper",
                        yref="paper",
                        text=f"<b>Cross-entropy</b><br>{loss_hist[t]:.6f}",
                        showarrow=False,
                        xanchor="right",
                        yanchor="top",
                        font=dict(size=11, color="black"),
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=1,
                        borderpad=4,
                    )
                )
            ann.append(
                dict(
                    x=0.85,
                    y=0.88,
                    xref="paper",
                    yref="paper",
                    text="Cross-entropy",
                    showarrow=False,
                    xanchor="center",
                    yanchor="bottom",
                    font=dict(size=14, color="white"),
                )
            )
        return ann

    frames = []
    for t in range(steps_n):
        trace = (
            go.Scatter(
                x=[step if i <= t else None for i, step in enumerate(ep_list)],
                y=[val if i <= t else None for i, val in enumerate(loss_hist_list)]
            )
            if show_loss else go.Scatter(x=[], y=[])
        )
        frames.append(
            go.Frame(
                name=str(t),
                data=[trace],
                traces=[0],
                layout=go.Layout(annotations=make_annotations(t)),
            )
        )
    fig.frames = frames

    fig.update_layout(
        **get_base_layout(title=title, margin_t=110, height=720 if d > 12 else 600, theme=theme),
        showlegend=False,
        sliders=get_sliders(steps_n, theme=theme),
        updatemenus=get_updatemenus(frame_duration, theme=theme),
        annotations=make_annotations(0),
    )

    fig.update_xaxes(visible=False, row=1, col=1, range=[0, 1])
    fig.update_yaxes(visible=False, row=1, col=1, range=[0, 1])

    if show_loss:
        fig.update_xaxes(title="Step", range=[0, steps_n - 1], row=1, col=2)
        fig.update_yaxes(range=[loss_min - loss_pad, loss_max + loss_pad], domain=[0.15, 0.85], row=1, col=2)
    return fig


__all__ = ["build_multiclass_multivar_logistic_figure"]
