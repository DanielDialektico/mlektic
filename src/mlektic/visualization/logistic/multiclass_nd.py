"""Multiclass logistic-regression (multivariate) figure builder."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..theme import (
    get_base_layout,
    get_updatemenus,
    get_sliders,
)

def build_multiclass_multivar_logistic_figure(
    X,
    y,
    w_hist,
    b_hist,
    *,
    loss_hist=None,
    show_loss=True,
    history_kind="iterative",
    title=None,
    strict_loss=False,
    dec=3,
    example_class=0,
    max_features_in_z=3,
    frame_duration=80,
    theme=None,
):
    """Internal method to build build_multiclass_multivar_logistic_figure."""
    if show_loss and history_kind != "iterative":
        if strict_loss:
            raise ValueError("show_loss=True is only allowed for iterative histories.")
        show_loss = False
        loss_hist = None

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

    if show_loss:
        loss_min, loss_max = float(loss_hist.min()), float(loss_hist.max())
        loss_pad = 0.08 * ((loss_max - loss_min) + 1e-9)

    def model_formula_latex():
        return r"$$\mathbf{z}=\Theta^\top\mathbf{x},\qquad \hat{\mathbf{p}}=\mathrm{softmax}(\mathbf{z})$$"

    def softmax_def_latex():
        return rf"$$\mathrm{{softmax}}(\mathbf{{z}})_k=\dfrac{{e^{{z_k}}}}{{\sum_{{j=1}}^{{{K}}}e^{{z_j}}}},\;\;k=1,\dots,{K}$$"

    def Theta_definition_latex():
        return (
            rf"$$\Theta\in\mathbb{{R}}^{{({d}+1)\times {K}}},\quad "
            rf"z_k(\mathbf{{x}})=\sum_{{j=1}}^{{{d + 1}}}\theta_{{j,k}}x_j$$"
        )

    def x_vector_latex_capped(d_local, max_rows=7, max_cols=4):
        entries = [rf"x_{{{j}}}" for j in range(1, d_local + 1)] + [r"1"]
        D = d_local + 1
        capacity = max_rows * max_cols

        def vdots_row():
            return " & ".join([r"\vdots"] * max_cols)

        lines = []

        if D <= capacity:
            padded = entries + [r"\;"] * (capacity - D)
            M = np.array(padded, dtype=object).reshape(max_rows, max_cols)
            for r in range(max_rows):
                lines.append(" & ".join(M[r, c] for c in range(max_cols)))
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
        Theta = np.vstack([w_hist[t], b_hist[t].reshape(1, -1)])  # (d+1, K)
        R, C = Theta.shape

        def fmt(v):
            return rf"{v:+.{dec}f}"

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

        return (
            r"$$"
            r"\Theta=\left[\begin{array}{" + cols_spec + r"}" + body + r"\end{array}\right]"
            r"$$"
        )

    def z_numeric_expr(Theta, class_idx, d_local, max_feat=max_features_in_z, dec=dec):
        def num(v):
            return f"{v:+.{dec}f}"

        feat_count = min(d_local, max_feat)
        terms = [rf"\left({num(Theta[j, class_idx])}\right)x_{{{j + 1}}}" for j in range(feat_count)]
        if d_local > max_feat:
            terms.append(r"\cdots")
        terms.append(rf"\left({num(Theta[d_local, class_idx])}\right)")
        return r" + ".join(terms)

    def final_prob_example_latex(t, class_k=example_class, max_feat=max_features_in_z, dec=dec):
        Theta = np.vstack([w_hist[t], b_hist[t].reshape(1, -1)])
        D, K_local = Theta.shape
        d_local = D - 1

        k = int(class_k)
        k = max(0, min(k, K_local - 1))

        z_k = z_numeric_expr(Theta, k, d_local, max_feat=max_feat, dec=dec)
        num_tex = rf"e^{{{z_k}}}"

        z_first = z_numeric_expr(Theta, 0, d_local, max_feat=max_feat, dec=dec)
        z_last = z_numeric_expr(Theta, K_local - 1, d_local, max_feat=max_feat, dec=dec)

        if K_local == 1:
            denom_tex = rf"e^{{{z_first}}}"
        else:
            denom_tex = rf"e^{{{z_first}}} + \cdots + e^{{{z_last}}}"

        return (
            r"$$"
            r"\begin{aligned}"
            + rf"\hat{{p}}(y=1\mid \mathbf{{x}}) &= \frac{{e^{{z_1(\mathbf{{x}})}}}}{{\sum_{{j=1}}^{{{K_local}}} e^{{z_j(\mathbf{{x}})}}}} \\[6pt]"
            + rf"&= \frac{{{num_tex}}}{{{denom_tex}}}"
            r"\end{aligned}"
            r"$$"
        )

    def vertical_dots_latex():
        return r"$$\vdots$$"

    def last_class_tail_latex(t, max_feat=max_features_in_z, dec=dec):
        Theta = np.vstack([w_hist[t], b_hist[t].reshape(1, -1)])
        D, K_local = Theta.shape
        d_local = D - 1

        last_idx = K_local - 1
        z_last = z_numeric_expr(Theta, last_idx, d_local, max_feat=max_feat, dec=dec)
        num_tex = rf"e^{{{z_last}}}"

        z_first = z_numeric_expr(Theta, 0, d_local, max_feat=max_feat, dec=dec)
        if K_local == 1:
            denom_tex = rf"e^{{{z_first}}}"
        else:
            denom_tex = rf"e^{{{z_first}}} + \cdots + e^{{{z_last}}}"

        return (
            r"$$"
            r"\begin{aligned}" + rf"\hat{{p}}(y={K_local}\mid \mathbf{{x}}) &= \frac{{{num_tex}}}{{{denom_tex}}}"
            r"\end{aligned}"
            r"$$"
        )

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.55, 0.45],
        horizontal_spacing=0.06,
        specs=[[{"type": "xy"}, {"type": "xy"}]],
    )

    if show_loss:
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="Cross-entropy", line=dict(width=3)), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="Cross-entropy", line=dict(width=3)), row=1, col=1)

    def make_annotations(t):
        ann = [
            dict(
                x=0.74,
                y=0.965,
                xref="paper",
                yref="paper",
                text=model_formula_latex(),
                showarrow=False,
                xanchor="center",
                yanchor="top",
                font=dict(size=20, color="white"),
            ),
            dict(
                x=0.74,
                y=0.885,
                xref="paper",
                yref="paper",
                text=softmax_def_latex(),
                showarrow=False,
                xanchor="center",
                yanchor="top",
                font=dict(size=18, color="white"),
            ),
            dict(
                x=0.74,
                y=0.775,
                xref="paper",
                yref="paper",
                text=Theta_definition_latex(),
                showarrow=False,
                xanchor="center",
                yanchor="top",
                font=dict(size=18, color="white"),
            ),
            dict(
                x=0.64,
                y=0.49,
                xref="paper",
                yref="paper",
                text=x_vector_latex_capped(d, max_rows=7, max_cols=4),
                showarrow=False,
                xanchor="center",
                yanchor="middle",
                font=dict(size=16, color="white"),
            ),
            dict(
                x=0.84,
                y=0.49,
                xref="paper",
                yref="paper",
                text=Theta_matrix_latex_capped(t, max_rows=7, max_cols=6, dec=dec),
                showarrow=False,
                xanchor="center",
                yanchor="middle",
                font=dict(size=16, color="white"),
            ),
            dict(
                x=0.76,
                y=0.22,
                xref="paper",
                yref="paper",
                text=final_prob_example_latex(t, class_k=example_class, max_feat=max_features_in_z, dec=dec),
                showarrow=False,
                xanchor="center",
                yanchor="middle",
                font=dict(size=16, color="white"),
            ),
            dict(
                x=0.81,
                y=0.095,
                xref="paper",
                yref="paper",
                text=vertical_dots_latex(),
                showarrow=False,
                xanchor="center",
                yanchor="middle",
                font=dict(size=22, color="white"),
            ),
            dict(
                x=0.76,
                y=0.013,
                xref="paper",
                yref="paper",
                text=last_class_tail_latex(t, max_feat=max_features_in_z, dec=dec),
                showarrow=False,
                xanchor="center",
                yanchor="middle",
                font=dict(size=16, color="white"),
            ),
        ]

        if show_loss:
            ann.append(
                dict(
                    x=0.535,
                    y=0.94,
                    xref="paper",
                    yref="paper",
                    text=f"<b>Cross-entropy</b><br>{loss_hist[t]:.6f}",
                    showarrow=False,
                    xanchor="right",
                    yanchor="top",
                    font=dict(size=16, color="black"),
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=8,
                )
            )
        return ann

    frames = []
    for t in range(steps_n):
        trace = go.Scatter(x=ep[: t + 1], y=loss_hist[: t + 1]) if show_loss else go.Scatter(x=[], y=[])
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
        **get_base_layout(title=title, margin_t=110, theme=theme),
        showlegend=False,
        sliders=get_sliders(steps_n, theme=theme),
        updatemenus=get_updatemenus(frame_duration, theme=theme),
        annotations=make_annotations(0),
    )

    if show_loss:
        fig.update_xaxes(title="Step", range=[0, steps_n - 1], row=1, col=1)
        fig.update_yaxes(title="Cross-entropy", range=[loss_min - loss_pad, loss_max + loss_pad], row=1, col=1)
    else:
        fig.update_xaxes(title="Step", row=1, col=1)
        fig.update_yaxes(title="Cross-entropy", row=1, col=1)

    fig.update_xaxes(visible=False, row=1, col=2, range=[0, 1])
    fig.update_yaxes(visible=False, row=1, col=2, range=[0, 1])
    return fig


__all__ = ["build_multiclass_multivar_logistic_figure"]
