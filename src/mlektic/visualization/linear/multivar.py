"""Multivariate linear-regression visualization builder."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..theme import (
    get_base_layout,
    get_updatemenus,
    get_sliders,
    create_annotation,
)

def _needs_single_col(values, max_digits=5):
    vals = np.asarray(values, dtype=float).ravel()
    for v in vals:
        if not np.isfinite(v):
            return True
        int_digits = len(str(int(abs(float(v)))))
        if int_digits > max_digits:
            return True
    return False

def _theta_is_big_for_t(w_hist, t, max_digits=5):
    return _needs_single_col(w_hist[t], max_digits=max_digits)

def _metric_box(title, val, y_pos, fmt="6f"):
    return dict(
        x=0.33, y=y_pos, xref="paper", yref="paper",
        text=f"<b>{title}</b><br>{val:{fmt}}",
        showarrow=False, xanchor="left", yanchor="top",
        font=dict(size=16, color="black"), bgcolor="white",
        bordercolor="black", borderwidth=1, borderpad=8,
    )

def _metric_box_matrix(title, val, y_pos, fmt="6f"):
    return dict(
        x=0.25, y=y_pos, xref="paper", yref="paper",
        text=f"<b>{title}</b><br>{val:{fmt}}",
        showarrow=False, xanchor="left", yanchor="top",
        font=dict(size=16, color="black"), bgcolor="white",
        bordercolor="black", borderwidth=1, borderpad=8,
    )

def _make_expansion_annotations(t, d, w_hist, b_hist, dec, terms_per_line, show_loss, metrics_hist):
    def model_header_latex():
        return rf"$$\hat{{y}} = \sum_{{j=1}}^{{{d}}} \theta_j x_j + \theta_0$$"

    def full_scalar_model_multiline_latex(t_idx):
        w = w_hist[t_idx]
        b = float(b_hist[t_idx])
        terms = [rf"({w[i]:.{dec}f})x_{{{i + 1}}}" for i in range(d)]
        chunks = [terms[i : i + terms_per_line] for i in range(0, len(terms), terms_per_line)]
        lines = [r"\hat{y} = " + " + ".join(chunks[0])]
        for ch in chunks[1:]:
            lines.append(r"\quad " + " + ".join(ch))
        lines[-1] = lines[-1] + rf" + ({b:.{dec}f})"
        body = r" \\ ".join(lines)
        return r"$$\begin{aligned}" + body + r"\end{aligned}$$"

    ann = [
        create_annotation(model_header_latex(), x=0.68, y=0.93, size=22, yanchor="top"),
        create_annotation(full_scalar_model_multiline_latex(t), x=0.68, y=0.78, size=17, yanchor="top"),
    ]

    if show_loss and metrics_hist is not None:
        for i, (name, hist) in enumerate(metrics_hist.items()):
            y_pos = 0.94 - (i * 0.13)
            fmt = ".6f" if name.lower() == "loss" else ".4f"
            ann.append(_metric_box(name, hist[t], y_pos, fmt))
    return ann

def _make_matrix_annotations(t, d, w_hist, b_hist, dec, force_theta_one_col, show_loss, metrics_hist):
    rows = 15
    x_cols = 5
    capacity_x = rows * x_cols

    th_cols = 1 if force_theta_one_col else 5

    def model_formula_latex():
        return r"$$\hat{y} = \theta_0 + \operatorname{vec}(\boldsymbol{\theta})^\top \operatorname{vec}(\mathbf{x})$$"

    def bias_latex(t_idx):
        return rf"$$\theta_0 = {float(b_hist[t_idx]):.{dec}f}$$"

    def x_dim_latex():
        return rf"$$\mathbf{{x}} \in \mathbb{{R}}^{{{d}\times {x_cols}}}$$"

    def theta_dim_latex():
        return rf"$$\boldsymbol{{\theta}} \in \mathbb{{R}}^{{{d}\times {th_cols}}}$$"

    def x_vector_latex():
        def cell(j): return rf"x_{{{j}}}"
        def vdots_row(): return " & ".join([r"\vdots"] * x_cols)

        lines = []
        if d <= capacity_x:
            items = [cell(j) for j in range(1, d + 1)] + [r"\;"] * (capacity_x - d)
            M = np.array(items, dtype=object).reshape(rows, x_cols)
            for r in range(rows):
                lines.append(" & ".join(M[r, c] for c in range(x_cols)))
        else:
            head_rows = rows // 2
            tail_rows = rows - head_rows - 1

            head_js = list(range(1, head_rows * x_cols + 1))
            H = np.array([cell(j) for j in head_js], dtype=object).reshape(head_rows, x_cols)
            for r in range(head_rows):
                lines.append(" & ".join(H[r, c] for c in range(x_cols)))

            lines.append(vdots_row())

            tail_count = tail_rows * x_cols
            tail_js = list(range(d - tail_count + 1, d + 1))
            T = np.array([cell(j) for j in tail_js], dtype=object).reshape(tail_rows, x_cols)
            for r in range(tail_rows):
                lines.append(" & ".join(T[r, c] for c in range(x_cols)))

        body = r" \\ ".join(lines)
        return rf"$$\mathbf{{x}} = \begin{{bmatrix}} {body} \end{{bmatrix}}$$"

    def w_matrix_latex(t_idx):
        w = np.asarray(w_hist[t_idx], dtype=float).ravel()
        d_local = w.size

        def fmt(x): return rf"{x:+.{dec}f}"

        if th_cols == 1:
            if d_local <= rows:
                lines = [fmt(w[i]) for i in range(d_local)] + [r"\;"] * (rows - d_local)
            else:
                head_rows = rows // 2
                tail_rows = rows - head_rows - 1
                lines = [fmt(v) for v in w[:head_rows]] + [r"\vdots"] + [fmt(v) for v in w[-tail_rows:]]
            body = r" \\ ".join(lines)
            return rf"$$\boldsymbol{{\theta}} = \begin{{bmatrix}} {body} \end{{bmatrix}}$$"

        th_capacity = rows * th_cols
        def vdots_row(): return " & ".join([r"\vdots"] * th_cols)

        lines = []
        if d_local <= th_capacity:
            padded = np.full(th_capacity, np.nan, dtype=float)
            padded[:d_local] = w
            W = padded.reshape(rows, th_cols)
            for r in range(rows):
                row_items = [r"\;" if np.isnan(W[r, c]) else fmt(W[r, c]) for c in range(th_cols)]
                lines.append(" & ".join(row_items))
        else:
            head_rows = rows // 2
            tail_rows = rows - head_rows - 1
            H = w[: head_rows * th_cols].reshape(head_rows, th_cols)
            for r in range(head_rows):
                lines.append(" & ".join(fmt(H[r, c]) for c in range(th_cols)))
            lines.append(vdots_row())
            T = w[-(tail_rows * th_cols) :].reshape(tail_rows, th_cols)
            for r in range(tail_rows):
                lines.append(" & ".join(fmt(T[r, c]) for c in range(th_cols)))

        body = r" \\ ".join(lines)
        return rf"$$\boldsymbol{{\theta}} = \begin{{bmatrix}} {body} \end{{bmatrix}}$$"

    def scalar_model_compact_latex(t_idx):
        w = np.asarray(w_hist[t_idx], dtype=float).ravel()
        b = float(b_hist[t_idx])
        last = d
        if th_cols == 1:
            return r"$$\hat{y} = " + rf"({w[0]:.{dec}f})x_1 " + rf"+ \cdots + ({w[last - 1]:.{dec}f})x_{{{last}}} " + rf"+ ({b:.{dec}f}) $$"
        return r"$$\hat{y} = " + rf"({w[0]:.{dec}f})x_1 " + rf"+ ({w[1]:.{dec}f})x_2 " + rf"+ ({w[2]:.{dec}f})x_3 " + rf"+ ({w[3]:.{dec}f})x_4 " + rf"+ \cdots + ({w[last - 1]:.{dec}f})x_{{{last}}} " + rf"+ ({b:.{dec}f}) $$"

    ann = [
        create_annotation(model_formula_latex(), x=0.68, y=0.995, size=22, yanchor="top"),
        create_annotation(bias_latex(t), x=0.68, y=0.938, size=18, yanchor="top"),
        create_annotation(x_dim_latex(), x=0.55, y=0.83, size=14, yanchor="bottom"),
        create_annotation(theta_dim_latex(), x=0.83, y=0.83, size=14, yanchor="bottom"),
        create_annotation(x_vector_latex(), x=0.52, y=0.48, size=15, yanchor="middle"),
        create_annotation(w_matrix_latex(t), x=0.80, y=0.48, size=15, yanchor="middle"),
        create_annotation(scalar_model_compact_latex(t), x=0.71, y=0.01, size=16, yanchor="middle"),
    ]

    if show_loss:
        y_loss = 0.98 if th_cols == 1 else 0.86
        if metrics_hist is not None:
            for i, (name, hist) in enumerate(metrics_hist.items()):
                y_p = y_loss - (i * 0.13)
                fmt = ".6f" if name.lower() == "loss" else ".4f"
                ann.append(_metric_box_matrix(name, hist[t], y_p, fmt))
    return ann

def build_multivar_lr_figure(
    X,
    y,
    w_hist,
    b_hist,
    *,
    loss_hist=None,
    metrics_hist=None,
    show_loss=True,
    history_kind="iterative",
    title=None,
    strict_loss=False,
    terms_per_line=6,
    dec=4,
    frame_duration=80,
    threshold_dense=100,
    theme=None,
):
    """
    Multivariable visualization for d > 2 (parameter display).
    """
    if show_loss and history_kind != "iterative":
        if strict_loss:
            raise ValueError("show_loss=True is only allowed for iterative histories.")
        show_loss = False
        loss_hist = None

    X = np.asarray(X)
    y = np.asarray(y).ravel()
    w_hist = np.asarray(w_hist, dtype=float)
    b_hist = np.asarray(b_hist, dtype=float).ravel()

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if w_hist.ndim == 1:
        steps_n = int(b_hist.size)
        if steps_n < 1:
            raise ValueError("Need at least 1 step to animate.")
        if w_hist.size % steps_n != 0:
            raise ValueError("w_hist is 1D but cannot be reshaped to (steps, d) using b_hist length.")
        d = int(w_hist.size // steps_n)
        w_hist = w_hist.reshape(steps_n, d)
    else:
        steps_n = int(b_hist.size)
        if steps_n < 1:
            raise ValueError("Need at least 1 step to animate.")
        if w_hist.ndim != 2:
            raise ValueError("w_hist must have shape (steps, d).")
        if w_hist.shape[0] != steps_n:
            raise ValueError("w_hist must have shape (steps, d) and match b_hist length.")
        d = int(w_hist.shape[1])

    d_X = int(X.shape[1])
    if d_X != d:
        raise ValueError(
            f"X has d={d_X} features but w_hist has d={d}. For d>2 visualization we require theta compatible with X."
        )

    if d <= 2:
        raise ValueError("This figure is intended for d > 2. Use 1D/2D figures for d<=2.")

    if title is None:
        title = f"Multivariable Linear Regression Model ({d} variables)"

    if show_loss:
        if loss_hist is None:
            raise ValueError("show_loss=True requires loss_hist.")
        loss_hist = np.asarray(loss_hist, dtype=float).ravel()
        if loss_hist.size != steps_n:
            raise ValueError("loss_hist must have the same length as b_hist.")

    step_axis_list = np.arange(steps_n).tolist()

    if show_loss:
        loss_hist_list = loss_hist.tolist()
        lmin, lmax = float(loss_hist.min()), float(loss_hist.max())
        lpad = 0.08 * ((lmax - lmin) + 1e-9)

    force_matrix_for_dense = False
    for t in range(steps_n):
        if _theta_is_big_for_t(w_hist, t):
            force_matrix_for_dense = True
            break

    is_expansion_mode = (d <= threshold_dense and not force_matrix_for_dense)

    if is_expansion_mode:
        def _get_ann(t):
            return _make_expansion_annotations(t, d, w_hist, b_hist, dec, terms_per_line, show_loss, metrics_hist)
        updatemenus_y = 1.10
    else:
        force_theta_one_col = force_matrix_for_dense
        def _get_ann(t):
            return _make_matrix_annotations(t, d, w_hist, b_hist, dec, force_theta_one_col, show_loss, metrics_hist)
        updatemenus_y = 1.11

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.42, 0.58],
        horizontal_spacing=0.06,
        specs=[[{"type": "xy"}, {"type": "xy"}]],
    )

    fig.add_trace(
        go.Scatter(
            x=[step if i == 0 else None for i, step in enumerate(step_axis_list)] if show_loss else [],
            y=[val if i == 0 else None for i, val in enumerate(loss_hist_list)] if show_loss else [],
            mode="lines",
            name="Loss",
            line=dict(width=3),
            uid="LOSS_LINE",
        ),
        row=1,
        col=1,
    )

    frames = []
    for t in range(steps_n):
        if show_loss:
            loss_trace = go.Scatter(
                x=[step if i <= t else None for i, step in enumerate(step_axis_list)],
                y=[val if i <= t else None for i, val in enumerate(loss_hist_list)],
                mode="lines",
                line=dict(width=3),
                uid="LOSS_LINE",
            )
        else:
            loss_trace = go.Scatter(x=[], y=[], uid="LOSS_LINE")

        frames.append(
            go.Frame(
                name=str(t),
                data=[loss_trace],
                traces=[0],
                layout=go.Layout(annotations=_get_ann(t)),
            )
        )
    fig.frames = frames

    fig.update_layout(
        **get_base_layout(title=title, margin_t=110, height=760, theme=theme),
        showlegend=True,
        legend=dict(x=0.40, y=0.01, xanchor="right", yanchor="bottom"),
        sliders=get_sliders(steps_n, theme=theme),
        updatemenus=get_updatemenus(frame_duration, y=updatemenus_y, theme=theme),
        annotations=_get_ann(0),
    )

    fig.update_xaxes(title="Step", range=[0, steps_n - 1], row=1, col=1)
    if show_loss:
        fig.update_yaxes(title="Loss", range=[lmin - lpad, lmax + lpad], row=1, col=1)
    else:
        fig.update_yaxes(title="Loss", row=1, col=1)

    fig.update_xaxes(visible=False, row=1, col=2, range=[0, 1])
    fig.update_yaxes(visible=False, row=1, col=2, range=[0, 1])

    return fig

__all__ = ["build_multivar_lr_figure"]
