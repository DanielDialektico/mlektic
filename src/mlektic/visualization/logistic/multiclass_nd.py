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
    MULTICLASS_PROBABILITY_FONT_SIZE,
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
    size="default",
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
        return rf"$$\mathbf{{z}}=\Theta^\top\mathbf{{x}}+\boldsymbol{{\theta}}_0,\quad \mathbf{{x}}\in\mathbb{{R}}^{{{d}}},\quad \Theta\in\mathbb{{R}}^{{{d}\times {K}}},\quad \boldsymbol{{\theta}}_0\in\mathbb{{R}}^{{{K}}},\quad K={K}$$"

    def row3_formula_latex():
        definition = multiclass_link_latex(probability_link, K)
        if probability_link == "ovr":
            link, probability = definition.split(r",\qquad ", 1)
            definition = rf"{probability},\quad {link}"
        return rf"$${definition},\quad z_k(\mathbf{{x}})=\theta_{{0,k}}+\sum_{{j=1}}^{{{d}}}\theta_{{j,k}}x_j,\quad k=0,\ldots,{K - 1}$$"

    def x_vector_latex_capped(d_local, max_rows=7):
        entries = [rf"x_{{{j}}}" for j in range(1, d_local + 1)]
        if d_local <= max_rows:
            lines = entries
        else:
            head_rows = (max_rows - 1) // 2
            tail_rows = max_rows - head_rows - 1
            lines = entries[:head_rows] + [r"\vdots"] + entries[-tail_rows:]
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
        elif max_cols == 1:
            col_slots = [0, None]
        elif max_cols == 2:
            col_slots = [0, None, C - 1]
        else:
            # The ellipsis marks omitted classes; max_cols caps numeric columns.
            head_c = max(1, (max_cols - 1) // 2)
            tail_c = max(1, max_cols - head_c - 1)
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
            r"$$\begin{aligned}\Theta&=\left[\begin{array}{"
            + cols_spec
            + r"}"
            + body
            + rf"\end{{array}}\right]\in\mathbb{{R}}^{{{d}\times {K}}}\\[5pt]"
            + r"\boldsymbol{\theta}_0&=\begin{bmatrix}"
            + bias_body
            + rf"\end{{bmatrix}}\in\mathbb{{R}}^{{{K}}}"
            + r"\end{aligned}$$"
        )

    def linked_term(expression):
        if probability_link == "ovr":
            return rf"\sigma\!\left({expression}\right)"
        return rf"e^{{{expression}}}"

    def z_numeric_expr(Theta, class_idx, d_local, max_feat=max_features_in_z, dec=dec):
        def num(v):
            return f"{v:.{dec}f}"

        feat_count = min(d_local, max_feat, 2 if d_local > 12 else max_feat)
        terms = [rf"\left({num(Theta[d_local, class_idx])}\right)"]
        terms.extend(rf"\left({num(Theta[j, class_idx])}\right)x_{{{j + 1}}}" for j in range(feat_count))
        if d_local > feat_count:
            terms.append(r"\cdots")
            terms.append(rf"\left({num(Theta[d_local - 1, class_idx])}\right)x_{{{d_local}}}")
        return r" + ".join(terms)

    first_class = max(0, min(int(example_class), K - 1))
    expanded_canvas = size in {"default", "wide", "classroom"}

    def score_examples_latex(t, max_feat=max_features_in_z, dec=dec):
        Theta = np.vstack([w_hist[t], b_hist[t].reshape(1, -1)])
        D, K_local = Theta.shape
        d_local = D - 1
        if not expanded_canvas:
            max_feat = min(max_feat, 1)
        elif show_loss:
            max_feat = min(max_feat, 1)
        first = rf"z_{{{first_class}}}(\mathbf{{x}})={z_numeric_expr(Theta, first_class, d_local, max_feat=max_feat, dec=dec)}"
        if K_local == 1:
            return rf"$${first}$$"
        last = rf"z_{{{K_local - 1}}}(\mathbf{{x}})={z_numeric_expr(Theta, K_local - 1, d_local, max_feat=max_feat, dec=dec)}"
        if not expanded_canvas:
            widest_score = max((first, last), key=len)
            centered_ellipsis = rf"\smash{{\overset{{\hphantom{{\displaystyle {widest_score}}}}}{{\vdots}}}}"
            return (
                r"$$\begin{array}{l}"
                + rf"{first}\\[1pt]{centered_ellipsis}\\[1pt]{last}"
                + r"\end{array}$$"
            )
        separator = r"\;\cdots\; " if K_local > 2 else r"\qquad "
        return rf"$${first}{separator}{last}$$"

    def exponent_score(t, class_k):
        def fmt(value):
            return f"{value:.{dec}f}"

        intercept = rf"\left({fmt(b_hist[t, class_k])}\right)"
        first = rf"\left({fmt(w_hist[t, 0, class_k])}\right)x_1"
        last = rf"\left({fmt(w_hist[t, d - 1, class_k])}\right)x_{{{d}}}"
        return rf"{intercept}+{first}+\cdots+{last}"

    def probability_expression_latex(t, class_k):
        k = int(class_k)
        k = max(0, min(k, K - 1))
        if probability_link == "ovr":
            probability = compact_probability_fraction_latex(k, K, probability_link)
        elif expanded_canvas:
            numerator = linked_term(exponent_score(t, k))
            if K <= 3:
                denominator = " + ".join(linked_term(exponent_score(t, j)) for j in range(K))
            else:
                denominator = (
                    linked_term(exponent_score(t, 0))
                    + r" + \cdots + "
                    + linked_term(exponent_score(t, K - 1))
                )
            probability = rf"\frac{{{numerator}}}{{{denominator}}}"
        elif K <= 3:
            numerator = linked_term(rf"z_{{{k}}}")
            denominator = " + ".join(linked_term(rf"z_{{{j}}}") for j in range(K))
            probability = rf"\frac{{{numerator}}}{{{denominator}}}"
        else:
            probability = compact_probability_fraction_latex(k, K, probability_link)
        return probability

    def probability_stack_latex(t):
        """Typeset endpoint probabilities with a self-centering omission mark."""
        first_probability = probability_expression_latex(t, first_class)
        last_class = K - 1
        last_probability = probability_expression_latex(t, last_class)
        if first_class == last_class:
            return rf"$$\hat{{p}}(Y=c_{{{first_class}}}\mid\mathbf{{x}})={first_probability}$$"
        widest_probability = max((first_probability, last_probability), key=len)
        # Only reserve width: a full phantom fraction also reserves its height
        # and pushes the endpoint rows unnecessarily far apart.
        centered_ellipsis = rf"\smash{{\overset{{\hphantom{{\displaystyle {widest_probability}}}}}{{\vdots}}}}"
        row_gap = 6 if expanded_canvas else 0
        return (
            r"$$\begin{array}{r@{\,}c@{\,}l}"
            + rf"\hat{{p}}(Y=c_{{{first_class}}}\mid\mathbf{{x}})&=&\displaystyle {first_probability}\\[{row_gap}pt]"
            + rf"&&{centered_ellipsis}\\[{row_gap}pt]"
            + rf"\hat{{p}}(Y=c_{{{last_class}}}\mid\mathbf{{x}})&=&\displaystyle {last_probability}"
            + r"\end{array}$$"
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
        narrow_canvas = not expanded_canvas
        math_center = 0.46 if show_loss else 0.5
        theta_x = 0.10 if show_loss else 0.12
        matrix_y = 0.70 if dense_layout else 0.74
        if show_loss and narrow_canvas:
            matrix_y = 0.66 if dense_layout else 0.70
        # Lift the matrices slightly in expanded views. Narrow views move the
        # entire math block together to free the space above the timeline.
        block_shift = 0.04 if show_loss and narrow_canvas else 0.03
        matrix_y += block_shift
        x_vector_y = matrix_y + (0.05 if show_loss and narrow_canvas else 0.0)
        equation_x = 0.015 if show_loss else 0.025
        if dense_layout:
            row3_y, scores_y, probability_stack_y = 0.44, 0.285, 0.17
        else:
            row3_y, scores_y, probability_stack_y = 0.50, 0.345, 0.22
        if show_loss and narrow_canvas:
            row3_y = (0.42 if dense_layout else 0.44) + block_shift
            scores_y = (0.285 if dense_layout else 0.30) + block_shift
            probability_stack_y = (0.08 if dense_layout else 0.12) + block_shift
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
                y=x_vector_y,
                xref="paper",
                yref="paper",
                text=x_vector_latex_capped(d, max_rows=7),
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
                x=equation_x,
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
                x=equation_x,
                y=scores_y,
                xref="paper",
                yref="paper",
                text=score_examples_latex(t),
                showarrow=False,
                xanchor="left",
                yanchor="top",
                font=dict(size=14 if expanded_canvas else MULTICLASS_PROBABILITY_FONT_SIZE, color="white"),
            ),
            dict(
                x=equation_x,
                y=probability_stack_y,
                xref="paper",
                yref="paper",
                text=probability_stack_latex(t),
                showarrow=False,
                xanchor="left",
                yanchor="top",
                font=dict(size=16 if expanded_canvas else MULTICLASS_PROBABILITY_FONT_SIZE, color="white"),
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
