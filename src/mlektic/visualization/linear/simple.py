"""1D linear-regression visualization builder."""

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
    data_marker_style,
    model_line_style,
    loss_line_style,
)

def build_simple_lr_figure(
    x1,
    y,
    w_hist=None,
    b_hist=None,
    *,
    # --- robust inputs (preferred) ---
    y_line_hist=None,  # (T, G)
    x1_grid=None,  # (G,)
    # --- loss ---
    loss_hist=None,
    metrics_hist=None,
    show_loss=False,
    history_kind="iterative",
    title="Linear Regression (Simple, 1 variable)",
    strict_loss=False,
    dec=4,
    frame_duration=80,
    theme=None,
):
    """
    Simple (1D) visualization.

    Robust mode:
      - Provide y_line_hist + x1_grid => plot uses predictions, works with ANY sklearn Pipeline/transform.

    Legacy mode:
      - Provide w_hist,b_hist => plot uses y = w*x + b (only correct for pure linear model in original space)
    """
    # --- enforce inside the library ---
    if show_loss and history_kind != "iterative":
        if strict_loss:
            raise ValueError("show_loss=True is only allowed for iterative histories.")
        show_loss = False
        loss_hist = None

    x1 = np.asarray(x1).ravel()
    y = np.asarray(y).ravel()

    use_pred_grid = y_line_hist is not None

    # -------------------------
    # Select mode + validate inputs
    # -------------------------
    if use_pred_grid:
        y_line_hist = np.asarray(y_line_hist, dtype=float)
        if x1_grid is None:
            raise ValueError("If y_line_hist is provided, x1_grid must be provided.")
        x1_grid = np.asarray(x1_grid, dtype=float).ravel()

        if y_line_hist.ndim != 2:
            raise ValueError("y_line_hist must have shape (steps, grid_points).")
        if y_line_hist.shape[1] != x1_grid.size:
            raise ValueError("y_line_hist second dim must match x1_grid size.")
        steps_n = int(y_line_hist.shape[0])

        def y_line(t: int):
            return y_line_hist[t]

        # If theta history is also provided, show numeric equation; else show generic
        w_disp = None
        b_disp = None
        if w_hist is not None and b_hist is not None:
            w_arr = np.asarray(w_hist, dtype=float)
            b_arr = np.asarray(b_hist, dtype=float).ravel()

            # accept shapes: (T,), (T,1)
            if w_arr.ndim == 2 and w_arr.shape[1] == 1:
                w_arr = w_arr[:, 0]
            if w_arr.ndim == 1 and w_arr.size == steps_n and b_arr.size == steps_n:
                w_disp = w_arr
                b_disp = b_arr

        def theta_formula_text():
            return r"$\hat{y}=\theta_0+\theta_1 x_1$"

        def eq_text(t: int):
            if w_disp is None:
                return r"$\hat{y} = f(x_1)$"
            return rf"$\hat{{y}} = ({w_disp[t]:.{dec}f})x_1 + ({b_disp[t]:.{dec}f})$"

        x_min, x_max = float(x1_grid.min()), float(x1_grid.max())

    else:
        # legacy path
        if w_hist is None or b_hist is None:
            raise ValueError("Legacy mode requires w_hist and b_hist. Prefer providing y_line_hist + x1_grid.")

        w_hist = np.asarray(w_hist, dtype=float)
        b_hist = np.asarray(b_hist, dtype=float).ravel()
        steps_n = int(b_hist.size)

        # allow w_hist shape flexibility
        if w_hist.ndim == 1:
            w_hist = w_hist.reshape(-1, 1)
        if w_hist.shape[0] != steps_n:
            raise ValueError("w_hist and b_hist must have the same number of steps.")
        if w_hist.shape[1] != 1:
            raise ValueError(f"Simple LR expects 1 weight, got d={w_hist.shape[1]}.")

        x_min, x_max = float(x1.min()), float(x1.max())
        x1_grid = np.linspace(x_min, x_max, 250)

        def y_line(t: int):
            w1 = float(w_hist[t, 0])
            b = float(b_hist[t])
            return w1 * x1_grid + b

        def theta_formula_text():
            return r"$\hat{y}=\theta_0+\theta_1 x_1$"

        def eq_text(t: int):
            w1 = float(w_hist[t, 0])
            b = float(b_hist[t])
            return rf"$\hat{{y}} = ({w1:.{dec}f})x_1 + ({b:.{dec}f})$"

    if steps_n < 1:
        raise ValueError("Need at least 1 step to animate.")

    # validate loss
    if show_loss:
        if loss_hist is None:
            raise ValueError("show_loss=True requires loss_hist.")
        loss_hist = np.asarray(loss_hist, dtype=float).ravel()
        if loss_hist.size != steps_n:
            raise ValueError("loss_hist must have the same length as steps.")

    step_axis = np.arange(steps_n)

    # -------------------------
    # Annotations (paper coords)
    # -------------------------
    # NOTE: for subplots we use xref/yref="paper" too; it's fine because it's global paper.
    if show_loss:
        theta_y = 1.18
        eq_y = 1.10
        margin_t = 160
    else:
        theta_y = 1.15
        eq_y = 1.05
        margin_t = 150

    def theta_formula_annotation():
        return create_annotation(theta_formula_text(), y=theta_y, theme=theme)

    def eq_annotation(t):
        return create_annotation(eq_text(t), y=eq_y, theme=theme)

    # -------------------------
    # Stable ranges
    # -------------------------
    # (use step 0 and last to stabilize y-range)
    y_all = np.concatenate(
        [
            y,
            np.asarray(y_line(0)).ravel(),
            np.asarray(y_line(steps_n - 1)).ravel(),
        ]
    )
    y_min, y_max = float(y_all.min()), float(y_all.max())
    y_pad = 0.08 * (y_max - y_min + 1e-9)

    def _pad(lo, hi, frac=0.10):
        span = (hi - lo) + 1e-9
        return [lo - frac * span, hi + frac * span]

    x_range = _pad(x_min, x_max)

    if show_loss:
        lmin, lmax = float(loss_hist.min()), float(loss_hist.max())
        lpad = 0.10 * (lmax - lmin + 1e-9)

    # =====================================================================
    # CASE A) show_loss=True
    # =====================================================================
    if show_loss:
        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.62, 0.38],
            horizontal_spacing=0.08,
            specs=[[{"type": "xy"}, {"type": "xy"}]],
        )

        def metrics_annotations(t):
            ann = []
            if metrics_hist is not None:
                for i, (name, hist) in enumerate(metrics_hist.items()):
                    val = hist[t]
                    y_pos = 0.95 - (i * 0.18)
                    fmt = ".6f" if name.lower() == "loss" else ".4f"
                    ann.append(dict(
                        x=0.98, y=y_pos, xref="paper", yref="paper", 
                        text=f"<b>{name}</b><br>{val:{fmt}}", showarrow=False, 
                        xanchor="right", yanchor="top", font=dict(size=13, color="black"), 
                        bgcolor="white", bordercolor="black", borderwidth=1, borderpad=5
                    ))
            return ann

        # Data
        fig.add_trace(
            go.Scatter(
                x=x1,
                y=y,
                mode="markers",
                name="Data",
                marker=data_marker_style(theme=theme),
                legendgroup="fit",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # Model line
        fig.add_trace(
            go.Scatter(
                x=x1_grid,
                y=y_line(0),
                mode="lines",
                name="Model",
                line=model_line_style(theme=theme),
                hoverlabel=dict(bgcolor="white", font=dict(color="black")),
                legendgroup="fit",
                showlegend=True,
                uid="MODEL_LINE",
            ),
            row=1,
            col=1,
        )

        step_axis_list = step_axis.tolist()
        loss_hist_list = loss_hist.tolist()

        # Loss line (start as a single point)
        fig.add_trace(
            go.Scatter(
                x=[step if i == 0 else None for i, step in enumerate(step_axis_list)],
                y=[val if i == 0 else None for i, val in enumerate(loss_hist_list)],
                mode="lines",
                name="Loss",
                line=loss_line_style(theme=theme),
                legendgroup="loss",
                showlegend=True,
                uid="LOSS_LINE",
            ),
            row=1,
            col=2,
        )

        # Frames
        frames = []
        for t in range(steps_n):
            frames.append(
                go.Frame(
                    name=str(t),
                    data=[
                    go.Scatter(x=x1_grid, y=y_line(t), mode="lines", name="Model", line=model_line_style(theme=theme), hoverlabel=dict(bgcolor="white", font=dict(color="black")), uid="MODEL_LINE"),
                        go.Scatter(
                            x=[step if i <= t else None for i, step in enumerate(step_axis_list)],
                            y=[val if i <= t else None for i, val in enumerate(loss_hist_list)],
                            mode="lines",
                            line=loss_line_style(theme=theme),
                            uid="LOSS_LINE",
                        ),
                    ],
                    traces=[1, 2],  # update model + loss
                    layout=go.Layout(annotations=[theta_formula_annotation(), eq_annotation(t)] + metrics_annotations(t)),
                )
            )
        fig.frames = frames

        fig.update_layout(
            **get_base_layout(title=title, margin_t=margin_t, theme=theme),
            annotations=[theta_formula_annotation(), eq_annotation(0)] + metrics_annotations(0),
            legend=dict(orientation="v", **get_legend_props(x=0.49, theme=theme)),
            legend2=dict(orientation="v", **get_legend_props(x=0.985, y=0.05, theme=theme)),
            sliders=get_sliders(steps_n, theme=theme),
            updatemenus=get_updatemenus(frame_duration, theme=theme),
        )

        # Put loss on legend2
        fig.data[2].update(legend="legend2")

        fig.update_xaxes(title="x₁", range=x_range, row=1, col=1)
        fig.update_yaxes(title="ŷ", range=[y_min - y_pad, y_max + y_pad], row=1, col=1)

        fig.update_xaxes(title="Step", range=[0, steps_n - 1], row=1, col=2)
        fig.update_yaxes(title="Loss", range=[lmin - lpad, lmax + lpad], row=1, col=2)

        return fig

    # =====================================================================
    # CASE B) show_loss=False
    # =====================================================================
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x1,
            y=y,
            mode="markers",
            name="Data",
            marker=data_marker_style(theme=theme),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x1_grid,
            y=y_line(0),
            mode="lines",
            name="Model",
            line=model_line_style(theme=theme),
            hoverlabel=dict(bgcolor="white", font=dict(color="black")),
            uid="MODEL_LINE",
        )
    )

    frames = []
    for t in range(steps_n):
        frames.append(
            go.Frame(
                name=str(t),
                data=[go.Scatter(x=x1_grid, y=y_line(t), mode="lines", name="Model", line=model_line_style(theme=theme), hoverlabel=dict(bgcolor="white", font=dict(color="black")), uid="MODEL_LINE")],
                traces=[1],
                layout=go.Layout(annotations=[theta_formula_annotation(), eq_annotation(t)]),
            )
        )
    fig.frames = frames

    fig.update_layout(
        **get_base_layout(title=title, margin_t=margin_t, theme=theme),
        annotations=[theta_formula_annotation(), eq_annotation(0)],
        legend=get_legend_props(theme=theme),
        xaxis=dict(title="x₁", range=x_range),
        yaxis=dict(title="ŷ", range=[y_min - y_pad, y_max + y_pad]),
        sliders=get_sliders(steps_n, theme=theme),
        updatemenus=get_updatemenus(frame_duration, theme=theme),
    )

    return fig


__all__ = ["build_simple_lr_figure"]
