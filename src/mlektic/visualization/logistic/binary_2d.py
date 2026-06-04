"""Binary logistic-regression (2D) figure builder."""

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
    data_3d_marker_style,
    surface_style,
)
from ...utils.math import _sigmoid

def build_binary_plane_logistic_figure(
    x1,
    x2,
    y,
    w_hist=None,
    b_hist=None,
    *,
    p_plane_hist=None,
    X1g=None,
    X2g=None,
    loss_hist=None,
    metrics_hist=None,
    show_loss=False,
    history_kind="iterative",
    title="Binary Logistic Regression (2 variables)",
    strict_loss=False,
    dec=4,
    jitter=0.03,
    frame_duration=80,
    theme=None,
):
    """Internal method to build build_binary_plane_logistic_figure."""
    if show_loss and history_kind != "iterative":
        if strict_loss:
            raise ValueError("show_loss=True is only allowed for iterative histories.")
        show_loss = False
        loss_hist = None

    x1 = np.asarray(x1).ravel()
    x2 = np.asarray(x2).ravel()
    y = np.asarray(y).ravel().astype(float)
    y_jitter = y + np.random.uniform(-jitter, jitter, size=y.size)

    use_pred_grid = p_plane_hist is not None

    if use_pred_grid:
        p_plane_hist = np.asarray(p_plane_hist, dtype=float)
        if X1g is None or X2g is None:
            raise ValueError("If p_plane_hist is provided, X1g and X2g must be provided.")

        X1g = np.asarray(X1g, dtype=float)
        X2g = np.asarray(X2g, dtype=float)

        if X1g.shape != X2g.shape:
            raise ValueError("X1g and X2g must have same shape.")
        if p_plane_hist.ndim != 3:
            raise ValueError("p_plane_hist must have shape (steps, H, W).")
        if p_plane_hist.shape[1:] != X1g.shape:
            raise ValueError("p_plane_hist grid shape must match X1g/X2g.")

        steps_n = int(p_plane_hist.shape[0])

        def p_plane(t):
            return p_plane_hist[t]

        w_disp = None
        b_disp = None
        if w_hist is not None and b_hist is not None:
            w_arr = np.asarray(w_hist, dtype=float)
            b_arr = np.asarray(b_hist, dtype=float).ravel()
            if w_arr.ndim == 2 and w_arr.shape == (steps_n, 2) and b_arr.size == steps_n:
                w_disp = w_arr
                b_disp = b_arr

        def formula_text():
            return r"$\hat{p}(y=1\mid \mathbf{x})=\sigma(z),\;\; z=\theta_1x_1+\theta_2x_2+\theta_0$"

        def eq_text(t):
            if w_disp is None:
                return r"$\hat{p}(y=1\mid \mathbf{x})=f(x_1,x_2)$"
            w1 = float(w_disp[t, 0])
            w2 = float(w_disp[t, 1])
            b = float(b_disp[t])
            return r"$\sigma(z)=\dfrac{1}{1+e^{-z}}" + rf",\;\; z=({w1:.{dec}f})x_1+({w2:.{dec}f})x_2+({b:.{dec}f})$"

        x1_min, x1_max = float(np.min(X1g)), float(np.max(X1g))
        x2_min, x2_max = float(np.min(X2g)), float(np.max(X2g))

    else:
        if w_hist is None or b_hist is None:
            raise ValueError("Legacy mode requires w_hist and b_hist. Prefer p_plane_hist + X1g/X2g.")

        w_hist = np.asarray(w_hist, dtype=float)
        b_hist = np.asarray(b_hist, dtype=float).ravel()
        steps_n = int(b_hist.size)

        if w_hist.ndim == 1:
            if w_hist.size == steps_n * 2:
                w_hist = w_hist.reshape(steps_n, 2)
            else:
                raise ValueError("Expected w_hist shape (steps, 2).")
        if w_hist.ndim != 2 or w_hist.shape != (steps_n, 2):
            raise ValueError("Expected w_hist shape (steps, 2) and b_hist shape (steps,).")

        x1_grid = np.linspace(float(x1.min()), float(x1.max()), 40)
        x2_grid = np.linspace(float(x2.min()), float(x2.max()), 40)
        X1g, X2g = np.meshgrid(x1_grid, x2_grid)

        def p_plane(t):
            w1 = float(w_hist[t, 0])
            w2 = float(w_hist[t, 1])
            b = float(b_hist[t])
            return _sigmoid(w1 * X1g + w2 * X2g + b)

        def formula_text():
            return r"$\hat{p}(y=1\mid \mathbf{x})=\sigma(z),\;\; z=\theta_1x_1+\theta_2x_2+\theta_0$"

        def eq_text(t):
            w1 = float(w_hist[t, 0])
            w2 = float(w_hist[t, 1])
            b = float(b_hist[t])
            return r"$\sigma(z)=\dfrac{1}{1+e^{-z}}" + rf",\;\; z=({w1:.{dec}f})x_1+({w2:.{dec}f})x_2+({b:.{dec}f})$"

        x1_min, x1_max = float(np.min(X1g)), float(np.max(X1g))
        x2_min, x2_max = float(np.min(X2g)), float(np.max(X2g))

    if steps_n < 1:
        raise ValueError("Need at least 1 step to animate.")

    if show_loss:
        if loss_hist is None:
            raise ValueError("show_loss=True requires loss_hist.")
        loss_hist = np.asarray(loss_hist, dtype=float).ravel()
        if loss_hist.size != steps_n:
            raise ValueError("loss_hist must match steps.")

    step_axis = np.arange(steps_n)

    if show_loss:
        theta_y = 1.16
        eq_y = 1.08
        margin_t = 150
    else:
        theta_y = 1.15
        eq_y = 1.05
        margin_t = 150

    def formula_annotation():
        return create_annotation(formula_text(), y=theta_y)

    def eq_annotation(t):
        return create_annotation(eq_text(t), y=eq_y)

    z0 = np.asarray(p_plane(0), dtype=float)
    zL = np.asarray(p_plane(steps_n - 1), dtype=float)
    z_all = np.concatenate([y_jitter, z0.ravel(), zL.ravel()])
    z_min, z_max = float(z_all.min()), float(z_all.max())

    def _pad(lo, hi, frac=0.10):
        span = (hi - lo) + 1e-9
        return [lo - frac * span, hi + frac * span]

    x1_range = _pad(x1_min, x1_max)
    x2_range = _pad(x2_min, x2_max)
    y_range = [min(-0.08, z_min - 0.03), max(1.08, z_max + 0.03)]

    CAMERA = dict(eye=dict(x=1.55, y=1.55, z=1.15))

    if show_loss:
        lmin, lmax = float(loss_hist.min()), float(loss_hist.max())
        lpad = 0.10 * (lmax - lmin + 1e-9)

    if show_loss:
        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.60, 0.30],
            horizontal_spacing=0.06,
            specs=[[{"type": "scene"}, {"type": "xy"}]],
        )

        def metrics_annotations(t):
            ann = []
            if metrics_hist is not None:
                for i, (name, hist) in enumerate(metrics_hist.items()):
                    val = hist[t]
                    y_pos = 0.95 - (i * 0.13)
                    fmt = ".6f" if name.lower() == "log-loss" or name.lower() == "loss" else ".4f"
                    ann.append(dict(
                        x=0.98, y=y_pos, xref="paper", yref="paper", 
                        text=f"<b>{name}</b><br>{val:{fmt}}", showarrow=False, 
                        xanchor="right", yanchor="top", font=dict(size=14, color="black"), 
                        bgcolor="white", bordercolor="black", borderwidth=1, borderpad=6
                    ))
            return ann

        fig.add_trace(
            go.Scatter3d(
                x=x1,
                y=x2,
                z=y_jitter,
                mode="markers",
                name="Data",
                marker=data_3d_marker_style(theme=theme),
                legendgroup="fit",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Surface(
                x=X1g,
                y=X2g,
                z=p_plane(0),
                name="Model",
                **surface_style(theme=theme),
                legendgroup="fit",
                showlegend=True,
                uid="MODEL_SURFACE",
            ),
            row=1,
            col=1,
        )

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
                uid="LOSS_LINE",
            ),
            row=1,
            col=2,
        )

        frames = []
        for t in range(steps_n):
            frames.append(
                go.Frame(
                    name=str(t),
                    data=[
                        go.Surface(
                            x=X1g,
                            y=X2g,
                            z=p_plane(t),
                            **surface_style(theme=theme),
                            showlegend=True,
                            uid="MODEL_SURFACE",
                        ),
                        go.Scatter(
                            x=[step if i <= t else None for i, step in enumerate(step_axis_list)],
                            y=[val if i <= t else None for i, val in enumerate(loss_hist_list)],
                            mode="lines",
                            line=loss_line_style(theme=theme),
                            uid="LOSS_LINE",
                        ),
                    ],
                    traces=[1, 2],
                    layout=go.Layout(
                        annotations=[formula_annotation(), eq_annotation(t)] + metrics_annotations(t),
                    ),
                )
            )
        fig.frames = frames

        fig.update_layout(
            **get_base_layout(title=title, margin_t=margin_t, theme=theme),
            annotations=[formula_annotation(), eq_annotation(0)] + metrics_annotations(0),
            legend=dict(orientation="v", **get_legend_props(x=0.585, y=0.82, theme=theme)),
            legend2=dict(orientation="v", **get_legend_props(x=0.995, y=0.05, theme=theme)),
            scene=dict(
                xaxis=dict(title="x₁", range=x1_range),
                yaxis=dict(title="x₂", range=x2_range),
                zaxis=dict(title="σ(z)", range=y_range),
                aspectmode="cube",
                camera=CAMERA,
            ),
            sliders=get_sliders(steps_n, theme=theme),
            updatemenus=get_updatemenus(frame_duration, theme=theme),
        )

        fig.data[2].update(legend="legend2")
        fig.update_xaxes(title="Step", range=[0, steps_n - 1], row=1, col=2)
        fig.update_yaxes(title="Log-loss", range=[lmin - lpad, lmax + lpad], row=1, col=2)
        return fig

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=x1,
            y=x2,
            z=y_jitter,
            mode="markers",
            name="Data",
            marker=data_3d_marker_style(theme=theme),
        )
    )

    fig.add_trace(
        go.Surface(
            x=X1g,
            y=X2g,
            z=p_plane(0),
            name="Model",
            **surface_style(theme=theme),
            showlegend=True,
            uid="MODEL_SURFACE",
        )
    )

    frames = []
    for t in range(steps_n):
        frames.append(
            go.Frame(
                name=str(t),
                data=[
                    go.Surface(
                        x=X1g,
                        y=X2g,
                        z=p_plane(t),
                        opacity=0.55,
                        showscale=False,
                        showlegend=True,
                        uid="MODEL_SURFACE",
                    )
                ],
                traces=[1],
                layout=go.Layout(
                    annotations=[formula_annotation(), eq_annotation(t)],
                    scene=dict(camera=CAMERA),
                ),
            )
        )
    fig.frames = frames

    fig.update_layout(
        **get_base_layout(title=title, margin_t=margin_t, theme=theme),
        annotations=[formula_annotation(), eq_annotation(0)],
        showlegend=True,
        legend=get_legend_props(theme=theme),
        scene=dict(
            xaxis=dict(title="x₁", range=x1_range),
            yaxis=dict(title="x₂", range=x2_range),
            zaxis=dict(title="σ(z)", range=y_range),
            aspectmode="cube",
            camera=CAMERA,
        ),
        sliders=get_sliders(steps_n, theme=theme),
        updatemenus=get_updatemenus(frame_duration, theme=theme),
    )

    return fig


__all__ = ["build_binary_plane_logistic_figure"]
