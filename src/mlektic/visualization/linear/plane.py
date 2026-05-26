"""2D linear-regression visualization builder."""

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

def build_plane_lr_figure(
    x1,
    x2,
    y,
    w_hist=None,
    b_hist=None,
    *,
    # --- robust inputs (preferred) ---
    z_plane_hist=None,  # (T, H, W) predictions over grid
    X1g=None,
    X2g=None,
    loss_hist=None,
    metrics_hist=None,
    show_loss=False,
    history_kind="iterative",
    title="Linear Regression (2 variables)",
    strict_loss=False,
    dec=4,
    frame_duration=80,
    theme=None,
):
    """
    2D (plane) LR visualization.

    Robust mode (preferred):
      - Provide z_plane_hist + (X1g, X2g) => plot uses predictions, works with ANY sklearn Pipeline/transform.

    Legacy mode:
      - Provide w_hist, b_hist => plot uses z = w1*x1 + w2*x2 + b (only correct for pure linear model in original space)

    Notes:
      - If z_plane_hist is given, w_hist/b_hist are OPTIONAL and only used for display in the equation text.
      - show_loss is only allowed for iterative histories (same rule as 1D).
    """
    # --- enforce inside the library ---
    if show_loss and history_kind != "iterative":
        if strict_loss:
            raise ValueError("show_loss=True is only allowed for iterative histories.")
        show_loss = False
        loss_hist = None

    x1 = np.asarray(x1).ravel()
    x2 = np.asarray(x2).ravel()
    y = np.asarray(y).ravel()

    use_pred_grid = z_plane_hist is not None

    # -------------------------
    # Mode A: robust prediction-grid
    # -------------------------
    if use_pred_grid:
        z_plane_hist = np.asarray(z_plane_hist, dtype=float)

        if X1g is None or X2g is None:
            raise ValueError("If z_plane_hist is provided, X1g and X2g must be provided.")

        X1g = np.asarray(X1g, dtype=float)
        X2g = np.asarray(X2g, dtype=float)

        if X1g.shape != X2g.shape:
            raise ValueError("X1g and X2g must have the same shape.")
        if z_plane_hist.ndim != 3:
            raise ValueError("z_plane_hist must have shape (steps, H, W).")
        if z_plane_hist.shape[1:] != X1g.shape:
            raise ValueError("z_plane_hist grid shape must match X1g/X2g shape.")

        steps_n = int(z_plane_hist.shape[0])
        if steps_n < 1:
            raise ValueError("Need at least 1 step to animate.")

        def z_plane(t: int):
            return z_plane_hist[t]

        def theta_formula_text():
            return r"$\hat{y}=\theta_0+\theta_1 x_1+\theta_2 x_2$"

        # Optional theta display if consistent (accept (T,2) only)
        w_disp = None
        b_disp = None
        if w_hist is not None and b_hist is not None:
            w_arr = np.asarray(w_hist, dtype=float)
            b_arr = np.asarray(b_hist, dtype=float).ravel()
            if w_arr.ndim == 2 and w_arr.shape == (steps_n, 2) and b_arr.size == steps_n:
                w_disp = w_arr
                b_disp = b_arr

        def eq_text(t: int):
            if w_disp is None:
                return r"$\hat{y} = f(x_1,x_2)$"
            w1 = float(w_disp[t, 0])
            w2 = float(w_disp[t, 1])
            b = float(b_disp[t])
            return rf"$\hat{{y}} = ({w1:.{dec}f})x_1 + ({w2:.{dec}f})x_2 + ({b:.{dec}f})$"

        # Ranges driven by GRID (stable + matches surface)
        x1_min, x1_max = float(np.min(X1g)), float(np.max(X1g))
        x2_min, x2_max = float(np.min(X2g)), float(np.max(X2g))

    # -------------------------
    # Mode B: legacy parameter-plane
    # -------------------------
    else:
        if w_hist is None or b_hist is None:
            raise ValueError("Legacy mode requires w_hist and b_hist. Prefer providing z_plane_hist + X1g/X2g.")

        w_hist = np.asarray(w_hist, dtype=float)
        b_hist = np.asarray(b_hist, dtype=float).ravel()
        steps_n = int(b_hist.size)

        if steps_n < 1:
            raise ValueError("Need at least 1 step to animate.")

        # allow w_hist shape flexibility: (T,2) or (T*2,) (rare)
        if w_hist.ndim == 1:
            if w_hist.size == steps_n * 2:
                w_hist = w_hist.reshape(steps_n, 2)
            else:
                raise ValueError("Legacy plane expects w_hist shape (steps, 2) (or flat of length steps*2).")

        if w_hist.ndim != 2 or w_hist.shape != (steps_n, 2):
            raise ValueError("Legacy plane expects w_hist shape (steps, 2) and b_hist shape (steps,)")

        # Build default mesh
        x1_grid = np.linspace(float(x1.min()), float(x1.max()), 40)
        x2_grid = np.linspace(float(x2.min()), float(x2.max()), 40)
        X1g, X2g = np.meshgrid(x1_grid, x2_grid)

        def z_plane(t: int):
            w1 = float(w_hist[t, 0])
            w2 = float(w_hist[t, 1])
            b = float(b_hist[t])
            return w1 * X1g + w2 * X2g + b

        def theta_formula_text():
            return r"$\hat{y}=\theta_0+\theta_1 x_1+\theta_2 x_2$"

        def eq_text(t: int):
            w1 = float(w_hist[t, 0])
            w2 = float(w_hist[t, 1])
            b = float(b_hist[t])
            return rf"$\hat{{y}} = ({w1:.{dec}f})x_1 + ({w2:.{dec}f})x_2 + ({b:.{dec}f})$"

        x1_min, x1_max = float(np.min(X1g)), float(np.max(X1g))
        x2_min, x2_max = float(np.min(X2g)), float(np.max(X2g))

    # -------------------------
    # Validate loss
    # -------------------------
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
    if show_loss:
        theta_y = 1.16
        eq_y = 1.08
        margin_t = 150
    else:
        theta_y = 1.15
        eq_y = 1.05
        margin_t = 150

    def theta_formula_annotation():
        return create_annotation(theta_formula_text(), y=theta_y)

    def eq_annotation(t):
        return create_annotation(eq_text(t), y=eq_y)

    def metrics_annotations(t):
        ann = []
        if metrics_hist is not None:
            for i, (name, hist) in enumerate(metrics_hist.items()):
                val = hist[t]
                y_pos = 0.95 - (i * 0.13)
                fmt = ".6f" if name.lower() == "loss" else ".4f"
                ann.append(dict(
                    x=0.98, y=y_pos, xref="paper", yref="paper", 
                    text=f"<b>{name}</b><br>{val:{fmt}}", showarrow=False, 
                    xanchor="right", yanchor="top", font=dict(size=14, color="black"), 
                    bgcolor="white", bordercolor="black", borderwidth=1, borderpad=6
                ))
        return ann

    # -------------------------
    # Stable scene ranges (use data + plane endpoints)
    # -------------------------
    z0 = np.asarray(z_plane(0), dtype=float)
    zL = np.asarray(z_plane(steps_n - 1), dtype=float)
    z_all = np.concatenate([y, z0.ravel(), zL.ravel()])
    z_min, z_max = float(z_all.min()), float(z_all.max())

    def _pad(lo, hi, frac=0.10):
        span = (hi - lo) + 1e-9
        return [lo - frac * span, hi + frac * span]

    x1_range = _pad(x1_min, x1_max)
    x2_range = _pad(x2_min, x2_max)
    y_range = _pad(z_min, z_max)

    CAMERA = dict(eye=dict(x=1.55, y=1.55, z=1.15))

    if show_loss:
        lmin, lmax = float(loss_hist.min()), float(loss_hist.max())
        lpad = 0.10 * (lmax - lmin + 1e-9)

    # -------------------------
    # Build figure
    # -------------------------
    if show_loss:
        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.60, 0.30],
            horizontal_spacing=0.06,
            specs=[[{"type": "scene"}, {"type": "xy"}]],
        )

        fig.add_trace(
            go.Scatter3d(
                x=x1,
                y=x2,
                z=y,
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
                z=z_plane(0),
                name="Model",
                **surface_style(theme=theme),
                legendgroup="fit",
                showlegend=True,
                uid="MODEL_PLANE",
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
                name="Loss",
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
                            z=z_plane(t),
                            **surface_style(theme=theme),
                            showlegend=True,
                            uid="MODEL_PLANE",
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
                        annotations=[theta_formula_annotation(), eq_annotation(t)] + metrics_annotations(t),
                    ),
                )
            )
        fig.frames = frames

        fig.update_layout(
            **get_base_layout(title=title, margin_t=margin_t, theme=theme),
            annotations=[theta_formula_annotation(), eq_annotation(0)] + metrics_annotations(0),
            legend=dict(orientation="v", **get_legend_props(x=0.49, theme=theme)),
            legend2=dict(orientation="v", **get_legend_props(x=0.995, y=0.05, theme=theme)),
            scene=dict(
                xaxis=dict(title="x₁", range=x1_range),
                yaxis=dict(title="x₂", range=x2_range),
                zaxis=dict(title="y", range=y_range),
                aspectmode="cube",
                camera=CAMERA,
            ),
            sliders=get_sliders(steps_n, theme=theme),
            updatemenus=get_updatemenus(frame_duration, theme=theme),
        )

        # put loss trace in legend2
        fig.data[2].update(legend="legend2")

        fig.update_xaxes(title="Step", range=[0, steps_n - 1], row=1, col=2)
        fig.update_yaxes(title="Loss", range=[lmin - lpad, lmax + lpad], row=1, col=2)

        return fig

    # -------------------------
    # Without loss: single 3D plane
    # -------------------------
    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=x1,
            y=x2,
            z=y,
            mode="markers",
            name="Data",
            marker=data_3d_marker_style(theme=theme),
        )
    )

    fig.add_trace(
        go.Surface(
            x=X1g,
            y=X2g,
            z=z_plane(0),
            name="Model",
            **surface_style(theme=theme),
            showlegend=True,
            uid="MODEL_PLANE",
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
                        z=z_plane(t),
                        opacity=0.55,
                        showscale=False,
                        showlegend=True,
                        uid="MODEL_PLANE",
                    )
                ],
                traces=[1],
                layout=go.Layout(
                    annotations=[theta_formula_annotation(), eq_annotation(t)],
                    scene=dict(camera=CAMERA),
                ),
            )
        )
    fig.frames = frames

    fig.update_layout(
        **get_base_layout(title=title, margin_t=margin_t, theme=theme),
        annotations=[theta_formula_annotation(), eq_annotation(0)],
        showlegend=True,
        legend=get_legend_props(theme=theme),
        scene=dict(
            xaxis=dict(title="x₁", range=x1_range),
            yaxis=dict(title="x₂", range=x2_range),
            zaxis=dict(title="y", range=y_range),
            aspectmode="cube",
            camera=CAMERA,
        ),
        sliders=get_sliders(steps_n, theme=theme),
        updatemenus=get_updatemenus(frame_duration, theme=theme),
    )

    return fig


__all__ = ["build_plane_lr_figure"]
