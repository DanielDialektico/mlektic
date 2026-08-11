"""1D linear-regression visualization builder."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..theme import (
    create_annotation,
    data_marker_style,
    get_base_layout,
    get_legend_props,
    get_sliders,
    get_updatemenus,
    loss_line_style,
    model_line_style,
)

_MATH_TEXT_FONT = "STIX Two Math, Cambria Math, Times New Roman, serif"


def _interpolate_at(values, position):
    """Interpolate an array along its first, semantic-step axis."""
    array = np.asarray(values, dtype=float)
    lower = min(int(np.floor(position + 1e-12)), array.shape[0] - 1)
    upper = min(lower + 1, array.shape[0] - 1)
    alpha = float(np.clip(position - lower, 0.0, 1.0))
    return (1.0 - alpha) * array[lower] + alpha * array[upper]


def _hybrid_timeline(steps_n, interpolation_frames):
    """Return visual positions, frame names, and exact checkpoint names."""
    positions = [0.0]
    checkpoint_names = ["visual-0"]
    for step in range(steps_n - 1):
        for subframe in range(1, interpolation_frames + 1):
            positions.append(step + subframe / interpolation_frames)
            if subframe == interpolation_frames:
                checkpoint_names.append(f"visual-{len(positions) - 1}")
    frame_names = [f"visual-{index}" for index in range(len(positions))]
    return positions, frame_names, checkpoint_names


def _progressive_series(values, position):
    """Build a constant-length line ending at an interpolated position."""
    values = np.asarray(values, dtype=float).ravel()
    current = float(_interpolate_at(values, position))
    lower = min(int(np.floor(position + 1e-12)), values.size - 1)
    x_values = np.full(values.size, position, dtype=float)
    y_values = np.full(values.size, current, dtype=float)
    completed = lower + 1
    x_values[:completed] = np.arange(completed, dtype=float)
    y_values[:completed] = values[:completed]
    if position > lower + 1e-12 and completed < values.size:
        x_values[completed] = position
        y_values[completed] = current
    return x_values, y_values


def _format_math_number(value, dec):
    return f"{float(value):.{dec}f}".replace("-", "\u2212")


def _numeric_equation_text(w_values, b_values, position, dec):
    if w_values is None or b_values is None:
        return "\u0177 = f<sub>t</sub>(x<sub>1</sub>)"
    weight = _format_math_number(_interpolate_at(w_values, position), dec)
    bias = _format_math_number(_interpolate_at(b_values, position), dec)
    return f"\u0177 = ({weight})x<sub>1</sub> + ({bias})"


def _numeric_equation_latex(w_values, b_values, position, dec):
    """Return the interpolated fitted equation as MathJax-ready LaTeX."""
    if w_values is None or b_values is None:
        return r"$\hat{y}=f_t(x_1)$"
    weight = float(_interpolate_at(w_values, position))
    bias = float(_interpolate_at(b_values, position))
    return rf"$\hat{{y}}=({weight:.{dec}f})x_1+({bias:.{dec}f})$"


def _metric_card_texts(metrics_hist, position):
    if not metrics_hist:
        return []
    cards = []
    for name, values in metrics_hist.items():
        value = float(_interpolate_at(values, position))
        label = str(name).upper().replace("R2", "R\u00b2")
        precision = 6 if str(name).lower() == "loss" else 4
        cards.append(f"<b>{label}</b><br>{value:.{precision}f}")
    return cards


def _hybrid_sliders(checkpoint_names, *, theme=None):
    """Create a slider containing semantic checkpoints rather than subframes."""
    slider = get_sliders(len(checkpoint_names), theme=theme)[0]
    slider["steps"] = [
        dict(
            method="animate",
            args=[
                [frame_name],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": False},
                    "transition": {"duration": 0},
                },
            ],
            label=str(step),
        )
        for step, frame_name in enumerate(checkpoint_names)
    ]
    return [slider]


def _build_hybrid_figure(
    x1,
    y,
    x1_grid,
    line_history,
    w_values,
    b_values,
    *,
    loss_hist,
    metrics_hist,
    show_loss,
    title,
    dec,
    frame_duration,
    interpolation_frames,
    theta_formula_annotation,
    x_range,
    y_range,
    y_text,
    equation_location,
    theme,
):
    """Build a trace-only 1D animation with interpolated visual subframes."""
    steps_n = line_history.shape[0]
    positions, frame_names, checkpoint_names = _hybrid_timeline(steps_n, interpolation_frames)

    def formula_annotation():
        annotation = theta_formula_annotation()
        if equation_location == "math_band":
            annotation["y"] = 1.08
        return annotation

    def equation_trace(position):
        in_math_band = equation_location == "math_band"
        return go.Scatter(
            x=[0.5 if in_math_band else float(np.mean(x_range))],
            y=[0.5 if in_math_band else y_text],
            xaxis=("x4" if show_loss else "x2") if in_math_band else None,
            yaxis=("y4" if show_loss else "y2") if in_math_band else None,
            mode="text",
            text=[
                _numeric_equation_latex(w_values, b_values, position, dec)
                if in_math_band
                else _numeric_equation_text(w_values, b_values, position, dec)
            ],
            textfont=dict(family=_MATH_TEXT_FONT, size=18, color="white"),
            cliponaxis=False,
            hoverinfo="skip",
            showlegend=False,
            uid="NUMERIC_EQUATION",
        )

    def model_trace(position):
        return go.Scatter(
            x=x1_grid,
            y=_interpolate_at(line_history, position),
            mode="lines",
            name="Model",
            line=model_line_style(theme=theme),
            hoverlabel=dict(bgcolor="white", font=dict(color="black")),
            legendgroup="fit",
            showlegend=True,
            uid="MODEL_LINE",
        )

    if show_loss:
        loss_hist = np.asarray(loss_hist, dtype=float).ravel()
        lmin, lmax = float(loss_hist.min()), float(loss_hist.max())
        lspan = lmax - lmin + 1e-9
        lpad = 0.10 * lspan
        metric_count = len(metrics_hist or {})
        if metric_count > 1:
            metric_y = np.linspace(0.84, 0.16, metric_count)
        elif metric_count == 1:
            metric_y = np.asarray([0.5])
        else:
            metric_y = np.asarray([])

        def loss_trace(position):
            loss_x, loss_y = _progressive_series(loss_hist, position)
            return go.Scatter(
                x=loss_x,
                y=loss_y,
                mode="lines+markers",
                name="Loss",
                line=loss_line_style(theme=theme),
                marker=dict(size=3),
                legendgroup="loss",
                showlegend=True,
                uid="LOSS_LINE",
            )

        def metric_trace(position):
            return go.Scatter(
                x=np.full(metric_count, 0.5),
                y=metric_y,
                mode="markers+text",
                text=_metric_card_texts(metrics_hist, position),
                textposition="middle center",
                textfont=dict(family="Helvetica", size=12, color="black"),
                marker=dict(
                    symbol="square",
                    size=68,
                    color="white",
                    line=dict(color="#641E2E", width=1.5),
                ),
                cliponaxis=False,
                hoverinfo="skip",
                showlegend=False,
                uid="METRIC_VALUES",
            )

        fig = make_subplots(
            rows=1,
            cols=3,
            column_widths=[0.58, 0.30, 0.12],
            horizontal_spacing=0.05,
            specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
        )
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
        fig.add_trace(model_trace(0.0), row=1, col=1)
        fig.add_trace(loss_trace(0.0), row=1, col=2)
        if equation_location == "math_band":
            fig.add_trace(equation_trace(0.0))
            fig.data[-1].update(xaxis="x4", yaxis="y4")
        else:
            fig.add_trace(equation_trace(0.0), row=1, col=1)
        fig.add_trace(metric_trace(0.0), row=1, col=3)

        frames = []
        for name, position in zip(frame_names, positions):
            frames.append(
                go.Frame(
                    name=name,
                    data=[
                        model_trace(position),
                        loss_trace(position),
                        equation_trace(position),
                        metric_trace(position),
                    ],
                    traces=[1, 2, 3, 4],
                )
            )
        fig.frames = frames
        fig.update_layout(
            **get_base_layout(
                title=title,
                margin_t=145 if equation_location == "math_band" else 170,
                theme=theme,
            ),
            annotations=[formula_annotation()],
            legend=dict(orientation="v", **get_legend_props(x=0.49, theme=theme)),
            legend2=dict(orientation="v", **get_legend_props(x=0.85, y=0.05, theme=theme)),
            sliders=_hybrid_sliders(checkpoint_names, theme=theme),
            updatemenus=get_updatemenus(frame_duration, theme=theme),
        )
        fig.data[2].update(legend="legend2")
        fig.update_xaxes(title="x\u2081", range=x_range, row=1, col=1)
        fig.update_yaxes(title="\u0177", range=y_range, row=1, col=1)
        fig.update_xaxes(title="Step", range=[0, steps_n - 1], row=1, col=2)
        fig.update_yaxes(title="Loss", range=[lmin - lpad, lmax + lpad], row=1, col=2)
        fig.update_xaxes(visible=False, range=[0, 1], row=1, col=3)
        fig.update_yaxes(visible=False, range=[0, 1], row=1, col=3)
        if equation_location == "math_band":
            fig.update_layout(
                xaxis4=dict(domain=[0.0, 1.0], range=[0, 1], visible=False, anchor="y4", fixedrange=True),
                yaxis4=dict(domain=[0.91, 1.0], range=[0, 1], visible=False, anchor="x4", fixedrange=True),
            )
            for axis_name in ("yaxis", "yaxis2", "yaxis3"):
                fig.layout[axis_name].domain = [0.0, 0.86]
        return fig

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
    fig.add_trace(model_trace(0.0))
    fig.add_trace(equation_trace(0.0))
    if equation_location == "math_band":
        fig.data[-1].update(xaxis="x2", yaxis="y2")
    fig.frames = [
        go.Frame(
            name=name,
            data=[model_trace(position), equation_trace(position)],
            traces=[1, 2],
        )
        for name, position in zip(frame_names, positions)
    ]
    fig.update_layout(
        **get_base_layout(
            title=title,
            margin_t=140 if equation_location == "math_band" else 160,
            theme=theme,
        ),
        annotations=[formula_annotation()],
        legend=get_legend_props(theme=theme),
        xaxis=dict(title="x\u2081", range=x_range),
        yaxis=dict(title="\u0177", range=y_range),
        sliders=_hybrid_sliders(checkpoint_names, theme=theme),
        updatemenus=get_updatemenus(frame_duration, theme=theme),
    )
    if equation_location == "math_band":
        fig.update_layout(
            xaxis2=dict(domain=[0.0, 1.0], range=[0, 1], visible=False, anchor="y2", fixedrange=True),
            yaxis2=dict(domain=[0.91, 1.0], range=[0, 1], visible=False, anchor="x2", fixedrange=True),
        )
        fig.layout.yaxis.domain = [0.0, 0.86]
    return fig


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
    animation_mode="native",
    interpolation_frames=3,
    equation_location="plot",
    theme=None,
):
    """
    Simple (1D) visualization.

    Robust mode:
      - Provide y_line_hist + x1_grid => plot uses predictions, works with ANY sklearn Pipeline/transform.

    Legacy mode:
      - Provide w_hist,b_hist => plot uses y = w*x + b (only correct for pure linear model in original space)
    """
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

        w_disp = w_hist[:, 0]
        b_disp = b_hist

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
    y_range = [y_min - y_pad, y_max + y_pad]

    if animation_mode == "hybrid":
        line_history = np.vstack([np.asarray(y_line(t), dtype=float).ravel() for t in range(steps_n)])
        y_span = y_max - y_min + 1e-9
        return _build_hybrid_figure(
            x1,
            y,
            x1_grid,
            line_history,
            w_disp,
            b_disp,
            loss_hist=loss_hist,
            metrics_hist=metrics_hist,
            show_loss=show_loss,
            title=title,
            dec=dec,
            frame_duration=frame_duration,
            interpolation_frames=interpolation_frames,
            theta_formula_annotation=theta_formula_annotation,
            x_range=x_range,
            y_range=[y_range[0], y_max + 0.32 * y_span],
            y_text=y_max + 0.22 * y_span,
            equation_location=equation_location,
            theme=theme,
        )

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
        fig.update_yaxes(title="ŷ", range=y_range, row=1, col=1)

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
        yaxis=dict(title="ŷ", range=y_range),
        sliders=get_sliders(steps_n, theme=theme),
        updatemenus=get_updatemenus(frame_duration, theme=theme),
    )

    return fig


__all__ = ["build_simple_lr_figure"]
