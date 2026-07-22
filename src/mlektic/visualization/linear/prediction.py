import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..theme import (
    _resolve,
    data_3d_marker_style,
    data_marker_style,
    get_base_layout,
    model_line_style,
    surface_style,
)


def _fmt(val, dec=4):
    number = float(val)
    if number != 0 and abs(number) >= 1_000_000:
        exponent = int(np.floor(np.log10(abs(number))))
        mantissa = number / (10**exponent)
        mantissa_text = f"{mantissa:.{min(dec, 4)}f}".rstrip('0').rstrip('.')
        return rf"{mantissa_text}\times 10^{{{exponent}}}"
    s = f"{number:.{dec}f}"
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s

def _get_last_estimator(est):
    if hasattr(est, "steps"):
        return est.steps[-1][1]
    return est

def _extract_linear_theta(est):
    last = _get_last_estimator(est)
    if not (hasattr(last, "coef_") and hasattr(last, "intercept_")):
        raise ValueError("Estimator must expose coef_ and intercept_.")
    w = np.asarray(last.coef_, dtype=float).ravel()
    b = np.asarray(last.intercept_, dtype=float).ravel()
    b0 = float(b[0]) if b.size else float(last.intercept_)
    return w, b0

def _find_standard_scaler(est):
    if not hasattr(est, "steps"):
        return None
    for _, step in est.steps:
        if hasattr(step, "transform") and hasattr(step, "mean_") and (hasattr(step, "scale_") or hasattr(step, "var_")):
            return step
    return None

def _safe_get_scale(scaler):
    if scaler is None:
        return None, None, True, True
    mu = getattr(scaler, "mean_", None)
    scale = getattr(scaler, "scale_", None)
    if scale is None:
        var = getattr(scaler, "var_", None)
        if var is not None:
            scale = np.sqrt(np.asarray(var, dtype=float))
        else:
            scale = None
    with_mean = bool(getattr(scaler, "with_mean", True))
    with_std = bool(getattr(scaler, "with_std", True))
    return mu, scale, with_mean, with_std

def _to_scaled_x(x, scaler):
    if scaler is None:
        return np.asarray(x, dtype=float).ravel()
    try:
        return np.asarray(scaler.transform(np.asarray(x, dtype=float).reshape(1, -1)), dtype=float).ravel()
    except Exception:
        mu, scale, with_mean, with_std = _safe_get_scale(scaler)
        xs = np.asarray(x, dtype=float).ravel().copy()
        if with_mean and mu is not None:
            xs = xs - np.asarray(mu, dtype=float).ravel()
        if with_std and scale is not None:
            sc = np.asarray(scale, dtype=float).ravel()
            xs = xs / (sc + 1e-12)
        return xs

def _theta_to_original(w_s, b_s, scaler):
    w_s = np.asarray(w_s, dtype=float).ravel()
    b_s = float(b_s)
    if scaler is None:
        return w_s.copy(), b_s
    mu, scale, with_mean, with_std = _safe_get_scale(scaler)
    dloc = w_s.size

    if (not with_std) or (scale is None):
        scale = np.ones(dloc, dtype=float)
    else:
        scale = np.asarray(scale, dtype=float).ravel()
    if (not with_mean) or (mu is None):
        mu = np.zeros(dloc, dtype=float)
    else:
        mu = np.asarray(mu, dtype=float).ravel()

    w_o = w_s / (scale + 1e-12)
    b_o = float(b_s - np.sum(w_s * mu / (scale + 1e-12)))
    return w_o, b_o

def _custom_updatemenus(buttons, btn_bg, btn_border, btn_font_color, x=0.5, y=1.10, xanchor="center"):
    return [dict(
        type="buttons",
        direction="left",
        x=x, y=y, xanchor=xanchor,
        bgcolor=btn_bg,
        bordercolor=btn_border,
        borderwidth=1,
        font=dict(color=btn_font_color, size=14),
        buttons=buttons,
    )]

def _custom_sliders(steps, slider_font_color):
    cv = dict(prefix="Stage: ")
    if slider_font_color:
        cv["font"] = dict(color=slider_font_color)
    return [dict(
        active=0,
        currentvalue=cv,
        pad=dict(t=55),
        steps=steps,
    )]

def _explain_lr_1d(X_train, y_train, x_disp, w_disp, b_disp, yhat, title, dec, grid_points, theme, p, text_color, ann_color, btn_bg, btn_border, btn_font_color):
    x1_train = X_train[:, 0].ravel()
    xq1_disp = float(x_disp[0])

    x_min, x_max = float(x1_train.min()), float(x1_train.max())
    x_grid = np.linspace(x_min, x_max, int(grid_points))
    y_grid = x_grid * w_disp[0] + b_disp

    y_all = np.concatenate([y_train, y_grid, np.array([yhat], dtype=float)])
    y_min, y_max = float(y_all.min()), float(y_all.max())
    y_pad = 0.08 * (y_max - y_min + 1e-9)

    def _pad(lo, hi, frac=0.10):
        span = (hi - lo) + 1e-9
        return [lo - frac * span, hi + frac * span]

    x_range = _pad(x_min, x_max)
    y_range = [y_min - y_pad, y_max + y_pad]

    vars_tex = r"$\begin{aligned}" + rf"x_1 &= {_fmt(xq1_disp, dec)}" + r"\end{aligned}$"
    subst_tex = (
        r"$\begin{aligned}"
        r"\hat{y} &= \theta_0 + \theta_1 x_1\\"
        rf"\hat{{y}} &= ({_fmt(b_disp, dec)})\;+ \\"
        rf"&\quad ({_fmt(w_disp[0], dec)}) \cdot ({_fmt(xq1_disp, dec)})"
        r"\end{aligned}$"
    )
    res_tex = (
        r"$\begin{aligned}"
        rf"\hat{{y}} &= {_fmt(yhat, dec)}\\"
        rf"(x_1, \hat{{y}}) &= ({_fmt(xq1_disp, dec)}, {_fmt(yhat, dec)})"
        r"\end{aligned}$"
    )

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.36, 0.64],
        horizontal_spacing=0.14,
        specs=[[{"type": "xy"}, {"type": "xy"}]],
    )
    fig.update_xaxes(visible=False, range=[0, 1], row=1, col=1)
    fig.update_yaxes(visible=False, range=[0, 1], row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x1_train, y=y_train,
        mode="markers",
        name="Data",
        marker=data_marker_style(theme=theme),
        legendgroup="fit",
        showlegend=True,
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=x_grid, y=y_grid,
        mode="lines",
        name="Model",
        line=model_line_style(theme=theme),
        hoverlabel=dict(bgcolor="white", font=dict(color="black")),
        legendgroup="fit",
        showlegend=True,
        uid="MODEL_LINE",
    ), row=1, col=2)

    pred_color = p.get("loss_line", "#00cc96")
    fig.add_trace(go.Scatter(
        x=[x_disp[0]], y=[yhat],
        mode="markers",
        name="Prediction",
        marker=dict(size=12, symbol="circle", color=pred_color),
        legendgroup="fit",
        showlegend=True,
        visible=False,
        uid="PRED_POINT",
    ), row=1, col=2)

    def block_rect(y0, y1):
        return dict(
            type="rect", xref="x1", yref="y1",
            x0=0.02, x1=0.98, y0=y0, y1=y1,
            line=dict(width=1),
            fillcolor="rgba(220,220,220,0.10)", layer="below",
        )
    shapes = [block_rect(0.68, 0.98), block_rect(0.35, 0.65), block_rect(0.02, 0.32)]

    def title_annot(tex_title, y):
        return dict(
            x=0.06, y=y, xref="x1", yref="y1",
            text=rf"$\bf{{{tex_title}}}$", showarrow=False,
            xanchor="left", yanchor="top",
            font=dict(size=16, color=text_color),
        )

    def body_annot(tex_body, y):
        return dict(
            x=0.06, y=y, xref="x1", yref="y1",
            text=tex_body, showarrow=False,
            xanchor="left", yanchor="top", align="left",
            font=dict(size=15, color=text_color),
        )

    T1, T2, T3 = r"Variables\ (Input)", r"Substitution", r"Result\ (Output)"

    def ann_slots(stage: int):
        v_body = "" if stage < 1 else vars_tex
        s_body = "" if stage < 2 else subst_tex
        r_body = "" if stage < 3 else res_tex
        ann = [
            title_annot(T1, 0.96), body_annot(v_body, 0.89),
            title_annot(T2, 0.63), body_annot(s_body, 0.56),
            title_annot(T3, 0.30), body_annot(r_body, 0.23),
        ]
        if stage == 3:
            ann.append(dict(
                x=x_disp[0], y=yhat, xref="x2", yref="y2",
                text=rf"$\hat{{y}}={_fmt(yhat, dec)}$",
                showarrow=True, arrowhead=2, ax=25, ay=-35,
                font=dict(size=14, color=ann_color),
            ))
        return ann

    stage_pred_visible = [False, False, False, True]
    slider_steps = []
    for s in [0, 1, 2, 3]:
        slider_steps.append(dict(
            label=str(s), method="update",
            args=[{"visible": [True, True, stage_pred_visible[s]]}, {"annotations": ann_slots(s)}],
        ))

    buttons = [
        dict(label="Input", method="update", args=[{"visible": [True, True, False]}, {"annotations": ann_slots(1)}]),
        dict(label="Substitution", method="update", args=[{"visible": [True, True, False]}, {"annotations": ann_slots(2)}]),
        dict(label="Output", method="update", args=[{"visible": [True, True, True]}, {"annotations": ann_slots(3)}]),
        dict(label="Reset", method="update", args=[{"visible": [True, True, False]}, {"annotations": ann_slots(0)}]),
    ]

    layout_kwargs = get_base_layout(title=title, margin_t=110, theme=theme)
    layout_kwargs["margin"] = dict(t=110, r=50, l=60, b=80)

    fig.update_layout(
        **layout_kwargs,
        shapes=shapes,
        annotations=ann_slots(0),
        updatemenus=_custom_updatemenus(buttons, btn_bg, btn_border, btn_font_color),
    )
    fig.update_xaxes(title="x₁", range=x_range, row=1, col=2)
    fig.update_yaxes(title="ŷ", range=y_range, row=1, col=2)
    return fig

def _explain_lr_2d(X_train, y_train, x_disp, w_disp, b_disp, yhat, title, dec, grid_2d_points, theme, p, text_color, ann_color, btn_bg, btn_border, btn_font_color):
    x1, x2 = X_train[:, 0].ravel(), X_train[:, 1].ravel()
    xq1_disp, xq2_disp = float(x_disp[0]), float(x_disp[1])

    x1_min, x1_max = float(x1.min()), float(x1.max())
    x2_min, x2_max = float(x2.min()), float(x2.max())

    X1g, X2g = np.meshgrid(
        np.linspace(x1_min, x1_max, int(grid_2d_points)),
        np.linspace(x2_min, x2_max, int(grid_2d_points))
    )
    Zg = X1g * w_disp[0] + X2g * w_disp[1] + b_disp

    z_all = np.concatenate([y_train, Zg.ravel(), [yhat]])
    z_min, z_max = float(z_all.min()), float(z_all.max())

    def _pad(lo, hi, frac=0.10):
        span = (hi - lo) + 1e-9
        return [lo - frac * span, hi + frac * span]

    x1_range, x2_range, z_range = _pad(x1_min, x1_max), _pad(x2_min, x2_max), _pad(z_min, z_max)
    CAMERA = dict(eye=dict(x=1.55, y=1.55, z=1.15))

    vars_tex = r"$\begin{aligned}" + rf"&x_1 = {_fmt(xq1_disp, dec)}\\" + rf"&x_2 = {_fmt(xq2_disp, dec)}" + r"\end{aligned}$"
    subst_tex = (
        r"$\begin{aligned}"
        r"\hat{y} &= \theta_0 + \theta_1 x_1 + \theta_2 x_2\\"
        rf"\hat{{y}} &= ({_fmt(b_disp, dec)})\;+ \\"
        rf"&\quad ({_fmt(w_disp[0], dec)}) \cdot ({_fmt(xq1_disp, dec)})\;+ \\"
        rf"&\quad ({_fmt(w_disp[1], dec)}) \cdot ({_fmt(xq2_disp, dec)})"
        r"\end{aligned}$"
    )
    res_tex = (
        r"$\begin{aligned}"
        rf"\hat{{y}} &= {_fmt(yhat, dec)}\\"
        r"(x_1, x_2, \hat{y}) &= \\"
        rf"&\quad ({_fmt(xq1_disp, dec)}, {_fmt(xq2_disp, dec)}, {_fmt(yhat, dec)})"
        r"\end{aligned}$"
    )

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.40, 0.60],
        horizontal_spacing=0.10, specs=[[{"type": "xy"}, {"type": "scene"}]],
    )
    fig.update_xaxes(visible=False, range=[0, 1], row=1, col=1)
    fig.update_yaxes(visible=False, range=[0, 1], row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=x1, y=x2, z=y_train,
        mode="markers", name="Data",
        marker=data_3d_marker_style(theme=theme),
        hovertemplate="<b>Data</b><br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>",
        legendgroup="fit", showlegend=True,
    ), row=1, col=2)

    fig.add_trace(go.Surface(
        x=X1g, y=X2g, z=Zg,
        name="Model",
        **surface_style(theme=theme),
        showlegend=True, legendgroup="fit",
        uid="MODEL_PLANE",
    ), row=1, col=2)

    pred_color = p.get("loss_line", "#00cc96")
    fig.add_trace(go.Scatter3d(
        x=[x_disp[0]], y=[x_disp[1]], z=[yhat],
        mode="markers", name="Prediction",
        marker=dict(size=6, color=pred_color),
        hovertemplate="<b>Prediction</b><br>x₁: %{x}<br>x₂: %{y}<br>ŷ: %{z}<extra></extra>",
        legendgroup="fit", showlegend=True,
        visible=False, uid="PRED_POINT_3D",
    ), row=1, col=2)

    def block_rect(y0, y1):
        return dict(
            type="rect", xref="x1", yref="y1",
            x0=0.02, x1=0.98, y0=y0, y1=y1,
            line=dict(width=1),
            fillcolor="rgba(220,220,220,0.10)", layer="below",
        )
    shapes = [block_rect(0.68, 0.98), block_rect(0.35, 0.65), block_rect(0.02, 0.32)]

    def title_annot(tex_title, y):
        return dict(
            x=0.06, y=y, xref="x1", yref="y1",
            text=rf"$\bf{{{tex_title}}}$", showarrow=False,
            xanchor="left", yanchor="top",
            font=dict(size=16, color=text_color),
        )

    def body_annot(tex_body, y, size=15):
        return dict(
            x=0.06, y=y, xref="x1", yref="y1",
            text=tex_body, showarrow=False,
            xanchor="left", yanchor="top", align="left",
            font=dict(size=size, color=text_color),
        )

    T1, T2, T3 = r"Variables\ (Input)", r"Substitution", r"Result\ (Output)"

    def ann_slots(stage: int):
        v_body = "" if stage < 1 else vars_tex
        s_body = "" if stage < 2 else subst_tex
        r_body = "" if stage < 3 else res_tex
        return [
            title_annot(T1, 0.96), body_annot(v_body, 0.89),
            title_annot(T2, 0.63), body_annot(s_body, 0.56, size=13),
            title_annot(T3, 0.30), body_annot(r_body, 0.23, size=12),
        ]

    def scene_ann(stage: int):
        if stage == 3:
            return [dict(
                x=x_disp[0], y=x_disp[1], z=yhat,
                text=rf"$\hat{{y}}={_fmt(yhat, dec)}$",
                showarrow=True, arrowhead=2, ax=25, ay=-35,
                font=dict(size=14, color=ann_color),
            )]
        return []

    stage_pred_visible = [False, False, False, True]
    slider_steps = []
    for s in [0, 1, 2, 3]:
        slider_steps.append(dict(
            label=str(s), method="update",
            args=[{"visible": [True, True, stage_pred_visible[s]]}, {"annotations": ann_slots(s), "scene.annotations": scene_ann(s)}],
        ))

    buttons = [
        dict(label="Input", method="update", args=[{"visible": [True, True, False]}, {"annotations": ann_slots(1), "scene.annotations": scene_ann(1)}]),
        dict(label="Substitution", method="update", args=[{"visible": [True, True, False]}, {"annotations": ann_slots(2), "scene.annotations": scene_ann(2)}]),
        dict(label="Output", method="update", args=[{"visible": [True, True, True]}, {"annotations": ann_slots(3), "scene.annotations": scene_ann(3)}]),
        dict(label="Reset", method="update", args=[{"visible": [True, True, False]}, {"annotations": ann_slots(0), "scene.annotations": scene_ann(0)}]),
    ]

    layout_kwargs = get_base_layout(title=title, margin_t=110, theme=theme)
    layout_kwargs["margin"] = dict(t=110, r=50, l=60, b=80)

    fig.update_layout(
        **layout_kwargs,
        shapes=shapes,
        annotations=ann_slots(0),
        updatemenus=_custom_updatemenus(buttons, btn_bg, btn_border, btn_font_color),
        scene=dict(
            xaxis=dict(title="x₁", range=x1_range),
            yaxis=dict(title="x₂", range=x2_range),
            zaxis=dict(title="ŷ", range=z_range),
            aspectmode="cube",
            camera=CAMERA,
            annotations=scene_ann(0),
        ),
    )
    return fig

def _matrix_compact(items, rows, cols, head_rows, tail_rows):
    cap = rows * cols
    items = list(items)
    if len(items) <= cap:
        padded = items + [r"\;"] * (cap - len(items))
        M = np.array(padded, dtype=object).reshape(rows, cols)
        return [" & ".join(M[r, c] for c in range(cols)) for r in range(rows)]
    head, tail = items[:head_rows * cols], items[-tail_rows * cols:]
    H, T = np.array(head, dtype=object).reshape(head_rows, cols), np.array(tail, dtype=object).reshape(tail_rows, cols)
    lines = [" & ".join(H[r, c] for c in range(cols)) for r in range(head_rows)]
    lines.append(" & ".join([r"\vdots"] * cols))
    lines.extend(" & ".join(T[r, c] for c in range(cols)) for r in range(tail_rows))
    if len(lines) < rows:
        lines += [" & ".join([r"\;"] * cols)] * (rows - len(lines))
    elif len(lines) > rows:
        lines = lines[:rows]
    return lines

def _explain_lr_nd(d, x_disp, w_disp, b_disp, yhat, title, dec, theme, p, text_color, ann_color, btn_bg, btn_border, btn_font_color):
    if d <= 12:
        model_formula_tex = rf"$\hat{{y}}=\theta_0+\sum_{{j=1}}^{{{d}}}\theta_j x_j$"
        vars_lines = [rf"x_{{{j+1}}} = {_fmt(x_disp[j], dec)}" for j in range(d)]
        vars_tex = r"$\begin{aligned}" + r"\\ ".join([rf"&{ln}" for ln in vars_lines]) + r"\end{aligned}$"

        subst_lines = []
        for j in range(d):
            subst_lines.append(rf"({_fmt(w_disp[j], dec)})\cdot({_fmt(x_disp[j], dec)})\;+")
        subst_lines.append(rf"({_fmt(b_disp, dec)})")

        subst_tex = r"$\begin{aligned}" + r"&\hat{y} = " + subst_lines[0] + r"\\ " + r"\\ ".join([rf"&\quad {ln}" for ln in subst_lines[1:]]) + r"\end{aligned}$"
        res_yhat_tex = r"$\begin{aligned}" + rf"\hat{{y}} &= {_fmt(yhat, dec)}" + r"\end{aligned}$"
        vals = [_fmt(v, dec) for v in x_disp] + [_fmt(yhat, dec)]
        pairs = []
        for i in range(0, len(vals), 2):
            pairs.append(", ".join(vals[i:i+2]))

        rhs_lines = []
        for i, pair in enumerate(pairs):
            if i == 0:
                line = rf"&\quad ({pair}" + ("," if len(pairs) > 1 else ")")
            elif i == len(pairs) - 1:
                line = rf"&\quad {pair})"
            else:
                line = rf"&\quad {pair},"
            rhs_lines.append(line)

        rhs_tex = r" \\ ".join(rhs_lines)

        point_tex = (
            r"$\begin{aligned}"
            rf"&(x_1, x_2, \dots, \hat{{y}}) = \\"
            rf"{rhs_tex}"
            r"\end{aligned}$"
        )

        fig = make_subplots(
            rows=1, cols=3, column_widths=[0.33, 0.34, 0.33],
            horizontal_spacing=0.04, specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
        )
        for c in (1, 2, 3):
            fig.update_xaxes(visible=False, range=[0, 1], row=1, col=c)
            fig.update_yaxes(visible=False, range=[0, 1], row=1, col=c)
            fig.add_trace(go.Scatter(x=[], y=[], showlegend=False, hoverinfo="skip"), row=1, col=c)

        DY = -0.05
        def title_annot(col, tex_title, y):
            return dict(x=0.5, y=y + DY, xref=f"x{col}", yref=f"y{col}", text=rf"$\bf{{{tex_title}}}$", showarrow=False, xanchor="center", yanchor="top", font=dict(size=16, color=text_color))
        def paper_top_center(tex_body, y=1.05, size=18):
            return dict(x=0.5, y=y + DY, xref="paper", yref="paper", text=tex_body, showarrow=False, xanchor="center", yanchor="bottom", align="center", font=dict(size=size, color=text_color))
        def body_annot(col, tex_body, y, size=15, align="center", x_pos=None):
            if x_pos is None:
                x_pos = 0.5 if align == "center" else 0.10
            x_anch = "center" if align == "center" else "left"
            return dict(x=x_pos, y=y + DY, xref=f"x{col}", yref=f"y{col}", text=tex_body, showarrow=False, xanchor=x_anch, yanchor="top", align="left", font=dict(size=size, color=text_color))
        def block_rect(col, y0, y1):
            return dict(type="rect", xref=f"x{col}", yref=f"y{col}", x0=0.02, x1=0.98, y0=y0, y1=y1 + DY, line=dict(width=1), fillcolor="rgba(220,220,220,0.10)", layer="below")

        shapes = [block_rect(1, 0.02, 0.98), block_rect(2, 0.02, 0.98), block_rect(3, 0.02, 0.98)]
        T1, T2, T3 = r"Variables\ (Input)", r"Substitution", r"Result\ (Output)"

        def ann_slots(stage: int):
            ann = [paper_top_center(model_formula_tex, y=1.05, size=18)]
            ann.append(title_annot(1, T1, 0.96))
            ann.append(body_annot(1, "" if stage < 1 else vars_tex, 0.90, size=15, align="center"))
            ann.append(title_annot(2, T2, 0.96))
            ann.append(body_annot(2, "" if stage < 2 else subst_tex, 0.90, size=15, align="left", x_pos=0.15))
            ann.append(title_annot(3, T3, 0.96))
            if stage < 3:
                ann.append(body_annot(3, "", 0.88, size=15, align="center"))
                ann.append(body_annot(3, "", 0.74, size=15, align="left", x_pos=0.25))
            else:
                ann.append(body_annot(3, res_yhat_tex, 0.88, size=15, align="center"))
                ann.append(body_annot(3, point_tex, 0.79, size=15, align="left", x_pos=0.25))
            return ann

        slider_steps = []
        for s in [0, 1, 2, 3]:
            slider_steps.append(dict(label=str(s), method="update", args=[{"visible": [True, True, True]}, {"annotations": ann_slots(s)}]))

        buttons = [
            dict(label="Input", method="update", args=[{"visible": [True, True, True]}, {"annotations": ann_slots(1)}]),
            dict(label="Substitution", method="update", args=[{"visible": [True, True, True]}, {"annotations": ann_slots(2)}]),
            dict(label="Output", method="update", args=[{"visible": [True, True, True]}, {"annotations": ann_slots(3)}]),
            dict(label="Reset", method="update", args=[{"visible": [True, True, True]}, {"annotations": ann_slots(0)}]),
        ]

        layout_kwargs = get_base_layout(title=title, margin_t=110, theme=theme)
        layout_kwargs["margin"] = dict(t=110, r=50, l=60, b=80)

        up_kwargs = dict(x=0.08, xanchor="left") if d >= 10 else {}

        fig.update_layout(
            **layout_kwargs, shapes=shapes, annotations=ann_slots(0),
            updatemenus=_custom_updatemenus(buttons, btn_bg, btn_border, btn_font_color, **up_kwargs),
        )
        return fig

    # ---- MODE: d > 12 => matrix view ----
    x_rows, x_cols = 11, 1
    x_items = [rf"{_fmt(x_disp[j], dec)}" for j in range(d)]
    x_mat_inner = r" \\ ".join(_matrix_compact(x_items, x_rows, x_cols, 5, 5))
    x_mat_tex = rf"$\mathbf{{x}}=\begin{{bmatrix}} {x_mat_inner} \end{{bmatrix}}$"

    th_rows, th_cols = 9, 1
    th_items = [rf"{_fmt(w_disp[j], dec)}" for j in range(d)]
    th_mat_inner = r" \\ ".join(_matrix_compact(th_items, th_rows, th_cols, 4, 4))
    th_mat_tex = rf"$\boldsymbol{{\theta}}=\begin{{bmatrix}} {th_mat_inner} \end{{bmatrix}}$"

    x_dim_tex = rf"$\mathbf{{x}}\in\mathbb{{R}}^{{{d}}}$"
    th_dim_tex = rf"$\boldsymbol{{\theta}}\in\mathbb{{R}}^{{{d}}}$"
    y_dim_tex = r"$\hat{y}\in\mathbb{R}$"
    theta0_tex = rf"$\theta_0 = {_fmt(b_disp, dec)}$"

    model_formula_tex = r"$\hat{y}=\boldsymbol{\theta}^{\top}\mathbf{x}+\theta_0$"
    subst_eq_tex = (
        r"$\begin{aligned}"
        rf"&\hat{{y}} = ({_fmt(w_disp[0], dec)})\cdot({_fmt(x_disp[0], dec)}) \\"
        rf"&\quad + ({_fmt(w_disp[1], dec)})\cdot({_fmt(x_disp[1], dec)}) + \cdots\\"
        rf"&\quad + ({_fmt(b_disp, dec)})"
        r"\end{aligned}$"
    )
    res_yhat_tex = r"$\begin{aligned}" + rf"\hat{{y}} &= {_fmt(yhat, dec)}" + r"\end{aligned}$"
    point_tex = (
        r"$\begin{aligned}"
        r"&(x_1, x_2, \dots, \hat{y}) = \\"
        r"&\quad (" + rf"{_fmt(x_disp[0], dec)}, {_fmt(x_disp[1], dec)}, \dots," + r"\\"
        r"&\quad " + rf"{_fmt(yhat, dec)})"
        r"\end{aligned}$"
    )

    fig = make_subplots(
        rows=1, cols=3, column_widths=[0.33, 0.34, 0.33],
        horizontal_spacing=0.04, specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
    )
    for c in (1, 2, 3):
        fig.update_xaxes(visible=False, range=[0, 1], row=1, col=c)
        fig.update_yaxes(visible=False, range=[0, 1], row=1, col=c)
        fig.add_trace(go.Scatter(x=[], y=[], showlegend=False, hoverinfo="skip"), row=1, col=c)

    DY = -0.05
    def title_annot(col, tex_title, y):
        return dict(x=0.5, y=y + DY, xref=f"x{col}", yref=f"y{col}", text=rf"$\bf{{{tex_title}}}$", showarrow=False, xanchor="center", yanchor="top", font=dict(size=16, color=text_color))
    def top_center_annot(col, tex_body, y):
        return dict(x=0.5, y=y + DY, xref=f"x{col}", yref=f"y{col}", text=tex_body, showarrow=False, xanchor="center", yanchor="bottom", align="center", font=dict(size=15, color=text_color))
    def paper_top_center(tex_body, y=1.05, size=18):
        return dict(x=0.5, y=y + DY, xref="paper", yref="paper", text=tex_body, showarrow=False, xanchor="center", yanchor="bottom", align="center", font=dict(size=size, color=text_color))
    def body_annot(col, tex_body, y, size=15, align="center", x_pos=None):
        if x_pos is None:
            x_pos = 0.5 if align == "center" else 0.08
        x_anch = "center" if align == "center" else "left"
        return dict(x=x_pos, y=y + DY, xref=f"x{col}", yref=f"y{col}", text=tex_body, showarrow=False, xanchor=x_anch, yanchor="top", align="left", font=dict(size=size, color=text_color))
    def block_rect(col, y0, y1):
        return dict(type="rect", xref=f"x{col}", yref=f"y{col}", x0=0.02, x1=0.98, y0=y0, y1=y1 + DY, line=dict(width=1), fillcolor="rgba(220,220,220,0.10)", layer="below")

    shapes = [block_rect(1, 0.02, 0.98), block_rect(2, 0.02, 0.98), block_rect(3, 0.02, 0.98)]
    T1, T2, T3 = r"Variables\ (Input)", r"Substitution", r"Result\ (Output)"

    def ann_slots(stage: int):
        ann = [paper_top_center(model_formula_tex, y=1.07, size=18)]
        ann.append(top_center_annot(1, x_dim_tex, 0.995))
        ann.append(top_center_annot(2, th_dim_tex, 0.995))
        ann.append(top_center_annot(3, y_dim_tex, 0.995))
        ann.append(title_annot(1, T1, 0.96))
        ann.append(body_annot(1, "" if stage < 1 else x_mat_tex, 0.88, size=14, align="left"))
        ann.append(title_annot(2, T2, 0.96))
        if stage < 2:
            ann.append(body_annot(2, "", 0.88, align="center"))
            ann.append(body_annot(2, "", 0.38, align="center"))
        else:
            ann.append(body_annot(2, th_mat_tex, 0.90, size=14, align="left"))
            ann.append(body_annot(2, theta0_tex, 0.35, size=15, align="center"))
            ann.append(body_annot(2, subst_eq_tex, 0.27, size=15, align="center"))
        ann.append(title_annot(3, T3, 0.96))
        if stage < 3:
            ann.append(body_annot(3, "", 0.88, size=15, align="center"))
            ann.append(body_annot(3, "", 0.74, size=15, align="left", x_pos=0.25))
        else:
            ann.append(body_annot(3, res_yhat_tex, 0.88, size=15, align="center"))
            ann.append(body_annot(3, point_tex, 0.79, size=15, align="left", x_pos=0.25))
        return ann

    slider_steps = []
    for s in [0, 1, 2, 3]:
        slider_steps.append(dict(label=str(s), method="update", args=[{"visible": [True, True, True]}, {"annotations": ann_slots(s)}]))

    buttons = [
        dict(label="Input", method="update", args=[{"visible": [True, True, True]}, {"annotations": ann_slots(1)}]),
        dict(label="Substitution", method="update", args=[{"visible": [True, True, True]}, {"annotations": ann_slots(2)}]),
        dict(label="Output", method="update", args=[{"visible": [True, True, True]}, {"annotations": ann_slots(3)}]),
        dict(label="Reset", method="update", args=[{"visible": [True, True, True]}, {"annotations": ann_slots(0)}]),
    ]

    layout_kwargs = get_base_layout(title=title, margin_t=110, theme=theme)
    layout_kwargs["margin"] = dict(t=110, r=50, l=60, b=80)

    fig.update_layout(
        **layout_kwargs, shapes=shapes, annotations=ann_slots(0),
        updatemenus=_custom_updatemenus(buttons, btn_bg, btn_border, btn_font_color, x=0.08, xanchor="left"),
    )
    return fig


def explain_lr_prediction(
    trained_estimator,
    X_train, y_train,
    *,
    x_query=None,
    yhat=None,
    title=None,
    dec=4,
    grid_points=250,
    grid_2d_points=40,
    display_space="original",
    theme=None,
):
    """Build a visual explanation for a linear-regression prediction."""
    X_train = np.asarray(X_train, dtype=float)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    y_train = np.asarray(y_train, dtype=float).ravel()

    n, d = X_train.shape
    if d < 1:
        raise ValueError("X_train must have at least 1 feature.")

    if display_space not in ("original", "scaled"):
        raise ValueError("display_space must be 'original' or 'scaled'.")

    if title is None:
        if d == 1:
            title = "Prediction (Linear Regression)"
        elif d == 2:
            title = "Prediction (Linear Regression, 2 variables)"
        else:
            title = f"Prediction (Linear Regression, {d} variables)"

    if x_query is None:
        raise ValueError("You must provide x_query, e.g. x_query=np.array([[...]]).")

    x_query = np.asarray(x_query, dtype=float)
    if x_query.ndim == 1:
        x_query = x_query.reshape(1, -1)
    if x_query.shape[1] != d:
        raise ValueError(f"x_query must have shape (m, {d}). Got {x_query.shape}.")

    xq = x_query[0].astype(float).ravel()

    if yhat is None:
        yhat = trained_estimator.predict(xq.reshape(1, -1))
        yhat = float(np.asarray(yhat, dtype=float).ravel()[0])
    else:
        yhat = float(np.asarray(yhat, dtype=float).ravel()[0])

    w_scaled, b_scaled = _extract_linear_theta(trained_estimator)

    if w_scaled.size != d:
        raise ValueError(
            f"coef_ dimension mismatch: got {w_scaled.size} coefficients but X has d={d} features."
        )

    scaler = _find_standard_scaler(trained_estimator)
    xq_scaled = _to_scaled_x(xq, scaler)
    w_orig, b_orig = _theta_to_original(w_scaled, b_scaled, scaler)

    if display_space == "scaled":
        x_disp, w_disp, b_disp = xq_scaled, w_scaled, b_scaled
        if scaler is not None:
            X_train_scaled = np.zeros_like(X_train)
            for i in range(n):
                X_train_scaled[i] = _to_scaled_x(X_train[i], scaler)
            X_train = X_train_scaled
    else:
        x_disp, w_disp, b_disp = xq, w_orig, b_orig

    p = _resolve(theme)
    ann_color = p.get("annotation_color", "white")
    text_color = p.get("text", "white")
    btn_bg = p.get("btn_bg", "white")
    btn_border = p.get("btn_border", "black")
    btn_font_color = p.get("btn_font_color", "black")

    if d == 1:
        return _explain_lr_1d(X_train, y_train, x_disp, w_disp, b_disp, yhat, title, dec, grid_points, theme, p, text_color, ann_color, btn_bg, btn_border, btn_font_color)
    elif d == 2:
        return _explain_lr_2d(X_train, y_train, x_disp, w_disp, b_disp, yhat, title, dec, grid_2d_points, theme, p, text_color, ann_color, btn_bg, btn_border, btn_font_color)
    else:
        return _explain_lr_nd(d, x_disp, w_disp, b_disp, yhat, title, dec, theme, p, text_color, ann_color, btn_bg, btn_border, btn_font_color)

__all__ = ["explain_lr_prediction"]
