import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...adapters.sklearn import SklearnAdapter
from ...utils.math import _sigmoid
from ...utils.probability import multiclass_probabilities
from ..linear.prediction import (
    _custom_updatemenus,
    _extract_linear_theta,
    _find_standard_scaler,
    _fmt,
    _matrix_compact,
    _outside_training_range,
    _theta_to_original,
    _to_scaled_x,
    _validate_prediction_options,
    _validate_prediction_source,
)
from ..theme import (
    _resolve,
    data_marker_style,
    get_base_layout,
    model_line_style,
)


def _class_label_text(value):
    """Format numeric and string labels safely inside a LaTeX expression."""
    if isinstance(value, (str, np.str_)):
        escaped = str(value).replace("_", r"\_")
        return rf"\mathrm{{{escaped}}}"
    return str(value)


def _binary_class_context(classes, y_train):
    """Return fitted class order and numeric targets for probability plots.

    Scikit-learn's binary coefficient represents ``classes_[1]``. Raw string
    targets would make Plotly treat a shared probability axis as categorical,
    hiding numeric model geometry. The view therefore preserves the fitted
    class order in its labels and maps observed targets to 0 and 1 for plotting.
    """
    classes = np.asarray(classes).ravel()
    if classes.size != 2:
        raise ValueError("Binary prediction views require exactly two fitted classes.")

    y_train = np.asarray(y_train).ravel()
    numeric_targets = np.full(y_train.shape, np.nan, dtype=float)
    for index, class_label in enumerate(classes):
        numeric_targets[y_train == class_label] = float(index)
    if np.any(np.isnan(numeric_targets)):
        raise ValueError("y_train contains labels absent from the estimator's fitted classes.")
    return classes, numeric_targets


def _binary_result_tex(p_hat, y_hat, classes, dec, show_class_labels):
    """Build an explicit binary probability comparison and class decision."""
    winner_index = int(np.flatnonzero(classes == y_hat)[0])
    winner_label = (
        rf"\;({_class_label_text(y_hat)})" if show_class_labels else ""
    )
    return (
        r"$\begin{aligned}"
        rf"\hat{{\mathbf{{p}}}} &=(\hat{{p}}_0,\hat{{p}}_1)="
        rf"({_fmt(1.0 - p_hat, dec)},\,{_fmt(p_hat, dec)})\\"
        rf"\hat{{y}} &= \arg\max_{{k\in\{{0,1\}}}}\hat{{p}}_k={winner_index}{winner_label}"
        r"\end{aligned}$"
    )


def _multiclass_winner_tex(p_hat, y_hat, classes, dec, show_class_labels):
    """Build an indexed multiclass decision with optional semantic label."""
    winner_index = int(np.argmax(p_hat))
    winner_label = (
        rf"\;({_class_label_text(y_hat)})" if show_class_labels else ""
    )
    return (
        r"$\begin{aligned}"
        rf"&\max(\hat{{\mathbf{{p}}}}) = {_fmt(float(np.max(p_hat)), dec)} \\"
        rf"&\hat{{y}} = \arg\max_{{k\in\{{0,\ldots,{len(classes) - 1}\}}}}"
        rf"\hat{{p}}_k = {winner_index}{winner_label}"
        r"\end{aligned}$"
    )


def _explain_log_1d(
    X_train, y_train,
    x_disp, w_disp, b_disp,
    p_hat, y_hat, classes, show_class_labels,
    title, dec, grid_points, theme,
    p, text_color, ann_color, btn_bg, btn_border, btn_font_color
):
    classes, y_targets = _binary_class_context(classes, y_train)
    x1_train = X_train[:, 0].ravel()
    xq1_disp = float(x_disp[0])

    x_min, x_max = float(x1_train.min()), float(x1_train.max())
    # Expand slightly beyond training data
    pad_x = 0.1 * (x_max - x_min + 1e-9)
    x_min_plot, x_max_plot = x_min - pad_x, x_max + pad_x

    # Ensure the query point is within the grid
    x_min_plot = min(x_min_plot, xq1_disp - pad_x)
    x_max_plot = max(x_max_plot, xq1_disp + pad_x)

    x_grid = np.linspace(x_min_plot, x_max_plot, int(grid_points))
    z_grid = x_grid * w_disp[0] + b_disp
    y_grid = _sigmoid(z_grid)

    y_min, y_max = 0.0, 1.0
    y_pad = 0.08
    y_range = [y_min - y_pad, y_max + y_pad]
    x_range = [x_min_plot, x_max_plot]

    # Math formulas
    class_mapping = (
        rf"\\&\text{{Class }}0={_class_label_text(classes[0])},\quad "
        rf"\text{{Class }}1={_class_label_text(classes[1])}"
        if show_class_labels
        else ""
    )
    vars_tex = (
        r"$\begin{aligned}"
        + rf"x_1 &= {_fmt(xq1_disp, dec)}{class_mapping}"
        + r"\end{aligned}$"
    )

    subst_tex = (
        r"$\begin{aligned}"
        r"&\hat{p}_1 = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \theta_1x + \theta_0 \\[5pt]"
        rf"&z = ({_fmt(w_disp[0], dec)})\cdot({_fmt(xq1_disp, dec)}) + ({_fmt(b_disp, dec)}) \\[5pt]"
        rf"&\sigma(z) = \frac{{1}}{{1 + e^{{-\left(({_fmt(w_disp[0], dec)})\cdot({_fmt(xq1_disp, dec)}) + ({_fmt(b_disp, dec)})\right)}}}}"
        r"\end{aligned}$"
    )

    res_tex = _binary_result_tex(p_hat, y_hat, classes, dec, show_class_labels)

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.48, 0.52],
        horizontal_spacing=0.10,
        specs=[[{"type": "xy"}, {"type": "xy"}]],
    )
    fig.update_xaxes(visible=False, range=[0, 1], row=1, col=1)
    fig.update_yaxes(visible=False, range=[0, 1], row=1, col=1)

    # Traces
    # 1. Data
    fig.add_trace(go.Scatter(
        x=x1_train, y=y_targets,
        mode="markers",
        name="Data",
        marker=data_marker_style(theme=theme),
        legendgroup="fit",
        showlegend=True,
    ), row=1, col=2)

    # 2. Decision boundary
    w1 = float(w_disp[0])
    b = float(b_disp)
    if abs(w1) > 1e-12:
        x_decision = -b / w1
        if x_min_plot <= x_decision <= x_max_plot:
            fig.add_trace(go.Scatter(
                x=[x_decision, x_decision],
                y=y_range,
                mode="lines",
                name=(
                    f"Decision boundary: P({classes[1]} | x) = 0.5"
                    if show_class_labels else "Decision boundary: p1 = 0.5"
                ),
                line=dict(color="gray", dash="dash", width=1.5),
                legendgroup="fit",
                showlegend=True,
            ), row=1, col=2)

    # 3. Model curve
    fig.add_trace(go.Scatter(
        x=x_grid, y=y_grid,
        mode="lines",
        name=(f"Model: P({classes[1]} | x)" if show_class_labels else "Model: p1"),
        line=model_line_style(theme=theme),
        hoverlabel=dict(bgcolor="white", font=dict(color="black")),
        legendgroup="fit",
        showlegend=True,
        uid="MODEL_LINE",
    ), row=1, col=2)

    # 4. Prediction point
    pred_color = p.get("loss_line", "#00cc96")
    fig.add_trace(go.Scatter(
        x=[x_disp[0]], y=[p_hat],
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

    shapes = [block_rect(0.69, 0.98), block_rect(0.29, 0.67), block_rect(0.02, 0.27)]

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
            title_annot(T2, 0.65), body_annot(s_body, 0.61),
            title_annot(T3, 0.25), body_annot(r_body, 0.20),
        ]
        if stage == 3:
            ann.append(dict(
                x=x_disp[0], y=p_hat, xref="x2", yref="y2",
                text=rf"$\hat{{p}}={_fmt(p_hat, dec)}$",
                showarrow=True, arrowhead=2, ax=35, ay=-35,
                font=dict(size=14, color="white"),
            ))
        return ann

    stage_pred_visible = [False, False, False, True]
    slider_steps = []
    for s in [0, 1, 2, 3]:
        # trace visibility: data (0), boundary (1 - if exists), model (2), pred (3)
        # However, boundary trace only exists if w1 != 0 and within plot.
        # Let's count traces.
        n_traces = len(fig.data)
        # last trace is always prediction. All other traces should be visible=True
        vis_array = [True] * (n_traces - 1) + [stage_pred_visible[s]]
        slider_steps.append(dict(
            label=str(s), method="update",
            args=[{"visible": vis_array}, {"annotations": ann_slots(s)}],
        ))

    def make_btn(label, s):
        vis_array = [True] * (len(fig.data) - 1) + [stage_pred_visible[s]]
        return dict(label=label, method="update", args=[{"visible": vis_array}, {"annotations": ann_slots(s)}])

    buttons = [
        make_btn("Input", 1),
        make_btn("Substitution", 2),
        make_btn("Output", 3),
        make_btn("Reset", 0),
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
    fig.update_yaxes(
        title=(f"P({classes[1]} | x)" if show_class_labels else "p1"),
        range=y_range,
        tickvals=[0.0, 0.5, 1.0],
        ticktext=(
            [f"0 - {classes[0]}", "0.5 threshold", f"1 - {classes[1]}"]
            if show_class_labels else ["0", "0.5 threshold", "1"]
        ),
        row=1,
        col=2,
    )
    return fig

def _explain_log_2d(
    X_train, y_train,
    x_disp, w_disp, b_disp,
    p_hat, y_hat, classes, show_class_labels,
    title, dec, grid_2d_points, theme,
    p, text_color, ann_color, btn_bg, btn_border, btn_font_color
):
    from ..theme import data_3d_marker_style, surface_style

    classes, y_targets = _binary_class_context(classes, y_train)
    x1, x2 = X_train[:, 0].ravel(), X_train[:, 1].ravel()
    xq1_disp, xq2_disp = float(x_disp[0]), float(x_disp[1])

    x1_min, x1_max = min(float(x1.min()), float(x_disp[0])), max(float(x1.max()), float(x_disp[0]))
    x2_min, x2_max = min(float(x2.min()), float(x_disp[1])), max(float(x2.max()), float(x_disp[1]))

    X1g, X2g = np.meshgrid(
        np.linspace(x1_min, x1_max, int(grid_2d_points)),
        np.linspace(x2_min, x2_max, int(grid_2d_points))
    )
    Zg = X1g * w_disp[0] + X2g * w_disp[1] + b_disp
    Pg = _sigmoid(Zg)

    def _pad(lo, hi, frac=0.10):
        span = (hi - lo) + 1e-9
        return [lo - frac * span, hi + frac * span]

    x1_range, x2_range = _pad(x1_min, x1_max), _pad(x2_min, x2_max)
    z_range = [-0.08, 1.08]
    CAMERA = dict(eye=dict(x=1.55, y=1.55, z=1.15))

    class_mapping = (
        rf"\\&\text{{Class }}0={_class_label_text(classes[0])},\quad "
        rf"\text{{Class }}1={_class_label_text(classes[1])}"
        if show_class_labels
        else ""
    )
    vars_tex = (
        r"$\begin{aligned}"
        + rf"&x_1 = {_fmt(xq1_disp, dec)},\quad x_2 = {_fmt(xq2_disp, dec)}{class_mapping}"
        + r"\end{aligned}$"
    )

    subst_tex = (
        r"$\begin{aligned}"
        r"&\hat{p}_1=\sigma(z)=\frac{1}{1+e^{-z}},\quad z=\boldsymbol{\theta}^{\top}\mathbf{x}+\theta_0\\[5pt]"
        rf"&z = ({_fmt(w_disp[0], dec)})\cdot({_fmt(xq1_disp, dec)}) + ({_fmt(w_disp[1], dec)})\cdot({_fmt(xq2_disp, dec)}) + ({_fmt(b_disp, dec)}) \\[5pt]"
        rf"&\sigma(z) = \frac{{1}}{{1 + e^{{-\left(({_fmt(w_disp[0], dec)})\cdot({_fmt(xq1_disp, dec)}) + ({_fmt(w_disp[1], dec)})\cdot({_fmt(xq2_disp, dec)}) + ({_fmt(b_disp, dec)})\right)}}}}"
        r"\end{aligned}$"
    )

    res_tex = _binary_result_tex(p_hat, y_hat, classes, dec, show_class_labels)

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.64, 0.36],
        horizontal_spacing=0.08, specs=[[{"type": "xy"}, {"type": "scene"}]],
    )
    fig.update_xaxes(visible=False, range=[0, 1], row=1, col=1)
    fig.update_yaxes(visible=False, range=[0, 1], row=1, col=1)

    # 1. Data
    fig.add_trace(go.Scatter3d(
        x=x1, y=x2, z=y_targets,
        mode="markers", name="Data",
        marker=data_3d_marker_style(theme=theme),
        hovertemplate="<b>Data</b><br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>",
        legendgroup="fit", showlegend=True,
    ), row=1, col=2)

    # 2. Surface
    fig.add_trace(go.Surface(
        x=X1g, y=X2g, z=Pg,
        name=(f"Model: P({classes[1]} | x)" if show_class_labels else "Model: p1"),
        **surface_style(theme=theme),
        showlegend=True, legendgroup="fit",
        uid="MODEL_PLANE",
    ), row=1, col=2)

    # 2.5 The decision boundary is z(x)=0, equivalently P(Y=c_1|x)=0.5.
    # With two features it is a line where the probability surface crosses 0.5,
    # not a horizontal plane spanning every feature coordinate.
    boundary_x1 = np.array([], dtype=float)
    boundary_x2 = np.array([], dtype=float)
    if abs(w_disp[1]) >= abs(w_disp[0]) and abs(w_disp[1]) > 1e-12:
        candidate_x1 = np.linspace(x1_min, x1_max, 160)
        candidate_x2 = -(w_disp[0] * candidate_x1 + b_disp) / w_disp[1]
        inside = (candidate_x2 >= x2_min) & (candidate_x2 <= x2_max)
        boundary_x1, boundary_x2 = candidate_x1[inside], candidate_x2[inside]
    elif abs(w_disp[0]) > 1e-12:
        candidate_x2 = np.linspace(x2_min, x2_max, 160)
        candidate_x1 = -(w_disp[1] * candidate_x2 + b_disp) / w_disp[0]
        inside = (candidate_x1 >= x1_min) & (candidate_x1 <= x1_max)
        boundary_x1, boundary_x2 = candidate_x1[inside], candidate_x2[inside]

    fig.add_trace(go.Scatter3d(
        x=boundary_x1,
        y=boundary_x2,
        z=np.full(boundary_x1.shape, 0.5),
        mode="lines",
        name=(
            f"Decision boundary: P({classes[1]} | x) = 0.5"
            if show_class_labels else "Decision boundary: p1 = 0.5"
        ),
        line=dict(color="gray", dash="dash", width=6),
        showlegend=True,
        legendgroup="fit",
        hovertemplate="<b>Decision boundary</b><br>P(class 1 | x) = 0.5<extra></extra>",
        uid="DECISION_BOUNDARY_3D",
    ), row=1, col=2)

    # 3. Prediction point
    pred_color = p.get("loss_line", "#00cc96")
    fig.add_trace(go.Scatter3d(
        x=[x_disp[0]], y=[x_disp[1]], z=[p_hat],
        mode="markers", name="Prediction",
        marker=dict(size=6, color=pred_color),
        hovertemplate="<b>Prediction</b><br>x₁: %{x}<br>x₂: %{y}<br>p̂: %{z}<extra></extra>",
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
    shapes = [block_rect(0.69, 0.98), block_rect(0.29, 0.67), block_rect(0.02, 0.27)]

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
        return [
            title_annot(T1, 0.96), body_annot(v_body, 0.89),
            title_annot(T2, 0.65), body_annot(s_body, 0.61),
            title_annot(T3, 0.25), body_annot(r_body, 0.20),
        ]

    def scene_ann(stage: int):
        if stage == 3:
            return [dict(
                x=x_disp[0], y=x_disp[1], z=p_hat,
                text=rf"$\hat{{p}}={_fmt(p_hat, dec)}$",
                showarrow=True, arrowhead=2, ax=35, ay=-35,
                font=dict(size=14, color="white"),
            )]
        return []

    stage_pred_visible = [False, False, False, True]
    slider_steps = []
    for s in [0, 1, 2, 3]:
        # trace visibility: data (0), model (1), boundary (2), pred (3)
        vis_array = [True, True, True, stage_pred_visible[s]]
        slider_steps.append(dict(
            label=str(s), method="update",
            args=[{"visible": vis_array}, {"annotations": ann_slots(s), "scene.annotations": scene_ann(s)}],
        ))

    def make_btn(label, s):
        vis_array = [True, True, True, stage_pred_visible[s]]
        return dict(label=label, method="update", args=[{"visible": vis_array}, {"annotations": ann_slots(s), "scene.annotations": scene_ann(s)}])

    buttons = [
        make_btn("Input", 1),
        make_btn("Substitution", 2),
        make_btn("Output", 3),
        make_btn("Reset", 0),
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
            zaxis=dict(
                title=(f"P({classes[1]} | x)" if show_class_labels else "p1"),
                range=z_range,
                tickvals=[0.0, 0.5, 1.0],
                ticktext=(
                    [f"0 - {classes[0]}", "0.5", f"1 - {classes[1]}" ]
                    if show_class_labels else ["0", "0.5", "1"]
                ),
            ),
            aspectmode="cube",
            camera=CAMERA,
            annotations=scene_ann(0),
        ),
    )
    return fig

def _explain_log_multiclass_1d(
    X_train, y_train,
    x_disp, w_disp, b_disp,
    p_hat, y_hat, classes, show_class_labels,
    title, dec, grid_points, theme,
    p, text_color, ann_color, btn_bg, btn_border, btn_font_color,
    probability_link,
):
    import plotly.express as px

    x1_train = X_train[:, 0].ravel()
    xq1_disp = float(x_disp[0])
    K = w_disp.shape[0]

    x_min, x_max = float(x1_train.min()), float(x1_train.max())
    pad_x = 0.1 * (x_max - x_min + 1e-9)
    x_min_plot = min(x_min - pad_x, xq1_disp - pad_x)
    x_max_plot = max(x_max + pad_x, xq1_disp + pad_x)
    x_grid = np.linspace(x_min_plot, x_max_plot, int(grid_points))

    z_grid = w_disp @ x_grid.reshape(1, -1) + b_disp.reshape(-1, 1)
    p_curves = multiclass_probabilities(z_grid.T, probability_link).T

    x_tex = rf"x_1 = {_fmt(xq1_disp, dec)}"
    vars_tex = r"$\begin{aligned}" + rf"{x_tex}" + r"\end{aligned}$"

    y_hat_idx = int(np.argmax(p_hat))


    def exp_tex(k):
        return rf"({_fmt(w_disp[k, 0], dec)})({_fmt(xq1_disp, dec)}) + ({_fmt(b_disp[k], dec)})"

    scores = w_disp[:, 0] * xq1_disp + b_disp
    if probability_link == "ovr":
        components = _sigmoid(scores)
        component_symbol = "q"
        component_definition = r"q_k=\sigma(z_k)"
        normalizer_symbol = "Q"
    else:
        components = np.exp(scores - np.max(scores))
        component_symbol = "r"
        component_definition = r"r_k=e^{z_k-z_{\max}}"
        normalizer_symbol = "R"
    normalizer = float(np.sum(components))

    def p_subst(k):
        return (
            rf"z_{{{k + 1}}}={exp_tex(k)},\quad {component_symbol}_{{{k + 1}}}="
            rf"{_fmt(components[k], dec)},\quad\hat{{p}}(Y=c_{{{k + 1}}}\mid x)="
            rf"\frac{{{component_symbol}_{{{k + 1}}}}}{{{normalizer_symbol}}}={_fmt(p_hat[k], dec)}"
        )

    subst_lines = []
    header1 = (
        component_definition
        + rf",\quad {normalizer_symbol}=\sum_{{j=1}}^K{component_symbol}_j={_fmt(normalizer, dec)},\quad "
    )
    header2 = r"z_k(\mathbf{x}) = \theta_{1,k} x_1 + \theta_{0,k}"
    subst_lines.append(header1 + header2 + r" \\[-6pt]")

    y_hat_idx = int(np.argmax(p_hat))
    if K <= 3:
        for k in range(K):
            subst_lines.append(p_subst(k))
    else:
        subst_lines.append(p_subst(0))
        subst_lines.append(r"\vdots")
        if 0 < y_hat_idx < K - 1:
            subst_lines.append(p_subst(y_hat_idx))
            subst_lines.append(r"\vdots")
        subst_lines.append(p_subst(K - 1))

    subst_tex = r"$\begin{gathered}" + r" \\[2pt]".join(subst_lines) + r"\end{gathered}$"

    res_tex = _multiclass_winner_tex(p_hat, y_hat, classes, dec, show_class_labels)

    fig = make_subplots(
        rows=3, cols=2,
        row_heights=[0.11, 0.71, 0.18],
        column_widths=[0.55, 0.45],
        vertical_spacing=0.02, horizontal_spacing=0.05,
        specs=[
            [{"type": "xy"}, {"type": "xy", "rowspan": 3}],
            [{"type": "xy"}, None],
            [{"type": "xy"}, None]
        ]
    )

    # Hide axes for the math boxes: box 1 -> (1,1), box 2 -> (2,1)[idx 3], box 3 -> (3,1)[idx 4]
    for idx in (1, 3, 4):
        fig.update_xaxes(visible=False, row=(1 if idx==1 else 2 if idx==3 else 3), col=1)
        fig.update_yaxes(visible=False, row=(1 if idx==1 else 2 if idx==3 else 3), col=1)

    fig.update_xaxes(title="x₁", range=[x_min_plot, x_max_plot], row=1, col=2)
    fig.update_yaxes(title_text="p(y=k|x)", title_standoff=5, range=[-0.05, 1.05], row=1, col=2)

    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False), row=1, col=2)

    sorted_k = np.argsort(p_hat)[::-1]
    if K > 15:
        selected_classes = sorted(list(sorted_k[:8]) + list(sorted_k[-7:]))
    else:
        selected_classes = list(range(K))

    colors = px.colors.qualitative.Plotly
    for k in selected_classes:
        color = colors[k % len(colors)]
        fig.add_trace(go.Scatter(
            x=x_grid, y=p_curves[k],
            mode="lines",
            name=(f"Class {k} - {classes[k]}" if show_class_labels else f"Class {k}"),
            line=dict(color=color, width=2),
            legendgroup=f"class_{k}", showlegend=True,
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=[xq1_disp], y=[p_hat[k]],
            mode="markers", name=f"Pred {k}",
            marker=dict(color=color, size=10, symbol="circle"),
            legendgroup=f"class_{k}", showlegend=False,
            visible=False,
        ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=[xq1_disp, xq1_disp], y=[-0.05, 1.05],
        mode="lines", name="x_query",
        line=dict(color="gray", width=1, dash="dash"),
        showlegend=False, visible=False,
    ), row=1, col=2)

    def block_rect(idx):
        return dict(
            type="rect", xref=f"x{idx}", yref=f"y{idx}",
            x0=0.01, x1=0.99, y0=0.01, y1=0.99,
            line=dict(width=1),
            fillcolor="rgba(220,220,220,0.10)", layer="below",
        )
    shapes = [block_rect(1), block_rect(3), block_rect(4)]

    def title_annot(idx, tex_title):
        y_pos = 0.98 if idx == 3 else 0.92
        return dict(x=0.5, y=y_pos, xref=f"x{idx}", yref=f"y{idx}", text=rf"$\bf{{{tex_title}}}$", showarrow=False, xanchor="center", yanchor="top", font=dict(size=16, color=text_color))
    def body_annot(idx, tex_body):
        if idx == 1:
            y_pos = 0.60
        elif idx == 3:
            y_pos = 0.91
        else:
            y_pos = 0.75
        font_size = 12 if idx == 3 and K >= 4 else 14
        return dict(x=0.04, y=y_pos, xref=f"x{idx}", yref=f"y{idx}", text=tex_body, showarrow=False, xanchor="left", yanchor="top", align="center", font=dict(size=font_size, color=text_color))

    T1, T2, T3 = r"Variables\ (Input)", r"Substitution", r"Result\ (Output)"

    def ann_slots(stage: int):
        v_body = "" if stage < 1 else vars_tex
        s_body = "" if stage < 2 else subst_tex
        r_body = "" if stage < 3 else res_tex
        ann = [
            title_annot(1, T1), body_annot(1, v_body),
            title_annot(3, T2), body_annot(3, s_body),
            title_annot(4, T3), body_annot(4, r_body),
        ]
        if stage == 3:
            # Place annotations without overlap in data coordinates using bottom-up layout
            sorted_selected = sorted(selected_classes, key=lambda c: p_hat[c], reverse=True)

            text_ys_rev = []
            for k in reversed(sorted_selected):  # lowest prob to highest
                target_y = p_hat[k]
                if target_y < 0.08:
                    target_y = 0.08
                if text_ys_rev and target_y < text_ys_rev[-1] + 0.045:
                    target_y = text_ys_rev[-1] + 0.045
                text_ys_rev.append(target_y)
            text_ys = list(reversed(text_ys_rev))

            x_range = x_max_plot - x_min_plot
            x_offset = 0.05 * x_range

            for i, k in enumerate(sorted_selected):
                ann.append(dict(
                    x=xq1_disp, y=p_hat[k], xref="x2", yref="y2",
                    ax=xq1_disp + x_offset, ay=text_ys[i], axref="x2", ayref="y2",
                    text=rf"$\hat{{p}}_{{{k}}}={_fmt(p_hat[k], dec)}$",
                    showarrow=True, arrowhead=2,
                    font=dict(size=12, color="white"),
                    bgcolor="rgba(0,0,0,0.3)", borderpad=2,
                ))
        return ann

    stage_pred_visible = [False, False, False, True]
    slider_steps = []

    for s in [0, 1, 2, 3]:
        vis_array = [False]
        show_pred = stage_pred_visible[s]
        for _ in selected_classes:
            vis_array.append(True)
            vis_array.append(show_pred)
        vis_array.append(show_pred)
        slider_steps.append(dict(
            label=str(s), method="update",
            args=[{"visible": vis_array}, {"annotations": ann_slots(s)}],
        ))

    def make_btn(label, s):
        vis_array = [False]
        show_pred = stage_pred_visible[s]
        for _ in selected_classes:
            vis_array.append(True)
            vis_array.append(show_pred)
        vis_array.append(show_pred)
        return dict(label=label, method="update", args=[{"visible": vis_array}, {"annotations": ann_slots(s)}])

    buttons = [
        make_btn("Input", 1),
        make_btn("Substitution", 2),
        make_btn("Output", 3),
        make_btn("Reset", 0),
    ]

    layout_kwargs = get_base_layout(title=title, margin_t=110, theme=theme)
    layout_kwargs["margin"] = dict(t=110, r=50, l=60, b=40)

    fig.update_layout(
        **layout_kwargs, shapes=shapes, annotations=ann_slots(0),
        updatemenus=_custom_updatemenus(buttons, btn_bg, btn_border, btn_font_color),
        legend=dict(yanchor="middle", y=0.50, xanchor="left", x=1.02),
    )
    return fig


def _explain_log_multiclass_nd(
    d, K, x_disp, w_disp, b_disp, p_hat, y_hat, classes, show_class_labels,
    title, dec, theme, p, text_color, ann_color, btn_bg, btn_border, btn_font_color,
    probability_link,
):
    link_name = r"\operatorname{OvR}_{\sigma}" if probability_link == "ovr" else r"\operatorname{softmax}"
    model_formula_tex = rf"$\mathbf{{z}}=\Theta^\top\mathbf{{x}}+\boldsymbol{{\theta}}_0,\quad \hat{{\mathbf{{p}}}}={link_name}(\mathbf{{z}})$"

    x_rows, x_cols = 15, 1
    x_items = [rf"{_fmt(x_disp[j], dec)}" for j in range(d)]
    x_mat_inner = r" \\ ".join(_matrix_compact(x_items, x_rows, x_cols, 7, 7))
    x_mat_tex = rf"$\mathbf{{x}}=\begin{{bmatrix}} {x_mat_inner} \end{{bmatrix}}$"
    x_dim_tex = rf"$\mathbf{{x}}\in\mathbb{{R}}^{{{d}}}$"

    def exp_tex(k):
        return rf"({_fmt(w_disp[k, 0], dec)})({_fmt(x_disp[0], dec)}) + \dots + ({_fmt(b_disp[k], dec)})"

    scores = w_disp @ x_disp + b_disp
    if probability_link == "ovr":
        components = _sigmoid(scores)
        component_symbol = "q"
        component_definition = r"q_k=\sigma(z_k)"
        normalizer_symbol = "Q"
    else:
        components = np.exp(scores - np.max(scores))
        component_symbol = "r"
        component_definition = r"r_k=e^{z_k-z_{\max}}"
        normalizer_symbol = "R"
    normalizer = float(np.sum(components))

    def p_subst(k):
        return (
            rf"z_{{{k + 1}}}={exp_tex(k)},\quad {component_symbol}_{{{k + 1}}}="
            rf"{_fmt(components[k], dec)},\quad\hat{{p}}(Y=c_{{{k + 1}}}\mid\mathbf{{x}})="
            rf"\frac{{{component_symbol}_{{{k + 1}}}}}{{{normalizer_symbol}}}={_fmt(p_hat[k], dec)}"
        )

    subst_lines = []
    header1 = (
        component_definition
        + rf",\quad {normalizer_symbol}=\sum_{{j=1}}^K{component_symbol}_j={_fmt(normalizer, dec)},\quad "
    )
    header2 = r"z_k(\mathbf{x}) = \sum_{j=1}^{D} \theta_{j,k} x_j + \theta_{0,k}"
    subst_lines.append(header1 + header2 + r" \\[8pt]")

    y_hat_idx = int(np.argmax(p_hat))
    if K <= 3:
        for k in range(K):
            subst_lines.append(p_subst(k))
    else:
        subst_lines.append(p_subst(0))
        subst_lines.append(r"\vdots")
        if 0 < y_hat_idx < K - 1:
            subst_lines.append(p_subst(y_hat_idx))
            subst_lines.append(r"\vdots")
        subst_lines.append(p_subst(K - 1))

    subst_tex = r"$\begin{gathered}" + r" \\[-2pt]".join(subst_lines) + r"\end{gathered}$"

    res_tex = _multiclass_winner_tex(p_hat, y_hat, classes, dec, show_class_labels)

    y_dim_tex = rf"$\hat{{\mathbf{{p}}}}\in\Delta^{{{K - 1}}}$"

    fig = make_subplots(
        rows=1, cols=3, column_widths=[0.28, 0.50, 0.22],
        horizontal_spacing=0.01, specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
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
    def body_annot(col, tex_body, y, size=15, x=0.06):
        return dict(x=x, y=y + DY, xref=f"x{col}", yref=f"y{col}", text=tex_body, showarrow=False, xanchor="left", yanchor="top", align="left", font=dict(size=size, color=text_color))
    def block_rect(col, y0, y1):
        return dict(type="rect", xref=f"x{col}", yref=f"y{col}", x0=-0.02, x1=1.02, y0=0.00, y1=1.00 + DY, line=dict(width=1), fillcolor="rgba(220,220,220,0.10)", layer="below")

    shapes = [block_rect(1, 0.02, 0.98), block_rect(2, 0.02, 0.98), block_rect(3, 0.02, 0.98)]
    T1, T2, T3 = r"Variables\ (Input)", r"Substitution", r"Result\ (Output)"

    def ann_slots(stage: int):
        ann = [paper_top_center(model_formula_tex, y=1.02, size=18)]
        ann.append(top_center_annot(1, x_dim_tex, 1.02))
        ann.append(top_center_annot(3, y_dim_tex, 1.02))

        ann.append(title_annot(1, T1, 0.96))
        ann.append(body_annot(1, "" if stage < 1 else x_mat_tex, 0.89, size=14, x=0.03))

        ann.append(title_annot(2, T2, 0.96))
        ann.append(dict(x=0.5, y=0.90 + DY, xref="x2", yref="y2", text="" if stage < 2 else subst_tex, showarrow=False, xanchor="center", yanchor="top", align="center", font=dict(size=13, color=text_color)))

        ann.append(title_annot(3, T3, 0.96))
        if stage < 3:
            ann.append(dict(x=0.05, y=0.85 + DY, xref="x3", yref="y3", text="", showarrow=False, xanchor="left", yanchor="top", font=dict(size=16, color=text_color)))
        else:
            ann.append(dict(x=0.05, y=0.85 + DY, xref="x3", yref="y3", text=res_tex, showarrow=False, xanchor="left", yanchor="top", font=dict(size=16, color=text_color)))

        return ann

    buttons = [
        dict(label="Input", method="update", args=[{"visible": [True, True, True]}, {"annotations": ann_slots(1)}]),
        dict(label="Substitution", method="update", args=[{"visible": [True, True, True]}, {"annotations": ann_slots(2)}]),
        dict(label="Output", method="update", args=[{"visible": [True, True, True]}, {"annotations": ann_slots(3)}]),
        dict(label="Reset", method="update", args=[{"visible": [True, True, True]}, {"annotations": ann_slots(0)}]),
    ]

    layout_kwargs = get_base_layout(title=title, margin_t=110, theme=theme)
    layout_kwargs["margin"] = dict(t=110, r=50, l=60, b=40)

    up_kwargs = dict(x=0.08, xanchor="left") if d >= 10 else {}

    fig.update_layout(
        **layout_kwargs, shapes=shapes, annotations=ann_slots(0),
        updatemenus=_custom_updatemenus(buttons, btn_bg, btn_border, btn_font_color, **up_kwargs),
    )
    return fig

def explain_logistic_prediction(
    trained_estimator,
    X_train, y_train,
    *,
    x_query=None,
    p_hat=None,
    y_hat=None,
    title=None,
    dec=4,
    grid_points=250,
    show_class_labels=False,
    display_space="original",
    multiclass_link="auto",
    prediction_source="model",
    validation_rtol=1e-7,
    validation_atol=1e-9,
    theme=None,
):
    """Create a link-aware step-by-step logistic prediction visualization.

    For binary estimators, probability indices 0 and 1 follow ``classes_[0]``
    and ``classes_[1]`` exactly. The scalar ``p_hat`` and sigmoid surface mean
    ``p_1``; the complementary class probability is ``1 - p_hat``.
    String targets are converted to 0/1 only in plot coordinates, never in the
    fitted estimator or reported class identity.

    ``show_class_labels=False`` keeps the mathematical view indexed and omits
    semantic labels from equations, axes, and legends. Setting it to ``True``
    appends fitted labels while retaining the class index. Labels and fitted
    order are always preserved in ``layout.meta``.

    ``multiclass_link="auto"`` uses the estimator's exact probability semantics;
    explicit ``"softmax"`` and ``"ovr"`` values support custom estimators.
    Supplied probabilities and labels are verified against the estimator unless
    ``prediction_source="provided"`` explicitly requests a counterfactual.
    """
    _validate_prediction_options(
        dec=dec,
        grid_points=grid_points,
        validation_rtol=validation_rtol,
        validation_atol=validation_atol,
    )
    if not isinstance(show_class_labels, bool):
        raise TypeError("show_class_labels must be a boolean value.")
    if not isinstance(multiclass_link, str) or multiclass_link not in {"auto", "softmax", "ovr"}:
        raise ValueError("multiclass_link must be 'auto', 'softmax', or 'ovr'.")
    def _extract_logistic_multiclass_theta(est):
        from ..linear.prediction import _get_last_estimator

        last = _get_last_estimator(est)
        if not (hasattr(last, "coef_") and hasattr(last, "intercept_")):
            raise ValueError("Estimator must expose coef_ and intercept_.")
        w = np.asarray(last.coef_, dtype=float)
        b = np.asarray(last.intercept_, dtype=float).ravel()
        return w, b

    def _theta_to_original_multiclass(w_s, b_s, scaler):
        from ..linear.prediction import _safe_get_scale
        w_s = np.asarray(w_s, dtype=float)
        b_s = np.asarray(b_s, dtype=float).ravel()
        if scaler is None:
            return w_s.copy(), b_s.copy()

        mu, scale, with_mean, with_std = _safe_get_scale(scaler)
        dloc = w_s.shape[1]

        if (not with_std) or (scale is None):
            scale = np.ones(dloc, dtype=float)
        else:
            scale = np.asarray(scale, dtype=float).ravel()

        if (not with_mean) or (mu is None):
            mu = np.zeros(dloc, dtype=float)
        else:
            mu = np.asarray(mu, dtype=float).ravel()

        w_o = w_s / (scale + 1e-12)
        b_o = b_s - np.sum(w_s * mu / (scale + 1e-12), axis=1)
        return w_o, b_o

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_train.ndim != 2 or X_train.shape[0] == 0 or X_train.shape[1] == 0:
        raise ValueError("X_train must be a non-empty two-dimensional feature matrix.")
    if not np.all(np.isfinite(X_train)):
        raise ValueError("X_train must contain only finite values.")
    if y_train.ravel().size != X_train.shape[0]:
        raise ValueError("X_train and y_train must contain the same number of samples.")
    y_train = y_train.ravel()

    d = X_train.shape[1]

    if x_query is None:
        raise ValueError("Must provide x_query.")

    x_query_array = np.asarray(x_query, dtype=float)
    if x_query_array.ndim == 2 and x_query_array.shape[0] != 1:
        raise ValueError(f"x_query must describe exactly one sample; got shape {x_query_array.shape}.")
    x_query = x_query_array.ravel()
    if x_query.size != d:
        raise ValueError(f"x_query must have {d} elements.")
    if not np.all(np.isfinite(x_query)):
        raise ValueError("x_query must contain only finite values.")
    if not isinstance(display_space, str) or display_space not in {"original", "scaled"}:
        raise ValueError("display_space must be 'original' or 'scaled'.")
    _validate_prediction_source(prediction_source)

    scaler = _find_standard_scaler(trained_estimator)
    adapter = SklearnAdapter(trained_estimator)
    classes = adapter.classes if adapter.classes is not None else np.unique(y_train)
    is_multiclass = len(classes) > 2
    probability_link = (
        adapter.resolve_multiclass_link(X_train, multiclass_link) if is_multiclass else "sigmoid"
    )

    if is_multiclass:
        w_s, b_s = _extract_logistic_multiclass_theta(trained_estimator)
    else:
        w_s, b_s = _extract_linear_theta(trained_estimator)

    x_query_scaled = _to_scaled_x(x_query, scaler)
    model_probabilities = adapter.predict_proba(x_query.reshape(1, -1), classes).ravel()
    model_p_hat = model_probabilities if is_multiclass else float(model_probabilities[1])
    model_y_hat = np.asarray(trained_estimator.predict(x_query.reshape(1, -1))).ravel()[0]

    if p_hat is None:
        p_hat = model_p_hat.copy() if is_multiclass else model_p_hat
    elif is_multiclass:
        p_hat = np.asarray(p_hat, dtype=float).ravel()
        if p_hat.size != len(classes):
            raise ValueError(f"p_hat must contain one probability for each of the {len(classes)} classes.")
        if not np.all(np.isfinite(p_hat)) or np.any(p_hat < 0) or not np.isclose(np.sum(p_hat), 1.0, atol=1e-8):
            raise ValueError("p_hat must be a finite non-negative probability vector that sums to 1.")
        if prediction_source == "model" and not np.allclose(
            p_hat, model_p_hat, rtol=validation_rtol, atol=validation_atol
        ):
            raise ValueError(
                "Provided p_hat does not match estimator.predict_proba. Use prediction_source='provided' "
                "only for an intentional counterfactual."
            )
    else:
        p_hat_array = np.asarray(p_hat, dtype=float).ravel()
        if p_hat_array.size != 1:
            raise ValueError("Binary p_hat must be a single scalar probability.")
        p_hat = float(p_hat_array[0])
        if not np.isfinite(p_hat) or not 0 <= p_hat <= 1:
            raise ValueError("Binary p_hat must be a finite probability in [0, 1].")
        if prediction_source == "model" and not np.isclose(
            p_hat, model_p_hat, rtol=validation_rtol, atol=validation_atol
        ):
            raise ValueError(
                f"Provided p_hat={p_hat!r} does not match estimator.predict_proba value {model_p_hat!r}. "
                "Use prediction_source='provided' only for an intentional counterfactual."
            )

    derived_y_hat = classes[np.argmax(p_hat)] if is_multiclass else classes[1] if p_hat >= 0.5 else classes[0]
    if y_hat is None:
        y_hat = model_y_hat if prediction_source == "model" else derived_y_hat
    else:
        y_hat_array = np.asarray(y_hat).ravel()
        if y_hat_array.size != 1:
            raise ValueError("y_hat must be a single fitted class label.")
        y_hat = y_hat_array[0]
        if not np.any(classes == y_hat):
            raise ValueError(f"y_hat must be one of the fitted classes: {classes.tolist()}.")
        if prediction_source == "model" and y_hat != model_y_hat:
            raise ValueError(
                f"Provided y_hat={y_hat!r} does not match estimator.predict value {model_y_hat!r}. "
                "Use prediction_source='provided' only for an intentional counterfactual."
            )

    outside_features = _outside_training_range(X_train, x_query)

    if display_space == "scaled":
        X_disp = np.array([_to_scaled_x(x, scaler) for x in X_train])
        w_disp = w_s
        b_disp = b_s
        x_disp = x_query_scaled
        if title is None:
            title = f"Logistic Regression Prediction ({'Multiclass' if is_multiclass else 'Binary'}) - Scaled Space"
    else:
        X_disp = X_train.copy()
        if is_multiclass:
            w_disp, b_disp = _theta_to_original_multiclass(w_s, b_s, scaler)
        else:
            w_disp, b_disp = _theta_to_original(w_s, b_s, scaler)
        x_disp = x_query
        if title is None:
            title = f"Logistic Regression Prediction ({'Multiclass' if is_multiclass else 'Binary'}) - Original Space"

    scope = (
        f"Extrapolation outside training range in feature(s): {', '.join(str(index + 1) for index in outside_features)}"
        if outside_features
        else "Query lies within every observed training feature range"
    )
    source_label = "model-verified" if prediction_source == "model" else "user-provided counterfactual"
    title = f'{title}<br><sup><span style="color:#B8C1CC">Prediction source: {source_label} · {scope}</span></sup>'

    p = _resolve(theme)
    text_color = p.get("text", "#333333")
    ann_color = p.get("prediction_text", "#e0245e")
    btn_bg = p.get("button_bg", "#f4f4f4")
    btn_border = p.get("button_border", "#cccccc")
    btn_font_color = p.get("button_text", "#333333")

    if is_multiclass:
        K = len(classes)
        if d == 1:
            fig = _explain_log_multiclass_1d(
                X_disp, y_train, x_disp, w_disp, b_disp, p_hat, y_hat,
                classes, show_class_labels,
                title, dec, grid_points, theme,
                p, text_color, ann_color, btn_bg, btn_border, btn_font_color,
                probability_link,
            )
        else:
            fig = _explain_log_multiclass_nd(
                d, K, x_disp, w_disp, b_disp, p_hat, y_hat,
                classes, show_class_labels,
                title, dec, theme, p, text_color, ann_color, btn_bg, btn_border, btn_font_color,
                probability_link,
            )
    else:
        if d == 1:
            fig = _explain_log_1d(
                X_disp, y_train, x_disp, w_disp, b_disp, p_hat, y_hat,
                classes, show_class_labels,
                title, dec, grid_points, theme,
                p, text_color, ann_color, btn_bg, btn_border, btn_font_color
            )
        else:
            grid_2d_points = 40
            fig = _explain_log_2d(
                X_disp, y_train, x_disp, w_disp, b_disp, p_hat, y_hat,
                classes, show_class_labels,
                title, dec, grid_2d_points, theme,
                p, text_color, ann_color, btn_bg, btn_border, btn_font_color
            )

    fig.update_layout(
        meta={
            "mlektic_prediction": {
                "source": prediction_source,
                "show_class_labels": show_class_labels,
                "classes": [value.item() if hasattr(value, "item") else value for value in classes],
                "probability_target_class_index": None if is_multiclass else 1,
                "probability_target_class": (
                    None
                    if is_multiclass
                    else classes[1].item() if hasattr(classes[1], "item") else classes[1]
                ),
                "decision_threshold": None if is_multiclass else 0.5,
                "model_class_probabilities": model_probabilities.tolist(),
                "displayed_class_probabilities": (
                    np.asarray(p_hat).tolist()
                    if is_multiclass
                    else [1.0 - float(p_hat), float(p_hat)]
                ),
                "model_probability": np.asarray(model_p_hat).tolist(),
                "displayed_probability": np.asarray(p_hat).tolist(),
                "model_class": model_y_hat.item() if hasattr(model_y_hat, "item") else model_y_hat,
                "displayed_class": y_hat.item() if hasattr(y_hat, "item") else y_hat,
                "outside_training_feature_indices": outside_features,
                "probability_link": probability_link,
            }
        }
    )
    return fig

__all__ = ["explain_logistic_prediction"]
