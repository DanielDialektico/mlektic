"""Legends move off the data and off the labels they would hide."""

import numpy as np
import plotly.graph_objects as go
import pytest
from sklearn.linear_model import LogisticRegression

from mlektic import visualize_logistic
from mlektic.visualization.design import get_theme_tokens
from mlektic.visualization.legend_layout import place_legends
from mlektic.visualization.skins import get_skin, is_skin


def _figure(xs, ys, **legend):
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="markers", name="Data"))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Model"))
    fig.update_layout(width=900, height=600, margin=dict(l=60, r=40, t=60, b=60), legend=legend)
    fig.update_xaxes(range=[0, 10])
    fig.update_yaxes(range=[0, 10])
    return fig


def test_legend_leaves_a_crowded_corner():
    x = np.linspace(6, 10, 30)
    fig = _figure(x, np.full_like(x, 0.5), x=0.99, y=0.01, xanchor="right", yanchor="bottom")
    report = place_legends(fig)
    assert report["legend"]["previously_covered_marks"] > 0
    assert report["legend"]["covered_marks"] == 0
    assert report["legend"]["position"] != "bottom right"


def test_legend_avoids_annotations():
    x = np.linspace(0, 10, 40)
    fig = _figure(x, 10 - x, x=0.99, y=0.01, xanchor="right", yanchor="bottom")
    fig.add_annotation(
        x=0.99,
        y=0.99,
        xref="paper",
        yref="paper",
        text="<b>Log-loss</b><br>0.6931",
        xanchor="right",
        yanchor="top",
        showarrow=False,
    )
    report = place_legends(fig)
    assert report["legend"]["covered_labels"] == 0
    assert report["legend"]["position"] != "top right"


def test_legend_outside_the_plot_is_left_alone():
    x = np.linspace(0, 10, 40)
    fig = _figure(x, x)  # Plotly's default legend: x=1.02, left-anchored, beside the plot
    assert place_legends(fig) == {}
    assert fig.layout.legend.x is None


def test_threshold_metrics_figure_keeps_legends_off_cards_and_curves():
    X = np.linspace(-3.2, 3.2, 90).reshape(-1, 1)
    y = (1.5 * X[:, 0] - 0.2 > 0).astype(int)
    fig = visualize_logistic(LogisticRegression().fit(X, y), X, y, steps=6, threshold=0.8, show_loss=True)
    placement = fig.layout.meta["mlektic_visual"]["legend_placement"]
    assert placement, "the in-plot legends were not examined"
    for report in placement.values():
        assert report["covered_labels"] == 0
        assert report["covered_marks"] == 0


@pytest.mark.parametrize("theme", ["dusk", "editorial", "davinci"])
def test_threshold_line_reads_on_every_paper(theme):
    X = np.linspace(-3.2, 3.2, 90).reshape(-1, 1)
    y = (X[:, 0] > 0).astype(int)
    fig = visualize_logistic(LogisticRegression().fit(X, y), X, y, steps=4, threshold=0.8, theme=theme)
    line = next(shape for shape in fig.layout.shapes if shape.y0 == 0.8)
    expected = get_skin(theme).palette.text if is_skin(theme) else get_theme_tokens(theme).text
    assert line.line.color == expected
    assert line.opacity == pytest.approx(0.8)


@pytest.mark.parametrize("show_loss", [True, False])
def test_multiclass_one_input_legend_sits_beside_the_probability_panel(show_loss):
    X = np.linspace(-3.0, 3.0, 90).reshape(-1, 1)
    y = np.digitize(X[:, 0], [-1.0, 1.0])
    fig = visualize_logistic(
        LogisticRegression(max_iter=500).fit(X, y), X, y, steps=5, show_loss=show_loss, show_class_labels=True
    )
    legend = fig.layout.legend
    # Outside the panel, to its right, level with its top: the curves sweep the whole panel.
    assert legend.x > 1.0 and legend.xanchor == "left" and legend.yanchor == "top"
    assert legend.y == pytest.approx(fig.layout.yaxis2.domain[1])
    names = [trace.name for trace in fig.data if str(trace.name).startswith("p(class")]
    assert names and all("<br>" in name for name in names)  # labels on their own line keep it narrow
