"""Spacing of the fitted-model derivation and placement of explainer inputs."""

import numpy as np
import plotly.graph_objects as go
import pytest
from sklearn.linear_model import LogisticRegression

from mlektic import explain_logistic_prediction
from mlektic.mathematics import _panel_line_height, attach_math_contract
from mlektic.visualization.logistic.prediction import _input_vector_fits_centered

BS = chr(92)
ROW = BS + BS + "[4pt]"
BRACE = BS + "underbrace{(0.3725)(1.0650)}_{x_1}"


def _aligned(*rows):
    return "$" + BS + "begin{aligned}" + ROW.join(rows) + BS + "end{aligned}$"


def test_rows_with_underbraces_count_as_tall_rows():
    plain = _aligned("z&=a", "&=b", "&=c", "&=d")
    braced = _aligned("z&=a", "&=" + BRACE, "&" + BS + "quad+" + BRACE, "&=d")
    assert _panel_line_height(plain, 4) == pytest.approx(4 * 28 + 3 * 4)
    # Two labelled rows are each as tall as a row plus its labels.
    assert _panel_line_height(braced, 4) == pytest.approx(2 * 28 + 2 * 46 + 3 * 4)
    assert _panel_line_height(braced, 4) > 52 + 24 * 3  # the previous estimate ended against the next line


def test_the_line_after_a_derivation_starts_below_its_last_row():
    derivation = _aligned("z&=a", "&=" + BRACE, "&" + BS + "quad+" + BRACE, "&=d")
    loss = "$" + BS + "underbrace{" + BS + "text{Log-loss}=0.4669}_{data}$"
    figure = go.Figure(layout={"height": 700, "margin": {"t": 100, "b": 300}})
    contract = {"detail": "complete", "family": "binary_logistic_regression", "panel": [
        "heading", {"text": derivation, "visual_lines": 4}, loss, "Regularization",
    ]}
    attach_math_contract(figure, contract)
    plot_height = 700 - 100 - 300
    centers = [-float(a.y) * plot_height for a in figure.layout.annotations]
    derivation_bottom = centers[1] + _panel_line_height(derivation, 4) / 2
    loss_top = centers[2] - _panel_line_height(loss, 1) / 2
    loss_bottom = centers[2] + _panel_line_height(loss, 1) / 2
    regularization_top = centers[3] - _panel_line_height("Regularization", 1) / 2
    assert derivation_bottom < loss_top
    # Log-loss sits midway: the same gap above and below it.
    assert loss_top - derivation_bottom == pytest.approx(regularization_top - loss_bottom)


def test_input_vector_is_centred_when_it_fits():
    assert _input_vector_fits_centered(["-1.9738", "-1.0125"], column_width=0.28, right_margin=50)
    assert not _input_vector_fits_centered(["-123456789.123456789"], column_width=0.28, right_margin=50)


def test_multiclass_explainer_centres_a_short_input_vector():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(90, 2))
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0)
    figure = explain_logistic_prediction(LogisticRegression().fit(X, y), X, y, x_query=X[:1])
    button = next(b for menu in figure.layout.updatemenus for b in menu.buttons if b.label == "Input")
    vector = next(a for a in button.args[1]["annotations"] if "bmatrix" in str(a.get("text", "")))
    assert vector["xanchor"] == "center" and vector["x"] == 0.5
