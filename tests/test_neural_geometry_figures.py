"""Decision surface, representation, training step, gradient flow, and Hinton views."""

import numpy as np
import pytest
import torch

from mlektic import (
    TorchTrainingRecorder,
    export_figure,
    visualize_nn,
    visualize_nn_decision,
    visualize_nn_gradient_flow,
    visualize_nn_graph,
    visualize_nn_representation,
    visualize_nn_training,
    visualize_nn_training_step,
    visualize_nn_weights,
)
from mlektic.neural.introspection import run_torch_forward
from mlektic.visualization.neural.math_format import parameter_snapshot
from mlektic.visualization.skins import get_skin
from mlektic.visualization.skins.neural import color_mapper


def _blobs(seed=3, n=80):
    rng = np.random.default_rng(seed)
    centers = np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]], dtype=np.float32)
    index = rng.integers(0, 4, n)
    X = (centers[index] + rng.normal(0, 0.35, (n, 2))).astype(np.float32)
    y = np.array([0, 0, 1, 1])[index].astype(np.float32)[:, None]
    return torch.tensor(X), torch.tensor(y)


def _record(model, X, y, criterion, *, steps=40, every=5, lr=0.05):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    recorder = TorchTrainingRecorder(model, optimizer=optimizer, loss_fn=criterion)
    for step in range(steps):
        optimizer.zero_grad()
        prediction = model(X)
        loss = criterion(prediction, y)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % every == 0:
            recorder.record(
                step + 1, loss=loss.detach(), predictions=prediction.detach(), targets=y, task="classification"
            )
    recorder.close()
    return recorder.to_history()


@pytest.fixture
def xor_network():
    torch.manual_seed(11)
    X, y = _blobs()
    model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1), torch.nn.Sigmoid())
    criterion = torch.nn.BCELoss()
    return model, X, y, criterion, _record(model, X, y, criterion)


def _dequantize(values, low, high):
    return low + np.asarray(values, dtype=float) / 255.0 * (high - low)


def test_decision_field_is_the_exact_network_output(xor_network):
    model, X, y, _, history = xor_network
    figure = visualize_nn_decision(model, X, history, targets=y, grid_size=20, unit_grid_size=12, max_frames=4)
    meta = figure.layout.meta["mlektic_neural_decision"]

    assert meta["output_semantics"] == "probability"
    assert len(figure.frames) == 4
    assert meta["recorded_steps"][-1] == int(history["steps"][-1])
    field = figure.frames[-1].data[0]
    axis_x, axis_y = np.asarray(figure.data[0].x), np.asarray(figure.data[0].y)
    grid_x, grid_y = np.meshgrid(axis_x, axis_y)
    output, _ = run_torch_forward(
        model, np.c_[grid_x.ravel(), grid_y.ravel()], parameter_snapshot(history, len(history["steps"]) - 1)
    )
    expected = output.detach().numpy()[:, 0].reshape(20, 20)
    assert _dequantize(field.z, 0.0, 1.0) == pytest.approx(expected, abs=1 / 255)
    # Frames are plain integer lists: no base64 payload a renderer could leave undecoded.
    assert isinstance(field.z, (list, tuple)) and isinstance(field.z[0][0], int)


def test_decision_fold_lines_are_exact_zero_pre_activation_lines(xor_network):
    model, X, y, _, history = xor_network
    figure = visualize_nn_decision(model, X, history, targets=y, grid_size=16, unit_grid_size=12, max_frames=3)
    meta = figure.layout.meta["mlektic_neural_decision"]
    assert meta["fold_lines_exact"] is True
    parameters = parameter_snapshot(history, len(history["steps"]) - 1)
    weight, bias = parameters["0.weight"], parameters["0.bias"]
    folds = [trace for trace in figure.data if str(trace.uid).startswith("NN_DECISION_FOLD_")]
    assert len(folds) == len(meta["units_shown"])
    final_frame = figure.frames[-1]
    by_uid = dict(zip(final_frame.traces, final_frame.data))
    for trace_index, trace in enumerate(figure.data):
        if not str(trace.uid).startswith("NN_DECISION_FOLD_"):
            continue
        unit = int(str(trace.uid).rsplit("_", 1)[1]) - 1
        line = by_uid[trace_index]
        for x1, x2 in zip(line.x, line.y):
            assert weight[unit] @ np.array([x1, x2]) + bias[unit] == pytest.approx(0.0, abs=1e-6)


def test_decision_supports_logits_multiclass_and_regression():
    X, y = _blobs()
    torch.manual_seed(0)
    logits = torch.nn.Sequential(
        torch.nn.Linear(2, 5), torch.nn.ReLU(), torch.nn.Linear(5, 5), torch.nn.ReLU(), torch.nn.Linear(5, 1)
    )
    history = _record(logits, X, y, torch.nn.BCEWithLogitsLoss(), steps=10)
    figure = visualize_nn_decision(logits, X, history, targets=y, grid_size=16, unit_grid_size=12)
    assert figure.layout.meta["mlektic_neural_decision"]["output_semantics"] == "logit"
    assert figure.layout.meta["mlektic_neural_decision"]["fold_lines_exact"] is False

    classes = (X[:, 0] > 0).long() + (X[:, 1] > 0).long()
    multiclass = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 3))
    history = _record(multiclass, X, classes, torch.nn.CrossEntropyLoss(), steps=10)
    figure = visualize_nn_decision(multiclass, X, history, targets=classes, grid_size=16, unit_grid_size=12)
    assert figure.layout.meta["mlektic_neural_decision"]["output_semantics"] == "multiclass"
    assert any(trace.uid == "NN_DECISION_CLASS_2" for trace in figure.data)

    values = (X[:, :1] * X[:, 1:]).float()
    regression = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
    history = _record(regression, X, values, torch.nn.MSELoss(), steps=10)
    figure = visualize_nn_decision(regression, X, history, targets=values, grid_size=16, unit_grid_size=12)
    assert figure.layout.meta["mlektic_neural_decision"]["output_semantics"] == "regression"
    assert not any(trace.uid == "NN_DECISION_BOUNDARY" for trace in figure.data)


def test_decision_rejects_non_dense_models_and_bad_features(xor_network):
    model, X, y, _, history = xor_network
    with pytest.raises(ValueError, match="features"):
        visualize_nn_decision(model, X, history, features=(0, 0))
    conv = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.BatchNorm1d(4), torch.nn.Linear(4, 1))
    with pytest.raises(ValueError, match="dense chain"):
        visualize_nn_decision(conv, X, history)


def test_representation_stages_and_exact_readout_boundary(xor_network):
    model, X, y, _, history = xor_network
    figure = visualize_nn_representation(model, X, history, targets=y)
    meta = figure.layout.meta["mlektic_neural_representation"]

    assert meta["stages"] == ["x", "z⁽¹⁾", "a⁽¹⁾", "ŷ"]
    assert "isometric" in meta["projections"][1]
    assert "readout direction" in meta["projections"][2]
    assert len(figure.frames) == 4
    # Plotly transitions tween positions, so stages keep point counts and skip redraws.
    assert figure.layout.updatemenus[0].buttons[0].args[1]["frame"]["redraw"] is False
    boundary_index = next(i for i, trace in enumerate(figure.data) if trace.uid == "NN_REPRESENTATION_BOUNDARY")
    separator_index = next(i for i, trace in enumerate(figure.data) if trace.uid == "NN_REPRESENTATION_SEPARATOR")
    stage = figure.frames[2]
    boundary_x = np.asarray(stage.data[boundary_index].x, dtype=float)
    separator_x = float(stage.data[separator_index].x[0])
    # Points of the input-space boundary land on the vertical readout boundary.
    assert boundary_x == pytest.approx(np.full_like(boundary_x, separator_x), abs=0.02)
    assert [len(f.data[boundary_index].x) for f in figure.frames] == [len(figure.frames[0].data[boundary_index].x)] * 4


def test_training_step_values_follow_backpropagation(xor_network):
    model, X, y, criterion, history = xor_network
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    figure = visualize_nn_training_step(model, X[0], y[0], criterion, history, checkpoint=1, learning_rate=0.05)
    meta = figure.layout.meta["mlektic_neural_training_step"]

    for name, value in model.named_parameters():
        assert torch.equal(value, before[name])
    assert meta["loss_after"] < meta["loss_before"]
    assert "not the recorded optimizer" in meta["update_rule"]
    names = [frame.name for frame in figure.frames]
    assert names[0] == "start" and names[-1] == "update" and "loss" in names and "gradients" in names

    parameters = parameter_snapshot(history, 1)
    output, records = run_torch_forward(model, X[:1], parameters)
    y_hat = float(output.detach()[0, 0])
    delta_output = y_hat - float(y[0, 0])  # sigmoid + log-loss: δ⁽ᴸ⁾ = ŷ − y
    node_index = next(i for i, trace in enumerate(figure.data) if trace.uid == "NN_STEP_NODES")
    backward = figure.frames[names.index("backward_output")]
    nodes = dict(zip(backward.traces, backward.data))[node_index]
    assert float(nodes.text[-1]) == pytest.approx(delta_output, abs=1e-3)
    gradient_frame = figure.frames[names.index("gradients")]
    assert "δ<sub>j</sub>a<sub>i</sub>" in gradient_frame.data[-1].text[0]


def test_training_step_needs_a_loss(xor_network):
    model, X, y, _, history = xor_network
    with pytest.raises(ValueError, match="loss_fn"):
        visualize_nn_training_step(model, X[0], y[0], None, history)


def test_gradient_flow_is_static_and_compares_runs(xor_network):
    _, X, y, _, history = xor_network
    torch.manual_seed(5)
    deep = torch.nn.Sequential(
        *[layer for _ in range(5) for layer in (torch.nn.Linear(2 if _ == 0 else 8, 8), torch.nn.Sigmoid())],
        torch.nn.Linear(8, 1),
        torch.nn.Sigmoid(),
    )
    deep_history = _record(deep, X, y, torch.nn.BCELoss(), steps=10, every=2)
    figure = visualize_nn_gradient_flow({"shallow": history, "deep sigmoid": deep_history})
    meta = figure.layout.meta["mlektic_neural_gradient_flow"]

    assert not figure.frames
    assert meta["runs"] == ["shallow", "deep sigmoid"]
    heatmaps = [trace for trace in figure.data if trace.type == "heatmap"]
    assert len(heatmaps) == 2 and len(heatmaps[1].y) == 6
    assert heatmaps[0].zmin == heatmaps[1].zmin and heatmaps[0].zmax == heatmaps[1].zmax
    assert meta["first_to_last_mean_log10_ratio"]["deep sigmoid"] < meta["first_to_last_mean_log10_ratio"]["shallow"]
    assert visualize_nn(None, history=history, view="gradient_flow").layout.meta["mlektic_neural_gradient_flow"]


def test_hinton_area_tracks_absolute_value(xor_network):
    _, _, _, _, history = xor_network
    figure = visualize_nn_weights(history, style="hinton", max_frames=3)
    meta = figure.layout.meta["mlektic_neural_weights"]

    assert meta["style"] == "hinton"
    assert meta["parameters"] == ["0.weight", "2.weight"]
    final = figure.frames[-1].data[0]
    values = np.r_[np.ravel(history["parameters"]["0.weight"][-1]), np.ravel(history["parameters"]["0.bias"][-1])]
    sizes = np.asarray(final.marker.size, dtype=float)
    order = np.argsort(np.abs(values))
    assert np.all(np.diff(sizes[order]) >= -1e-9)
    assert [float(v) for v in final.customdata] == pytest.approx(values, abs=1e-4)
    with pytest.raises(ValueError, match="style"):
        visualize_nn_weights(history, style="unknown")


def test_light_skin_translates_neural_colours(xor_network):
    model, X, y, _, history = xor_network
    translate = color_mapper(get_skin("editorial").palette)
    # The dark design canvas becomes the skin background; light text becomes skin text.
    assert translate("#17181c") == get_skin("editorial").palette.background.lower()
    assert translate("#f5f7fa") == get_skin("editorial").palette.text.lower()
    # Hues keep their meaning but are darkened enough to read on paper.
    teal = translate("#55d6be")
    assert teal != "#55d6be" and int(teal[1:3], 16) < 0x55 + 1
    figure = visualize_nn_training(history, max_frames=3, theme="davinci")
    for trace in figure.data:
        color = getattr(getattr(trace, "line", None), "color", None)
        if isinstance(color, str) and color.startswith("#"):
            assert sum(int(color[i : i + 2], 16) for i in (1, 3, 5)) / 3 < 160
    graph = visualize_nn_graph(model, X[0], history, max_frames=3, theme="editorial")
    scales = [
        trace.marker.colorscale
        for trace in graph.data
        if getattr(trace, "marker", None) is not None and trace.marker.colorscale
    ]
    assert scales  # colour scales were translated, not dropped
    dark = visualize_nn_decision(model, X, history, targets=y, max_frames=2, grid_size=16, theme="aurora")
    assert dark.data[0].colorscale[-1][1].lower() == "#55d6be"  # dark skins keep the design hues


def test_export_uses_short_slash_escape(tmp_path, xor_network):
    model, X, y, _, history = xor_network
    figure = visualize_nn_decision(model, X, history, targets=y, max_frames=2, grid_size=16)
    path = export_figure(figure, tmp_path / "decision.html", include_plotly="cdn")
    html = path.read_text(encoding="utf-8")
    assert (chr(92) + "u002f") not in html
    assert "bdata" in html  # the initial field still travels as a compact typed array
