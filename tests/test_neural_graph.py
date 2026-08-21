"""Execution-graph and extensibility tests for the neural foundation."""

import numpy as np
import plotly.graph_objects as go
import pytest

from mlektic import (
    TorchTrainingRecorder,
    inspect_nn,
    register_neural_descriptor,
    visualize_nn,
    visualize_nn_architecture,
    visualize_nn_blocks,
    visualize_nn_training,
)
from mlektic.neural.introspection import run_torch_forward
from mlektic.visualization.neural.math_format import buffer_snapshot, parameter_snapshot

torch = pytest.importorskip("torch")


class ResidualNetwork(torch.nn.Module):
    def __init__(self):
        """Create a small residual test model."""
        super().__init__()
        self.first = torch.nn.Linear(4, 4)
        self.second = torch.nn.Linear(4, 4)

    def forward(self, values):
        return values + self.second(torch.relu(self.first(values)))


class SharedNetwork(torch.nn.Module):
    def __init__(self):
        """Create a model that executes one module twice."""
        super().__init__()
        self.shared = torch.nn.Linear(4, 4)

    def forward(self, values):
        return self.shared(self.shared(values))


def test_fx_capture_preserves_functional_residual_topology():
    graph = inspect_nn(ResidualNetwork(), torch.zeros(1, 4))
    add_node = next(node for node in graph.nodes if node.label == "Add")
    incoming = [edge for edge in graph.edges if edge.target == add_node.id]

    assert graph.provenance.backend == "torch.fx"
    assert graph.provenance.includes_functional_ops is True
    assert len(incoming) == 2
    assert add_node.role == "merge"
    assert add_node.math_status == "specialized"
    assert r"\mathbf{x}_1+\mathbf{x}_2" in add_node.formula


def test_reused_module_has_a_distinct_execution_call_index():
    graph = inspect_nn(SharedNetwork(), torch.zeros(1, 4))
    calls = [node for node in graph.nodes if node.module_path == "shared"]

    assert [node.call_index for node in calls] == [1, 2]
    assert len({node.id for node in calls}) == 2


def test_embedding_preserves_integer_input_dtype():
    model = torch.nn.Sequential(torch.nn.Embedding(20, 8), torch.nn.Flatten())
    graph = inspect_nn(model, torch.tensor([[1, 2, 3]], dtype=torch.long))
    embedding = next(node for node in graph.nodes if node.module_type == "Embedding")

    assert graph.provenance.backend == "torch.fx"
    assert embedding.inputs[0].dtype == "int64"
    assert embedding.role == "embedding"
    assert r"E[x_t]" in embedding.formula


def test_root_recurrent_module_uses_observed_fallback():
    model = torch.nn.LSTM(3, 5, batch_first=True, bidirectional=True)
    graph = inspect_nn(model, torch.zeros(1, 4, 3))
    recurrent = next(node for node in graph.nodes if node.module_type == "LSTM")

    assert graph.provenance.backend == "eager-hooks"
    assert graph.provenance.includes_dynamic_control_flow is True
    assert recurrent.role == "recurrent"
    assert len(recurrent.outputs) == 3
    assert recurrent.hyperparameters["hidden_size"] == 5
    assert recurrent.hyperparameters["bidirectional"] is True


def test_convolutional_blocks_keep_spatial_roles_and_hyperparameters():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 4, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(4),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(4 * 4 * 4, 3),
    )
    graph = inspect_nn(model, torch.zeros(1, 1, 8, 8))
    modules = {node.module_type: node for node in graph.nodes if node.module_type}

    assert modules["Conv2d"].role == "learnable"
    assert modules["Conv2d"].hyperparameters["kernel_size"] == (3, 3)
    assert modules["BatchNorm2d"].role == "normalization"
    assert modules["MaxPool2d"].role == "pooling"
    assert modules["Flatten"].outputs[0].shape == (1, 64)


def test_attention_root_accepts_multiple_tensor_arguments():
    model = torch.nn.MultiheadAttention(8, 2, batch_first=True)
    values = torch.randn(2, 4, 8)
    inputs = (values, values.clone(), values.clone())
    graph = inspect_nn(model, inputs)
    attention = next(node for node in graph.nodes if node.module_type == "MultiheadAttention")
    output = graph.node_map()[graph.output_nodes[0]]
    figure = visualize_nn_blocks(model, inputs)
    hover = next(
        str(value) for trace in figure.data for value in (trace.customdata or []) if "MultiheadAttention" in str(value)
    )
    output_labels = [
        str(value) for trace in figure.data for value in (trace.text or []) if "<b>Output</b>" in str(value)
    ]

    assert attention.role == "attention"
    assert attention.hyperparameters["embed_dim"] == 8
    assert attention.hyperparameters["num_heads"] == 2
    assert len(attention.inputs) == 3
    assert len(attention.outputs) == 2
    assert len(output.outputs) == 2
    assert r"\quadhead" not in attention.formula
    assert r"\quad head_i" in attention.formula
    assert r"\operatorname" not in hover
    assert "MHA(Q,K,V)" in hover
    assert output_labels and all("?" not in label for label in output_labels)


def test_multiple_inputs_outputs_and_tensor_merges_are_captured():
    class Siamese(torch.nn.Module):
        def __init__(self):
            """Create a shared-input test head."""
            super().__init__()
            self.head = torch.nn.Linear(6, 1)

        def forward(self, left, right):
            joined = torch.cat((left, right), dim=-1)
            return {"score": self.head(joined), "difference": left - right}

    graph = inspect_nn(Siamese(), (torch.zeros(2, 3), torch.ones(2, 3)))

    assert len(graph.input_nodes) == 2
    assert any(node.label == "Concatenate" for node in graph.nodes)
    assert len(graph.node_map()[graph.output_nodes[0]].inputs) == 2


def test_custom_descriptor_extends_unknown_module_semantics():
    class PedagogicalScale(torch.nn.Module):
        def forward(self, values):
            return 2.0 * values

    register_neural_descriptor(
        "PedagogicalScale",
        role="operation",
        label="Scale by two",
        formula=r"\mathbf{y}=2\mathbf{x}",
        replace=True,
    )
    graph = inspect_nn(PedagogicalScale(), torch.ones(1, 2), backend="hooks")
    node = next(node for node in graph.nodes if node.module_type == "PedagogicalScale")

    assert node.label == "Scale by two"
    assert node.formula == r"\mathbf{y}=2\mathbf{x}"
    assert node.math_status == "specialized"


def test_block_renderer_is_opt_in_and_legacy_default_is_unchanged():
    model = ResidualNetwork()
    sample = torch.zeros(1, 4)
    default_figure = visualize_nn_architecture(model, sample)
    explicit_legacy = visualize_nn_architecture(model, sample, architecture_mode="legacy")
    blocks = visualize_nn_architecture(model, sample, architecture_mode="blocks")
    direct = visualize_nn_blocks(model, sample)
    routed = visualize_nn(model, sample, view="blocks")

    assert isinstance(blocks, go.Figure)
    assert len(default_figure.data) == len(explicit_legacy.data)
    assert default_figure.layout.title.text == explicit_legacy.layout.title.text
    assert blocks.layout.title.text == "Neural execution graph"
    assert direct.layout.title.text == routed.layout.title.text == "Neural execution graph"
    assert any("Add" in str(value) for trace in blocks.data for value in (trace.text or []))
    assert blocks.layout.meta["mlektic_neural_graph"]["capture"]["backend"] == "torch.fx"
    formulas = [annotation for annotation in blocks.layout.annotations if str(annotation.text).startswith("$")]
    assert formulas
    assert [annotation.yshift > 0 for annotation in formulas] == [index % 2 == 0 for index in range(len(formulas))]
    assert [annotation.yanchor for annotation in formulas] == [
        "bottom" if index % 2 == 0 else "top" for index in range(len(formulas))
    ]
    assert {annotation.font.size for annotation in formulas} == {13}
    assert blocks.layout.meta["mlektic_neural_graph"]["formula_layout"].startswith("alternating")


def test_recorder_v2_exposes_temporal_semantics_buffers_and_optimizer_groups():
    torch.manual_seed(4)
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3), torch.nn.ReLU())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    recorder = TorchTrainingRecorder(
        model,
        optimizer=optimizer,
        loss_fn=torch.nn.MSELoss(),
        capture_optimizer_state=True,
    )
    values = torch.randn(6, 3)
    optimizer.zero_grad()
    predictions = model(values)
    loss = predictions.square().mean()
    loss.backward()
    optimizer.step()
    recorder.record(1, loss=loss, predictions=predictions, targets=torch.zeros_like(predictions), task="regression")
    recorder.close()
    history = recorder.to_history()

    assert history["history_schema_version"] == 2
    assert history["frame_semantics"][0]["parameter_phase"] == "post_step"
    assert history["frame_semantics"][0]["observation_phase"] == "pre_step"
    assert "1.running_mean" in history["buffers"]
    assert history["optimizer_groups"][0][0]["lr"] == pytest.approx(0.02)
    assert "exp_avg" in history["optimizer_state_norms"][0]["0.weight"]
    figure = visualize_nn_training(history, max_frames=None)
    metadata = figure.layout.meta["mlektic_neural_history"]
    assert metadata["history_schema_version"] == 2
    assert metadata["captured"]["buffers"] is True


def test_historical_forward_replays_captured_buffers_instead_of_live_state():
    torch.manual_seed(5)
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    recorder = TorchTrainingRecorder(model, capture_activations=False)
    model.train()
    model(torch.randn(8, 2))
    recorder.record(0, capture_phase="post_forward", observation_phase="post_forward")
    recorder.close()
    history = recorder.to_history()
    parameters = parameter_snapshot(history, 0)
    buffers = buffer_snapshot(history, 0)
    query = torch.tensor([[0.2, -0.1]])
    expected, _ = run_torch_forward(model, query, parameters, buffers)

    with torch.no_grad():
        model[0].weight.add_(10.0)
        model[1].running_mean.add_(20.0)
    restored, _ = run_torch_forward(model, query, parameters, buffers)

    assert np.allclose(expected.detach().numpy(), restored.detach().numpy())


def test_data_dependent_control_flow_reports_eager_fallback():
    class DynamicBranch(torch.nn.Module):
        def __init__(self):
            """Create two data-dependent branches."""
            super().__init__()
            self.positive = torch.nn.Linear(2, 2)
            self.negative = torch.nn.Linear(2, 2)

        def forward(self, values):
            return self.positive(values) if values.sum() > 0 else self.negative(values)

    graph = inspect_nn(DynamicBranch(), torch.ones(1, 2))

    assert graph.provenance.backend == "eager-hooks"
    assert graph.provenance.exact_for_input is False
    assert "torch.fx fallback reason" in " ".join(graph.provenance.notes)


def test_keyword_inputs_and_graph_serialization_preserve_ports():
    class KeywordModel(torch.nn.Module):
        def forward(self, values, *, scale):
            return values * scale

    graph = inspect_nn(
        KeywordModel(),
        torch.ones(1, 3),
        input_kwargs={"scale": torch.full((1, 3), 2.0)},
    )
    payload = graph.to_dict()

    assert len(graph.input_nodes) == 2
    assert any(node.label == "Multiply" for node in graph.nodes)
    assert payload["schema_version"] == 1
    assert payload["provenance"]["backend"] == "torch.fx"


def test_unknown_root_module_is_visible_as_generic_mathematics():
    class UnknownOperation(torch.nn.Module):
        def forward(self, values):
            return values.square()

    graph = inspect_nn(UnknownOperation(), torch.ones(1, 3), backend="hooks")
    node = next(node for node in graph.nodes if node.module_type == "UnknownOperation")

    assert node.math_status == "generic"
    assert node.role == "operation"
    assert node.formula


def test_large_block_rendering_discloses_visual_collapse():
    layers = []
    for _ in range(18):
        layers.extend((torch.nn.Linear(8, 8), torch.nn.ReLU()))
    model = torch.nn.Sequential(*layers)
    full_graph = inspect_nn(model, torch.zeros(1, 8))
    figure = visualize_nn_blocks(model, torch.zeros(1, 8), max_nodes=12)
    visible_text = [str(value) for trace in figure.data for value in (trace.text or [])]
    hover_text = [str(value) for trace in figure.data for value in (trace.customdata or [])]

    assert len(full_graph.nodes) > 12
    assert any("ops" in value and "summarized" in value for value in visible_text)
    assert any("collapsed operations" in value for value in hover_text)


def test_dense_parallel_column_moves_formulas_to_hover():
    class Parallel(torch.nn.Module):
        def __init__(self):
            """Create four operations at the same graph depth."""
            super().__init__()
            self.branches = torch.nn.ModuleList([torch.nn.Linear(4, 2) for _ in range(4)])

        def forward(self, values):
            return tuple(branch(values) for branch in self.branches)

    figure = visualize_nn_blocks(Parallel(), torch.zeros(1, 4))

    assert figure.layout.meta["mlektic_neural_graph"]["inline_formulas"] is False
    assert any("hidden to prevent overlap" in str(item.text) for item in figure.layout.annotations)


def test_compound_transformer_primitive_remains_one_semantic_block():
    model = torch.nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, batch_first=True)
    graph = inspect_nn(model, torch.zeros(2, 4, 8))
    block = next(node for node in graph.nodes if node.module_type == "TransformerEncoderLayer")

    assert graph.provenance.backend == "eager-hooks"
    assert block.role == "attention"
    assert block.hyperparameters["num_heads"] == 2
    assert block.hyperparameters["dim_feedforward"] == 16
