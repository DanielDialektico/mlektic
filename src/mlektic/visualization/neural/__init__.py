"""Plotly builders for PyTorch neural-network visualizations."""

from .architecture import build_nn_architecture_figure
from .backpropagation import build_nn_backpropagation_figure
from .blocks import build_nn_block_figure
from .decision import build_nn_decision_figure
from .gradient_flow import build_nn_gradient_flow_figure
from .graph import build_nn_graph_figure
from .hinton import build_nn_hinton_figure
from .hyperparameters import build_nn_hyperparameter_figure
from .landscape import build_nn_loss_landscape_figure
from .math_view import build_nn_prediction_figure
from .representation import build_nn_representation_figure
from .training import build_nn_activation_figure, build_nn_training_figure, build_nn_weight_figure
from .training_step import build_nn_training_step_figure

__all__ = [
    "build_nn_activation_figure",
    "build_nn_architecture_figure",
    "build_nn_backpropagation_figure",
    "build_nn_block_figure",
    "build_nn_decision_figure",
    "build_nn_gradient_flow_figure",
    "build_nn_graph_figure",
    "build_nn_hinton_figure",
    "build_nn_hyperparameter_figure",
    "build_nn_loss_landscape_figure",
    "build_nn_prediction_figure",
    "build_nn_representation_figure",
    "build_nn_training_figure",
    "build_nn_training_step_figure",
    "build_nn_weight_figure",
]
