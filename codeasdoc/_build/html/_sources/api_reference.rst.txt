=============
API reference
=============

Public tabular visualization
============================

.. module:: mlektic.api.linear

.. autofunction:: mlektic.api.linear.visualize_lr

.. autofunction:: mlektic.api.linear.explain_lr_prediction

.. module:: mlektic.api.logistic

.. autofunction:: mlektic.api.logistic.visualize_logistic

.. autofunction:: mlektic.api.logistic.explain_logistic_prediction

History services
================

.. module:: mlektic.services.linear_history

.. autofunction:: mlektic.services.linear_history.fit_history

.. autofunction:: mlektic.services.linear_history.fit_history_logistic

Configuration and payloads
==========================

.. module:: mlektic.domain.config

.. autoclass:: mlektic.domain.config.LinearHistoryConfig
   :members:

.. autoclass:: mlektic.domain.config.LogisticHistoryConfig
   :members:

.. module:: mlektic.domain.history

.. autoclass:: mlektic.domain.history.HistoryMetadata
   :members:

.. autoclass:: mlektic.domain.history.LinearHistoryPayload
   :members:

.. autoclass:: mlektic.domain.history.LogisticHistoryPayload
   :members:

History implementation
======================

.. module:: mlektic.history.engine

.. autoclass:: mlektic.history.engine.HistoryEngine
   :members:

.. module:: mlektic.history.strategy_iterative

.. autoclass:: mlektic.history.strategy_iterative.IterativeCapture
   :members:

.. module:: mlektic.history.strategy_interp

.. autoclass:: mlektic.history.strategy_interp.InterpolationCapture
   :members:

.. module:: mlektic.history.metrics

.. autofunction:: mlektic.history.metrics.build_linear_metrics

.. autofunction:: mlektic.history.metrics.build_logistic_metrics

.. module:: mlektic.history.sampling

.. autofunction:: mlektic.history.sampling.decimate_history

Adapters
========

.. module:: mlektic.adapters.sklearn

.. autoclass:: mlektic.adapters.sklearn.SklearnAdapter
   :members:

Low-level figure builders
=========================

.. module:: mlektic.visualization.linear.router

.. autofunction:: mlektic.visualization.linear.router.build_lr_figure

.. module:: mlektic.visualization.logistic.router

.. autofunction:: mlektic.visualization.logistic.router.build_logistic_figure

Theme and animation helpers
===========================

.. module:: mlektic.visualization.design

.. autoclass:: mlektic.visualization.design.VisualTokens

.. autoclass:: mlektic.visualization.design.VisualSpec

.. autofunction:: mlektic.visualization.design.available_themes

.. autofunction:: mlektic.visualization.design.get_theme_tokens

.. module:: mlektic.visualization.theme

.. autofunction:: mlektic.visualization.theme.get_base_layout

.. autofunction:: mlektic.visualization.theme.get_sliders

.. autofunction:: mlektic.visualization.theme.configure_animation

.. autofunction:: mlektic.visualization.theme.annotate_history_semantics

Export
======

.. module:: mlektic.api.optimize

.. autofunction:: mlektic.api.optimize.export_figure

.. autofunction:: mlektic.api.optimize.show_optimized

Neural networks
===============

.. module:: mlektic.api.neural

.. autoclass:: mlektic.neural.recorder.TorchTrainingRecorder
   :members:

.. autofunction:: mlektic.api.neural.visualize_nn

.. autofunction:: mlektic.api.neural.visualize_nn_architecture

.. autofunction:: mlektic.api.neural.inspect_nn

.. autofunction:: mlektic.api.neural.visualize_nn_blocks

.. autofunction:: mlektic.api.neural.visualize_nn_hyperparameters

.. autofunction:: mlektic.api.neural.register_neural_descriptor

.. autofunction:: mlektic.api.neural.visualize_nn_graph

.. autofunction:: mlektic.api.neural.visualize_nn_backpropagation

.. autofunction:: mlektic.api.neural.visualize_nn_loss_landscape

.. autofunction:: mlektic.api.neural.visualize_nn_training

.. autofunction:: mlektic.api.neural.visualize_nn_weights

.. autofunction:: mlektic.api.neural.explain_nn_prediction

.. autofunction:: mlektic.api.neural.build_nn_math_report

.. autofunction:: mlektic.api.neural.display_nn_math_report

.. autofunction:: mlektic.api.neural.export_nn_math_report

.. module:: mlektic.neural.graph_ir

.. autoclass:: mlektic.neural.graph_ir.NeuralGraph
   :members:

.. autoclass:: mlektic.neural.graph_ir.NeuralNode

.. autoclass:: mlektic.neural.graph_ir.NeuralEdge

.. autoclass:: mlektic.neural.graph_ir.CaptureProvenance
