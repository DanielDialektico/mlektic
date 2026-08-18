===============================
Mlektic technical documentation
===============================

**Mlektic** is a Python library for learning how machine-learning models map
inputs to outputs and how their mathematical state can be represented over
time. It provides interactive Plotly views for Scikit-learn linear and logistic
models and architecture, training, parameter, activation, forward-pass, and
prediction views for PyTorch networks.

Mlektic treats provenance as part of the mathematics. A tabular animation may
be a replay over a cloned estimator or a synthetic interpolation to a fitted
model; it is not described as a recording unless the states were captured from
the actual training process.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   introduction
   installation
   getting_started
   linear_lesson
   logistic_lesson
   neural_lesson
   neural_execution_graphs
   history_semantics
   mathematical_parity
   mathematical_conventions
   animation_performance
   visual_design
   themes_formats_sizes
   model_hyperparameters
   prediction_explanations
   export
   gallery
   compatibility
   limitations
   architecture
   api_reference
   visualization
   advanced
   contributing_visual_qa
   changelog

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
