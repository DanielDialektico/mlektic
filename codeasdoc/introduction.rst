============
Introduction
============
this page is ``SMOKE-LR-1D`` in ``notebooks/qa/qa_00_smoke_matrix.ipynb``.
Mlektic is a teaching library for inspecting the mathematics of fitted
Scikit-learn linear and logistic models and genuinely recorded PyTorch neural
networks. It connects an estimator's numerical state to equations, geometry,
probabilities, empirical metrics, and individual predictions in Plotly.

What makes the library different
================================

Mlektic treats provenance as part of mathematical correctness:

* ``recorded`` means a state was captured during the actual PyTorch training
  loop;
* ``replayed`` means an incremental Scikit-learn estimator was trained again on
  a clone under documented replay settings;
* ``interpolated`` means the figure follows a declared baseline-to-fitted-model
  parameter path;
* ``fitted`` identifies the exact supplied estimator state.

The animation never renames perceptual subframes as optimizer updates. This
allows motion to remain fluid without teaching a false training history.

Supported public routes
=======================

The tabular APIs route one, two, and higher-dimensional data to suitable
geometries or symbolic summaries. Neural APIs cover architecture, computational
graph, recorded training metrics, weights, activations, and forward-pass
prediction explanations. See :doc:`compatibility` for the exact matrix.

Start with :doc:`installation`, then choose :doc:`linear_lesson`,
:doc:`logistic_lesson`, or :doc:`neural_lesson`. The human-QA reference case for
this page is ``SMOKE-LR-1D`` in ``notebooks/qa/qa_00_smoke_matrix.ipynb``.
