===============
Getting started
===============

Installation
============

From the repository root:

.. code-block:: bash

   pip install -e .

Core dependencies are NumPy, Scikit-learn, and Plotly. PyTorch integration is
optional:

.. code-block:: bash

   pip install -e ".[torch]"

Linear regression
=================

.. code-block:: python

   import numpy as np
   from sklearn.linear_model import SGDRegressor
   from mlektic import visualize_lr

   X = np.linspace(-2, 2, 80).reshape(-1, 1)
   y = 1.5 + 2.2 * X[:, 0]
   model = SGDRegressor(max_iter=200, random_state=7).fit(X, y)

   fig = visualize_lr(model, X, y, steps=80, max_frames=30)
   fig.show()

Because ``SGDRegressor`` supports ``partial_fit``, Mlektic reconstructs a replay
over a clone. The subtitle and slider state that the checkpoints are replayed;
they do not claim to be the original ``fit`` history.

For a closed-form estimator:

.. code-block:: python

   from sklearn.linear_model import LinearRegression

   model = LinearRegression().fit(X, y)
   fig = visualize_lr(model, X, y, steps=30)

The second figure uses a synthetic baseline-to-model interpolation and labels
its slider as interpolation progress.

Logistic regression
===================

.. code-block:: python

   from sklearn.linear_model import LogisticRegression
   from mlektic import visualize_logistic

   labels = np.where(X[:, 0] >= 0, "accepted", "rejected")
   classifier = LogisticRegression().fit(X, labels)
   fig = visualize_logistic(
       classifier,
       X,
       labels,
       steps=30,
       show_class_labels=False,
   )
   fig.show()

Logistic figures display class indices by default. Pass
``show_class_labels=True`` when the fitted semantic labels are relevant to the
lesson; they remain available in figure metadata when hidden.

Prediction explanations
=======================

.. code-block:: python

   from mlektic import explain_lr_prediction

   explanation = explain_lr_prediction(
       model,
       X,
       y,
       x_query=[[1.25]],
   )
   explanation.show()

The explainer computes the estimator output, marks extrapolation, and stores
the source and range assessment in ``figure.layout.meta``. If you supply
``yhat`` or logistic ``p_hat``/``y_hat``, it is verified against the estimator
by default. Use ``prediction_source="provided"`` only for an intentional
counterfactual lesson.

Neural networks
===============

.. code-block:: python

   import torch
   from torch import nn
   from mlektic import visualize_nn_architecture

   network = nn.Sequential(
       nn.Linear(4, 8),
       nn.ReLU(),
       nn.Linear(8, 1),
   )
   figure = visualize_nn_architecture(network, input_shape=(4,))
   figure.show()

Use :class:`mlektic.neural.recorder.TorchTrainingRecorder` when real neural
training checkpoints must be captured. The recorder is fundamentally different
from reconstructing a history after a model has already been fitted.

HTML export
===========

.. code-block:: python

   from mlektic import export_figure

   export_figure(fig, "linear-lesson.html")

The default export inlines Plotly and loads MathJax from a CDN. Plotly is
available offline, but equation rendering still requires network access. Pass
``include_mathjax=False`` only when preserving raw LaTeX without guaranteed
rendering is acceptable. Current fixed figure dimensions remain the default;
``responsive=True`` is explicit.

Next steps
==========

Read :doc:`history_semantics` before interpreting any animated timeline, then
use :doc:`visualization` for model-specific mathematics and :doc:`advanced`
for sampling, smoothing, pipelines, and performance.
