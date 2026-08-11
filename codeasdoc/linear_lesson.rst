First rigorous linear-regression lesson
============================================================

For a fitted model with ``d`` features, Mlektic uses

.. math::

   \hat y = \theta_0 + \mathbf{x}^{\mathsf T}\boldsymbol\theta.

The numerical equation shown above the geometry evolves with the displayed
state. The academic panel below the slider is deliberately separate: it derives
one prediction from the exact fitted model and reports empirical MSE, MAE, and
:math:`R^2`.

.. code-block:: python

   fig = visualize_lr(
       model,
       X,
       y,
       steps=30,
       max_frames=10,
       detail="complete",
       feature_names=["study_hours"],
       sample_index=12,
   )

What is exact?
==============

The supplied estimator's final coefficients, fitted prediction, and endpoint
metrics are exact. Earlier Scikit-learn states are either reconstructed replay
checkpoints or synthetic interpolation states, as the subtitle and slider say.
``Interpolation MSE`` evaluates the dataset along the declared parameter path;
it is not a private optimizer-loss trace.

For an affine ``StandardScaler`` pipeline, original-space coefficients are
verified by algebraic inversion. For ``PolynomialFeatures``, the mathematics is
correctly reported in transformed-feature space; linearity refers to the
coefficients, not necessarily to a straight raw-space curve.

Inspect ``LEARN-LINEAR-PLANE`` and ``LEARN-LINEAR-POLY`` in
``notebooks/learn/learn_01_linear_regression.ipynb``.
