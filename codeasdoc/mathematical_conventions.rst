========================
Mathematical conventions
========================
“rejected.” Inspect ``LR-POLYNOMIAL`` and ``LOG-MULTI-FOCUS`` in the QA suite.
Mlektic uses :math:`\theta_0` for an intercept, bold lower-case symbols for
vectors, upper-case symbols for matrices, and fitted class indices in the order
of ``classes_``. A displayed coordinate belongs to its declared feature space:
original, affine-scaled, or transformed.

Exactness labels
================

``fitted`` and ``recorded`` identify estimator or recorder values. ``replayed``
and ``interpolated`` identify constructed states. Empirical evaluation along a
constructed path is mathematically valid but is not evidence of the estimator's
private optimization process.

Objectives and metrics
======================

MSE, MAE, :math:`R^2`, log-loss, accuracy, precision, and recall are reported
with their empirical roles. Regularization panels expose only supported public
hyperparameters and explicitly say when intercept penalties or private
normalization are not introspected.

Class labels
============

Indexed output is the mathematical default. ``show_class_labels=True`` adds
semantic labels for context; it does not invent meanings such as “accepted” or
“rejected.” Inspect ``LR-POLYNOMIAL`` and ``LOG-MULTI-FOCUS`` in the QA suite.
