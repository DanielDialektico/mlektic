=======================
Prediction explanations
=======================

Prediction explainers use the estimator output by default and verify supplied
values within documented tolerances. ``prediction_source="provided"`` is an
explicit counterfactual lesson and is labeled as such.

Linear output
=============

The panel connects inputs, symbolic form, numerical substitution, scalar
prediction, and the plotted coordinate. Queries outside any observed per-feature
training range remain visible and are labeled extrapolations. The distinct
Input, Substitution, Output, and Reset controls remain available for
high-dimensional predictions, including ten-variable views. In ``report`` or
``reduced_motion`` mode the complete Output content is applied before controls
are removed.

Logistic output
===============

Binary explanations show both probabilities and the winning fitted class index.
Multiclass explanations show the probability vector and argmax. Semantic labels
appear only when ``show_class_labels=True``. Borderless, high-contrast
annotations keep plotted prediction values readable over dense geometry.

Inspect ``LR-PRED-EXTRAP``, ``LR-PRED-10D-STAGES``, and ``LOG-PRED-BINARY`` in
the QA notebooks.
