=======================
Prediction explanations
=======================
prediction values readable over dense geometry.
Inspect ``LR-PRED-EXTRAP`` and ``LOG-PRED-BINARY`` in the QA notebooks.
Prediction explainers use the estimator output by default and verify supplied
values within documented tolerances. ``prediction_source="provided"`` is an
explicit counterfactual lesson and is labeled as such.

Linear output
=============

The panel connects inputs, symbolic form, numerical substitution, scalar
prediction, and the plotted coordinate. Queries outside any observed per-feature
training range remain visible and are labeled extrapolations.

Logistic output
===============

Binary explanations show both probabilities and the winning fitted class index.
Multiclass explanations show the probability vector and argmax. Semantic labels
appear only when ``show_class_labels=True``. High-contrast boxes keep plotted
prediction values readable over dense geometry.

Inspect ``LR-PRED-EXTRAP`` and ``LOG-PRED-BINARY`` in the QA notebooks.
