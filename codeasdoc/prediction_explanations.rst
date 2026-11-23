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

Pipelines with feature expansion
================================

Both explainers accept pipelines whose preprocessing expands features, such as
``PolynomialFeatures``. The Input stage shows the query :math:`\mathbf x` and
its transformed features :math:`\mathbf u=\phi(\mathbf x)`; Substitution writes
:math:`\theta_0+\sum_j\theta_j u_j` (followed by :math:`\sigma` or softmax for
classification) with the fitted numbers. The substituted value is checked
against ``predict`` or ``predict_proba`` before anything is drawn, and the
check is recorded as ``substitution_matches_predict`` in
``layout.meta["mlektic_prediction"]``. Curves, surfaces, and decision
boundaries come from the estimator over the raw inputs, so a polynomial
boundary can have several pieces. Wider maps use the coefficient views on
:math:`\mathbf u`, in which every coordinate enters the substitution.

Inspect ``LR-PRED-EXTRAP``, ``LR-PRED-10D-STAGES``, and ``LOG-PRED-BINARY`` in
the QA notebooks.
