====================================
Tabular mathematical detail contract
====================================

Phase 1 gives linear and logistic figures the same kind of traceable
mathematical explanation used by the neural views. The implementation is
additive: the default ``detail="essential"`` keeps the compact main figure,
classic visual language, controls, and animation cadence. ``"academic"`` and
``"complete"`` add a fitted-model reference panel below the existing slider.
All detail levels share the same main mathematical animation.

Detail levels
=============

``essential``
   Compact visualization without a visible fitted-model reference panel. The
   complete estimator-backed contract is still available under
   ``figure.layout.meta["mlektic_math"]``.

``academic``
   Adds dimensions, feature space, one numerical substitution, the model
   transformation, the decision, and the empirical objective. The panel is
   explicitly labeled *Fitted-model derivation*: it is a stable reference
   while the animation continues to show replayed or interpolated states.

``complete``
   Adds preprocessing claims, regularization settings, evaluation metrics,
   and the distinction between a canonical gradient update and an exact
   estimator-private optimization rule.

The reference panel is intentionally fixed during playback. This preserves
trace-only hybrid interpolation for one-dimensional linear regression and
avoids turning smooth visual subframes into expensive MathJax layout redraws.

Evolving equation placement
============================

The symbolic one-dimensional linear definition remains above the figure. Its
numerical fitted equation is interpolated at every hybrid visual subframe and
rendered as a LaTeX text trace in a dedicated mathematical band above the data
and loss axes. It is not an annotation inside the data coordinate system.
Consequently it cannot cover observations or the fitted line, and it remains
synchronized with the model, loss, and metric-card traces while Plotly keeps
``redraw=False``. The cards retain their dedicated vertical side column.

This main-figure contract is identical for ``essential``, ``academic``, and
``complete``. The academic/complete fitted-model panel remains a separate,
stable derivation below the slider. Other linear and logistic dimensionalities
already use evolving LaTeX substitutions outside their data regions.

Linear contract
===============

For a selected observation :math:`i`, the contract stores and displays

.. math::

   \hat y_i=\theta_0+\mathbf{x}_i^\top\boldsymbol{\theta}
   =\theta_0+\sum_j \theta_j x_{ij}.

``sample_index`` selects the observation; ``None`` deterministically selects
the first row. ``feature_names`` accepts original feature names, and pandas
column names are used automatically. The metadata contains each
observation-specific contribution :math:`\theta_jx_{ij}`, their sum, the
reconstructed prediction, the estimator prediction, and a numerical match
flag. Coefficient magnitude is therefore not confused with a contribution.

The fitted-model panel formats these products as a multiline LaTeX derivation
with at most three products per row. Up to nine features are shown in full.
For larger models, the nine largest absolute observation-specific
contributions are displayed across bounded rows, an independent note reports
the selection, and the complete ordered contribution vector remains in
``layout.meta["mlektic_math"]``. This is deliberate semantic reduction rather
than accidental clipping at the canvas boundary.

The displayed MSE is always the estimator-verifiable empirical convention

.. math::

   \operatorname{MSE}=\frac{1}{n}\sum_i(y_i-\hat y_i)^2.

MAE and :math:`R^2` remain evaluation metrics. For ``LinearRegression``, the
documentation states that residual sum of squares and MSE have the same
minimizer but different normalization. Other estimators are not assigned a
private loss normalization that Mlektic cannot verify.

Binary logistic contract
========================

The binary chain is

.. math::

   z_i=\theta_0+\mathbf{x}_i^\top\boldsymbol{\theta},\qquad
   p_i=\sigma(z_i),\qquad
   \hat y_i=\begin{cases}c_1,&p_i\ge\tau\\c_0,&p_i<\tau.\end{cases}

``classes_[1]`` is the positive class associated with the public binary
coefficient vector. ``threshold`` must be strictly between zero and one. A
custom threshold updates the indexed probability tick and the academic
threshold geometry. Semantic class labels remain hidden unless
``show_class_labels=True``; fitted label order is always stored in metadata.

The binary log-loss maps targets through
:math:`y_i'=\mathbb{1}[y_i=c_1]`. The stored probabilities and empirical loss
are compared with ``predict_proba`` using fitted class order, so numeric,
string, and other Scikit-learn-compatible scalar labels are not reinterpreted.

Multiclass logistic contract
============================

For class column :math:`k`,

.. math::

   z_k(\mathbf{x})=b_k+\mathbf{x}^\top\mathbf{w}_k.

The resolved link is either Softmax or normalized one-vs-rest sigmoid. Every
reconstructed probability vector is checked against ``predict_proba``. The
winning class uses ``argmax`` in fitted ``classes_`` order.

``class_focus`` accepts a fitted label or zero-based class index. In a 1D or
2D multiclass view it keeps one probability curve or surface visible and adds
``class focus c_k (1/K)`` to the title. This avoids competing translucent
surfaces while preserving all class columns and fitted labels in metadata.

Preprocessing and feature spaces
================================

Mlektic probes the fitted preprocessing map on the supplied feature space. If
the map is affine, it verifies

.. math::

   \mathbf{u}=A\mathbf{x}+\mathbf{c}

and converts final coefficients back to original units:

.. math::

   \mathbf{w}_{o}=A^\top\mathbf{w}_{m},\qquad
   b_o=b_m+\mathbf{c}^\top\mathbf{w}_{m}.

The reconstructed raw-space prediction must equal the pipeline prediction.
This covers recognized standard scaling and other numerically verified affine
maps. If preprocessing is non-affine, such as polynomial feature expansion,
Mlektic uses transformed-feature names from ``get_feature_names_out`` when
available and explicitly refuses to claim a raw-space coefficient vector.

A polynomial pipeline is therefore still linear in its fitted coefficients
and transformed features, but it need not be a straight line in the original
input. For example,

.. math::

   \phi(x)=(x,x^2),\qquad
   \hat y=\theta_0+\theta_1x+\theta_2x^2

is a plane in transformed feature space and a parabola when plotted against
the original scalar ``x``. The curve represents fitted-model geometry; the
separate ``Interpolation MSE`` curve represents empirical evaluation along the
synthetic parameter path, not optimizer history.

Objective and regularization honesty
=====================================

``show_objective="auto"`` enables the empirical objective for academic and
complete detail. ``show_regularization="auto"`` enables the regularization
line in complete detail. Either option also accepts a boolean override.

The penalty family, ``alpha`` or ``C``, and ``l1_ratio`` are included only when
they are public estimator parameters. ``C`` is labeled as inverse strength.
The exact internal normalization and intercept penalty convention are reported
as *not introspected* whenever they cannot be established safely. The
canonical gradient expression is a teaching reference, never labeled as the
estimator's exact update unless all relevant scheduling, batching, averaging,
preprocessing, and penalty behavior is known.

Interpolation parity
====================

For coefficient-bearing logistic estimators, synthetic interpolation now
occurs in parameter space. At every semantic state Mlektic derives score,
sigmoid/Softmax/OvR probability, surface or curve, and empirical loss from the
same interpolated coefficients. ``source_detail.interpolation_target`` is
``"parameters"``. Estimators that expose probabilities without compatible
public coefficients retain probability-space interpolation and are labeled
``"probabilities"`` instead.

This change does not alter the number of semantic checkpoints, frame sampling,
playback controls, transition duration, or hybrid subframe count. It removes
a mathematical mismatch without reducing visual continuity.

Metadata example
================

.. code-block:: python

   figure = visualize_lr(
       model,
       X,
       y,
       detail="complete",
       feature_names=["length", "width"],
       sample_index=4,
   )

   contract = figure.layout.meta["mlektic_math"]
   contract["sample"]["contributions"]
   contract["sample"]["matches_model"]
   contract["objective"]
   contract["regularization"]

The metadata is the machine-readable source of truth. The visible panel is a
compact rendering of that contract, not an independent calculation.
