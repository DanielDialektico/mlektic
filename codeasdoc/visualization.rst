=========================
Visualization mathematics
=========================

Linear regression
=================

For d features, the fitted mapping is

.. math::

   \hat y_i = \theta_0 + \mathbf{x}_i^\top\boldsymbol{\theta}
   = \theta_0 + \sum_{j=1}^{d}\theta_j x_{ij}.

The 1D builder animates a line, the 2D builder animates a plane, and the nD
builder uses a report-style parameter view. ``display_space="original"``
converts coefficients through a recognized affine scaler when possible;
``display_space="scaled"`` shows learned-space parameters.

The displayed empirical MSE is

.. math::

   \operatorname{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2.

MSE, MAE, and R-squared are available as checkpoint metrics. They should not be
assumed to reproduce every estimator's private objective normalization or
regularization term.

Binary logistic regression
==========================

.. math::

   z_i = \theta_0 + \mathbf{x}_i^\top\boldsymbol{\theta},
   \qquad
   p_i = \sigma(z_i)=\frac{1}{1+e^{-z_i}}.

The positive class is ``classes_[1]`` for the binary coefficient convention
used by supported Scikit-learn estimators. The default decision is the fitted
estimator's prediction; a canonical 0.5 threshold is shown in the mathematical
explanation where appropriate.

Binary log-loss is computed after mapping labels to the positive class:

.. math::

   -\frac{1}{n}\sum_i
   [y_i'\log p_i + (1-y_i')\log(1-p_i)].

Multiclass logistic regression
==============================

For each fitted class column k:

.. math::

   z_k(\mathbf{x})=b_k+\mathbf{x}^{\top}\mathbf{w}_k.

Mlektic resolves multinomial Softmax or normalized one-vs-rest sigmoid
semantics by comparing available scores and probabilities when
``multiclass_link="auto"``. Explicit ``"softmax"`` and ``"ovr"`` values are
available for supported custom cases. Class identity always follows
``classes_`` order.

Replay and interpolation
========================

Incremental Scikit-learn models are replayed over a clone. Non-incremental
models are interpolated from a task-specific baseline. See
:doc:`history_semantics` for the exact contract and limitations.

Smoothing
=========

With ``smooth="ema"``:

.. math::

   s_t=\beta s_{t-1}+(1-\beta)\ell_t.

``loss_raw`` keeps \(\ell_t\); ``loss_display`` stores \(s_t\) for replay or
the unchanged empirical path evaluation for synthetic interpolation;
``loss_hist`` is a compatibility alias for the display values. The visible
metric and curve use the same series. Synthetic paths are already smooth, so
EMA is not applied and their final curve value remains exact.

Prediction explanations
=======================

The linear explainer substitutes a single query into the coefficient equation.
The logistic explainer additionally shows score, probability link, and class
decision. Both compute estimator output and verify any supplied display values.

Logistic figures use indexed classes by default. For a binary estimator,
probability indices 0 and 1 follow ``classes_[0]`` and ``classes_[1]``. The
sigmoid value is :math:`\hat p_1`; the result compares both probabilities and
reports only the winning index. ``show_class_labels=True`` appends fitted
semantic labels to applicable axes, legends, and prediction results. Fitted
labels and order remain present in ``layout.meta`` even when hidden.

In a two-feature view, string targets are mapped to numeric endpoints 0 and 1
only for plotting. This keeps the probability axis numeric, so the trained
probability surface remains visible. The decision boundary is rendered as the
line where the surface crosses :math:`\hat p_1=0.5`.

Queries outside a per-feature observed range are identified as extrapolations.
This is a featurewise domain warning, not a claim that the point lies outside a
statistical support model or convex hull.

Two-feature linear prediction results keep ordinary coordinate tuples inline
at the standard mathematical font size. A compact wrapped result is selected
only when formatted coordinates, including scientific notation, exceed the
available mathematical panel width.

Neural networks
===============

Neural views cover:

- architecture and tensor shape flow;
- computational graph structure;
- recorded training checkpoints;
- parameter and weight summaries;
- layerwise activations;
- mathematical forward-pass reports;
- prediction explanations.

For a dense layer:

.. math::

   \mathbf{z}^{(l)}=\mathbf{W}^{(l)}\mathbf{a}^{(l-1)}+\mathbf{b}^{(l)},
   \qquad
   \mathbf{a}^{(l)}=\phi^{(l)}(\mathbf{z}^{(l)}).

``TorchTrainingRecorder`` captures genuine training states when called from the
training loop. ``record_every`` controls semantic checkpoint capture; later
visual sampling should be interpreted separately.

Mathematical detail
===================

``detail="essential"`` retains the compact size, classic visual language, and
motion. ``detail="academic"`` adds a compact fitted-model derivation;
``detail="complete"`` also exposes preprocessing, objective, regularization,
and optimizer caveats. In every level, one-dimensional linear playback uses
the same evolving LaTeX equation band above the plotting axes. See
:doc:`mathematical_parity` for the exact contract.

The academic panel is a stable final-model reference while the existing
animation continues to show its labeled replay or interpolation path. This
keeps hybrid trace-only motion fluid and prevents visual subframes from being
misidentified as new mathematical states.

Classic visual contract
=======================

The original classic theme, base size, trace colors, line widths, and motion
remain the default. The one-dimensional equation placement is the documented
exception: it now occupies a reserved math band instead of the data axes.
Academic detail increases the canvas only when explicitly requested. Compact,
classroom, accessible, and responsive/reflow formats remain later-phase work.
