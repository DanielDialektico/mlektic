================
Advanced usage
================

Build history separately
========================

.. code-block:: python

   from mlektic import fit_history, build_lr_figure

   history = fit_history(
       model,
       X,
       y,
       steps=500,
       max_frames=40,
       smooth="ema",
       smooth_beta=0.85,
       metrics=["loss", "mse", "r2", "mae"],
   )
   figure = build_lr_figure(X, y, history=history)

Low-level builders do not automatically attach the public provenance subtitle.
Use the public ``visualize_lr``/``visualize_logistic`` functions for the full
contract, or call ``annotate_history_semantics`` explicitly.

Custom metrics
==============

Custom metric mappings receive ``(y_true, y_pred)`` and return one scalar per
checkpoint:

.. code-block:: python

   def median_absolute_error(y_true, y_pred):
       return float(np.median(np.abs(y_true - y_pred)))

   history = fit_history(
       model,
       X,
       y,
       metrics={"Median AE": median_absolute_error},
   )

Unknown built-in names and non-callable mapping values raise immediately.
Visible metric capacity is currently limited by figure design; phase 1/3 will
formalize truncation and detail controls.

Temporal decimation
===================

``steps`` constructs K semantic states. ``max_frames`` uniformly retains at
most N states. If ``max_frames=None``, ``frame_step`` selects a stride and
retains the final endpoint.

.. code-block:: python

   history = fit_history(
       model,
       X,
       y,
       steps=1000,
       max_frames=50,
   )

   print(history["metadata"]["captured_steps"])   # 1000
   print(history["metadata"]["displayed_steps"])  # 50
   print(history["metadata"]["displayed_step_indices"])

Linear 1D hybrid animation
==========================

``animation_mode="auto"`` selects hybrid trace-only motion for one-dimensional
linear regression and native animation elsewhere. ``interpolation_frames``
creates perceptual intervals between retained semantic checkpoints. These
visual frames do not change K or N.

.. code-block:: python

   fig = visualize_lr(
       model,
       X,
       y,
       animation_mode="hybrid",
       interpolation_frames=3,
       fps=36,
   )

Use ``animation_mode="native"`` when every Plotly frame must correspond to one
retained semantic checkpoint or when inspecting dynamic layout equations.

Pipelines and feature space
===========================

With a recognized affine scaler, Mlektic can convert learned coefficients to
original feature units. The estimator still predicts through its full pipeline.
For non-affine transforms, an exact raw-space coefficient equation may not
exist; use transformed-space interpretation and document the preprocessing.

Counterfactual prediction lessons
=================================

By default, supplied values are verified:

.. code-block:: python

   explanation = explain_lr_prediction(
       model, X, y,
       x_query=[[3.0]],
       yhat=model.predict([[3.0]])[0],
   )

For an intentional comparison value:

.. code-block:: python

   explanation = explain_lr_prediction(
       model, X, y,
       x_query=[[3.0]],
       yhat=0.0,
       prediction_source="provided",
   )

The subtitle and figure metadata identify the result as user-provided. Do not
use this mode to bypass estimator verification in ordinary explanations.

History subtitle visibility
===========================

The provenance and N/K subtitle is visible by default. It can be omitted in a
compact embedding without removing the underlying context:

.. code-block:: python

   fig = visualize_lr(
       model,
       X,
       y,
       show_history_context=False,
   )

The slider still identifies replay or interpolation and ``fig.layout.meta``
still contains the complete ``mlektic_history`` contract.

HTML size and dependencies
==========================

.. code-block:: python

   export_figure(
       fig,
       "lesson.html",
       include_plotly="inline",
       include_mathjax="cdn",
       responsive=False,
       auto_play=False,
   )

Inline Plotly increases file size but removes its network dependency. MathJax
CDN keeps equation rendering reliable but requires network access. A fully
self-contained MathJax export is not currently promised.

Neural recording
================

Create a ``TorchTrainingRecorder`` before training and call it at a documented
point in the loop (normally after the parameter update and metric calculation).
Record loss/metrics with consistent step or epoch coordinates. The recorder's
``record_every`` reduces captured semantic states and must not be confused with
later display sampling.

Performance guidance
====================

- reduce K/N before increasing 3D surface grid density;
- keep 2D trace topology stable for smooth interpolation;
- avoid large simultaneous multiclass surfaces;
- use the optimized notebook renderer in Colab when widget overhead dominates;
- export representative cases and inspect actual HTML size;
- retain motion unless a user explicitly selects a static/reduced-motion form.
