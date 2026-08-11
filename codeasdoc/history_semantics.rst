=====================================
History provenance and time semantics
=====================================

Why provenance matters
======================

An animated sequence is not automatically a training history. Mlektic accepts
already fitted Scikit-learn estimators, so it usually cannot observe the states
that existed during the original ``fit`` call. It uses explicit source labels
to avoid teaching a synthetic or reconstructed sequence as empirical history.

Sources
=======

``recorded``
   States captured during the actual training process. This term is reserved
   for recorder-based workflows, such as the neural training recorder.

``replayed``
   States reconstructed by cloning an incremental estimator, applying replay
   overrides where supported, fitting the clone, and calling ``partial_fit``.
   The original fitted estimator is not mutated. Because the reconstruction
   can differ from its final parameters, the public history reserves its last
   state for the exact supplied estimator and labels that origin ``fitted``.
   The preceding states remain explicitly reconstructed, not recorded.

``interpolated``
   Synthetic states between a documented baseline and the fitted model. The
   coordinate is alpha in ``[0, 1]``. Intermediate states are not optimizer
   updates.

For coefficient-bearing models, the synthetic parameter state is

.. math::

   \boldsymbol{\theta}(\alpha)
   =(1-\alpha)\boldsymbol{\theta}_{\mathrm{base}}
   +\alpha\boldsymbol{\theta}_{\mathrm{fitted}},
   \qquad 0\leq\alpha\leq1,

with the intercept treated in the same way. Predictions, geometry, and
empirical evaluation are recomputed from that same state. This construction is
used when the fitted estimator does not expose an observable training history;
it must not be interpreted as gradient descent. In particular,
``LinearRegression`` normally solves least squares directly rather than
exposing gradient updates.

Timeline quantities
===================

``T``
   Training updates reported or captured by the training process, when known.

``K``
   Semantic checkpoints constructed or recorded before display sampling.

``N``
   Checkpoints retained for display after sampling, with ``N <= K``.

``q``
   Perceptual intervals inserted between displayed checkpoints.

``F``
   Rendered frames. In a simple hybrid animation,
   ``F = (N - 1) q + 1``.

Only K and N describe semantic model states. Perceptual frames improve visual
continuity but are not additional training events.

Payload contract
================

.. code-block:: python

   history["metadata"]
   # {
   #   "schema_version": 2,
   #   "source": "replayed" | "interpolated",
   #   "source_detail": {...},
   #   "requested_mode": "auto",
   #   "resolved_mode": "iterative",
   #   "requested_steps": 100,
   #   "training_total_steps": 200,
   #   "captured_steps": 100,
   #   "displayed_steps": 30,
   #   "step_indices": array([...]),
   #   "displayed_step_indices": array([...]),
   #   "state_origins": array(["replayed", ..., "fitted_estimator"]),
   #   "displayed_state_origins": array([...]),
   #   "final_state_matches_estimator": True,
   #   "display_space": "original",
   #   "smoothing": {"method": "ema", "beta": 0.85},
   #   "decimation": {"max_frames": 30, "frame_step": 10},
   #   "warnings": [...],
   # }

``step_indices`` in metadata describes all K source coordinates.
``displayed_step_indices`` describes the retained N coordinates. The slider
uses retained source coordinates rather than renumbering them ``0..N-1``.
``state_origins`` prevents the exact fitted endpoint from being described as a
replay update; the corresponding slider label is ``fitted``.

Loss contract
=============

.. code-block:: python

   history["loss_raw"]
   history["loss_display"]
   history["loss_hist"]  # backward-compatible alias of loss_display

EMA never overwrites ``loss_raw``. It is applied to reconstructed replay
sequences when requested. Synthetic interpolation is already a smooth
mathematical path, so ``loss_display`` remains the raw empirical evaluation and
reaches the exact fitted endpoint. The smoothing metadata records the requested
method, applied method, beta, and the reason EMA was unnecessary when relevant.

With ``show_loss=True``, both replay and interpolation figures display this
quantity by default where the layout supports a curve. Labels distinguish
``Replay MSE``, ``Interpolation MSE``, replay/interpolation log-loss, and EMA
variants. The machine-readable ``loss_display_semantics`` states the quantity,
its role, and
``optimizer_loss=False``: these values are independently evaluated on the
supplied data, not claimed as a solver-private training objective.

Final-state comparison
======================

When coefficients can be extracted, Mlektic compares the final constructed
state with the supplied fitted estimator. In public replay histories, the last
state is the exact supplied estimator, so the animated equations, geometry,
probabilities, raw loss, and evaluation metrics finish at model truth.
``True`` means the parameters match within the documented numerical tolerance,
``False`` exposes an unexpected mismatch, and ``None`` means the comparison
could not be made.

This endpoint is not silently presented as another optimizer update. The
subtitle says ``Reconstructed replay + fitted endpoint``, its slider label is
``fitted``, and ``source_detail.endpoint_policy`` is
``"supplied_fitted_estimator"``. Hybrid subframes interpolate continuously from
the last retained replay state to this endpoint; those perceptual frames are
not additional training events.

Practical interpretation
========================

Before discussing convergence, ask:

1. What is the source?
2. How many semantic states existed and how many are displayed?
3. Is a slider state replayed, fitted, recorded, or interpolated?
4. Is the loss raw or smoothed for display?
5. Does the sequence end at the supplied fitted estimator?

These questions are part of model literacy, not implementation trivia.
