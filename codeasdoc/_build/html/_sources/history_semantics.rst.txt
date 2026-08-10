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
   The original fitted estimator is not mutated. Replay results can differ from
   its final parameters.

``interpolated``
   Synthetic states between a documented baseline and the fitted model. The
   coordinate is alpha in ``[0, 1]``. Intermediate states are not optimizer
   updates.

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
   #   "schema_version": 1,
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
   #   "final_state_matches_estimator": False,
   #   "display_space": "original",
   #   "smoothing": {"method": "ema", "beta": 0.85},
   #   "decimation": {"max_frames": 30, "frame_step": 10},
   #   "warnings": [...],
   # }

``step_indices`` in metadata describes all K source coordinates.
``displayed_step_indices`` describes the retained N coordinates. The slider
uses retained source coordinates rather than renumbering them ``0..N-1``.

Loss contract
=============

.. code-block:: python

   history["loss_raw"]
   history["loss_display"]
   history["loss_hist"]  # backward-compatible alias of loss_display

EMA never overwrites ``loss_raw``. A visible loss metric and the plotted loss
curve use ``loss_display`` so they do not disagree. Other evaluation metrics
are computed from checkpoint predictions.

Final-state comparison
======================

When coefficients can be extracted, Mlektic compares the final constructed
state with the supplied fitted estimator. ``True`` means the parameters match
within the documented numerical tolerance, ``False`` exposes a mismatch, and
``None`` means the comparison could not be made. A fitted state is never
silently appended to a replay as though it were another update.

Practical interpretation
========================

Before discussing convergence, ask:

1. What is the source?
2. How many checkpoints existed and how many are displayed?
3. Are the slider coordinates updates, replay checkpoints, or alpha values?
4. Is the loss raw or smoothed for display?
5. Does the final replay match the fitted estimator?

These questions are part of model literacy, not implementation trivia.
