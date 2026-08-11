===================================
Animation and performance semantics
===================================

Five quantities must remain distinct:

* ``T`` — actual or estimator-reported training updates, when known;
* ``K`` — semantic checkpoints recorded or constructed;
* ``N`` — checkpoints retained for display;
* ``q`` — perceptual intervals inserted between retained checkpoints;
* ``F`` — Plotly frames, which are not additional optimizer updates.

``steps`` controls K. ``max_frames`` bounds N by uniform endpoint-preserving
sampling. When ``max_frames=None``, ``frame_step`` applies a source-position
stride. Hybrid one-dimensional linear animation uses ``interpolation_frames``
or ``fps`` to improve motion between semantic checkpoints.

Playback speed
==============

``frame_duration`` is milliseconds per native frame. Larger values play more
slowly. ``transition_duration`` controls supported Plotly trace interpolation;
it does not create mathematical states or hide the model. Three-dimensional
traces retain redraw semantics.

Lesson format
=============

Re-executing a ``format="lesson"`` cell intentionally resets the composition to
stage ``1 Data``. Choose ``2 Model`` or ``4 Complete`` to reveal the model before
pressing Play. This is visibility state, not a lost fitted line.

Inspect ``MOTION-FRAME-STEP`` and ``MOTION-LESSON`` in
``notebooks/qa/qa_04_motion.ipynb``.
