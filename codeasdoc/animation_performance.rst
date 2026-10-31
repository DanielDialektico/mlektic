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

Figure-owned controls
=====================

The dashboard starts with the complete data, model, objective, and metric
composition. Training figures preserve Play/Pause and the checkpoint slider.
Prediction explainers retain their distinct Input, Substitution, Output, and
Reset controls because those change the mathematical explanation rather than
duplicate the training timeline.

Inspect ``MOTION-FRAME-STEP`` and ``MOTION-NATIVE-SLOW`` in
``notebooks/qa/qa_04_motion.ipynb``.
