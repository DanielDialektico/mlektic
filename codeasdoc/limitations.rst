===============================
Limitations and troubleshooting
===============================
CDN may show LaTeX source until MathJax becomes available. See :doc:`export`.
Scikit-learn history
====================

A fitted Scikit-learn estimator normally does not retain its original optimizer
trajectory. Replay and interpolation are pedagogical constructions with explicit
provenance. Use ``TorchTrainingRecorder`` when genuine recorded neural history is
required.

Large equations and models
==========================

High-dimensional views summarize contributions and use ellipses. The complete
values remain in estimator state and figure metadata. Use ``feature_names``,
``max_theta_cols``, neural matrix limits, ``size="wide"``, or a separate report;
do not shrink text until it becomes unreadable. Inspect ``LR-MANY-FEATURES`` and
``LOG-MULTI-MANY-CLASSES``.

Rendering and performance
=========================

Reduce ``max_frames`` before reducing mathematical fidelity. Three-dimensional
Plotly scenes redraw per frame and can be more expensive than two-dimensional
traces. ``frame_duration`` changes speed, not checkpoint count. If a lesson view
shows only data after re-execution, select its Model or Complete stage.

Export
======

Inline Plotly does not imply inline MathJax. A browser that blocks the configured
CDN may show LaTeX source until MathJax becomes available. See :doc:`export`.
