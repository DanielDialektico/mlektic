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

Neural program coverage
=======================

The block view captures an execution path, not every path a dynamic PyTorch
program may take. FX preserves supported functional operations and static
branches; eager hooks preserve actually executed module calls but may expose an
``Uncaptured operation`` for arbitrary tensor code. Custom autograd,
distributed or quantized wrappers, opaque native extensions, and compiler
internals do not yet have guaranteed exact semantic expansion. Generic blocks
remain visible and are explicitly labeled instead of receiving speculative
formulas. Large renderings collapse middle nodes only after the complete
intermediate graph has been captured. See :doc:`neural_execution_graphs`.

Export
======

Inline Plotly does not imply inline MathJax. A browser that blocks the configured
CDN may show LaTeX source until MathJax becomes available. See :doc:`export`.
