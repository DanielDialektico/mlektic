===================================
Animation and performance semantics
===================================

Five quantities must remain distinct:

* ``T`` — actual or estimator-reported training updates, when known;
* ``H`` — semantic checkpoints recorded or constructed (``K`` is the number of classes);
* ``N`` — checkpoints retained for display;
* ``q`` — perceptual intervals inserted between retained checkpoints;
* ``F`` — Plotly frames, which are not additional optimizer updates.

``steps`` controls H. ``max_frames`` bounds N by uniform endpoint-preserving
sampling. When ``max_frames=None``, ``frame_step`` applies a source-position
stride. Hybrid one-dimensional linear animation uses ``interpolation_frames``
or ``fps`` to improve motion between semantic checkpoints.

Playback speed
==============

``frame_duration`` is milliseconds per native frame. Larger values play more
slowly. ``transition_duration`` controls supported Plotly trace interpolation;
it does not create mathematical states or hide the model. Three-dimensional
traces retain redraw semantics.

Fluid playback
==============

Frames that change LaTeX annotations normally make Plotly redraw the whole
figure and MathJax re-typeset every equation, including the unchanged ones.
On a student laptop that can take most of a second per step, so Play falls far
behind ``frame_duration``.

Animated figures therefore carry a small browser controller
(``mlektic/visualization/assets/fluid_playback.js``). It does not change what
is drawn:

* the first time a frame is shown Plotly draws it normally, and the rendered
  equation nodes are kept;
* later visits move the traces without a full redraw and put back those exact
  nodes, so equations, positions, and fonts are pixel-identical;
* MathJax output is cached, and the equations of frames not shown yet are
  typeset one at a time while the page is idle, pausing whenever the student
  interacts, so a first visit only waits for Plotly's redraw.

The slider and Play/Pause stay Plotly's own controls. Plotly's ``execute: false``
attribute hands their commands to the controller while Plotly keeps drawing
them. Resizing, zooming, or toggling legend entries discards the kept nodes, and
they are rebuilt. Three-dimensional surfaces still rebuild their WebGL scene on
each step, but no longer re-typeset the equations.

The controller travels with the figure's HTML, so it works wherever Plotly emits
HTML: Google Colab, classic Jupyter, ``show_optimized``, and ``export_figure``.
Renderers that do not emit HTML, such as the JupyterLab and VS Code Plotly
mimetype renderers, keep Plotly's own playback; ``show_optimized(fig)`` gives
them the fluid path. Pass ``fluid_playback=False`` to ``show_optimized`` or
``export_figure`` to opt out.

Figure-owned controls
=====================

The dashboard starts with the complete data, model, objective, and metric
composition. Training figures preserve Play/Pause and the checkpoint slider.
Prediction explainers retain their distinct Input, Substitution, Output, and
Reset controls because those change the mathematical explanation rather than
duplicate the training timeline.

Inspect ``MOTION-FRAME-STEP`` and ``MOTION-NATIVE-SLOW`` in
``notebooks/qa/qa_04_motion.ipynb``.
