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

Step by step
============

Every animated figure has a **Next step** button beside Play/Pause. A click
stops playback and shows the next frame, so a student can read each change
before moving on; after the last frame it returns to the first. The slider
follows every step.

Where each frame carries many equations (multiclass models with several
classes and inputs, whose frames hold at least
``HEAVY_FRAME_TEX_CHARS`` LaTeX characters), continuous playback is slow and
hard to follow, so **Next step** leads: it takes Play's place and Play/Pause
sit below it (or after it, when the margin holds one row only). In every other
figure Play leads, and below it a **Step** button mirrors Play: the same label
style (``⏭  Step`` under ``▶  Play``), font, height, and left edge. Where the
controls must share Play's row, a slightly smaller **Next step** follows
Pause instead.

The controls share one left edge in the top margin, just above the plotting
area, and never cover an annotation: when an equation sits there they share
one row, or keep the place the figure gave Play (beside its slider, for
example). ``layout.meta["mlektic_visual"]["step_control"]`` records whether the
button leads, where it went, and the frame's LaTeX size.

Plotly has no command for "the next frame", so the button uses Plotly's
``"skip"`` method and the fluid controller acts on the click. Where the
controller cannot run (``fluid_playback=False``, or a mimetype renderer such
as JupyterLab's or VS Code's Plotly renderer) the button is left out rather
than shown without effect.

Raster frames and payload size
==============================

Heat maps and contours cannot move with a redraw-free animate: their image is
only rebuilt by a replot. The controller (version 2) replots those frames with
MathJax switched off and puts the kept equation nodes back, as it already did
for WebGL traces. It also decodes base64 typed arrays inside frames, which
Plotly.js only decodes in ``fig.data``.

Builders keep animated payloads small without depending on that decoder:

* a frame lists only the traces it changes (``traces=[...]``);
* raster fields are quantized to integers in ``[0, 255]`` on one global range
  and sent as plain lists in frames, while the first state travels as a compact
  ``uint8`` typed array;
* positions that move between stages are tweened by Plotly transitions rather
  than extra frames (the representation view has one frame per stage);
* edges are grouped into a fixed number of colour/width bins, so the trace
  count does not grow with the number of edges.

``export_figure`` and ``show_optimized`` also write Plotly's slash escape in its
two-character JSON form, which shortens typed-array payloads by about a third.

Several copies of Plotly.js on one page
=======================================

Notebook front ends that put every output in one page, such as VS Code and
classic Jupyter, load a copy of Plotly.js with each figure, and ``window.Plotly``
is the last one loaded. When that copy redraws a figure that another copy drew,
which 3-D, heatmap, and contour frames require, it gives the figure new event
emitters. The controller notices the change after each step and carries every
listener over: its own, so the slider and Play/Pause keep working, and Plotly's,
so the slider keeps following the animation. A slider move made while the
listeners were missing is applied afterwards.

Figure-owned controls
=====================

The dashboard starts with the complete data, model, objective, and metric
composition. Training figures preserve Play/Pause and the checkpoint slider.
Prediction explainers retain their distinct Input, Substitution, Output, and
Reset controls because those change the mathematical explanation rather than
duplicate the training timeline.

Inspect ``MOTION-FRAME-STEP`` and ``MOTION-NATIVE-SLOW`` in
``notebooks/qa/qa_04_motion.ipynb``.
