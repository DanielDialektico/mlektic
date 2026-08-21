========================================
Extensible neural-network execution maps
========================================

``notebooks/qa/qa_08_neural_structures.ipynb`` is the complete neural figure
gallery. Its assertion-free cases invoke every public Plotly figure route,
then exercise synthetic XOR and nonlinear regression, real Iris and breast
cancer tabular classification, real handwritten-digit CNN training, and the
structural block cases described below. The Scikit-learn datasets are bundled
locally and require no network download. Its short deterministic full-batch
runs create genuine recorder states for visual inspection; they are not
train/test benchmarks or estimates of generalization.

Mlektic has two complementary architecture views. The established
``architecture_mode="legacy"`` view remains the default and is optimized for
compact sequential lessons. The opt-in ``architecture_mode="blocks"`` view
captures the executed tensor graph and preserves branches, merges, repeated
module calls, multiple inputs, and multiple outputs.

.. code-block:: python

   from mlektic import inspect_nn, visualize_nn_blocks

   graph = inspect_nn(model, (left, right))
   figure = visualize_nn_blocks(
       model,
       (left, right),
       theme="academic",
       size="wide",
   )

Representation before rendering
===============================

The renderer does not inspect PyTorch directly. ``inspect_nn`` first creates a
versioned ``NeuralGraph`` intermediate representation containing:

* semantic nodes for model inputs, executed modules, tensor operations,
  parameters, outputs, and collapsed summaries;
* directed tensor edges with source and target ports, shape, dtype, device, and
  gradient requirements;
* module path, module type, call index, parameter and buffer shapes, stable
  public hyperparameters, and mathematical description per node;
* capture provenance and warnings.

This separation is the extension boundary. A future renderer, exporter, lesson
panel, or framework adapter can consume the same graph without duplicating
PyTorch introspection. ``graph.to_dict()`` provides a serializable form.
Rendered figures retain a compact audit summary under
``layout.meta["mlektic_neural_graph"]``.

Capture backends and truthfulness
=================================

``backend="auto"`` chooses the most informative safe route.

``torch.fx``
   Preserves functional operations such as ``add``, ``cat``, reshaping, and
   branches. Shape propagation describes the supplied example input. Static
   Python control flow is specialized to that capture.

``eager-hooks``
   Observes the module calls that actually execute for the supplied input. It
   supports data-dependent paths and primitives that FX cannot trace, but
   arbitrary functional tensor operations between modules may be represented
   as an explicit ``Uncaptured operation``. It never silently presents that
   partial graph as complete.

PyTorch attention and Transformer primitives are retained as semantic blocks
instead of exposing implementation-detail children. Integer inputs remain
integer, which is required for embeddings. Positional tuples and
``input_kwargs`` support multi-input signatures.

Mathematical coverage
=====================

Specialized roles and formulas cover dense and convolutional layers,
transposed convolution, common activations, normalization, embeddings,
recurrent layers, multi-head attention, Transformer blocks, pooling, dropout,
reshape operations, reductions, and common merge operations. Unsupported or
custom modules remain visible with a generic formula and
``math_status="generic"`` rather than receiving invented mathematics.

Projects can register an exact description without changing Mlektic internals:

.. code-block:: python

   from mlektic import register_neural_descriptor

   register_neural_descriptor(
       "MyResidualGate",
       role="merge",
       label="Residual gate",
       formula=r"\mathbf{y}=g\odot F(\mathbf{x})+\mathbf{x}",
   )

Registration changes semantic description only. It does not execute arbitrary
code, infer hidden behavior, or claim support for an unknown training rule.

Large networks
==============

``max_nodes`` bounds visual complexity, not capture fidelity. When the rendered
view is capped, middle operations become an explicit collapsed summary while
the complete graph remains available from ``inspect_nn``. For small graphs,
inline formulas alternate above and below successive nodes so adjacent
operations cannot share the same equation band; every node also retains its
formula, shapes, parameters, buffers, hyperparameters, and status in hover
data.
The dense neuron-and-edge replay has a separate ``max_neurons`` bound per
layer. The QA default displays up to eight representative neurons from both
ends of each wide dimension and derives marker diameter from the actual
pixel-space separation between adjacent visible nodes. It retains the complete
layer dimensions and declares the sampled counts and marker policy
under ``layout.meta["mlektic_neural_evolution"]["dense_scope"]``. This gives a
wide layer more visual presence without implying that eight glyphs are the
whole tensor. The final cases in ``qa_08_neural_structures.ipynb`` exercise a
network with hundreds of units per layer and many layers through both bounded
representations.

Convolutional models and dense replay
=====================================

``visualize_nn_blocks`` is the complete structural view for a convolutional
model. Convolution, normalization, activation, pooling, flattening, and the
dense classifier appear as distinct executed blocks with tensor shapes,
kernel/stride/padding hyperparameters, and specialized mathematics.

``visualize_nn_graph`` performs the same completeness check automatically. A
pure sequential dense network receives the animated neuron-and-edge lesson. If
that replay would omit an executed convolution, normalization, pooling,
Dropout, tensor operation, shared call, or branch, the request is routed to the
complete execution-block topology instead of rendering only the first and last
dense stages. ``layout.meta["mlektic_neural_graph_route"]`` records the requested
and rendered views and states that training animation was not applied. The
library never presents an incomplete classifier head as though it were the
whole network and never invents one dense edge per kernel application.

Recorded training semantics
===========================

``TorchTrainingRecorder`` history schema version 2 separates temporal facts
that older histories could conflate:

* parameters and buffers are captured independently;
* effective optimizer parameter groups are retained per frame;
* optimizer-state tensor norms are optional because full states can be large;
* every frame declares ``capture_phase`` and ``observation_phase``;
* replay restores recorded buffers as well as parameters without mutating the
  live model after the visualization call.

The recommended loop records after ``optimizer.step()``. In that arrangement,
the parameter state is post-step while loss, predictions, activations, and
gradients normally describe the preceding forward/backward observation. The
history states that distinction explicitly instead of implying simultaneity.
Training figures retain this non-geometric contract under
``layout.meta["mlektic_neural_history"]``.

Dense-graph section contract
============================

Every dense graph variant reserves invariant paper rows for the composed model,
retained parameter snapshot, forward phase, optional backpropagation equation,
heatmap legend, activity glow, optional update halo, and optional scope
disclosure. The invisible graph axis reaches every header row used by animated
parameter and step readouts, while node and edge content retains a separately
bounded network domain. Adding a loss/update panel therefore cannot clip the
evolving parameter snapshot, move ``t`` under the node legend, or put the
forward equation beside heatmap notation. Backpropagation owns its own row so
its MathJax box cannot enter the right colorbar at notebook width.

Update-aware views additionally reserve a 0.065 paper-coordinate clearance
between the update-halo explanation and the center of the uppermost node. This
clearance remains invariant when a loss or diagnostics panel changes the graph
subplot domain.

Making small parameter updates visible
=======================================

The animated mathematical graph keeps ``evolution_mode="absolute"`` as its
backward-compatible default. Absolute edge color answers *what value does this
weight or forward signal have?* Small optimizer updates can be numerically real
but almost invisible on that full-history scale, so two opt-in modes answer a
different question:

``evolution_mode="hybrid"``
   Preserves the absolute weight/signal edge and adds a signed update halo.

``evolution_mode="updates"``
   Uses a neutral edge as context and makes the signed update halo primary.

For a displayed checkpoint :math:`t`, the halo represents

.. math::

   \Delta\theta_t = \theta_t - \theta_{\mathrm{ref}}.

``update_reference="previous"`` uses the previous *displayed recorded
checkpoint*. This distinction matters when ``max_frames`` decimates a longer
history: the difference may aggregate several optimizer steps.
``update_reference="initial"`` instead shows cumulative displacement from the
first retained training state.

Halo color encodes update sign; width and opacity encode magnitude.
``update_scale="global"`` is the default because one scale is retained across
the animation, so magnitudes remain comparable over time. ``"frame"`` is an
explicit contrast mode: every frame is normalized independently and its color
intensity must not be compared with another frame. ``top_k_updates`` can
de-emphasize all but the largest *visible edge* updates; bias changes remain in
the numerical summary even though biases are not graph edges.

Every dense mathematical graph also includes a forward-activity glow based on
:math:`s_{ji}^{(\ell)}=\theta_{ji}^{(\ell)}a_i^{(\ell-1)}`. Its signed color
and globally scaled thickness/opacity make active computational paths visible
even when absolute weights change only slightly. This activity layer is present
in ``absolute``, ``updates``, and ``hybrid`` modes. It is distinct from the
update halo: activity answers *which paths carry the current forward signal?*;
the update halo answers *which parameters changed relative to the reference?*

The optional update panel reports :math:`\lVert\Theta_t\rVert_2`,
:math:`\lVert\Delta\Theta_t\rVert_2`, their relative ratio, the recorded
gradient norm, and directional agreement with :math:`-\nabla_\Theta\mathcal L`.
That last cosine is shown only when the reference is the immediately adjacent
recorded checkpoint; decimated and cumulative changes are not mislabeled as a
single optimizer update. The diagnostic panel occupies a dedicated band below
the semantic slider, leaving the network and its equations unobstructed.

Recorded backpropagation edges are independently opt-in through
``show_backpropagation=True``. The default is ``False`` because the overlay
adds one animated dotted trace per visible edge and can materially increase
browser rendering work. Disabling it does not remove forward activity, weight
evolution, update halos, or the optional loss curve. The QA gallery contains
otherwise equivalent cases with and without the overlay for direct playback
comparison.

.. code-block:: python

   figure = visualize_nn_graph(
       model,
       input_sample,
       history,
       evolution_mode="hybrid",
       update_reference="previous",
       update_scale="global",
       top_k_updates=8,
       interpolation_frames=3,
   )

``interpolation_frames`` inserts linearly interpolated parameter states and
recomputes their forward activations. These frames improve perceived motion;
they do not invent gradients or optimizer steps. When ``show_loss_panel=True``,
the loss curve contains only recorded objective values. A hollow marker moves
between them during perceptual subframes, but metadata explicitly declares
that those intermediate marker positions are not fresh loss evaluations. All
perceptual states are identified in the figure, excluded from the semantic
slider, and declared under ``layout.meta["mlektic_neural_evolution"]``.

Loss geometry and backpropagation
=================================

``visualize_nn_loss_landscape`` evaluates the supplied loss function on a real
affine two-direction section through the final recorded parameter state. When
the retained path spans two independent directions, history PCA defines the
plane; otherwise a deterministic orthogonal complement is used. The rendered
training trajectory is a projection onto that plane. The subtitle and metadata
state both facts: this is an exact batch-loss slice, not the complete objective
landscape in the network's high-dimensional parameter space.

The initial loss-slice state contains one checkpoint marker and no pre-drawn
optimization path. Playback reveals the projected path progressively. The
final recorded objective appears only on the final frame in a 15 px
paper-coordinate annotation, so camera perspective cannot shrink it and the
surface cannot occlude it. The checkpoint marker receives a disclosed, tiny
visual z-offset to prevent WebGL depth sorting from hiding a point that lies
exactly on the surface; the path and every reported loss remain exact. This
identifies the retained endpoint without claiming global convergence.

``visualize_nn_backpropagation`` complements the forward equation with the
dense-layer chain rule. Its backward line widths use genuine recorded
per-layer parameter-gradient norms on one global scale. Every displayed layer
also reports its gradient norm, adjacent update norm, relative update norm,
the current objective, and its adjacent change. Playback is deliberately slow
enough to read these quantities. It does not claim that an optimizer update
always equals :math:`-\eta\nabla\mathcal L`; that identity is exact for plain
SGD, while momentum and adaptive optimizers also depend on optimizer state.

``explain_nn_prediction`` is the numerical use-of-model view. It separates the
query input, the reusable forward rule, bounded per-layer substitutions, and
the numerical output. When recorded BCE or cross-entropy semantics identify a
classification task, it also displays the winning class; regression outputs
remain numeric without inventing a decision rule.

Prediction and training replay are deliberately separate. The default
``parameter_state="final"`` uses the fitted state and provides Input,
Substitution, Output, and Reset controls. Substitution is numerical: it shows
actual fitted coefficients multiplying the selected input values, followed by
the resulting first-unit pre-activation. This answers *how is the resulting
function used for this query?* It has no Play/Pause or checkpoint slider.

The initial prediction state is Reset. Each stage reveals a cumulative,
fixed paper-coordinate region; MathJax content does not determine card size.
Summary vectors use two representative endpoints and at most three decimals.
The detailed derivation applies a precision-aware coordinate limit and aligns
every layer's symbolic rule and computed vector inside an independently
reserved formula block. Multi-row ``Linear`` expansions cannot enter the
following activation block. These are display limits only: forward execution
and the reported values remain exact.

``parameter_state="training_replay"`` is an independent training-observation
figure. It answers the different question *how did the retained parameters and
signals evolve across checkpoints?* Its Play/Pause controls and slider are
explicitly training-time mechanisms. It has no Input, Substitution, or Output
stages, no prediction cards, and no repeated final prediction. The two control
systems are never combined.

Dropout and stochastic state
============================

Dropout is visible as a regularization block in the complete execution graph.
A ``view="graph"`` request for a model containing Dropout is routed to that
complete topology because recorder schema version 2 does not retain historical
random masks. The library therefore does not remove or brighten dense edges as
though it knew which units were dropped at an earlier checkpoint. In a pure
dense replay, activity glow always represents
:math:`\theta_{ji}^{(\ell)}a_i^{(\ell-1)}`, never dropout probability.

Mathematical typography
=======================

Neural public figures accept ``math_font_scale`` between ``0.75`` and ``2.0``.
The default ``1.0`` preserves existing compositions. Plotly viewport zoom does
not resize paper-anchored equations, so this explicit control is the reliable
way to enlarge LaTeX in notebooks or classrooms. Shared defaults now keep
essential equations at readable sizes; large networks are summarized through
bounded tensors and semantic blocks rather than globally shrinking text.

The neural QA gallery applies every supported figure family to the same trained
network with hundreds of units and many layers: architecture, complete
execution topology, prediction substitutions, performance, parameters,
activations, backpropagation, and a reduced-grid loss slice.

Current boundary
================

The graph foundation is designed to accept new blocks, but it does not promise
exact symbolic mathematics for every program that PyTorch can execute. Custom
autograd functions, opaque C++/CUDA extensions, distributed wrappers,
quantization internals, arbitrary Python side effects, data-dependent loops,
and compiler-generated graphs may require a future adapter or a user-supplied
descriptor. Capturing a representative input describes that execution path;
it is not a proof that every possible input follows the same path.

No external diagram service is required. The renderer is native Plotly, works
with the existing theme/format/size system, and keeps the legacy neural figure
as the compatibility default.
