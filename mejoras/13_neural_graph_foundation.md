# Neural execution-graph foundation — implementation record

## Product decision

Mlektic now owns its neural diagram model instead of integrating an external
diagram generator. The public Plotly renderer is one consumer of a versioned,
renderer-independent intermediate representation. This keeps the library
offline-capable, theme-compatible, inspectable, and open to future renderers.

The established sequential architecture figure remains the default. The new
block graph is opt-in through `architecture_mode="blocks"`, `view="blocks"`, or
`visualize_nn_blocks()`. Existing figures, sizes, motion, and notebook calls do
not change.

## Implemented architecture

### Versioned intermediate representation

`NeuralGraph` contains typed nodes, directed port-aware edges, tensor metadata,
parameter and buffer specifications, model hyperparameters, semantic formulas,
mathematical coverage status, capture provenance, warnings, and a schema
version. It can be serialized with `to_dict()`.

The representation is intentionally independent from Plotly and from the
capture backend. New framework adapters or renderers can target this contract.

### Dual PyTorch capture

- `torch.fx` is preferred for user-defined composed models because it preserves
  supported functional operations, branches, merges, and repeated calls.
- eager forward hooks are the transparent fallback for dynamic execution and
  trace-resistant modules.
- compound PyTorch primitives such as multi-head attention and Transformer
  blocks remain a single semantic block rather than exposing implementation
  children.
- integer inputs, nested outputs, multiple positional inputs, and keyword
  inputs retain their native tensor semantics.

FX capture is exact for the supplied example path and shape propagation, not a
proof over all possible program paths. Hook capture reports that functional
operations may be partial and inserts explicit uncaptured-operation nodes when
needed.

### Semantic registry

The registry supplies roles, labels, formulas, and coverage status for dense,
convolutional, transposed-convolution, activation, normalization, embedding,
recurrent, attention, Transformer, pooling, dropout, reshape, reduction, and
merge operations. Unknown modules receive a visible generic description.

`register_neural_descriptor()` is the public extension seam for custom modules
or operations. It extends presentation semantics without pretending to inspect
an unknown implementation.

### Native block renderer

The Plotly renderer uses topological columns, branch-aware edges, role-specific
shape/color encodings, tensor and parameter hover details, optional formulas,
and provenance in the subtitle. `max_nodes` collapses only the rendered middle;
the captured graph remains complete.

### Recorder schema version 2

The recorder now retains parameter and buffer snapshots independently,
effective optimizer groups per frame, optional optimizer-state tensor norms,
and explicit capture/observation phases. Historical replay restores both
parameters and buffers and restores the live model after evaluation.

This resolves an important temporal ambiguity in the standard loop: after-step
parameters commonly coexist with loss, activations, predictions, and gradients
observed before that step. The history declares both facts.

## Verified configurations

Automated tests cover residual branches, functional addition, shared modules,
integer embedding inputs, convolution and pooling, recurrent multiple outputs,
multi-head attention with three tensor arguments, Siamese multi-input and
multi-output graphs, custom descriptor registration, dynamic fallback,
large-graph collapse, backward-compatible legacy rendering, recorder temporal
metadata, optimizer groups/state, buffer evolution, and buffer-aware replay.

The exhaustive 46-case human gallery belongs in
`notebooks/qa/qa_08_neural_structures.ipynb`; the student-oriented introduction belongs in
`notebooks/learn/learn_04_neural_architectures.ipynb`.

## Deliberate boundaries

Exact semantic expansion is not guaranteed for arbitrary custom autograd
functions, C++/CUDA extensions, distributed wrappers, quantization internals,
compiler-generated graphs, Python side effects, or all data-dependent loops.
The architecture supports future adapters for those cases without changing the
public graph contract.

The dense mathematical graph now offers opt-in update-first and hybrid
animation. It visualizes exact recorded parameter differences, forward values,
and retained gradients for supported dense stages. It still does not animate
arbitrary internal tensors for every execution-graph block. Joining general
graph-node identities to recorder tensors remains future work and requires safe
retention/display policies for large activations.

## Dynamic parameter-evolution decision

The compatibility default remains `evolution_mode="absolute"`. Two opt-in modes
make small but real optimizer changes legible:

- `hybrid` keeps the absolute edge encoding and adds a signed update halo;
- `updates` uses neutral structural edges and makes the halo primary.

The halo is the actual `delta_theta = theta_current - theta_reference`; sign is
encoded by color and magnitude by width plus opacity. `update_reference` can be
the previous displayed recorded checkpoint or the initial state. The global
scale is the rigorous default because it is comparable over time; frame-level
normalization is explicitly labeled as a contrast-only view. Optional top-k
filtering affects only visual emphasis, never retained values or summary norms.

The numerical panel includes parameter norm, update norm, relative update,
gradient norm, and update/negative-gradient cosine. The cosine is deliberately
withheld when checkpoint decimation or an initial-state reference makes the
displayed delta an aggregate rather than one adjacent recorded update.
The panel uses a dedicated lower plotting band beneath the semantic timeline;
it does not compete with the equations or network geometry and its values can
still update through data traces without forcing frame redraws.

`interpolation_frames` linearly interpolates parameters solely for motion and
recomputes forward activations at those intermediate states. It never invents
optimizer steps or gradients: interpolated states are visibly labeled, omitted
from the semantic slider, and recorded as perceptual in figure metadata.

Neural graph traces are exempt from generic data-marker recoloring in the
additive visual-theme layer. Themes may change surrounding typography,
controls, and canvas treatment, but they do not replace activation, signal,
weight, update, or gradient encodings. This is especially important for the
`accessible` theme: mathematical color scales remain data-bearing rather than
becoming uniform decorative outlines.

All dense graphs now share a forward-activity glow. Its source is the exact
displayed signal `s_ji = theta_ji * a_i`; sign controls hue while globally
scaled magnitude controls thickness and opacity. This is intentionally separate
from the parameter-update halo so students can distinguish computational
activity from learning. The graph's vertical extent is capped below the legend
band to prevent nodes from covering these definitions.

## Permanent documentation and notebook rule

Every new or materially changed public neural documentation claim must add a
real assertion-free human inspection cell to its corresponding canonical
notebook. Unit tests validate machine invariants; visual notebooks validate
readability, hierarchy, hover content, and layout under representative models.

## Validation evidence

The completed implementation passed:

- 163 automated tests, including focused execution-graph, recorder, and dynamic
  parameter-evolution tests;
- Ruff across source, tests, scripts, and Sphinx sources;
- strict clean-environment Sphinx HTML generation with warnings as errors;
- the documentation/notebook policy with 143 cases and 23 public-page mappings;
- in-memory execution of all 14 canonical QA and student notebooks without
  modifying committed outputs;
- source and wheel package construction;
- raster inspection of the residual diagram, which led to semantic-color,
  stable-operation-name, and visible skip-path refinements.
- raster inspection of the hybrid update graph, which led to dedicated header,
  diagnostic, legend, and network bands plus stronger but bounded update halos.

The only notebook-execution warning was the established Windows ZMQ selector
thread notice; it did not fail or alter any notebook cell.

## Post-inspection spacing correction

Human inspection at notebook width showed that the three-line node caption and
the LaTeX annotation shared the same vertical band. Inline formulas now anchor
to the node with a fixed pixel offset below the complete caption, so resizing
does not collapse the gap. If a topological column contains more than three
parallel nodes, inline formulas are deliberately suppressed and remain in hover
content; a visible notice explains the decision. This density fallback avoids
moving equations onto adjacent branches.

Plotly hover cards do not run MathJax. They therefore present a dedicated
readable Unicode/plain-text form rather than the raw LaTeX retained by the graph
contract and rendered in the equation layer. Long formulas, parameter lists,
and configurations use explicit line breaks. Eager output nodes now expose the
observed tensor specifications as both consumed and displayed output metadata,
which removes the misleading unknown-shape marker for tuple outputs such as
multi-head attention.

## Objective, reverse-mode, stochastic-state, and scale decisions

The evolving dense graph now accepts `show_loss_panel=True`. The panel uses the
loss values actually supplied to `TorchTrainingRecorder.record()` and names the
configured loss class. Perceptual parameter subframes move a hollow marker by
visual interpolation only; they are not new objective evaluations. This keeps
smooth motion without converting visual frames into false empirical evidence.

`visualize_nn_loss_landscape()` evaluates the supplied model, batch, targets,
and loss function across an affine two-direction parameter plane centered at
the final captured state. History PCA selects two directions when possible; a
deterministic orthogonal complement handles rank-one paths. The surface is an
exact section and the recorded path is projected onto it. Both qualifications
are visible and stored in metadata.

`visualize_nn_backpropagation()` separates pedagogy from the compact graph
overlay. It presents the canonical dense-layer chain rule, globally scales
backward line widths by recorded per-layer parameter-gradient norms, and adds a
separate optimizer-update lane based on adjacent recorded parameter changes.
The current recorded loss is displayed at every step. It also states that
adaptive optimizer updates are not generally `-eta * gradient`.

The view now exposes the numerical result of the reverse phase for every
visible layer: gradient L2 norm, adjacent update L2 norm, relative update norm,
current loss, and adjacent loss change. Line width remains redundant visual
encoding rather than the only explanation. Default playback is 900 ms per
recorded checkpoint, and the QA lesson uses 1100 ms.

Per-edge gradient overlays in the dense graph are controlled by
`show_backpropagation`, which defaults to `False`. They add one animated dotted
trace per visible edge and therefore can materially increase Plotly rendering
work. This switch is independent from the forward-activity glow, update halos,
loss panel, and numerical update panel. The QA notebook includes matched cases
with the overlay disabled and enabled.

Dropout is represented structurally by the execution graph, but recorder v2
does not retain the stochastic mask or RNG state for each observation. A graph
request containing Dropout is therefore routed to the complete structural view
instead of pretending to reproduce historical masks in a dense replay.
Activity glow in eligible pure-dense replays remains the exact displayed
forward signal `theta_ji * a_i`.

Paper-anchored MathJax annotations do not scale with Plotly viewport zoom.
Neural APIs therefore expose additive `math_font_scale` in the range 0.75 to
2.0, with 1.0 preserving every established default. Parameter-evolution rows
were compacted and enlarged, and a bounded display now reports how many
intermediate tensors were omitted rather than rendering a disconnected
vertical ellipsis.

Large-network rendering follows a semantic-level-of-detail policy. A new QA
suite uses hundreds of units per layer and many layers. `max_nodes` bounds the
rendered middle while the captured graph, dimensions, parameters, buffers, and
hyperparameters remain available. The same trained fixture is exercised
through architecture, complete execution topology, forward substitution,
metrics, weights, activations, backpropagation, and loss-slice views. The
library deliberately avoids an all-neuron/all-edge view whose complexity grows
multiplicatively and becomes both unreadable and expensive.

The time-aware forward explanation now uses an explicit Input / Substitution /
Output contract. It preserves exact bounded layer computations and shows a
winning class only when recorded BCE or cross-entropy semantics justify that
decision rule.

The fitted-model explanation and training-time replay are now separate public
states. `parameter_state="final"` is the default and exposes staged Input,
Substitution, Output, and Reset controls over the final retained parameters.
`parameter_state="training_replay"` retains the historical Play/Pause and
checkpoint slider. This prevents a fitted prediction lesson from implying that
the model is still changing while it is being used.

### Post-inspection decision: prediction and training are not hybrid views

Human QA showed that combining prediction semantics with checkpoint animation
made the lesson ambiguous and caused summary cards to compete with evolving
equations. The final contract is strict:

- the fitted-prediction figure has Input, Substitution, Output, and Reset only;
- Substitution contains an actual numerical operation using the query and the
  fitted parameters, rather than only the generic layer identity;
- the training-replay figure has Play/Pause and a checkpoint slider only;
- replay has no fixed-query/result cards and no repeated final prediction;
- its title describes parameter and signal evolution rather than presenting a
  fixed query as the lesson objective;
- line-count-aware vertical allocation separates multi-row dense equations
  from the following activation equation.

Summary-card containment and graph legends follow the same rendered-content
rule. The Substitution equation carries an invisible MathJax width reserve so
its border is derived from the final typeset expression rather than Plotly's
earlier intrinsic estimate. In the dense graph, the Node/Edge heatmap and
Activity glow definitions use independent baselines with extra clearance for
stacked fractions; their separation is an asserted layout invariant.

The QA notebook keeps these as two independent cases:
`NN-TRAINING-QUERY-REPLAY` and `NN-GALLERY-FORWARD-SUBSTITUTION`.

### Deep-architecture connector geometry

Visual inspection of the eight-node collapsed architecture exposed a geometry
bug: semantic modules used fixed widths while their centers moved closer as
more modules were displayed. The old connector line and a separate arrow glyph
could consequently enter a module or appear on different sides of its border.

The architecture renderer now derives one density scale from the horizontal
module gap, applies it consistently to semantic shapes and hover targets, and
reserves a minimum connector corridor. Each connector is a single arrow whose
start and end are calculated from the actual adjacent shape boundaries. The
large 128-to-512 QA fixture is the canonical human visual check for this rule.

### Post-inspection decision: bounded prediction summaries

Large fitted networks can have hundreds of input coordinates and multiple
output classes. Rendering the full vectors in the three prediction summary
cards made Input, Substitution, and Output compete for the same horizontal
space. The summary band now has a separate density contract from the detailed
forward derivation:

- the initial render is Reset, so no prediction card is pre-populated before a
  learner selects a stage;
- Input, Substitution, and Output own fixed, non-overlapping paper-coordinate
  shapes. MathJax annotations provide content only and can no longer resize a
  border after a relayout button is clicked;
- Input and Output show at most two representative endpoints, use at most three
  decimals in the summary band, and preserve explicit ellipses for omitted
  coordinates;
- Substitution uses at most three rows: the first fitted coefficient-value
  term, an omitted/last term row when needed, and the fitted bias plus computed
  first-unit result;
- the Output card is centered inside a wider right-hand region;
- Output aligns the numeric result and decision at the same equals-sign column;
- the detached red duplicate of the final output is removed because Output is
  already the authoritative fitted prediction;
- the model formula, summary cards, and detailed derivation occupy independent
  vertical bands, so controls cannot drift into the title on tall figures;
- detailed layer equations retain exact computed values but apply a
  precision-aware inline width budget. Each symbolic definition and computed
  vector remains aligned inside its own layer block, and ellipsis discloses
  presentation truncation for wide tensors.

This is presentation-level truncation only. Shapes, winning-class computation,
the complete model execution, and the values used by prediction remain exact.

Detailed forward mathematics uses a row-aware vertical allocator. A block's
anchor advances by the number of MathJax baselines it renders and then by a
separate inter-layer corridor, so a multi-row Linear expansion cannot touch the
following activation. Training replay uses the otherwise empty upper band.
Four-layer lessons remain at 14 points; the 13-point fallback is limited to
layer selections whose rendered rows cannot fit the fixed derivation band. A
single expanded Linear block is capped at four rendered rows; larger layers use
the exact symbolic matrix equation and computed output vector instead of
consuming the corridor assigned to later layers.

### Post-inspection decision: graph section rows

Every dense graph variant uses the same paper-space teaching rows: composed
model, retained parameter snapshot, forward phase, optional backpropagation
phase, heatmap legend, activity glow, optional update halo, and optional scope
disclosure. Parameter and training-step readouts are animated traces on an
invisible graph axis that reaches every header row; node and edge content keeps
its independently bounded network domain below 0.92. A lower panel therefore
cannot clip the evolving ``Theta_t`` snapshot, move ``t`` beneath the node
legend, or place Feed forward beside the heatmap definitions.

Update-aware graphs also enforce a 0.065 paper-coordinate minimum between the
update-halo row and the center of the uppermost graph node. The network ceiling
is derived from this constraint after the final subplot domain is known, so a
loss or diagnostics panel cannot collapse the clearance.

Backpropagation uses its own row instead of extending the forward MathJax box.
This prevents the equation from entering the right colorbar at notebook width
without shrinking the essential mathematics.

### Post-inspection decision: progressive loss-slice trajectory

Plotly ``Scatter3d`` text is rendered in scene depth. As the animated marker
moved across the loss surface, perspective and surface occlusion made an
unchanged label appear progressively smaller or disappear. The initial frame
now contains only the first checkpoint marker: it does not reveal a path that
has not been played and it does not pre-populate a checkpoint card. Frames
reveal the projected trajectory cumulatively. The final checkpoint alone adds
a fixed 15 px paper-coordinate annotation with its loss, so the surface cannot
occlude it. A small metadata-disclosed visual z-offset keeps the endpoint
marker above WebGL's surface depth test; the path and reported loss use the
exact evaluated coordinate. The figure therefore identifies the retained
endpoint without claiming global convergence.

## Post-inspection density, CNN scope, and typography decisions

Human review established that five sampled neurons per wide dense layer did
not communicate architectural scale well enough. Pure dense graph cases now
request up to eight representative neurons per layer, but marker diameter is
not a fixed count-based constant: it is derived from the narrowest actual
pixel-space separation after graph and panel domains are known. The policy
leaves a visible gap between adjacent markers and is stored in `dense_scope`
metadata with the complete tensor dimensions.

Graph truthfulness takes priority over applying the same renderer to every
model. A `visualize_nn_graph()` request is routed to the complete execution
block view whenever dense replay would omit convolution, normalization,
pooling, Dropout, tensor operations, shared calls, or branches. The route is
declared in `mlektic_neural_graph_route`; a CNN can no longer appear as only its
first and last dense columns. Formulas in the complete execution view alternate
above and below successive blocks so neighboring operations such as Conv2d and
BatchNorm2d cannot share a notation band.

Recorder v2 still does not retain historical Dropout masks. Routing Dropout
models to the complete topology makes the regularization stage visible without
pretending that arbitrary edges are historically active or inactive.
Builder-reserved 12 px Play/Pause controls and 11 px numeric colorbar ticks are
preserved under every additive theme, including `classroom`. The
multi-head-attention formula retains a valid separator before `head_i`, and
tall matrix anchors retain explicit separation from their bias vectors.

### Post-inspection decision: invariant prediction and replay formula sections

Prediction cards use fixed, disjoint paper regions. Their borders never depend
on post-render MathJax measurement. The Substitution preview has a bounded
three-row contract (first product, omitted/last product, then bias and result),
while detailed layers use a row-aware allocator with an independent
inter-layer corridor. Activation outputs are aligned on the same equation row;
a numerical vector cannot drift into a neighboring layer. The same allocator
drives final-model prediction and training replay for every theme and named
size. When the fixed derivation band is genuinely dense, font size may fall
from 14 to 13 px, but the reserved layer corridor is never removed.

### Post-inspection decision: training performance has one interaction model

Neural performance plots always show loss and all three metric panels together.
The generic lesson-stage Data/Model/Objective/Complete filter is not attached,
including under `format="lesson"`; only Play/Pause and the semantic checkpoint
slider remain. This prevents a presentation-only control from making an
unchanged metric curve disappear.

### Post-inspection decision: instance-based hyperparameter contract

The library now exposes `visualize_nn_hyperparameters()` and
`view="hyperparameters"`. The figure enumerates every detected effective
configuration value for each leaf module, each optimizer parameter group, the
objective, and an optional learning-rate scheduler. Each row owns an
independent panel corridor and contains its concrete value, mathematical role,
and PyTorch-aligned definition; rows are never presentation-truncated.

This is intentionally an instance contract, not a static claim to display the
entire open-ended PyTorch API in one canvas. Live objects are authoritative,
while recorder history is the fallback. Execution-only flags remain visible
and are marked non-mathematical. `TorchTrainingRecorder` accepts an optional
`scheduler` so the scheduler identity and effective constructor settings
survive after training. A dedicated human-QA cell is mandatory whenever this
coverage is extended.

The same inspection established three small invariant refinements: graph
Play/Pause controls sit 0.03 paper units higher than before, the prediction
model equation uses paper row 0.955, and alternating execution-block formulas
use 13 px mathematics. These changes preserve the existing graph, prediction,
and block geometry.

### Post-inspection decision: bounded legacy architecture configuration

Legacy architecture captions no longer render module configuration as an
unbounded single line. Each visible module owns a semantic multiline column
whose character budget is derived from the actual horizontal module pitch.
The first and last captions anchor toward the interior of the canvas, so
convolution, normalization, pooling, and classifier settings cannot cross a
figure edge or enter a neighboring module column under classroom typography.

The visible caption remains an intentionally compact four-setting summary.
The complete detected PyTorch configuration is retained in the node hover and
the layout contract records the wrapping policy and character budget. This is
a presentation constraint only; model inspection and execution are unchanged.
The real Digits CNN architecture case in the neural QA gallery is the required
human visual check for this five-column boundary condition.

### Post-inspection decision: omission notices participate in row allocation

An omission notice is semantic content, not a free-floating annotation. In
`visualize_nn_weights()`, a bounded-history notice now consumes the same
vertical budget as a tall matrix and enforces a minimum corridor before the
next visible tensor. The allocation is recomputed once and reused by every
animation frame, so changing parameter values cannot move the notice back over
the mathematics.

For `visualize_nn_backpropagation()`, six displayed trainable layers are dense
enough that three-line numerical readouts cannot share one baseline. Their
anchors therefore alternate between two invariant rows. The count of omitted
trainable layers is derived from the complete inspected model and rendered in
a dedicated caption below all per-layer values, never inside their numerical
band. The exact `NN-LARGE-WEIGHTS` and `NN-LARGE-BACKPROPAGATION` QA cases are
the required human checks for the initial and final animation frames.
