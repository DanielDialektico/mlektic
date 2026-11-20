====================================
First rigorous neural-network lesson
====================================
``notebooks/learn/learn_03_neural_networks.ipynb``.
Neural visualizations separate architecture, forward propagation, recorded
training, parameters, activations, gradients, and prediction substitution. A
``TorchTrainingRecorder`` captures genuine states only when the training loop
calls ``record``.

.. code-block:: python

   recorder = TorchTrainingRecorder(model, optimizer=optimizer, loss_fn=criterion)
   for step in range(epochs):
       optimizer.zero_grad()
       prediction = model(X)
       loss = criterion(prediction, y)
       loss.backward()
       optimizer.step()
       recorder.record(
           step + 1,
           loss=loss,
           predictions=prediction,
           targets=y,
           task="classification",
       )
   recorder.close()

Record at a consistent point relative to ``optimizer.step()``. History schema
version 2 stores parameter and buffer snapshots separately, effective optimizer
parameter groups, gradients, activation vectors when enabled, temporal capture
semantics, training configuration, and supplied or inferred metrics. Optional
optimizer-state norms expose momentum or adaptive state without copying every
state tensor into the figure. ``max_frames`` only decimates displayed
checkpoints.

Execution structure
===================

The default architecture remains the established sequential lesson figure.
Its per-module configuration captions use bounded semantic lines derived from
the displayed column spacing. The visible four-setting summary therefore stays
inside its module column even under ``theme="classroom"``; hover preserves the
complete detected PyTorch configuration.
Use ``architecture_mode="blocks"`` to inspect an executed branch-aware graph:

.. code-block:: python

   figure = visualize_nn_architecture(
       model,
       X[:1],
       architecture_mode="blocks",
       theme="academic",
   )

The block view distinguishes specialized formulas from generic fallbacks and
reports whether capture used static ``torch.fx`` tracing or observed eager
module calls. See :doc:`neural_execution_graphs`.

What is exact?
==============

Retained recorder snapshots are genuine captured states. Node colors can encode
exact global values or relative per-layer contrast; edge colors can encode
weights or forward signal :math:`w_{ji}a_i`. The legend and metadata state the
selected meaning.

Inspect ``LEARN-NN-ARCHITECTURE`` and ``LEARN-NN-TRAINING`` in
``notebooks/learn/learn_03_neural_networks.ipynb``.
Inspect ``LEARN-NN-BRANCHES`` and ``LEARN-NN-ATTENTION`` in
``notebooks/learn/learn_04_neural_architectures.ipynb``.

Effective hyperparameters and their mathematics
================================================

``visualize_nn_hyperparameters`` provides a dedicated, non-animated contract
for the concrete PyTorch objects used by a lesson:

.. code-block:: python

   figure = visualize_nn_hyperparameters(
       model,
       optimizer=optimizer,
       loss_fn=criterion,
       scheduler=scheduler,
       theme="academic",
       size="wide",
   )

The figure lists every detected effective constructor value for every leaf
module, every value in every optimizer parameter group, every detected loss
argument, and every detected scheduler argument. Each row includes the value,
its mathematical role, and a concise definition. Options such as ``foreach``,
``fused``, ``capturable``, and ``inplace`` remain visible but are explicitly
identified as execution choices rather than changes to the mathematical map.

This is deliberately instance based. It does not present an unbounded global
catalogue of classes that were not used by the model. Live objects are the
most complete source; recorder history is a transparent fallback when those
objects are no longer available. ``TorchTrainingRecorder(...,
scheduler=scheduler)`` retains scheduler identity and effective constructor
settings for this purpose. BatchNorm momentum is described with PyTorch's
running-statistic convention, which is not optimizer momentum.

Inspect ``NN-ROUTER-HYPERPARAMETERS`` in
``notebooks/qa/qa_08_neural_structures.ipynb``.

Forward and backward are complementary
======================================

The animated dense graph shows forward activations, signals, retained
gradients, and parameter evolution together. For a formal step-by-step account
of the reverse pass, use ``visualize_nn_backpropagation``. It displays the
hidden-layer chain rule, the recorded gradient norm, and adjacent parameter
change for each trainable layer. Backpropagation computes gradients; the
optimizer converts them into parameter changes. The dedicated animation is
slower by default so these three phases can be compared. Each trainable layer
shows the recorded gradient norm, adjacent update norm, relative update,
current loss, and adjacent loss change rather than relying only on moving line
widths.
Use ``show_loss_panel=True`` on ``visualize_nn_graph`` to place the selected
recorded objective beside the update diagnostics, or
``visualize_nn_loss_landscape`` to evaluate a disclosed two-direction loss
slice in three dimensions.

Use ``show_backpropagation=True`` only when per-edge recorded gradients are the
lesson focus. Dense graphs default to ``False`` because those additional traces
can reduce notebook animation performance.

Large parameter and backpropagation views use bounded disclosure rather than
shrinking all mathematics until it becomes unreadable. ``max_rows`` and
``max_cols`` abbreviate a displayed tensor, while ``max_parameters`` and
``max_layers`` select a representative visible subset. These controls never
delete recorder history or alter model execution. ``visualize_nn_weights``
assigns the omitted-parameter count its own matrix-height row between retained
tensors. ``visualize_nn_backpropagation`` alternates crowded three-line layer
readouts across two fixed rows and places the omitted-layer count in a separate
lower caption. The complete and displayed counts, row policy, and scope remain
auditable through ``layout.meta`` in every frame.

``visualize_nn_training`` always keeps its four metric plots visible and
exposes only Play/Pause and the checkpoint slider. No generic trace filter is
added because hiding an individual trace changes the comparison the figure is
meant to teach.

Use ``explain_nn_prediction`` when the lesson asks how one query is actually
used. With the default ``parameter_state="final"``, Input, Substitution, and
Output are prediction-only stages: the Substitution stage inserts the query and
the fitted numerical parameters into the first retained unit before expanding
the remaining forward computations. There is no training slider in this mode.
For recorded BCE and cross-entropy tasks the output also identifies the winning
class, while regression remains numeric.

The prediction figure starts in Reset and reveals fixed, cumulative regions for
Input, Substitution, and Output. Summary vectors use two representative
endpoints and at most three decimals; detailed layer values use an explicit
precision-aware display bound with ellipsis. Symbolic definitions and computed
vectors share aligned, layer-owned formula blocks. These display limits never
change model execution or the computed prediction.

Detailed forward blocks are positioned from their rendered MathJax row count,
not from the number of modules alone. Each additional Linear row consumes an
explicit baseline pitch before an independent inter-layer corridor is applied.
Ordinary lessons retain 14-point mathematics; only a selection that cannot fit
the reserved derivation band uses the 13-point fallback. A single expanded
Linear substitution is limited to four rendered rows; larger layers retain the
exact computed vector through the symbolic matrix form.

Training observation is a separate interaction. Set
``parameter_state="training_replay"`` only when the lesson examines recorded
parameter and signal evolution. That figure has Play/Pause and a checkpoint
slider, but deliberately has no Input, Substitution, or Output controls, no
prediction summary cards, and no duplicated final prediction. The two modes
never combine prediction stages with a training animation.

What, why, and how the network learns
=====================================

Four geometric views answer the questions a first lesson usually leaves open.
Every value they draw is an exact forward pass or autograd computation with the
recorded parameters, and they use the course notation
(``course/notacion_matematica.html``): :math:`\Theta^{(\ell)}` with rows as
outputs, :math:`\boldsymbol\theta_0^{(\ell)}`, :math:`\mathbf z^{(\ell)}`,
:math:`\mathbf a^{(\ell)}`, :math:`\phi_\ell`, per-sample :math:`l`, and the
reduced :math:`\mathcal L`. They accept any dense chain of ``Linear`` layers and
elementwise activations; other modules raise an error that points to the block
view.

.. code-block:: python

   decision = visualize_nn_decision(model, X, history, targets=y)
   representation = visualize_nn_representation(model, X, history, targets=y)
   step = visualize_nn_training_step(model, X[0], y[0], criterion, history, checkpoint=0)
   flow = visualize_nn_gradient_flow({"sigmoid": history_sigmoid, "relu": history_relu})

**What it learns.** ``visualize_nn_decision`` evaluates
:math:`\hat y=f_\theta(\mathbf x)` on the plane of two input features (others
held at their median or at ``reference``) at every retained checkpoint. Up to
four units of the last hidden layer, :math:`a^{(L-1)}_j(\mathbf x)`, sit beside
it with the readout weights :math:`\Theta^{(L)}` that combine them. With one
hidden layer each unit folds the plane along the exact line
:math:`z^{(1)}_j(\mathbf x)=0`, drawn in the unit's colour on the main panel. A
sigmoid output shows the boundary :math:`\hat y=0.5`; a logit output shows
:math:`z^{(L)}` and its zero set; several outputs show the arg-max class.

**Why it works.** ``visualize_nn_representation`` pushes a grid, the data, and
the exact boundary through :math:`\mathbf x\to\mathbf z^{(1)}\to\mathbf a^{(1)}\to\cdots`.
The affine image of the plane is shown isometrically; the last hidden layer is
read along the readout direction of :math:`\Theta^{(L)}`, where a one-output
classifier's boundary is exactly a vertical line; any other stage uses its
principal axes, with the retained variance disclosed. Motion between stages is
a Plotly transition between exact stages, not a network state.

**How it learns.** ``visualize_nn_training_step`` follows one example: pulses
of :math:`s^{(\ell)}_{ji}=\theta^{(\ell)}_{ji}a^{(\ell-1)}_i`, the per-sample loss,
:math:`\boldsymbol\delta^{(\ell)}` travelling back, each edge gradient
:math:`\delta_j a_i`, and one step :math:`\Theta\leftarrow\Theta-\eta\nabla_\Theta l`
on that example. The step illustrates the rule; it is not a replay of the
recorded optimizer, and ``layout.meta`` says so. It complements
``visualize_nn_backpropagation``, which reports the recorded layerwise norms
over the whole run.

**Does the signal reach every layer?** ``visualize_nn_gradient_flow`` is a
static map of the recorded :math:`\log_{10}\|\nabla_{\Theta^{(\ell)}}\mathcal L\|`
per layer and checkpoint. Up to four runs share one colour scale, so a sigmoid
network whose first layer receives orders of magnitude less gradient than its
last stands out against a ReLU one.

Large matrices and the loss slice
=================================

``visualize_nn_weights(history, style="hinton")`` draws each recorded
:math:`\Theta^{(\ell)}` as a Hinton diagram: square area proportional to
:math:`|\theta^{(\ell)}_{ji}|` on one global scale, colour for the sign, and
:math:`\boldsymbol\theta_0^{(\ell)}` as a separate column. Matrices larger than
32 rows or columns keep their first and last entries, disclosed below the
matrix.

``visualize_nn_loss_landscape`` now adds a synchronized top view
(``show_contour=True``): the level sets of
:math:`\log_{10}\mathcal L_{slice}(u,v)` with the projected path and the slice
centre :math:`\theta_*`.

Inspect ``LEARN-NN-DECISION``, ``LEARN-NN-REPRESENTATION``,
``LEARN-NN-TRAINING-STEP``, and ``LEARN-NN-GRADIENT-FLOW`` in
``notebooks/learn/learn_03_neural_networks.ipynb``, and the theme matrix
``NN-GEOMETRY-THEMES`` in ``notebooks/qa/qa_08_neural_structures.ipynb``.
