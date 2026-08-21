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

``visualize_nn_training`` always keeps its four metric plots visible. Even
under ``format="lesson"`` it exposes only Play/Pause and the checkpoint slider;
the generic Data/Model/Objective/Complete stage filter is intentionally not
applied because hiding an individual training curve changes the comparison the
figure is meant to teach.

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
