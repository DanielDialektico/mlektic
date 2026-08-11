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

Record at a consistent point relative to ``optimizer.step()``. The recorder
stores parameter and gradient snapshots, activation vectors when enabled,
training configuration, and supplied or inferred metrics. ``max_frames`` only
decimates the displayed checkpoints.

What is exact?
==============

Retained recorder snapshots are genuine captured states. Node colors can encode
exact global values or relative per-layer contrast; edge colors can encode
weights or forward signal :math:`w_{ji}a_i`. The legend and metadata state the
selected meaning.

Inspect ``LEARN-NN-ARCHITECTURE`` and ``LEARN-NN-TRAINING`` in
``notebooks/learn/learn_03_neural_networks.ipynb``.
