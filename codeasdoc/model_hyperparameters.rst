====================================
Model parameters and hyperparameters
====================================
``HYPER-NN-REGRESSION`` in ``notebooks/qa/qa_07_hyperparameters.ipynb``.
Mlektic visualizes an already configured, fitted estimator. It does not choose
its regularization, solver, learning rate, class weighting, network topology,
activation, optimizer, or loss. These remain estimator-owned hyperparameters.
The figure reports only values that can be obtained from supported public model
attributes; it does not infer private optimizer conventions.

Tabular estimators
==================

The complete mathematical panel distinguishes a fitted parameter such as
:math:`\theta_j` from a training hyperparameter such as ``SGDRegressor.alpha``
or ``LogisticRegression.C``. ``C`` is inverse regularization strength, so smaller
values imply stronger regularization. Penalty and solver combinations must be
valid in Scikit-learn before Mlektic receives the model. Class weights affect the
fitted estimator but do not change the meaning of its output probabilities.

Neural networks
===============

Architecture views report layer topology, dimensions, activations, dropout, and
normalization roles. The block view additionally exposes stable public
configuration for convolution, embedding, recurrence, attention, Transformer,
pooling, and reshape modules. Genuine optimizer, loss, and task evolution comes
from ``TorchTrainingRecorder``. History schema version 2 retains effective
optimizer parameter groups per frame and can retain optimizer-state tensor
norms. It does not claim that a displayed module attribute fully describes a
custom optimizer or scheduler. The QA matrix includes binary classification,
multiclass cross-entropy, regression MSE, deep dense, convolutional, recurrent,
embedding, attention, branching, shared-module, and multi-I/O examples.

Use ``visualize_nn_hyperparameters`` when the lesson must audit the exact
configuration rather than infer it from architecture labels. The view reads
the supplied model, each optimizer parameter group, the objective, and the
learning-rate scheduler. It displays every detected effective value without
row truncation and pairs it with a PyTorch-aligned mathematical definition.
Runtime implementation switches remain present but are marked as
non-mathematical. Live objects take precedence over recorder metadata.

Coverage boundary
=================

The canonical suite tests combinations that change mathematics, geometry,
provenance, or layout. It does not form an infinite Cartesian product of every
numeric value: independently tested visual axes compose with independently
tested estimator axes. Add a case whenever a new estimator option creates a new
mathematical claim or visible condition.

Inspect ``HYPER-LR-L2``, ``HYPER-LOG-C-STRONG``, and
``HYPER-NN-REGRESSION`` in ``notebooks/qa/qa_07_hyperparameters.ipynb``.
Inspect ``NN-ROUTER-HYPERPARAMETERS`` in
``notebooks/qa/qa_08_neural_structures.ipynb`` for the complete neural
instance contract.
