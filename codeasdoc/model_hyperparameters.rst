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
normalization roles. Genuine optimizer, loss, and task evolution comes from
``TorchTrainingRecorder``. The QA matrix includes binary classification,
multiclass cross-entropy, regression MSE, deep dense, and convolutional examples.

Coverage boundary
=================

The canonical suite tests combinations that change mathematics, geometry,
provenance, or layout. It does not form an infinite Cartesian product of every
numeric value: independently tested visual axes compose with independently
tested estimator axes. Add a case whenever a new estimator option creates a new
mathematical claim or visible condition.

Inspect ``HYPER-LR-L2``, ``HYPER-LOG-C-STRONG``, and
``HYPER-NN-REGRESSION`` in ``notebooks/qa/qa_07_hyperparameters.ipynb``.
