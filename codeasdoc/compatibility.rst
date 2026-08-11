====================
Compatibility matrix
====================

.. list-table:: Supported estimator routes
   :header-rows: 1

   * - Family
     - Estimator or source
     - Dimensions
     - Temporal provenance
   * - Linear
     - Scikit-learn regressors exposing fitted coefficients; recognized pipelines
     - 1, 2, and d > 2
     - replay for ``partial_fit``; otherwise interpolation
   * - Logistic
     - classifiers exposing fitted coefficients, class order, and probabilities
     - 1, 2, and d > 2; binary and multiclass
     - replay for ``partial_fit``; otherwise interpolation
   * - Neural
     - supported PyTorch ``nn.Module`` leaf layers
     - architecture-dependent
     - genuine states supplied by ``TorchTrainingRecorder``

Python 3.9 through 3.13 are tested in the core CI matrix. PyTorch tests run in a
separate optional job. NumPy, Scikit-learn, and Plotly are runtime dependencies;
notebook and documentation tools are extras.

Affine preprocessing can be expressed in original or scaled units. Non-affine
transformers remain in transformed-feature space when names and coefficients can
be inspected safely. Unsupported estimators fail explicitly rather than
fabricating mathematics. Inspect ``LR-PIPE-SCALED`` and ``NN-RELU-ADAM``.
