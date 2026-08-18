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
     - PyTorch ``nn.Module`` execution capture with specialized common layers
       and transparent generic fallback
     - sequential, branched, shared, recurrent, attention, multi-input, and
       multi-output example paths
     - genuine states supplied by ``TorchTrainingRecorder``; static FX or
       observed eager provenance for architecture capture

Python 3.9 through 3.13 are tested in the core CI matrix. PyTorch tests run in a
separate optional job. NumPy, Scikit-learn, and Plotly are runtime dependencies;
notebook and documentation tools are extras.

Affine preprocessing can be expressed in original or scaled units. Non-affine
transformers remain in transformed-feature space when names and coefficients can
be inspected safely. Unsupported estimators fail explicitly rather than
fabricating mathematics. Inspect ``LR-PIPE-SCALED`` and ``NN-RELU-ADAM``.

The opt-in neural block view retains inputs, dtype, shapes, branches, merge
operations, and repeated calls. Exact coverage depends on the supplied example
path and capture backend. See :doc:`neural_execution_graphs` and inspect
``NN-BLOCK-RESIDUAL`` and ``NN-BLOCK-ATTENTION``.
