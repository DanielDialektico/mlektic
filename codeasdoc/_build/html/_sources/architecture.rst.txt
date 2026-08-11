============
Architecture
============

Design goals
============

Mlektic separates model introspection, history construction, mathematical
semantics, and Plotly rendering. This separation allows the library to test a
history payload without constructing a figure and to reuse the same semantics
across 1D, 2D, and high-dimensional views.

Package map
===========

.. code-block:: text

   mlektic/
     api/              public orchestration and export helpers
     services/         configuration construction and history facades
     adapters/         estimator/pipeline capability normalization
     domain/           validated configuration and payload contracts
     history/          replay, interpolation, metrics, sampling, metadata
     mathematics.py    estimator-backed tabular mathematical contracts
     visualization/
       linear/         1D, 2D, nD, and prediction builders
       logistic/       binary/multiclass builders and prediction explainers
       neural/         architecture, graph, training, and mathematical views
     neural/           recorder, introspection, taxonomy, and reports
     utils/            numerical utilities and probability semantics

Public orchestration
====================

``visualize_lr`` and ``visualize_logistic`` perform five operations:

1. validate public animation controls;
2. call a history service;
3. build an estimator-backed mathematical contract;
4. route the payload to a dimensional figure builder;
5. configure motion and attach history and mathematical semantics.

The figure builders do not fit the supplied estimator. Replay fitting occurs
only on a Scikit-learn clone inside the history strategy. Prediction explainers
call ``predict``/``predict_proba`` for the single query they explain.

Configuration and services
==========================

``LinearHistoryConfig`` and ``LogisticHistoryConfig`` are frozen validated
contracts. Service functions construct those objects and pass them to a
``HistoryEngine``. Unsupported enum values, invalid numeric ranges, unknown
metrics, and non-callable custom metrics fail before strategy execution.

Adapters
========

``SklearnAdapter`` normalizes:

- direct estimators and pipelines;
- prediction and probability access;
- decision scores and multiclass link resolution;
- linear/logistic coefficient extraction;
- affine scaler parameters;
- clone creation and incremental replay capability.

For pipelines, transformations before the final estimator are applied through
the pipeline for normal prediction. Learned-space coefficients can be converted
to original units only when the preprocessing has an affine scaler convention
that can be represented by mean and scale.

History strategies
==================

``IterativeCapture``
--------------------

The strategy clones the supplied estimator, sets supported replay parameters
(``warm_start=True``, ``max_iter=1``, ``tol=0``, ``shuffle=False``), performs
an initial clone fit, and then calls ``partial_fit`` for later checkpoints. It
records empirical predictions, loss, grid values, and learned-space parameters.
The final semantic state is reserved for the exact supplied estimator and has
origin ``fitted_estimator``; preceding states retain origin ``replayed``. Its
public sequence is therefore labeled ``Reconstructed replay + fitted
endpoint`` rather than being presented as recorded training.

``InterpolationCapture``
------------------------

The strategy computes baseline and fitted-model states using alpha. Linear
predictions follow interpolated linear parameters. For coefficient-bearing
logistic estimators, scores, probabilities, geometry, and loss are derived
from the same interpolated parameters at every state. Probability-only
estimators retain an explicitly labeled probability-space fallback. Its public
source is ``interpolated``; the path is pedagogical and is not an optimizer
trajectory.

History engine
==============

``HistoryEngine`` is responsible for:

- training-data validation;
- auto mode resolution;
- strategy selection;
- original/scaled coefficient conversion;
- checkpoint metric calculation;
- provenance and final-state metadata;
- aligned temporal decimation;
- raw/display loss separation and EMA;
- final retained-coordinate metadata.

The engine deliberately builds metrics from raw checkpoint state before
display smoothing. It then updates the visible loss metric from the same display
series as the loss curve, leaving other metrics unchanged.

Sampling
========

``decimate_history`` builds one retained-position vector. Every NumPy array
whose leading dimension equals K and every metric history with that length is
sampled by the same vector. Metadata holds the full coordinate vector outside
the top-level sampling loop; retained coordinates remain at the top level.

Visualization
=============

Routers select builders by feature count and class count. Builders create
Plotly data, frames, formulas, controls, and dimensional geometry. A final
semantic annotation layer updates title subtitles, slider labels, and loss-axis
coordinates without changing frame data or the classic geometry.

The Phase-1 mathematical layer independently reconstructs a selected fitted
prediction, objective, class decision, feature-space representation, and
public regularization settings. The default stores this contract only in
``layout.meta``. Academic modes add a stable fitted-model reference below the
slider, preserving hybrid trace-only motion.

Neural architecture
===================

Neural functionality uses an explicit recorder for real training checkpoints.
Introspection and taxonomy classify supported modules and tensor roles;
visualization builders consume those records. Mathematical reports and
prediction explanations are distinct from the interactive training plot, which
keeps complex derivations from overloading one figure.

Extension rules
===============

When adding an estimator family or view:

1. define exact source semantics;
2. create a validated configuration contract;
3. keep capture independent of Plotly;
4. preserve source coordinates before sampling;
5. define raw and display quantities;
6. add mathematical invariants;
7. route through a semantic view model;
8. add classic compatibility and new-format visual tests;
9. document unsupported assumptions explicitly.
