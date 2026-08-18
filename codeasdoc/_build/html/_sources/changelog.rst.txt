=========
Changelog
=========

Unreleased — neural execution-graph foundation
===============================================

Added
-----

- ``visualize_nn_hyperparameters`` and ``view="hyperparameters"`` now expose
  every detected effective module, optimizer-group, objective, and scheduler
  setting with its value, mathematical role, and PyTorch-aligned definition;
  the recorder can retain optional scheduler configuration for this lesson;
- a versioned renderer-independent neural graph with typed nodes, port-aware
  tensor edges, parameter/buffer specifications, provenance, warnings, and
  serializable metadata;
- automatic ``torch.fx`` capture with truthful eager-hook fallback, preserving
  supported branches, merges, shared calls, multiple inputs/outputs, and integer
  embedding inputs;
- an extensible semantic descriptor registry for common dense, convolutional,
  normalization, embedding, recurrent, attention, Transformer, pooling,
  reshape, reduction, activation, and merge operations;
- opt-in native Plotly block figures and public ``inspect_nn``,
  ``visualize_nn_blocks``, and ``register_neural_descriptor`` APIs;
- neural recorder history schema version 2 with buffer snapshots, effective
  optimizer groups, optional optimizer-state norms, and explicit temporal
  capture/observation phases;
- a dedicated execution-graph guide, implementation record, automated coverage,
  human visual-QA matrix, and student architecture lesson.
- an exhaustive 42-case neural QA gallery covering every public Plotly figure,
  synthetic classification/regression, real Iris and breast-cancer tabular
  data, a real handwritten-digit CNN, and simple-to-complex graph structures.
- opt-in ``updates`` and ``hybrid`` neural evolution modes with signed update
  halos, global or explicitly frame-normalized scales, previous/initial
  references, update diagnostics, top-k emphasis, and perceptual subframes
  that never masquerade as optimizer steps.

Fixed
-----

- bounded neural-weight omission notices now own a matrix-height row instead
  of floating over the next parameter tensor; crowded backpropagation values
  alternate across two fixed rows and place omitted-layer scope in a separate
  lower caption;
- legacy neural architecture configuration captions now wrap within semantic
  module columns, anchor inward at both canvas edges, and retain the complete
  detected PyTorch settings on hover.
- inline neural block equations now use a pixel-stable band below node labels;
  dense parallel columns move formulas to hover with an explicit notice instead
  of allowing labels, nodes, and equations to overlap.
- neural block hover cards now use readable Unicode/plain-text mathematics
  instead of exposing LaTeX commands in a Plotly surface that cannot render
  MathJax; eager multi-output nodes also report every observed output shape.

Changed
-------

- the README neural quick start now uses the recorder-v2 history contract and
  current public APIs for structure, replay, training, prediction,
  backpropagation, parameter tensors, and effective hyperparameters; it also
  explains temporal capture semantics and presentation-only bounds.

Compatibility
-------------

The established neural architecture figure remains the default. The block view
is opt-in, and current animation cadence, classic styling, dimensions, and
legacy history keys remain available.

Unreleased — phase 5 documentation and release readiness
=========================================================

Added
-----

- a complete English guide architecture covering installation, first lessons,
  mathematical conventions, motion, visual options, predictions, export,
  compatibility, limitations, gallery, and contribution policy;
- eight deterministic human visual-QA notebooks and four student notebooks with
  98 stable real-figure cases and cleared committed output;
- an enforced documentation-to-notebook manifest: new public documentation must
  add a corresponding assertion-free human inspection cell;
- notebook generation, validation, and non-mutating execution scripts;
- Python 3.9–3.13, PyTorch, docs, package, clean-import, notebook-smoke, and
  scheduled full-notebook CI workflows;
- complete package metadata and public version discovery.
- a dedicated estimator-hyperparameter guide and visual matrix spanning
  regularization, learning rates, weighting, neural topology, task, and loss.

Changed
-------

- earlier phase notebooks now live in the versioned archive, while the original
  large exploratory notebooks remain locally preserved with a tracked checksum
  inventory instead of bloating the release repository.
- postponed evaluation of Scikit-learn adapter type annotations preserves the
  documented Python 3.9 import contract without changing runtime behavior;
- strict Sphinx builds now start from a fresh environment and write to an
  ignored build directory, preventing committed artifacts from affecting CI.

Fixed
-----

- Python 3.9 test collection no longer fails while evaluating the
  ``np.ndarray | None`` return annotation;
- the compatibility guide no longer contains a duplicated fragment that Sphinx
  reports as unexpected indentation.

Compatibility
-------------

No visualization default changed. Classic dashboard styling, current sizes,
fluid animation, history semantics, and public figure behavior remain intact.

Unreleased — phase 3 visual design
==================================

Added
-----

- opt-in academic, classroom, compact, and accessible themes backed by public
  immutable design tokens;
- dashboard, lesson, compact, and report compositions;
- named sizes, explicit dimensions, responsive scaling, reduced-motion output,
  and responsive export inheritance;
- ``layout.meta["mlektic_visual"]`` audit metadata across tabular, prediction,
  and neural public figures;
- non-color marker and line redundancy for accessible views;
- a dedicated :doc:`visual_design` guide and Phase 3 showcase notebook.

Compatibility
-------------

Classic remains the default. Dashboard, lesson, and compact formats preserve
every retained frame and the existing animation cadence. Only explicit report
and reduced-motion choices freeze the exact final displayed state.

Unreleased — phase 1 mathematical parity
=========================================

Added
-----

- opt-in ``academic`` and ``complete`` mathematical detail for linear and
  logistic training figures while ``essential`` keeps the compact main view;
- versioned ``layout.meta["mlektic_math"]`` contracts with estimator-backed
  dimensions, feature spaces, contributions, predictions, probabilities,
  objectives, decisions, class order, and regularization settings;
- ``show_objective``, ``show_regularization``, ``feature_names``, and
  ``sample_index`` controls;
- binary ``threshold`` and multiclass ``class_focus`` controls;
- verified affine preprocessing conversion and explicit transformed-feature
  mathematics for non-affine pipelines;
- a dedicated :doc:`mathematical_parity` guide and Phase-1 implementation
  record.

Changed
-------

- ``show_loss=True`` now displays empirical MSE or log-loss along synthetic
  interpolation instead of silently removing the curve; labels explicitly
  reject optimizer-loss claims, interpolation endpoints remain exact, and
  compact low-dimensional nD linear layouts use their space more efficiently;
- fitted-model contribution products now wrap into dynamically spaced LaTeX
  rows; moderate-dimensional models show every term, while larger models use a
  disclosed contribution-ranked selection and retain the complete vector in
  metadata;
- synthetic interpolation now has an explicit baseline-to-fitted parameter
  equation, and polynomial-feature documentation distinguishes linearity in
  fitted coefficients from nonlinear geometry in the original input;
- incremental linear and logistic sequences now end at an explicitly labeled
  exact fitted endpoint; prior states remain reconstructed replay and metadata
  preserves every state origin;
- coefficient-bearing logistic interpolation now derives every intermediate
  probability, geometry, and empirical loss from the same parameter state;
- one-dimensional linear figures now place the interpolated fitted equation in
  a dedicated LaTeX band at every detail level, outside the data axes and
  synchronized as a redraw-free trace;
- history metadata schema version 2 records coefficient space and whether
  logistic interpolation targets parameters or probabilities.

Compatibility
-------------

The essential default retains compact styling, controls, frame counts, and
motion. Its evolving one-dimensional equation now uses the shared LaTeX math
band instead of covering the data region. Academic panels are separate
fitted-model references and remain fixed during playback so hybrid trace-only
animation stays fluid.

Unreleased — phase 0 integrity contract
=======================================

Added
-----

- ``show_class_labels=False`` for indexed-by-default logistic figures with
  optional fitted semantic labels and persistent class metadata;
- schema-versioned provenance and timeline metadata for tabular histories;
- explicit ``replayed`` and ``interpolated`` sources;
- full and displayed checkpoint coordinates;
- interpolation alpha values;
- raw/display loss separation;
- visible source and N/K timeline labels;
- verified prediction values, counterfactual opt-in, and extrapolation labels;
- public ``export_figure`` helper with explicit Plotly/MathJax dependencies;
- English history-semantics and implementation documentation;
- phase 0 contract tests.

Changed
-------

- invalid history enum values and unknown metrics now fail early;
- explicit iterative mode requires ``partial_fit``;
- unknown themes no longer silently fall back to classic;
- the loss metric uses the same display series as the visible loss curve;
- logistic prediction formulas support string labels;
- binary prediction views expose fitted class order and keep string-labeled
  probability axes numeric so model curves and surfaces remain visible;
- the two-feature binary decision boundary is the actual 0.5 intersection line,
  and the output identifies both probabilities and the winning fitted label;
- query ranges expand to keep 1D/2D extrapolations visible;
- constrained multiclass replay layouts use exact compact probability fractions
  so equations remain inside the mathematical panel;
- multiclass matrix, bias, score, and probability blocks use stable independent
  spacing across 1D, 2D, and nD routes;
- nD figures omit the loss subplot entirely when loss is not displayed;
- ``show_optimized`` documentation is English and IPython is imported lazily.

Compatibility
-------------

Classic colors, fixed dimensions, trace geometry, and animation behavior remain
the defaults. ``loss_hist`` remains available as an alias of
``loss_display``. New subtitles and slider labels are intentional transparency
changes. Previously ignored invalid values may now raise actionable exceptions.

Known limitations
-----------------

- tabular replay remains reconstructed even though its separately labeled
  endpoint is the exact fitted estimator;
- a fully offline MathJax bundle is not supplied;
- responsive scaling is implemented; structural reflow requires choosing a
  dedicated compact, lesson, or report format;
- full objective/regularization introspection belongs to the next phase.

Earlier development history
===========================

Earlier versions established linear 1D/2D/nD visualization, binary and
multiclass logistic visualization, Scikit-learn pipeline support, iterative and
interpolation strategies, metric histories, temporal decimation, hybrid linear
motion, and the PyTorch architecture/training/mathematical report family.

The repository-level ``CHANGELOG.md`` retains the detailed historical record.
Future entries should remain English-first and distinguish source semantics.
