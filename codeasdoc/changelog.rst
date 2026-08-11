=========
Changelog
=========

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
