=========
Changelog
=========

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

- tabular replay is reconstructed and can differ from the fitted estimator;
- a fully offline MathJax bundle is not supplied;
- responsive reflow and alternative themes are planned, not implemented;
- full objective/regularization introspection belongs to the next phase.

Earlier development history
===========================

Earlier versions established linear 1D/2D/nD visualization, binary and
multiclass logistic visualization, Scikit-learn pipeline support, iterative and
interpolation strategies, metric histories, temporal decimation, hybrid linear
motion, and the PyTorch architecture/training/mathematical report family.

The repository-level ``CHANGELOG.md`` retains the detailed historical record.
Future entries should remain English-first and distinguish source semantics.
