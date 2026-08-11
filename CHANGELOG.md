# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Added the opt-in Phase-3 visual system across linear, logistic, prediction,
  and neural public figures: immutable `VisualTokens`, `classic`, `academic`,
  `classroom`, `compact`, and `accessible` themes; `dashboard`, `lesson`,
  `compact`, and `report` formats; named sizes; explicit dimensions;
  responsive scaling; and reduced-motion final-state rendering.
- Added schema-versioned `layout.meta["mlektic_visual"]` metadata containing
  every resolved visual choice, token, dimension, motion decision,
  accessibility declaration, and responsive export setting.
- Added non-color redundancy in the accessible theme, staged pedagogical
  controls in lesson format, responsive HTML inheritance, a Phase-3 Sphinx
  guide, implementation record, and executable end-user showcase notebook.
- Added Phase-1 mathematical detail levels to `visualize_lr()` and `visualize_logistic()`: `detail="essential"` keeps the compact main visualization, while `"academic"` and `"complete"` add a stable fitted-model derivation without reducing animation frames or hybrid motion.
- Added estimator-backed `layout.meta["mlektic_math"]` contracts for linear, binary logistic, and multiclass logistic figures, including dimensions, feature space, parameters, one observation's contributions, reconstructed/model predictions, objective values, decisions, fitted class order, probability link, regularization settings, and canonical-versus-exact optimizer semantics.
- Added `show_objective`, `show_regularization`, `feature_names`, and `sample_index` controls to linear and logistic training figures; added binary `threshold` and multiclass `class_focus` controls to logistic figures.
- Added verified affine preprocessing conversion and transformed-feature mathematics for non-affine pipelines, using `get_feature_names_out` where available and explicitly declining to claim unavailable raw-space coefficients.
- Added a dedicated Sphinx mathematical-parity guide and an executable end-user Phase-1 notebook covering linear, binary, multiclass, pipeline, high-dimensional, and motion configurations.
- Added `show_class_labels=False` to logistic training and prediction figures. Indexed classes are the uncluttered default; fitted semantic labels can be revealed without removing their always-available metadata.
- Added `show_history_context=True` to linear and logistic public visualization APIs; setting it to `False` hides only the title subtitle while preserving slider and metadata context.
- Added a schema-versioned provenance contract for tabular histories, including replay/interpolation source detail, full and displayed checkpoint coordinates, estimator-reported iterations, smoothing/decimation settings, warnings, and final-state comparison.
- Added separate `loss_raw` and `loss_display` arrays while retaining `loss_hist` as a backward-compatible display alias.
- Added visible replay/interpolation subtitles and honest slider/loss-axis coordinates without changing classic geometry or animation defaults.
- Added strict validation for history configuration, training data, themes, prediction controls, supplied prediction/probability/class values, and unsupported explicit replay.
- Added extrapolation-aware linear and logistic prediction explanations, explicit counterfactual mode, string-label support, and auditable figure metadata.
- Added public `export_figure()` with explicit inline/CDN Plotly and MathJax dependency semantics.
- Added English-first history semantics, architecture, visualization, advanced-use, implementation-plan, and phase-0 verification documentation.
- Added hybrid 1D linear-regression animation with synchronized visual subframes for the model line, numeric coefficients, loss, and metrics; semantic checkpoints remain distinct in the slider and the symbolic definition stays in LaTeX.
- Added optional PyTorch support through `mlektic[torch]`, including `TorchTrainingRecorder` for frame-aligned loss, metrics, parameter values, gradients, compact activation vectors, and optimizer/loss metadata.
- Added a LaTeX-annotated architecture diagram with tensor dimensions, semantic layer shapes, formulas, configured hyperparameters, and compact summaries for large models.
- Added an animated mathematical network graph with one stable frame per training step, true global min/max scales for node outputs and edge weights by default, optional relative node contrast and forward-signal edge modes, simultaneous wine-red backpropagation overlays, final model tensors, and readable hover data without raw LaTeX syntax.
- Added a compact 2-by-2 learning-performance grid for loss and three independent metrics, automatic classification/regression metric inference from predictions and targets, and explicit empty metric panels when a history contains only loss.
- Moved neural parameter and forward-pass animation controls to a reserved upper-left area so buttons never cover the displayed equations.
- Fixed the mathematical-network parameter readout so weights and the training step update through animation traces without forcing a flickering redraw, and separated the title from the composed-function equation.
- Added explicit exact-zero and inactive-ReLU hover labels, inset the animated step readout, softened node outlines, and moved mathematical colorbar titles above their scales with wider outer margins.
- Added `explain_nn_prediction()` for time-aware layer-by-layer forward-pass mathematics, numerical substitutions for `z = Theta a + theta_0`, and summarized representations for deeper networks.
- Added standalone and notebook HTML mathematical reports with the complete taxonomy, definition, configuration, dimensions, parameter roles, and training evolution for every layer.
- Added reusable history metric builders for linear and logistic animations, including built-in support for `loss`, `mse`, `r2`, `mae`, `accuracy`, and `f1`, plus custom metric callables.
- Added reusable history sampling utilities to decimate long animation histories through `max_frames` or `frame_step`.
- Documented the adapter extension path for future model families, including non-Scikit-Learn estimators and upcoming neural-network visualization work.
- **2D Multiclass Logistic Regression Visualization**: Added full support for visualizing multi-class logistic regression in 2-dimensional feature spaces. The builder dynamically generates a 3D plot displaying the actual data points on the floor grid and $K$ distinct translucent, colored probability surfaces hovering and adjusting over time.
- Integrated a live LaTeX panel directly into the 2D multiclass layout, including the parameter matrix, bias vector, probability link, and step-by-step evaluation of representative class probabilities.
- Modified capture engines (`strategy_iterative.py` and `strategy_interp.py`) to systematically extract and cache a multidimensional probability surface history (`p_surfaces_hist`) required for $K$-class 3D rendering.
- Comprehensive `pytest` test suite covering `visualize_lr`, `visualize_logistic`, and `explain_lr_prediction` across 1D, 2D, and ND dimensional boundaries, including Pipeline scenarios.
- Moved legacy tests and old modules that were incompatible with Scikit-Learn to `old_mlektic_core`.
- Added an extreme complexity local test case (`local_test/log_test_2_var.py`) demonstrating `K=20` and `d=20` to validate robust mathematical rendering in maximum stress scenarios.
- `explain_lr_prediction()` and `explain_logistic_prediction()`: tools for mathematically and visually explaining predictions of already trained linear and logistic regression models.
- Dynamic metrics parameter (`metrics`) to display multiple variables (e.g., `loss`, `mse`, `r2`) simultaneously with smart number formatting to avoid visual overflows.
- Comprehensive initial documentation in `README.md`.
- Proper docstrings and enabled pydocstyle in ruff for Sphinx readiness.
- `frame_duration` control parameter to adjust animation speeds.
- Extensive local test cases matching notebook scenarios, including large multivariable tests (100 and 150 variables).

### Fixed
- Added theme-aware high-contrast boxes and borders to plotted prediction-value
  annotations (`y_hat` and `p_hat`) in linear, binary-logistic, multiclass, 2D,
  and 3D prediction explainers so values remain legible over model geometry.
- Prevented compact figures with academic/complete derivations from compressing
  the plot domain until the lower mathematics panel overlapped slider labels;
  the panel reserve and upper equation/title clearance now remain stable across
  named size presets and animated frames.
- Suppressed the expected Scikit-learn convergence warning produced only by Mlektic's intentional one-iteration replay initialization; warnings from the user's own estimator fit remain untouched.
- Synchronized coefficient-bearing logistic interpolation so every intermediate score, sigmoid/Softmax/OvR probability, curve or surface, and empirical loss is derived from the same interpolated parameter state; probability-only fallbacks are explicitly labeled.
- Restored normal 15-point typography and an inline coordinate tuple for ordinary two-feature linear prediction results; only genuinely long formatted coordinates now trigger a compact 13-point wrapped layout.
- Kept binary prediction axes numeric for string-labeled estimators so fitted sigmoid curves and probability surfaces remain visible; the two-feature boundary is now the actual `p=0.5` intersection line, and results identify both fitted-class probabilities and the winning label.
- Separated the multiclass 2D coefficient matrix and bias vector so the bias is centered between the matrix and class-score equation.
- Prevented normalized-OvR probability substitutions from crossing into loss panels by using exact compact fractions only in constrained replay layouts.
- Recentered multiclass ellipses in 1D and nD, removed empty rows from compact nD input vectors, and honored `max_theta_cols` in dense multiclass matrices.
- Removed empty loss subplots from linear and binary-logistic nD figures when `show_loss=False`.
- Shortened replay/interpolation provenance subtitles while preserving N/K, estimator iterations, and explicit endpoint-origin semantics.
- Balanced the multiclass probability stack by lowering only the first expanded fraction, keeping the ellipsis visually centered, and sharing the same fraction spacing and typography across 1D, 2D, and nD figures.
- Increased vertical separation between the definition and substituted sigmoid equations in binary logistic 2D figures so fraction numerators do not overlap the equation above.
- Fixed temporal decimation labels so retained checkpoints keep their source coordinates instead of being renumbered `0..N-1`.
- Fixed smoothed loss figures so the visible Loss/Log-loss metric uses the same display series while raw empirical values remain unchanged.
- Fixed 1D and 2D prediction ranges so out-of-range queries remain visible and are identified as extrapolations.
- Stabilized Play/Pause styling during 3D redraw animations by keeping buttons white with black text in every interaction state, without JavaScript state tracking or changes to interpolation.
- Moved hybrid 1D metric cards into a dedicated vertical side column so longer values cannot overlap each other or the loss plot.
- Prevented 2D prediction substitutions and results from overflowing by using compact LaTeX scientific notation, smaller panel typography, and a wrapped output tuple.
- Replaced display-only multicolumn grids for high-dimensional inputs and parameters with mathematically faithful truncated column vectors in linear and multiclass-logistic prediction explanations.
- Restored high-contrast metric cards in hybrid 1D animations using synchronized marker-and-text traces, preserving smooth playback without layout redraws.
- Restored animated LaTeX parameter and metric updates in Jupyter for layout-driven views; simple 1D regression now avoids layout redraws through synchronized numeric traces.
- Limited EMA smoothing to loss histories so displayed parameters always remain mathematically consistent with model geometry and probabilities.
- Prevented 2D regression lines from appearing partially drawn or blinking during playback by preserving every SVG line point and ensuring Plotly transitions finish before the next animation frame begins.
- Fixed multiclass logistic mathematics so figures and prediction explanations detect Softmax versus normalized one-vs-rest sigmoids instead of always claiming Softmax.
- Corrected ND parameter dimensions, matrix orientation, scalar output spaces, generic class notation, and the separation between `Theta` and `theta_0` across linear and logistic views.
- Replaced repeated expanded OvR denominators with compact exact normalizers to prevent LaTeX substitutions from overflowing into adjacent plots.
- Kept neural animation button labels readable when Plotly applies its light hover state.
- Fixed root-package exports so `from mlektic import explain_lr_prediction` matches the documented public API.
- Fixed logistic metric histories so classification metrics map predictions back to the original class labels instead of assuming zero-based label indexes.
- Preserved scaler metadata in interpolation histories so pipeline visualizations can evaluate metrics in the requested display space.
- **Plotly Visual Overflows**: Resolved massive text layout overlaps in logistic regression `prediction` views (1D, 2D, Multiclass 1D, and Multiclass ND) by precisely tuning Grid `column_widths`, `row_heights`, and expanding bounding boxes beyond default axis constraints.
- **Result Output Alignment**: Forced strict left-alignment on `\max` and `\hat{y} = \text{argmax}` math equations inside Multiclass result panels to prevent right-drifting.
- **Y-Axis Stand-Offs**: Prevented the $p(y=k \mid x)$ axis label from crashing into mathematical panels by directly shrinking `title_standoff` instead of disturbing layout proportions.
- **Plotly HTML Animation Rendering**: fixed an issue where exported HTML animations had unstable lines that cut or lengthened due to dynamic array resizing. Arrays (`loss_hist` and `step_axis`) are now padded with `None` to maintain a constant length across all frames.
- **Multivariable Prediction Formatting**: in `explain_lr_prediction` for $d \ge 3$, the output coordinate display string now properly appends $\hat{y}$ at the end (e.g., $(x_1, \ldots, x_d, \hat{y})$) to ensure mathematical consistency with 1D and 2D views.
- Fixed baseline value logic in `test_1_var.py`.

### Changed
- Preserved every retained animation frame in the new dashboard, compact, and
  lesson formats. Only the explicit report and reduced-motion alternatives
  freeze the exact final displayed state and remove temporal controls.
- Made `density` a validated alias of the Phase-1 `detail` contract for tabular
  training figures while leaving `detail` authoritative when density is
  omitted.
- Honored `show_loss=True` for synthetic interpolation across linear and logistic dimensionalities. These curves are labeled as interpolation MSE/log-loss rather than optimizer training loss; synthetic paths retain raw endpoint-exact values, replay-only EMA is declared explicitly, and compact linear models up to 10 variables use a shorter, better-balanced canvas with a wider loss panel.
- Wrapped fitted-model contribution expansions into dynamically spaced LaTeX rows. Moderate-dimensional panels show every coefficient-value product; higher-dimensional panels show a bounded, contribution-ranked selection and disclose the omitted count separately while preserving every value in metadata.
- Clarified synthetic interpolation with its baseline-to-fitted parameter equation and documented that polynomial-feature linear models can form nonlinear original-space geometry without implying gradient-descent training.
- Closed incremental linear and logistic replays with an explicitly labeled `fitted` endpoint from the supplied estimator. Intermediate states remain reconstructed, while the final equations, geometry, probabilities, raw loss, and evaluation metrics now match the model exactly; metadata records every state origin and the endpoint policy.
- Standardized one-dimensional linear-regression playback across all mathematical detail levels: the interpolated fitted equation now evolves in a dedicated LaTeX band above the plotting axes, metric cards retain their organized side column, and the equation no longer covers observations or the fitted line. The equation remains a trace update, so hybrid subframes keep `redraw=False`; academic panels remain separate below the slider.
- Advanced history metadata to schema version 2 with explicit coefficient-space semantics and parameter-versus-probability interpolation targets.
- Clarified that incremental Scikit-learn histories are reconstructed replays over clones and that non-incremental paths are synthetic interpolations, not recovered optimizer histories.
- Changed Sphinx documentation language to English and made the new public contract English-first.
- Changed `show_optimized()` to import IPython lazily so the core package import does not require notebook dependencies.
- Documented the animation rendering contract: hybrid 1D linear views keep symbolic LaTeX fixed and animate synchronized numeric traces, native views redraw dynamic MathJax substitutions, and notebook-friendly hybrid cadence is typically 30-45 FPS.
- Unified learnable-parameter notation across linear, logistic, and neural models: vectors use `theta`, matrices use `Theta`, and bias vectors use `theta_0`.
- Added configurable 2D animation transitions while preserving redraws for Plotly 3D traces; existing `steps`, `max_frames`, and `frame_step` controls remain available for temporal sampling.
- Completed the README, Sphinx guides, architecture reference, and public neural API docstrings for the PyTorch workflow, exact global heatmaps, optional relative/signal modes, inferred metrics, exact ReLU zeros, animated parameter readouts, and notebook/standalone HTML reports.
- Refactored `HistoryEngine` so it orchestrates capture, metric building, temporal sampling, smoothing, and parameter scaling through smaller focused helpers.
- Tightened Ruff configuration to lint the maintained package, tests, and Sphinx configuration while excluding generated, legacy, and manual-test directories.
- Updated README and Sphinx documentation for metrics, frame controls, `multiclass_2d`, generated API reference, and future adapter scalability.
- Replaced broad replay-configuration handling in the Scikit-Learn adapter with explicit parameter attempts and narrower exception handling.
- **Surface Smoothing (2D)**: Drastically enhanced rendering quality for Logistic Regression 2D probability surfaces and decision boundary planes by boosting the mesh grid resolution.
- **Layout Dimensions**: Enforced fixed `height` (and dimensions when appropriate) across visualization frames to guarantee consistent aspect ratios and eliminate bounding jitters.
- Refactored rendering core files (`prediction.py`, `multivar.py`, `binary_nd.py`, `multiclass_1d.py`) to eliminate Plotly trace code duplication, dynamically assemble subplot columns, and centralize LaTeX builders for better maintainability.
- Improved formatting and alignment of LaTeX mathematical annotations in the multivariable logistic regression visualization (`multiclass_nd.py`).
- Adjusted the layout of the logistic regression visualization for multi-class and multi-variable configurations to prevent overlapping of equations and matrices.
- Refined Multiclass Logistic prediction (1D and ND): replaced static matrix prediction display with dynamic mathematical argmax equations in the Result panel.
- Tightened mathematical substitution bounds and probability annotation bounds for high-cardinality multi-class classification (e.g. K=30) to eliminate vertical canvas overflow.
- Separated probability equations into aligned individual blocks (`\begin{aligned}`) allowing distinct configurations without compromising perfect equality alignment.
- Modified Y-axis positioning for numerical fraction and definitions to use available vertical canvas space properly.
- Re-synced formatting logic for the 1-graph vs 2-graphs layouts to make annotations mathematically identical (`multiclass_1d.py`).
- Refactored `last_class_tail_latex` to maintain proportional positioning relative to numerical and symbolic definitions.
- Automatically open HTML in browser for animations 1 and 2 in single variable tests.
- Set Plotly renderer to `notebook` by default in tests.
