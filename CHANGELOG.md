# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2026-06-04

### Added
- Added optional PyTorch support through `mlektic[torch]`, including `TorchTrainingRecorder` for frame-aligned loss, metrics, parameter values, gradients, compact activation vectors, and optimizer/loss metadata.
- Added a LaTeX-annotated architecture diagram with tensor dimensions, semantic layer shapes, formulas, configured hyperparameters, and compact summaries for large models.
- Added an animated mathematical network graph with one stable frame per training step, true global min/max scales for node outputs and edge weights by default, optional relative node contrast and forward-signal edge modes, simultaneous wine-red backpropagation overlays, final model tensors, and readable hover data without raw LaTeX syntax.
- Added a compact 2-by-2 learning-performance grid for loss and three independent metrics, automatic classification/regression metric inference from predictions and targets, and explicit empty metric panels when a history contains only loss.
- Moved neural parameter and forward-pass animation controls to a reserved upper-left area so buttons never cover the displayed equations.
- Added `explain_nn_prediction()` for time-aware layer-by-layer forward-pass mathematics, numerical substitutions for `z = Wa + b`, and summarized representations for deeper networks.
- Added standalone and notebook HTML mathematical reports with the complete taxonomy, definition, configuration, dimensions, parameter roles, and training evolution for every layer.
- Added reusable history metric builders for linear and logistic animations, including built-in support for `loss`, `mse`, `r2`, `mae`, `accuracy`, and `f1`, plus custom metric callables.
- Added reusable history sampling utilities to decimate long animation histories through `max_frames` or `frame_step`.
- Documented the adapter extension path for future model families, including non-Scikit-Learn estimators and upcoming neural-network visualization work.
- **2D Multiclass Logistic Regression Visualization**: Added full support for visualizing multi-class logistic regression in 2-dimensional feature spaces. The builder dynamically generates a 3D plot displaying the actual data points on the floor grid and $K$ distinct translucent, colored probability surfaces hovering and adjusting over time.
- Integrated a live LaTeX panel directly into the 2D Multiclass layout, showcasing the $\mathbf{z} = \Theta^\top\mathbf{x}$ formula, the dynamic parameter matrix $\Theta \in \mathbb{R}^{3 \times K}$, the explicit Softmax formulation, and a live step-by-step mathematical evaluation of a sample probability curve $\hat{p}(y=k \mid \mathbf{x})$.
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
