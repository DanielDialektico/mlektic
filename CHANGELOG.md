# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2026-05-26

### Added
- Added an extreme complexity local test case (`local_test/log_test_2_var.py`) demonstrating `K=20` and `d=20` to validate robust mathematical rendering in maximum stress scenarios.
- `explain_lr_prediction()`: tool for mathematically and visually explaining predictions of an already trained linear regression model.
- Dynamic metrics parameter (`metrics`) to display multiple variables (e.g., `loss`, `mse`, `r2`) simultaneously with smart number formatting to avoid visual overflows.
- Comprehensive initial documentation in `README.md`.
- Proper docstrings and enabled pydocstyle in ruff for Sphinx readiness.
- `frame_duration` control parameter to adjust animation speeds.
- Extensive local test cases matching notebook scenarios, including large multivariable tests (100 and 150 variables).

### Fixed
- **Plotly HTML Animation Rendering**: fixed an issue where exported HTML animations had unstable lines that cut or lengthened due to dynamic array resizing. Arrays (`loss_hist` and `step_axis`) are now padded with `None` to maintain a constant length across all frames.
- **Multivariable Prediction Formatting**: in `explain_lr_prediction` for $d \ge 3$, the output coordinate display string now properly appends $\hat{y}$ at the end (e.g., $(x_1, \ldots, x_d, \hat{y})$) to ensure mathematical consistency with 1D and 2D views.
- Fixed baseline value logic in `test_1_var.py`.

### Changed
- Improved formatting and alignment of LaTeX mathematical annotations in the multivariable logistic regression visualization (`multiclass_nd.py`).
- Adjusted the layout of the logistic regression visualization for multi-class and multi-variable configurations to prevent overlapping of equations and matrices.
- Separated probability equations into aligned individual blocks (`\begin{aligned}`) allowing distinct configurations without compromising perfect equality alignment.
- Modified Y-axis positioning for numerical fraction and definitions to use available vertical canvas space properly.
- Re-synced formatting logic for the 1-graph vs 2-graphs layouts to make annotations mathematically identical (`multiclass_1d.py`).
- Refactored `last_class_tail_latex` to maintain proportional positioning relative to numerical and symbolic definitions.
- Automatically open HTML in browser for animations 1 and 2 in single variable tests.
- Set Plotly renderer to `notebook` by default in tests.
