# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed
- Improved formatting and alignment of LaTeX mathematical annotations in the multivariable logistic regression visualization (`multiclass_nd.py`).
- Adjusted the layout of the logistic regression visualization for multi-class and multi-variable configurations to prevent overlapping of equations and matrices.
- Separated probability equations into aligned individual blocks (`\begin{aligned}`) allowing distinct configurations without compromising perfect equality alignment.
- Modified Y-axis positioning for numerical fraction and definitions to use available vertical canvas space properly.
- Re-synced formatting logic for the 1-graph vs 2-graphs layouts to make annotations mathematically identical (`multiclass_1d.py`).
- Refactored `last_class_tail_latex` to maintain proportional positioning relative to numerical and symbolic definitions.
- Ensured arrays used in HTML animations are properly padded with `None` elements matching the exact animation frames to solve out-of-bounds frame rendering errors in output HTML files.

### Added
- Added an extreme complexity local test case (`local_test/log_test_2_var.py`) demonstrating `K=20` and `d=20` to validate robust mathematical rendering in maximum stress scenarios.
