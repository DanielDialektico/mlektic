# Mlektic improvement master plan

## Purpose

This directory is the durable engineering and product record for turning Mlektic into a rigorous, transparent, elegant, and genuinely pedagogical public library. It separates verified facts from proposals and implementation decisions. All canonical planning and implementation notes are written in English so they can later become public documentation without translation drift.

The plans deliberately preserve the current visual behavior as the compatibility baseline. Motion, interpolation, the classic theme, current dimensions, and current figure formats remain available by default unless a phase explicitly documents and tests a backward-compatible extension.

## Evidence base

The review covered:

- the public linear, logistic, and neural APIs;
- history capture, replay, interpolation, temporal decimation, smoothing, metrics, and Plotly frame construction;
- mathematical prediction explainers and HTML export paths;
- all Sphinx pages and the repository README;
- the four active validation notebooks: `test_interpt.ipynb`, `test_linreg.ipynb`, `test_logreg.ipynb`, and `test_ann.ipynb`;
- representative configurations across task, dimensionality, estimator type, pipeline use, display space, capture mode, smoothing, loss panels, and animation settings;
- visual outputs, fixed sizing, typography, information density, and potential states that are not visible to a student.

## Documents

1. [Audit and design principles](00_audit_and_principles.md)
2. [Phase 0 — Mathematical integrity and history contract](01_phase_0_integrity_and_contract.md)
3. [Phase 1 — Mathematical parity](02_phase_1_mathematical_parity.md)
4. [Phase 2 — Motion and time](03_phase_2_motion_and_time.md)
5. [Phase 3 — Visual design system](04_phase_3_visual_design.md)
6. [Phase 4 — QA and learning notebooks](05_phase_4_notebooks.md)
7. [Phase 5 — Documentation and release](06_phase_5_documentation_and_release.md)
8. [Test and acceptance matrix](07_test_and_acceptance_matrix.md)
9. [Phase 0 implementation record](08_phase_0_implementation_record.md)
10. [Phase 1 implementation record](09_phase_1_implementation_record.md)
11. [Phase 3 implementation record](10_phase_3_implementation_record.md)

## Recommended order

The phases are intentionally sequenced by dependency:

1. Establish truthful data and API contracts.
2. Bring linear and logistic explanations to neural-level mathematical rigor.
3. Formalize semantic time, sampling, and perceptual interpolation.
4. Introduce optional visual formats, density levels, and responsive behavior.
5. Replace ad hoc notebooks with a QA matrix and student learning path.
6. Publish a coherent English documentation system and reproducible gallery.

Visual polish must not precede mathematical integrity. A polished animation with ambiguous provenance would teach the wrong concept more effectively.

## Decisions already adopted

- `classic` remains the default visual contract.
- Existing fixed sizes and animation behavior remain defaults during phases 0–2.
- Motion is a pedagogical feature, not decorative overhead.
- The library distinguishes optimizer history, replayed history, and synthetic interpolation.
- Raw empirical values are never overwritten by smoothing.
- Temporal reduction preserves original checkpoint coordinates.
- Linear and logistic regression will share a formal explanatory grammar with neural-network views.
- New canonical documentation is English-first.
- Public figures and exported artifacts must state important assumptions and dependencies.

## Expected outcome

A student should be able to answer, from the figure and its documentation:

- What model is being shown?
- Which mathematical mapping produces the prediction?
- Which objective or metric is plotted?
- Are the states recorded, reconstructed, or interpolated?
- How many states existed and how many are displayed?
- Was a displayed curve smoothed?
- Is the query inside the observed data domain?
- Are parameters shown in scaled or original feature space?
- Which parts are exact estimator behavior and which are explanatory approximations?

The library is ready to be shown publicly only when those answers are consistent across code, figures, notebooks, exports, and documentation.
