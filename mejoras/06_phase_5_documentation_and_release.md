# Phase 5 — Documentation, packaging, and public release

## Objective

Present Mlektic as a coherent public learning tool with an English-first documentation system, reproducible examples, explicit compatibility, and installation metadata that matches actual runtime behavior.

## Documentation architecture

### README

The README should answer in the first screen:

- what Mlektic teaches;
- which model families are supported;
- how history provenance is handled;
- one small installation command;
- one minimal example and image/GIF;
- links to learning guides, API reference, and limitations.

Avoid presenting a synthetic interpolation as training. Include a one-paragraph explanation of recorded, replayed, and interpolated sources near the first animated example.

### Guides

Proposed Sphinx structure:

```text
Introduction
Installation
First linear lesson
First logistic lesson
First neural lesson
History provenance and time
Mathematical conventions
Animation and performance
Themes, formats, and sizing
Prediction explanations
HTML export
API reference
Compatibility matrix
Limitations and troubleshooting
Contributing and visual QA
```

Every guide includes executable code, expected interpretation, and a “What is exact?” box distinguishing estimator behavior from explanatory construction.

## Language policy

- canonical public documentation, docstrings, errors, metadata fields, tests, and planning records are English;
- examples may later have Spanish translations, but translations must link to a canonical English source/version;
- code identifiers remain English;
- mathematical notation is language-neutral but every symbol is defined in English prose;
- existing mixed-language pages are migrated before the public release milestone.

## Compatibility tables

Maintain tables by model family and estimator capability:

| Family | Dimensionality | Fitted geometry | Replay | Interpolation | Prediction explainer | Pipeline scaling |
|---|---:|---:|---:|---:|---:|---:|
| Linear | 1D | yes | incremental only | yes | yes | affine scalers |
| Linear | 2D | yes | incremental only | yes | yes | affine scalers |
| Linear | nD | report | incremental only | yes | yes | affine scalers |
| Logistic binary | 1D/2D/nD | yes/report | incremental only | yes | yes | affine scalers |
| Logistic multiclass | 1D/2D/nD | yes/report | incremental only | yes | yes | affine scalers |
| Neural | architecture/training/report | recorder | n/a | display sampling | yes | model-defined |

Add estimator-specific notes for `SGDRegressor`, `SGDClassifier`, `LinearRegression`, `LogisticRegression`, pipelines, feature transforms, and probability fallbacks.

## Public API

- define one canonical import path per feature;
- retain compatibility bridges but mark them as such;
- publish complete signatures and accepted enum values;
- document defaults and their visual implications;
- include provenance metadata schemas;
- document whether a function mutates, clones, fits, predicts, or writes files;
- document tolerances and failure behavior for prediction verification.

## Packaging

### Python version

The declared `>=3.9` support must be tested or raised. All modules using modern annotations need `from __future__ import annotations` where Python 3.9 would otherwise evaluate unsupported syntax.

### Dependencies

Audit imports against metadata:

- core: NumPy, Scikit-learn, Plotly;
- notebook display: IPython should be optional and imported lazily;
- neural: Torch extra;
- docs: Sphinx, theme, intersphinx dependencies;
- QA: pytest, Ruff, notebook execution, screenshot tools as optional development dependencies.

Do not make a package import fail because an optional notebook renderer is absent.

### Metadata

Add accurate author/project URLs, classifiers, license metadata, keywords, issue tracker, documentation URL, and tested Python versions. Establish semantic versioning rules for figure defaults and metadata schema changes.

## CI

Required jobs:

1. unit tests across supported Python versions;
2. Ruff and documentation build with warnings treated as errors;
3. package build and installation smoke test in a clean environment;
4. core import without IPython or Torch;
5. optional neural tests with Torch;
6. lightweight notebook execution;
7. semantic visual invariants;
8. scheduled or release-time screenshot regression suite;
9. artifact size and performance budget checks.

## Public gallery

Curate a small intentional gallery:

- linear 1D replay with source subtitle;
- linear 2D interpolation with α slider;
- binary logistic sigmoid/boundary lesson;
- multiclass selected-class probability view;
- nD contribution report;
- neural architecture, training, and prediction explanation;
- classic vs academic/compact comparison;
- in-range vs extrapolation prediction explanation.

Each item provides code, version, configuration, data description, exact/synthetic status, and a static fallback image. Large HTML files are release/site artifacts, not tracked root files.

## Repository artifacts

- remove or relocate generated root HTML after confirming they are reproducible;
- keep generated Sphinx `_build` out of version control unless publication requires it;
- define notebook output policy;
- add scripts or documented commands for gallery regeneration;
- record artifact checksums/version metadata when published;
- never delete existing user artifacts during feature work without explicit approval.

## Notebook publication

Learning notebooks should be launchable through documented Jupyter/Colab links, pin or record the package version, run in a reasonable time, avoid hidden state, and provide static fallbacks. QA notebooks remain maintainer-facing.

## Release criteria

Phase 5 is complete when a clean install works on every supported Python version; English docs build without warnings; public examples reproduce; compatibility and limitations are explicit; generated artifacts are organized; optional dependencies are truly optional; gallery cases identify provenance; and a new student can complete the first lesson without repository-specific knowledge.

## Deliverable

A release candidate containing the package, English documentation site, curated gallery, learning notebooks, compatibility tables, changelog/migration notes, and CI evidence.
