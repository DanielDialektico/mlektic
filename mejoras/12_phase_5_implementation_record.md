# Phase 5 implementation record — documentation and release readiness

## Scope and compatibility decision

Phase 5 packages the existing mathematical and visual work as a public learning
tool. It does not change classic styling, default dimensions, animation cadence,
frame generation, model routing, or mathematical outputs. Fluid motion remains
the default where it already existed; static output requires explicit report or
reduced-motion selection.

## Notebook architecture

The former ad hoc notebook collection is replaced as the canonical review path
by:

- eight `notebooks/qa/qa_*.ipynb` matrices for routes, linear, logistic, neural,
  motion, visual-system presets, data edge cases, and estimator hyperparameters;
- four `notebooks/learn/learn_*.ipynb` student lessons;
- 98 stable case IDs, each showing a real public-API figure in an independent
  assertion-free cell;
- deterministic generation by `scripts/generate_notebooks.py` and non-mutating
  execution by `scripts/execute_notebooks.py`.

Meaningful orthogonal variants are covered without constructing a wasteful
Cartesian product. A case is distinct when it changes mathematical semantics,
geometry, temporal provenance, data regime, visible composition, accessibility,
or a documented control. Equivalent combinations inherit the independently
tested axes.

## Preservation decision

The three small phase notebooks moved to `notebooks/archive/phases/` through
Git-aware renames. Seven original exploratory notebooks moved locally to
`notebooks/archive/exploratory/` without clearing or deleting their outputs.
Several are tens of megabytes and `test_logreg.ipynb` exceeds GitHub's 100 MB
per-file limit, so they remain ignored. The tracked
`notebooks/archive/archive_manifest.json` records every filename, byte size, and
SHA-256 checksum.

## Documentation architecture

Sphinx now provides introduction, installation, first linear/logistic/neural
lessons, history provenance, mathematical conventions, animation/performance,
visual options, prediction explanation, export, gallery, compatibility,
limitations, API, architecture, advanced usage, and contribution guidance. All
new canonical text is English. Each behavioral guide identifies exact versus
recorded/replayed/interpolated content and maps to executable visual cases.

## Permanent policy

Every new or materially changed public documentation page must add a new human
visual-QA cell. `notebooks/visual_case_manifest.json`,
`scripts/validate_notebook_policy.py`, CI, `CONTRIBUTING.md`, and
`11_visual_qa_notebook_policy.md` make this an enforceable engineering rule.

## Packaging and automation

`pyproject.toml` now declares the Apache-2.0 license, classifiers, keywords,
project URLs, Python support, and explicit torch/notebooks/docs/dev extras.
`mlektic.__version__` uses installed distribution metadata. CI includes:

- core tests on Python 3.9, 3.10, 3.11, 3.12, and 3.13;
- Ruff, strict Sphinx, source/wheel build, and clean wheel import;
- optional PyTorch tests;
- pull-request notebook smoke execution;
- manual and weekly execution of the complete visual notebook matrix.

## Validation commands

```text
python scripts/generate_notebooks.py
python scripts/validate_notebook_policy.py
python scripts/execute_notebooks.py --group smoke
python -m pytest
python -m ruff check src tests scripts codeasdoc
sphinx-build -W -b html codeasdoc codeasdoc/_build/html
python -m build
```

## Validation evidence

The completed Phase 5 branch produced the following results on Windows with
Python 3.11:

- notebook policy: 98 cases and 22 public documentation mappings;
- human notebook execution: all eight QA notebooks and all four learning
  notebooks executed in memory without writing outputs;
- automated tests: 137 passed; the 39 warnings are existing expected
  Scikit-learn convergence warnings from deliberately short SGD fixtures;
- Ruff: all checks passed across `src`, `tests`, `scripts`, and `codeasdoc`;
- Sphinx: strict clean-environment HTML build succeeded with warnings treated as
  errors;
- packaging: lockfile check succeeded and both `mlektic-0.1.0.tar.gz` and the
  169 KB `mlektic-0.1.0-py3-none-any.whl` built successfully;
- clean install: the wheel installed in an isolated virtual environment,
  reported version `0.1.0`, and exposed the public linear API.

Jupyter emitted the standard Windows Proactor/ZMQ selector-thread runtime
warning during notebook execution. It did not affect a cell or figure. The
temporary isolated wheel environment was created outside the repository; the
tool sandbox allowed validation but declined its subsequent recursive cleanup.
