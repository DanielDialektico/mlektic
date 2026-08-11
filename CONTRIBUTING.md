# Contributing to Mlektic

Mlektic is an educational visualization library. Mathematical honesty,
provenance, readable composition, and fluid motion are part of the public
contract, not optional polish.

## Development setup

```bash
python -m pip install -e ".[dev,torch]"
python -m pytest
python -m ruff check src tests scripts codeasdoc
sphinx-build -W -b html codeasdoc codeasdoc/_build/html
```

## Required visual-documentation workflow

Every new or materially changed public documentation page must add at least one
new separately executable human visual-QA cell to the corresponding notebook in
`notebooks/qa/`. A compliant cell calls a public API, displays the real Plotly
figure, has a unique `metadata.mlektic_case_id`, explains what a reviewer should
inspect, contains no assertions, and is registered under that document in
`notebooks/visual_case_manifest.json`.

Add the case to `scripts/generate_notebooks.py`, regenerate the notebooks, and
run:

```bash
python scripts/generate_notebooks.py
python scripts/validate_notebook_policy.py
python scripts/execute_notebooks.py --group smoke
```

Machine expectations belong in `tests/`; human notebooks must remain readable
as end-user examples. Commit canonical notebooks with all outputs cleared.

## Visual review checklist

- Inspect the initial and final states and use Play/Pause.
- Confirm equations, geometry, cards, legends, and prediction labels agree.
- Confirm replay/interpolation/recorded provenance is visible and accurate.
- Check the relevant themes, formats, sizes, density, and motion settings.
- For a lesson composition, inspect Data, Model, Objective, and Complete stages.
- Check long feature names, large values, and high-dimensional truncation.
- Preserve the classic default unless a versioned compatibility change is approved.

See `codeasdoc/contributing_visual_qa.rst` and `notebooks/README.md` for the
enforced policy and notebook organization.
