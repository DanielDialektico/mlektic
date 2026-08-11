# Mlektic notebooks
reject undocumented visual behavior and visual documentation without a QA cell.
The notebooks have three distinct jobs:

- `learn/` contains focused, student-facing lessons. Each notebook explains
  what is exact, replayed, recorded, or synthetically interpolated.
- `qa/` contains the canonical human visual-QA matrix. Each meaningful
  configuration has a stable case ID and one separately executable display
  cell. These are real library figures, not mocks and not unit-test wrappers.
- `archive/` preserves earlier exploratory and phase notebooks. Large local
  exploratory notebooks remain ignored because some exceed GitHub's file-size
  limit; `archive/archive_manifest.json` records their names, sizes, and SHA-256
  hashes. The smaller phase notebooks remain versioned with their history.

Regenerate the canonical notebooks with:

```bash
python scripts/generate_notebooks.py
```

Validate the documentation-to-notebook contract with:

```bash
python scripts/validate_notebook_policy.py
```

## Mandatory documentation rule

Every new or materially changed public documentation page must add at least one
new, separately executable visual inspection cell to the corresponding
`notebooks/qa/` notebook. The cell must:

1. call a public Mlektic API and display the real returned figure;
2. have a unique `metadata.mlektic_case_id` value;
3. explain the human-visible condition being reviewed;
4. be registered in `notebooks/visual_case_manifest.json` under that document;
5. contain no assertions (machine assertions belong in `tests/`).

Purely textual edits may keep the existing mapping only when they do not add or
change a documented behavior, option, example, or visual claim. Reviewers must
reject undocumented visual behavior and visual documentation without a QA cell.
