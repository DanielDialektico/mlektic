# Permanent documentation-to-human-visual-QA policy
subframes are never presented as optimizer steps.
## Decision

Every new or materially changed public documentation page must add at least one
new, separately executable cell to the corresponding canonical notebook under
`notebooks/qa/`. This requirement is permanent and applies to documentation of
new model routes, parameters, hyperparameters, data regimes, mathematical
claims, layouts, themes, formats, sizes, motion behavior, prediction behavior,
exports, and limitations.

## Why this is required

Unit tests can prove that arrays, metadata, formulas, coordinates, and frame
contracts are internally consistent. They cannot decide whether two LaTeX rows
overlap, a prediction label disappears over a surface, a matrix becomes
unreadable, a lesson stage hides a model unexpectedly, or a layout wastes its
available space. Public visual documentation therefore requires both machine
invariants and human visual evidence.

## Cell contract

A compliant cell:

1. calls a public Mlektic API and displays the genuine returned figure;
2. is independently executable after the notebook setup cells;
3. has one unique `metadata.mlektic_case_id`;
4. is preceded by the visible condition that a reviewer must inspect;
5. contains no assertions;
6. is committed with cleared output;
7. is registered under the related documentation path in
   `notebooks/visual_case_manifest.json`.

Assertions belong in `tests/`. A notebook cell may compute deterministic data,
fit a real model, and display its figure, but it must remain understandable to a
library user.

## Enforcement

`scripts/generate_notebooks.py` is the canonical source for notebook cells and
the case manifest. `scripts/validate_notebook_policy.py` fails when a public RST
page is unmapped, a case is duplicated or missing, a case contains an assertion,
the figure is not explicitly displayed, or committed output is present. CI runs
this validator on every pull request.

New pages always require a new case. A wording-only edit to an existing page may
retain its existing case mapping only if it changes no documented behavior,
option, example, or visual claim. Reviewers make that determination explicitly.

## Human review protocol

Review the first and final states, Play/Pause, relevant slider positions,
equations, geometry, metric cards, legends, prediction boxes, long values,
responsive/static output, and lesson stages. Confirm that recorded, replayed,
interpolated, and fitted states are labeled honestly and that perceptual
subframes are never presented as optimizer steps.
