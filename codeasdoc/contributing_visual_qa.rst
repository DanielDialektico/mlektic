Contributing and human visual QA policy
============================================================

Every new or materially changed public documentation page must add at least one
new separately executable visual cell to the corresponding notebook under
``notebooks/qa/``. This is a release requirement.

The cell must call a public Mlektic API, display the genuine figure, describe the
visible condition, use a unique ``metadata.mlektic_case_id``, contain no
assertions, and be registered under the documentation page in
``notebooks/visual_case_manifest.json``. Machine invariants belong in
``tests/``; human inspection cells are not disguised unit tests.

Workflow
========

1. Add or change implementation and unit tests.
2. Add a case specification to ``scripts/generate_notebooks.py``.
3. Run ``python scripts/generate_notebooks.py``.
4. Register the new case under its page in the generated manifest mapping.
5. Run ``python scripts/validate_notebook_policy.py``.
6. Execute the relevant notebook and inspect text, spacing, first/final state,
   playback, lesson stages, and responsive/static behavior.
7. Commit notebooks with cleared outputs.

Existing documentation-only wording changes may retain an existing mapping only
when no behavior, option, example, or visual claim changed. New pages never
inherit a case silently. Reference cases for this policy are ``DATA-LR-SCALE``
and ``DATA-LOG-OVERLAP``.
