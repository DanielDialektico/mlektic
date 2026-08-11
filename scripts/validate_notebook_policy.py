"""Validate Mlektic's public-documentation to human visual-QA contract."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "notebooks" / "visual_case_manifest.json"
EXEMPT_DOCUMENTS = {
    "codeasdoc/api_reference.rst",  # generated API inventory; examples live in guides
    "codeasdoc/changelog.rst",  # release ledger; links to implementation records
    "codeasdoc/index.rst",  # navigation only
}


def validate() -> list[str]:
    """Return contract violations without mutating any notebooks."""
    errors: list[str] = []
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cases = payload.get("cases", {})
    documents = payload.get("documents", {})

    public_documents = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "codeasdoc").glob("*.rst")
        if path.relative_to(ROOT).as_posix() not in EXEMPT_DOCUMENTS
    }
    missing_documents = sorted(public_documents - set(documents))
    unknown_documents = sorted(set(documents) - public_documents)
    if missing_documents:
        errors.append(f"Public documentation has no visual-QA mapping: {missing_documents}")
    if unknown_documents:
        errors.append(f"Manifest references absent/exempt documentation: {unknown_documents}")

    discovered: dict[str, tuple[str, int]] = {}
    for relative in sorted({entry["notebook"] for entry in cases.values()}):
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"Notebook does not exist: {relative}")
            continue
        notebook = nbformat.read(path, as_version=4)
        for index, cell in enumerate(notebook.cells):
            case_id = cell.get("metadata", {}).get("mlektic_case_id")
            if not case_id:
                continue
            if case_id in discovered:
                errors.append(f"Duplicate case ID {case_id}: {discovered[case_id]} and {(relative, index)}")
            discovered[case_id] = (relative, index)
            if cell.cell_type != "code":
                errors.append(f"Case {case_id} must be a code cell.")
                continue
            try:
                tree = ast.parse(cell.source)
            except SyntaxError as exc:
                errors.append(f"Case {case_id} contains invalid Python: {exc}")
                continue
            if any(isinstance(node, ast.Assert) for node in ast.walk(tree)):
                errors.append(f"Case {case_id} contains an assert; assertions belong in tests/.")
            if "display(" not in cell.source:
                errors.append(f"Case {case_id} does not explicitly display a real figure.")
            if cell.get("outputs") or cell.get("execution_count") is not None:
                errors.append(f"Case {case_id} must be committed with cleared outputs.")

    if set(cases) != set(discovered):
        errors.append(
            "Manifest/notebook case IDs differ: "
            f"missing={sorted(set(cases) - set(discovered))}, "
            f"unregistered={sorted(set(discovered) - set(cases))}"
        )
    for case_id, expected in cases.items():
        actual = discovered.get(case_id)
        if actual and actual != (expected["notebook"], expected["cell_index"]):
            errors.append(f"Case {case_id} location changed: manifest={expected}, notebook={actual}.")
    for document, mapped_cases in documents.items():
        if not mapped_cases:
            errors.append(f"Document {document} must map to at least one case.")
        for case_id in mapped_cases:
            if case_id not in cases:
                errors.append(f"Document {document} references unknown case {case_id}.")
    return errors


def main() -> None:
    """Exit nonzero and print every policy violation."""
    errors = validate()
    if errors:
        raise SystemExit("\n".join(f"- {error}" for error in errors))
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    print(
        "Visual notebook policy valid: "
        f"{len(payload['cases'])} cases, {len(payload['documents'])} public documentation mappings."
    )


if __name__ == "__main__":
    main()
