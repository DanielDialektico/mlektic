"""Phase 5 notebook and public-documentation contract tests."""

import importlib.util
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_notebook_policy", ROOT / "scripts" / "validate_notebook_policy.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_documentation_visual_qa_policy_is_complete():
    """Every public guide maps to real, assertion-free visual cells."""
    validator = _load_validator()
    assert validator.validate() == []


def test_canonical_notebooks_are_small_generated_and_cleared():
    """Canonical notebooks remain reviewable source artifacts, not output archives."""
    for directory in (ROOT / "notebooks" / "qa", ROOT / "notebooks" / "learn"):
        for path in directory.glob("*.ipynb"):
            notebook = nbformat.read(path, as_version=4)
            assert notebook.metadata["mlektic"]["generated"] is True
            assert path.stat().st_size < 100_000
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    assert cell.execution_count is None
                    assert cell.outputs == []


def test_archive_manifest_preserves_original_exploratory_names():
    """The local archive inventory prevents silent loss of prior visual work."""
    manifest = json.loads((ROOT / "notebooks" / "archive" / "archive_manifest.json").read_text())
    names = {Path(item["path"]).name for item in manifest["notebooks"]}
    assert names == {
        "Bocetos_Lib.ipynb",
        "Bocetos_Lib_Log.ipynb",
        "Bocetos_Lib_RLin.ipynb",
        "test_ann.ipynb",
        "test_interpt.ipynb",
        "test_linreg.ipynb",
        "test_logreg.ipynb",
    }
    assert all(len(item["sha256"]) == 64 and item["bytes"] > 0 for item in manifest["notebooks"])
