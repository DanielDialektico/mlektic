"""Static safeguards for the supported Python 3.9 import contract."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src" / "mlektic"


def _annotation_nodes(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            yield node.annotation
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                yield node.returns
            arguments = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
            for argument in arguments:
                if argument.annotation is not None:
                    yield argument.annotation
            if node.args.vararg is not None and node.args.vararg.annotation is not None:
                yield node.args.vararg.annotation
            if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
                yield node.args.kwarg.annotation


def _uses_pep604_union(tree):
    return any(
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)
        for annotation in _annotation_nodes(tree)
        for node in ast.walk(annotation)
    )


def _postpones_annotation_evaluation(tree):
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )


def test_pep604_annotations_are_not_evaluated_during_python39_imports():
    """Modules using ``X | None`` annotations must postpone their evaluation."""
    incompatible_modules = []
    for path in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if _uses_pep604_union(tree) and not _postpones_annotation_evaluation(tree):
            incompatible_modules.append(path.relative_to(ROOT).as_posix())

    assert incompatible_modules == []
