"""Execute canonical notebooks without storing generated outputs in the repository."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def selected_notebooks(group: str) -> list[Path]:
    """Resolve a named CI execution group."""
    if group == "smoke":
        return [
            ROOT / "notebooks/qa/qa_00_smoke_matrix.ipynb",
            ROOT / "notebooks/learn/learn_00_getting_started.ipynb",
        ]
    if group == "core":
        return sorted((ROOT / "notebooks/qa").glob("*.ipynb")) + sorted(
            path for path in (ROOT / "notebooks/learn").glob("*.ipynb") if "neural" not in path.name
        )
    if group == "all":
        return sorted((ROOT / "notebooks/qa").glob("*.ipynb")) + sorted((ROOT / "notebooks/learn").glob("*.ipynb"))
    raise ValueError(f"Unknown notebook group: {group}")


def main() -> None:
    """Execute notebooks in memory and report progress."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("smoke", "core", "all"), default="smoke")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    paths = selected_notebooks(args.group)
    for path in paths:
        if args.group == "core" and "neural" in path.name:
            continue
        print(f"Executing {path.relative_to(ROOT)}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=args.timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        )
        try:
            client.execute()
        except Exception as exc:
            print(f"Notebook failed: {path.relative_to(ROOT)}\n{exc}", file=sys.stderr)
            raise
    print(f"Executed {len(paths)} notebook(s) without modifying committed outputs.")


if __name__ == "__main__":
    main()
