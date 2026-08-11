"""Write a reproducible inventory for locally preserved exploratory notebooks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "notebooks" / "archive" / "exploratory"
DESTINATION = ROOT / "notebooks" / "archive" / "archive_manifest.json"


def sha256(path: Path) -> str:
    """Return the SHA-256 checksum of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Record archived notebook identities, sizes, and checksums."""
    notebooks = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "tracked": False,
            "reason": (
                "Preserved exploratory output; excluded from Git because large "
                "embedded outputs are not release artifacts."
            ),
        }
        for path in sorted(ARCHIVE.glob("*.ipynb"))
    ]
    payload = {
        "schema_version": 1,
        "description": "Local historical notebooks preserved before the Phase 5 canonical QA replacement.",
        "notebooks": notebooks,
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
