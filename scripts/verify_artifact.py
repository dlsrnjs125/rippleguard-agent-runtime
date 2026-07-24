#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rippleguard_agent_runtime.adapters.digest import file_sha256  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: verify_artifact.py <manifest> <artifact-root>", file=sys.stderr)
        return 2
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    artifact = Path(sys.argv[2]) / str(manifest["modelBinaryArtifactReference"]).removeprefix("file://")
    actual = file_sha256(str(artifact))
    if actual != manifest["modelBinaryArtifactDigest"]:
        print(f"digest mismatch: expected {manifest['modelBinaryArtifactDigest']} actual {actual}", file=sys.stderr)
        return 1
    print(actual)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
