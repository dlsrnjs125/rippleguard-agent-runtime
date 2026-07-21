from __future__ import annotations

import json
from pathlib import Path


def test_readiness_loads_manifest_artifact_and_model(service: object) -> None:
    ready = service.readiness()  # type: ignore[attr-defined]
    assert ready["status"] == "ready"
    assert ready["modelVersion"] == "loan-model.v1.0.0"


def test_artifact_digest_matches_manifest() -> None:
    manifest = json.loads(Path("artifacts/manifests/phase2-loan-xgboost.v1.0.0.json").read_text())
    assert manifest["modelBinaryArtifactDigest"].startswith("sha256:")
    artifact = Path("artifacts/models") / manifest["modelBinaryArtifactReference"].removeprefix("file://")
    assert artifact.is_file()
