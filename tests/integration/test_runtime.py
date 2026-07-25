from __future__ import annotations

import json
from pathlib import Path

import pytest

from rippleguard_agent_runtime.adapters.contracts import ContractValidationError
from rippleguard_agent_runtime.adapters.digest import file_sha256
from rippleguard_agent_runtime.loan_decision.service import LoanDecisionAgentService

ROOT = Path(__file__).resolve().parents[2]


def test_readiness_loads_manifest_artifact_and_model(service: object) -> None:
    ready = service.readiness()  # type: ignore[attr-defined]
    assert ready["status"] == "ready"
    assert ready["modelVersion"] == "loan-model.v1.0.0"
    assert ready["provenanceStatus"] == "MATERIALIZED"


def test_artifact_digest_matches_manifest() -> None:
    manifest = json.loads((ROOT / "tests" / "fixtures" / "model-manifest-valid.json").read_text())
    assert manifest["modelBinaryArtifactDigest"].startswith("sha256:")
    artifact = ROOT / "artifacts" / "models" / manifest["modelBinaryArtifactReference"].removeprefix("file://")
    assert artifact.is_file()
    assert file_sha256(str(artifact)) == manifest["modelBinaryArtifactDigest"]


def test_template_manifest_fails_readiness() -> None:
    service = LoanDecisionAgentService(
        contracts_root=ROOT.parent / "rippleguard-contracts",
        manifest_path=ROOT / "artifacts" / "templates" / "phase2-loan-xgboost.v1.0.0.template.json",
        artifact_root=ROOT / "artifacts" / "models",
    )
    with pytest.raises(ContractValidationError):
        service.readiness()
