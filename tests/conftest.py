from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from rippleguard_agent_runtime.loan_decision.service import LoanDecisionAgentService
from rippleguard_agent_runtime.loan_decision.features import feature_payload_digest


ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT.parent / "rippleguard-contracts"


@pytest.fixture()
def service() -> LoanDecisionAgentService:
    return LoanDecisionAgentService(
        contracts_root=CONTRACTS,
        manifest_path=ROOT / "artifacts" / "manifests" / "phase2-loan-xgboost.v1.0.0.json",
        artifact_root=ROOT / "artifacts" / "models",
    )


@pytest.fixture()
def valid_request() -> dict[str, Any]:
    payload = load_contract_fixture("examples/valid/commands/v1.0.0/loan-decision-agent-request.json")
    payload = deepcopy(payload)
    now = datetime.now(timezone.utc)
    payload["snapshotReference"]["snapshotCreatedAt"] = (now - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    payload["requestedAt"] = (now - timedelta(seconds=5)).isoformat().replace("+00:00", "Z")
    payload["deadlineAt"] = (now + timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    manifest = json.loads((ROOT / "artifacts" / "manifests" / "phase2-loan-xgboost.v1.0.0.json").read_text())
    payload["modelArtifactDigest"] = manifest["modelBinaryArtifactDigest"]
    payload["featurePayload"]["featurePayloadDigest"] = feature_payload_digest(payload["featurePayload"])
    return payload


def load_contract_fixture(relative: str) -> dict[str, Any]:
    return json.loads((CONTRACTS / relative).read_text(encoding="utf-8"))
