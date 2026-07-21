from __future__ import annotations

import pytest

from rippleguard_agent_runtime.adapters.contracts import ContractValidationError


def test_valid_request_generates_schema_valid_completed_result(
    service: object, valid_request: dict[str, object]
) -> None:
    result = service.run(valid_request)  # type: ignore[attr-defined]
    assert result["resultStatus"] == "COMPLETED"
    assert "proposal" in result
    assert "explanationDigest" in result
    assert "localLlmPromptVersion" not in result


def test_malformed_request_is_rejected_before_failed_result(service: object, valid_request: dict[str, object]) -> None:
    del valid_request["agentRunId"]
    with pytest.raises(ContractValidationError):
        service.run(valid_request)  # type: ignore[attr-defined]


def test_digest_mismatch_returns_failed_result(service: object, valid_request: dict[str, object]) -> None:
    valid_request["modelArtifactDigest"] = "sha256:9999999999999999999999999999999999999999999999999999999999999999"
    result = service.run(valid_request)  # type: ignore[attr-defined]
    assert result["resultStatus"] == "FAILED"
    assert result["failure"]["reasonCode"] == "MODEL_ARTIFACT_DIGEST_MISMATCH"
    assert "proposal" not in result
