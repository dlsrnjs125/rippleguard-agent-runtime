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


def test_decline_result_is_schema_valid_completed_result(service: object, valid_request: dict[str, object]) -> None:
    payload = valid_request["featurePayload"]  # type: ignore[index]
    features = payload["features"]  # type: ignore[index]
    features["debtToIncomeRatio"] = 0.95  # type: ignore[index]
    features["existingDebtAmount"] = 180000000  # type: ignore[index]
    features["delinquencyCount"] = 3  # type: ignore[index]
    features["platformSettlementMonths"] = 3  # type: ignore[index]
    features["monthlyIncomeVolatility"] = 0.65  # type: ignore[index]
    from rippleguard_agent_runtime.loan_decision.features import feature_payload_digest

    payload["featurePayloadDigest"] = feature_payload_digest(payload)  # type: ignore[index]
    result = service.run(valid_request)  # type: ignore[attr-defined]
    assert result["resultStatus"] == "COMPLETED"
    assert result["proposal"]["proposalOutcome"] == "RECOMMEND_DECLINE"
    assert set(result["proposal"]["reasonCodes"]).issubset(
        {
            "LOW_DTI",
            "STABLE_INCOME",
            "HIGH_DTI",
            "RECENT_DELINQUENCY",
            "INSUFFICIENT_HISTORY",
            "VOLATILE_SETTLEMENTS",
        }
    )


def test_feature_payload_digest_mismatch_returns_failed_result(
    service: object, valid_request: dict[str, object]
) -> None:
    payload = valid_request["featurePayload"]  # type: ignore[index]
    payload["featurePayloadDigest"] = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"  # type: ignore[index]
    result = service.run(valid_request)  # type: ignore[attr-defined]
    assert result["resultStatus"] == "FAILED"
    assert result["failure"]["reasonCode"] == "SNAPSHOT_DIGEST_MISMATCH"
    assert "proposal" not in result
