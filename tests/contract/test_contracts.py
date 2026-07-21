from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Event
from time import sleep

import pytest

from rippleguard_agent_runtime.adapters.contracts import ContractValidationError
from rippleguard_agent_runtime.adapters.xgboost_model import XGBoostModelAdapter
from rippleguard_agent_runtime.domain.errors import AgentFailure
from rippleguard_agent_runtime.loan_decision.features import feature_payload_digest
import rippleguard_agent_runtime.loan_decision.service as service_module
from rippleguard_agent_runtime.loan_decision import result_builder
from rippleguard_agent_runtime.loan_decision.service import LoanDecisionAgentService


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


def test_duplicate_with_bad_declared_feature_digest_does_not_return_cached_success(
    service: object, valid_request: dict[str, object]
) -> None:
    first = service.run(valid_request)  # type: ignore[attr-defined]
    assert first["resultStatus"] == "COMPLETED"
    first_attempt = first["agentRun"]["attemptId"]
    payload = valid_request["featurePayload"]  # type: ignore[index]
    payload["featurePayloadDigest"] = "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"  # type: ignore[index]
    result = service.run(valid_request)  # type: ignore[attr-defined]
    assert result["resultStatus"] == "FAILED"
    assert result["agentRun"]["attemptId"] != first_attempt
    assert result["failure"]["classification"] == "BLOCKED"
    assert result["failure"]["reasonCode"] == "SNAPSHOT_DIGEST_MISMATCH"


def test_duplicate_identical_request_returns_existing_result(service: object, valid_request: dict[str, object]) -> None:
    first = service.run(valid_request)  # type: ignore[attr-defined]
    second = service.run(valid_request)  # type: ignore[attr-defined]
    assert second == first


def test_same_agent_run_different_snapshot_is_blocked(service: object, valid_request: dict[str, object]) -> None:
    first = service.run(valid_request)  # type: ignore[attr-defined]
    assert first["resultStatus"] == "COMPLETED"
    first_attempt = first["agentRun"]["attemptId"]
    valid_request["snapshotReference"]["snapshotDigest"] = (  # type: ignore[index]
        "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
    )
    result = service.run(valid_request)  # type: ignore[attr-defined]
    assert result["resultStatus"] == "FAILED"
    assert result["agentRun"]["attemptId"] != first_attempt
    assert result["failure"]["reasonCode"] == "AGENT_RUN_INPUT_CONFLICT"


def test_same_agent_run_different_feature_payload_is_blocked(
    service: object, valid_request: dict[str, object]
) -> None:
    first = service.run(valid_request)  # type: ignore[attr-defined]
    assert first["resultStatus"] == "COMPLETED"
    payload = valid_request["featurePayload"]  # type: ignore[index]
    payload["features"]["debtToIncomeRatio"] = 0.75  # type: ignore[index]
    payload["featurePayloadDigest"] = feature_payload_digest(payload)  # type: ignore[index]
    result = service.run(valid_request)  # type: ignore[attr-defined]
    assert result["resultStatus"] == "FAILED"
    assert result["failure"]["reasonCode"] == "AGENT_RUN_INPUT_CONFLICT"


def test_same_agent_run_different_model_is_blocked(service: object, valid_request: dict[str, object]) -> None:
    first = service.run(valid_request)  # type: ignore[attr-defined]
    assert first["resultStatus"] == "COMPLETED"
    valid_request["modelVersion"] = "loan-model.v9.9.9"
    result = service.run(valid_request)  # type: ignore[attr-defined]
    assert result["resultStatus"] == "FAILED"
    assert result["failure"]["reasonCode"] == "AGENT_RUN_INPUT_CONFLICT"


def test_same_agent_run_different_threshold_is_blocked(service: object, valid_request: dict[str, object]) -> None:
    first = service.run(valid_request)  # type: ignore[attr-defined]
    assert first["resultStatus"] == "COMPLETED"
    valid_request["thresholdVersion"] = "threshold.v9.9.9"
    result = service.run(valid_request)  # type: ignore[attr-defined]
    assert result["resultStatus"] == "FAILED"
    assert result["failure"]["reasonCode"] == "AGENT_RUN_INPUT_CONFLICT"


def test_concurrent_duplicate_identical_request_single_flights(
    monkeypatch: pytest.MonkeyPatch, service: object, valid_request: dict[str, object]
) -> None:
    calls = 0
    original_predict = XGBoostModelAdapter.predict

    def counted_predict(self: XGBoostModelAdapter, features: object) -> object:
        nonlocal calls
        calls += 1
        return original_predict(self, features)  # type: ignore[arg-type]

    monkeypatch.setattr(XGBoostModelAdapter, "predict", counted_predict)
    requests = [deepcopy(valid_request) for _ in range(10)]
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(lambda payload: service.run(payload), requests))  # type: ignore[attr-defined]

    assert calls == 1
    assert len({result["proposal"]["proposalId"] for result in results}) == 1
    assert all(result == results[0] for result in results)


def test_unexpected_exception_releases_waiting_duplicate_request(
    monkeypatch: pytest.MonkeyPatch, service: object, valid_request: dict[str, object]
) -> None:
    first_predict_started = Event()
    calls = 0
    original_predict = XGBoostModelAdapter.predict

    def flaky_predict(self: XGBoostModelAdapter, features: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            first_predict_started.set()
            sleep(0.1)
            raise RuntimeError("unexpected inference failure")
        return original_predict(self, features)  # type: ignore[arg-type]

    monkeypatch.setattr(XGBoostModelAdapter, "predict", flaky_predict)

    def run_once(payload: dict[str, object]) -> dict[str, object] | str:
        try:
            return service.run(payload)  # type: ignore[attr-defined,no-any-return]
        except RuntimeError:
            return "raised"

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(run_once, deepcopy(valid_request))
        assert first_predict_started.wait(timeout=5)
        second_future = executor.submit(run_once, deepcopy(valid_request))
        results = [first_future.result(timeout=5), second_future.result(timeout=5)]

    assert "raised" in results
    completed = [result for result in results if isinstance(result, dict)]
    assert len(completed) == 1
    assert completed[0]["resultStatus"] == "COMPLETED"
    assert calls == 2


def test_retryable_failure_reexecution_uses_next_attempt(
    monkeypatch: pytest.MonkeyPatch, service: object, valid_request: dict[str, object]
) -> None:
    calls = 0
    original_threshold = LoanDecisionAgentService._threshold

    def flaky_threshold(self: LoanDecisionAgentService, threshold_version: str) -> float:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise AgentFailure("RETRYABLE", "AGENT_RUNTIME_TEMPORARY_FAILURE", "Temporary threshold load failure.")
        return original_threshold(self, threshold_version)

    monkeypatch.setattr(LoanDecisionAgentService, "_threshold", flaky_threshold)
    first = service.run(valid_request)  # type: ignore[attr-defined]
    second = service.run(valid_request)  # type: ignore[attr-defined]

    assert first["resultStatus"] == "FAILED"
    assert first["failure"]["classification"] == "RETRYABLE"
    assert first["agentRun"]["attemptId"] == 1
    assert second["resultStatus"] == "COMPLETED"
    assert second["agentRun"]["attemptId"] == 2


def test_recoverable_blocked_failure_is_not_cached(
    monkeypatch: pytest.MonkeyPatch, service: object, valid_request: dict[str, object]
) -> None:
    calls = 0
    original_threshold = LoanDecisionAgentService._threshold

    def missing_then_present(self: LoanDecisionAgentService, threshold_version: str) -> float:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise AgentFailure("BLOCKED", "MODEL_MANIFEST_NOT_FOUND", "Model manifest was not found.")
        return original_threshold(self, threshold_version)

    monkeypatch.setattr(LoanDecisionAgentService, "_threshold", missing_then_present)
    first = service.run(valid_request)  # type: ignore[attr-defined]
    second = service.run(valid_request)  # type: ignore[attr-defined]

    assert first["resultStatus"] == "FAILED"
    assert first["failure"]["reasonCode"] == "MODEL_MANIFEST_NOT_FOUND"
    assert second["resultStatus"] == "COMPLETED"
    assert calls == 2


def test_invalid_completed_result_is_not_cached(
    monkeypatch: pytest.MonkeyPatch, service: object, valid_request: dict[str, object]
) -> None:
    calls = 0
    original_builder = result_builder.build_completed_result

    def invalid_then_valid(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        result = original_builder(*args, **kwargs)  # type: ignore[arg-type]
        if calls == 1:
            result["proposal"]["reasonCodes"] = ["MODEL_SCORE_BELOW_THRESHOLD"]  # type: ignore[index]
        return result

    monkeypatch.setattr(service_module, "build_completed_result", invalid_then_valid)
    with pytest.raises(ContractValidationError):
        service.run(valid_request)  # type: ignore[attr-defined]

    second = service.run(valid_request)  # type: ignore[attr-defined]
    assert second["resultStatus"] == "COMPLETED"
    assert calls == 2
