from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

from rippleguard_agent_runtime.domain.errors import AgentFailure
from rippleguard_agent_runtime.loan_decision.features import feature_payload_digest


@dataclass(frozen=True)
class AgentRunInputIdentity:
    decision_case_id: str
    evaluation_run_id: str
    request_idempotency_key: str
    snapshot_digest: str
    feature_schema_version: str
    feature_payload_digest: str
    preprocessing_version: str
    model_version: str
    model_artifact_digest: str
    threshold_version: str


class AgentRunStateRepository:
    def __init__(self) -> None:
        self._lock = Lock()
        self._attempts: dict[str, int] = {}
        self._identities: dict[str, AgentRunInputIdentity] = {}
        self._completed_results: dict[tuple[str, AgentRunInputIdentity], dict[str, Any]] = {}

    def begin(self, request: dict[str, Any], identity: AgentRunInputIdentity) -> tuple[int, dict[str, Any] | None]:
        agent_run_id = str(request["agentRunId"])
        with self._lock:
            existing_identity = self._identities.get(agent_run_id)
            if existing_identity is not None and existing_identity != identity:
                raise AgentFailure("BLOCKED", "AGENT_RUN_INPUT_CONFLICT", "Agent run input identity changed.")
            cached = self._completed_results.get((agent_run_id, identity))
            if cached is not None:
                return self._attempts.get(agent_run_id, 1), cached
            self._identities[agent_run_id] = identity
            attempt = self._attempts.get(agent_run_id, 0) + 1
            self._attempts[agent_run_id] = attempt
            return attempt, None

    def complete(self, request: dict[str, Any], identity: AgentRunInputIdentity, result: dict[str, Any]) -> None:
        if result.get("resultStatus") != "COMPLETED":
            return
        agent_run_id = str(request["agentRunId"])
        with self._lock:
            self._completed_results[(agent_run_id, identity)] = result


def input_identity(request: dict[str, Any]) -> AgentRunInputIdentity:
    feature_payload = request.get("featurePayload")
    if not isinstance(feature_payload, dict):
        raise AgentFailure("VALIDATION_REQUIRED", "FEATURE_REQUIRED_MISSING", "Feature payload is required.")
    try:
        actual_feature_digest = feature_payload_digest(feature_payload)
    except ValueError as error:
        raise AgentFailure("VALIDATION_REQUIRED", "FEATURE_TYPE_INVALID", "Feature payload digest input is invalid.") from error
    return AgentRunInputIdentity(
        decision_case_id=str(request["decisionCaseId"]),
        evaluation_run_id=str(request["evaluationRunId"]),
        request_idempotency_key=str(request["requestIdempotencyKey"]),
        snapshot_digest=str(request["snapshotReference"]["snapshotDigest"]),
        feature_schema_version=str(request["featureSchemaVersion"]),
        feature_payload_digest=actual_feature_digest,
        preprocessing_version=str(request["preprocessingVersion"]),
        model_version=str(request["modelVersion"]),
        model_artifact_digest=str(request["modelArtifactDigest"]),
        threshold_version=str(request["thresholdVersion"]),
    )
