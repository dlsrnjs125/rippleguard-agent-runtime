from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from threading import Event, Lock
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


class RunStatus(Enum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class RunEntry:
    identity: AgentRunInputIdentity
    status: RunStatus
    attempt_id: int
    result: dict[str, Any] | None = None
    done: Event = field(default_factory=Event)


class AgentRunInputConflict(AgentFailure):
    attempt_id: int

    def __init__(self, attempt_id: int) -> None:
        super().__init__("BLOCKED", "AGENT_RUN_INPUT_CONFLICT", "Agent run input identity changed.")
        object.__setattr__(self, "attempt_id", attempt_id)


class AgentRunStateRepository:
    def __init__(self) -> None:
        self._lock = Lock()
        self._attempts: dict[str, int] = {}
        self._identities: dict[str, AgentRunInputIdentity] = {}
        self._entries: dict[str, RunEntry] = {}

    def begin(self, request: dict[str, Any], identity: AgentRunInputIdentity) -> tuple[int, dict[str, Any] | None]:
        agent_run_id = str(request["agentRunId"])
        while True:
            with self._lock:
                entry = self._entries.get(agent_run_id)
                if entry is None:
                    attempt = self._attempts.get(agent_run_id, 0) + 1
                    self._attempts[agent_run_id] = attempt
                    self._identities[agent_run_id] = identity
                    self._entries[agent_run_id] = RunEntry(identity=identity, status=RunStatus.IN_PROGRESS, attempt_id=attempt)
                    return attempt, None
                if entry.identity != identity:
                    attempt = self._attempts.get(agent_run_id, 0) + 1
                    self._attempts[agent_run_id] = attempt
                    raise AgentRunInputConflict(attempt)
                if entry.status == RunStatus.IN_PROGRESS:
                    done = entry.done
                elif entry.result is not None and _is_cacheable(entry.result):
                    return entry.attempt_id, entry.result
                else:
                    attempt = self._attempts.get(agent_run_id, 0) + 1
                    self._attempts[agent_run_id] = attempt
                    entry.attempt_id = attempt
                    entry.status = RunStatus.IN_PROGRESS
                    entry.result = None
                    entry.done.clear()
                    return attempt, None
            done.wait()

    def complete(self, request: dict[str, Any], identity: AgentRunInputIdentity, result: dict[str, Any]) -> None:
        agent_run_id = str(request["agentRunId"])
        with self._lock:
            entry = self._entries.get(agent_run_id)
            if entry is None or entry.identity != identity:
                return
            entry.result = result
            entry.status = RunStatus.COMPLETED if result.get("resultStatus") == "COMPLETED" else RunStatus.FAILED
            entry.done.set()

    def abort(self, request: dict[str, Any], identity: AgentRunInputIdentity) -> None:
        agent_run_id = str(request["agentRunId"])
        with self._lock:
            entry = self._entries.get(agent_run_id)
            if entry is None or entry.identity != identity:
                return
            entry.result = None
            entry.status = RunStatus.FAILED
            entry.done.set()

    def allocate_attempt(self, request: dict[str, Any]) -> int:
        agent_run_id = str(request["agentRunId"])
        with self._lock:
            attempt = self._attempts.get(agent_run_id, 0) + 1
            self._attempts[agent_run_id] = attempt
            return attempt


def _is_cacheable(result: dict[str, Any]) -> bool:
    if result.get("resultStatus") == "COMPLETED":
        return True
    failure = result.get("failure")
    if not isinstance(failure, dict):
        return False
    return failure.get("reasonCode") in {
        "AGENT_RUN_INPUT_CONFLICT",
        "FEATURE_REQUIRED_MISSING",
        "FEATURE_SCHEMA_VERSION_UNSUPPORTED",
        "FEATURE_TYPE_INVALID",
        "FEATURE_UNKNOWN",
        "FEATURE_VALUE_OUT_OF_RANGE",
        "MODEL_ARTIFACT_DIGEST_MISMATCH",
        "SNAPSHOT_DIGEST_MISMATCH",
    }


def input_identity(request: dict[str, Any]) -> AgentRunInputIdentity:
    feature_payload = request.get("featurePayload")
    if not isinstance(feature_payload, dict):
        raise AgentFailure("VALIDATION_REQUIRED", "FEATURE_REQUIRED_MISSING", "Feature payload is required.")
    try:
        actual_feature_digest = feature_payload_digest(feature_payload)
    except ValueError as error:
        raise AgentFailure("VALIDATION_REQUIRED", "FEATURE_TYPE_INVALID", "Feature payload digest input is invalid.") from error
    if feature_payload.get("featurePayloadDigest") != actual_feature_digest:
        raise AgentFailure("BLOCKED", "SNAPSHOT_DIGEST_MISMATCH", "Feature payload digest mismatch.")
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
