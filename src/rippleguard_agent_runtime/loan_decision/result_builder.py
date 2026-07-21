from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from rippleguard_agent_runtime.domain.errors import AgentFailure
from rippleguard_agent_runtime.ports.model import ModelPrediction


def now_text() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_completed_result(
    request: dict[str, Any],
    manifest: dict[str, Any],
    prediction: ModelPrediction,
    explanation_ref: str,
    explanation_digest: str,
    attempt_id: int,
) -> dict[str, Any]:
    completed_at = now_text()
    outcome = "RECOMMEND_APPROVAL" if prediction.score >= prediction.threshold else "RECOMMEND_DECLINE"
    reason_codes = ["LOW_DTI", "STABLE_INCOME"] if outcome == "RECOMMEND_APPROVAL" else ["MODEL_SCORE_BELOW_THRESHOLD"]
    return {
        "schemaVersion": "1.0.0",
        "resultStatus": "COMPLETED",
        "agentRun": _agent_run(request, attempt_id, completed_at),
        "snapshotReference": request["snapshotReference"],
        "featureSchemaVersion": request["featureSchemaVersion"],
        "preprocessingVersion": request["preprocessingVersion"],
        "modelVersion": request["modelVersion"],
        "modelArtifactDigest": request["modelArtifactDigest"],
        "thresholdVersion": request["thresholdVersion"],
        "proposal": {
            "schemaVersion": "1.0.0",
            "proposalId": str(uuid4()),
            "proposalOutcome": outcome,
            "repaymentLikelihoodScore": prediction.score,
            "scoreSemantics": manifest["scoreSemantics"],
            "threshold": prediction.threshold,
            "thresholdVersion": prediction.threshold_version,
            "comparisonDirection": "score_greater_than_or_equal_threshold_supports_approval",
            "reasonCodes": reason_codes,
            "modelVersion": prediction.model_version,
            "featureSchemaVersion": request["featureSchemaVersion"],
            "generatedAt": completed_at,
        },
        "explanationRef": explanation_ref,
        "explanationDigest": explanation_digest,
        "evidenceRefs": [request["snapshotReference"]["snapshotReference"]],
        "completedAt": completed_at,
    }


def build_failed_result(request: dict[str, Any], failure: AgentFailure, attempt_id: int) -> dict[str, Any]:
    completed_at = now_text()
    return {
        "schemaVersion": "1.0.0",
        "resultStatus": "FAILED",
        "agentRun": _agent_run(request, attempt_id, completed_at),
        "snapshotReference": request["snapshotReference"],
        "featureSchemaVersion": request["featureSchemaVersion"],
        "preprocessingVersion": request["preprocessingVersion"],
        "modelVersion": request["modelVersion"],
        "modelArtifactDigest": request["modelArtifactDigest"],
        "thresholdVersion": request["thresholdVersion"],
        "failure": {
            "classification": failure.classification,
            "reasonCode": failure.reason_code,
            "safeMessage": failure.safe_message,
        },
        "completedAt": completed_at,
    }


def _agent_run(request: dict[str, Any], attempt_id: int, completed_at: str) -> dict[str, Any]:
    return {
        "schemaVersion": "1.0.0",
        "decisionCaseId": request["decisionCaseId"],
        "evaluationRunId": request["evaluationRunId"],
        "agentRunId": request["agentRunId"],
        "attemptId": attempt_id,
        "agentType": "LOAN_DECISION_AGENT",
        "requestIdempotencyKey": request["requestIdempotencyKey"],
        "startedAt": completed_at,
        "completedAt": completed_at,
        "runtimeVersion": "agent-runtime.v0.1.0",
    }
