from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rippleguard_agent_runtime.adapters.contracts import ContractValidationError, ContractValidator
from rippleguard_agent_runtime.adapters.xgboost_model import XGBoostModelAdapter
from rippleguard_agent_runtime.domain.errors import AgentFailure
from rippleguard_agent_runtime.loan_decision.features import feature_values, validate_and_prepare_features
from rippleguard_agent_runtime.loan_decision.manifest import load_manifest, verify_manifest_request, verify_runtime_compatibility
from rippleguard_agent_runtime.loan_decision.result_builder import build_completed_result, build_failed_result
from rippleguard_agent_runtime.loan_decision.run_state import AgentRunStateRepository, input_identity


class LoanDecisionAgentService:
    def __init__(self, contracts_root: Path, manifest_path: Path, artifact_root: Path) -> None:
        self.validator = ContractValidator(contracts_root)
        self.manifest_path = manifest_path
        self.artifact_root = artifact_root
        self.thresholds_path = manifest_path.parent / "thresholds.v1.0.0.json"
        self.run_state = AgentRunStateRepository()

    def readiness(self) -> dict[str, str]:
        manifest = load_manifest(self.manifest_path, self.validator)
        verify_runtime_compatibility(manifest)
        threshold = self._threshold(manifest["thresholdVersion"])
        artifact = verify_manifest_request(manifest, _request_from_manifest(manifest), self.artifact_root)
        XGBoostModelAdapter(artifact, manifest, threshold)
        return {"status": "ready", "modelVersion": str(manifest["modelVersion"])}

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        self.validator.validate("commands/loan-decision-agent-request.v1.0.0.schema.json", request)
        started_at = _now_text()
        attempt_id = 1
        try:
            identity = input_identity(request)
            attempt_id, cached_result = self.run_state.begin(request, identity)
            if cached_result is not None:
                return cached_result
            _validate_time_bounds(request)
            manifest = load_manifest(self.manifest_path, self.validator)
            verify_runtime_compatibility(manifest)
            threshold = self._threshold(manifest["thresholdVersion"])
            artifact = verify_manifest_request(manifest, request, self.artifact_root)
            features = validate_and_prepare_features(request)
            raw_features = feature_values(request)
            model = XGBoostModelAdapter(artifact, manifest, threshold)
            prediction = model.predict(features)
            explanation_ref, explanation_digest, explanation = model.explain(features)
            _validate_time_bounds(request)
            result = build_completed_result(
                request,
                manifest,
                prediction,
                explanation_ref,
                explanation_digest,
                explanation,
                raw_features,
                attempt_id,
                started_at,
            )
            self.run_state.complete(request, identity, result)
        except AgentFailure as failure:
            result = build_failed_result(request, failure, attempt_id, started_at)
        except (json.JSONDecodeError, ContractValidationError):
            configuration_failure = AgentFailure(
                "NON_RETRYABLE", "CONTRACT_VALIDATION_FAILED", "Runtime configuration is invalid."
            )
            result = build_failed_result(request, configuration_failure, attempt_id, started_at)
        self.validator.validate("agent-output/loan-decision-agent-result.v1.0.0.schema.json", result)
        return result

    def _threshold(self, threshold_version: str) -> float:
        if not self.thresholds_path.is_file():
            raise AgentFailure("BLOCKED", "MODEL_MANIFEST_NOT_FOUND", "Threshold configuration was not found.")
        try:
            thresholds = json.loads(self.thresholds_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise AgentFailure("NON_RETRYABLE", "CONTRACT_VALIDATION_FAILED", "Threshold configuration is invalid.") from error
        if not isinstance(thresholds, dict) or threshold_version not in thresholds:
            raise AgentFailure("VALIDATION_REQUIRED", "MODEL_VERSION_UNSUPPORTED", "Threshold version is unsupported.")
        value = thresholds[threshold_version]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise AgentFailure("NON_RETRYABLE", "CONTRACT_VALIDATION_FAILED", "Threshold value is invalid.")
        if float(value) < 0 or float(value) > 1:
            raise AgentFailure("BLOCKED", "CONTRACT_VALIDATION_FAILED", "Threshold value is invalid.")
        return float(value)


def _validate_time_bounds(request: dict[str, Any]) -> None:
    snapshot_time = _parse_time(request["snapshotReference"]["snapshotCreatedAt"])
    requested = _parse_time(request["requestedAt"])
    deadline = _parse_time(request["deadlineAt"])
    if deadline <= requested:
        raise AgentFailure("RETRYABLE", "AGENT_TIMEOUT", "Request deadline is not after requestedAt.")
    if snapshot_time > requested:
        raise AgentFailure("BLOCKED", "SNAPSHOT_DIGEST_MISMATCH", "Snapshot was created after request time.")
    if deadline <= datetime.now(deadline.tzinfo):
        raise AgentFailure("RETRYABLE", "AGENT_TIMEOUT", "Request deadline has already passed.")


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _now_text() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _request_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "modelVersion": manifest["modelVersion"],
        "featureSchemaVersion": manifest["featureSchemaVersion"],
        "preprocessingVersion": manifest["preprocessingVersion"],
        "thresholdVersion": manifest["thresholdVersion"],
        "modelArtifactDigest": manifest["modelBinaryArtifactDigest"],
    }
