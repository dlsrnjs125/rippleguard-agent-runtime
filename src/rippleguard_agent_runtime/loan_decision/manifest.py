from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import shap
import xgboost as xgb

from rippleguard_agent_runtime.adapters.contracts import ContractValidator
from rippleguard_agent_runtime.adapters.digest import file_sha256
from rippleguard_agent_runtime.domain.errors import AgentFailure


def load_manifest(path: Path, validator: ContractValidator) -> dict[str, Any]:
    if not path.is_file():
        raise AgentFailure("BLOCKED", "MODEL_MANIFEST_NOT_FOUND", "Model manifest was not found.")
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise AgentFailure("BLOCKED", "MODEL_MANIFEST_NOT_FOUND", "Model manifest is not an object.")
    validator.validate("agent-output/tabular-model-manifest.v1.0.0.schema.json", manifest)
    return manifest


def artifact_path(manifest: dict[str, Any], artifact_root: Path) -> Path:
    reference = str(manifest["modelBinaryArtifactReference"])
    if not reference.startswith("file://"):
        raise AgentFailure("BLOCKED", "MODEL_ARTIFACT_NOT_FOUND", "Only file model artifacts are supported locally.")
    relative = reference.removeprefix("file://")
    return artifact_root / relative


def verify_manifest_request(manifest: dict[str, Any], request: dict[str, Any], artifact_root: Path) -> Path:
    compared = (
        ("modelVersion", "modelVersion", "MODEL_VERSION_UNSUPPORTED"),
        ("featureSchemaVersion", "featureSchemaVersion", "FEATURE_SCHEMA_VERSION_UNSUPPORTED"),
        ("preprocessingVersion", "preprocessingVersion", "MODEL_VERSION_UNSUPPORTED"),
        ("thresholdVersion", "thresholdVersion", "MODEL_VERSION_UNSUPPORTED"),
    )
    for request_field, manifest_field, reason in compared:
        if request.get(request_field) != manifest.get(manifest_field):
            raise AgentFailure("VALIDATION_REQUIRED", reason, "Request and model manifest are inconsistent.")
    if request.get("modelArtifactDigest") != manifest.get("modelBinaryArtifactDigest"):
        raise AgentFailure("BLOCKED", "MODEL_ARTIFACT_DIGEST_MISMATCH", "Request model artifact digest does not match manifest.")
    path = artifact_path(manifest, artifact_root)
    if not path.is_file():
        raise AgentFailure("BLOCKED", "MODEL_ARTIFACT_NOT_FOUND", "Model artifact was not found.")
    actual = file_sha256(str(path))
    if actual != manifest.get("modelBinaryArtifactDigest"):
        raise AgentFailure("BLOCKED", "MODEL_ARTIFACT_DIGEST_MISMATCH", "Model artifact digest mismatch.")
    if manifest.get("framework") != "xgboost" or manifest.get("modelFormat") != "xgboost-json":
        raise AgentFailure("VALIDATION_REQUIRED", "MODEL_VERSION_UNSUPPORTED", "Only the selected XGBoost JSON baseline is supported.")
    return path


def verify_runtime_compatibility(manifest: dict[str, Any]) -> None:
    if manifest.get("frameworkVersion") != xgb.__version__:
        raise AgentFailure("VALIDATION_REQUIRED", "MODEL_VERSION_UNSUPPORTED", "Installed XGBoost version does not match manifest.")
    if manifest.get("shapExplainerVersion") != f"shap.v{shap.__version__}":
        raise AgentFailure("VALIDATION_REQUIRED", "MODEL_VERSION_UNSUPPORTED", "Installed SHAP version does not match manifest.")
    if manifest.get("threadCount") != 1:
        raise AgentFailure("VALIDATION_REQUIRED", "MODEL_VERSION_UNSUPPORTED", "Only single-threaded runtime execution is supported.")
