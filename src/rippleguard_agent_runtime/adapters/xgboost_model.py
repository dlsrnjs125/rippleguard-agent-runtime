from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import shap
import xgboost as xgb

from rippleguard_agent_runtime.adapters.digest import sha256_digest
from rippleguard_agent_runtime.domain.errors import AgentFailure
from rippleguard_agent_runtime.ports.model import ModelPrediction, PreparedFeatures


class XGBoostModelAdapter:
    def __init__(self, artifact_path: Path, manifest: dict[str, Any], threshold: float) -> None:
        self.manifest = manifest
        self.threshold = threshold
        self.booster = xgb.Booster()
        try:
            self.booster.load_model(str(artifact_path))
            self.explainer = shap.TreeExplainer(self.booster)
        except xgb.core.XGBoostError as error:
            raise AgentFailure("NON_RETRYABLE", "MODEL_VERSION_UNSUPPORTED", "Model artifact could not be loaded.") from error

    def predict(self, features: PreparedFeatures) -> ModelPrediction:
        matrix = xgb.DMatrix(features.values, feature_names=list(features.names))
        raw = self.booster.predict(matrix)
        if len(raw) == 0:
            raise AgentFailure("RETRYABLE", "AGENT_RUNTIME_TEMPORARY_FAILURE", "Model returned an empty prediction.")
        score = float(raw[0])
        if not np.isfinite(score):
            raise AgentFailure("RETRYABLE", "AGENT_RUNTIME_TEMPORARY_FAILURE", "Model returned a non-finite score.")
        return ModelPrediction(
            score=round(score, 6),
            threshold=self.threshold,
            threshold_version=str(self.manifest["thresholdVersion"]),
            model_version=str(self.manifest["modelVersion"]),
        )

    def explain(self, features: PreparedFeatures) -> tuple[str, str, list[dict[str, float | str]]]:
        try:
            values = self.explainer.shap_values(features.values)
        except Exception as error:  # noqa: BLE001
            raise AgentFailure("VALIDATION_REQUIRED", "SHAP_CALCULATION_FAILED", "SHAP explanation failed.") from error
        row = np.asarray(values)[0]
        contributions: list[dict[str, float | str]] = [
            {"featureName": name, "contribution": round(float(value), 8)}
            for name, value in zip(features.names, row)
        ]
        contributions.sort(key=lambda item: abs(cast(float, item["contribution"])), reverse=True)
        explanation = {
            "schemaVersion": "1.0.0",
            "type": "SHAP",
            "modelVersion": self.manifest["modelVersion"],
            "featureSchemaVersion": self.manifest["featureSchemaVersion"],
            "preprocessingVersion": self.manifest["preprocessingVersion"],
            "values": contributions[:8],
        }
        digest = sha256_digest(explanation)
        return "shap://phase2-loan-xgboost/local-explanation", digest, contributions[:8]
