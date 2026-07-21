from __future__ import annotations

from typing import Any

import numpy as np

from rippleguard_agent_runtime.domain.errors import AgentFailure
from rippleguard_agent_runtime.ports.model import PreparedFeatures


FEATURE_ORDER: tuple[str, ...] = (
    "annualIncome",
    "monthlyIncomeMean",
    "monthlyIncomeVolatility",
    "debtToIncomeRatio",
    "existingDebtAmount",
    "delinquencyCount",
    "platformSettlementMonths",
    "platformSettlementMean",
    "platformSettlementVolatility",
    "contractDurationMonths",
    "incomeDeclarationAvailable",
    "telecomPaymentDelinquencyCount",
)

FEATURE_SCHEMA_VERSION = "phase-2-loan-features.v1.0.0"
PREPROCESSING_VERSION = "preprocess.v1.0.0"


def validate_and_prepare_features(request: dict[str, Any]) -> PreparedFeatures:
    if request.get("featureSchemaVersion") != FEATURE_SCHEMA_VERSION:
        raise AgentFailure("VALIDATION_REQUIRED", "FEATURE_SCHEMA_VERSION_UNSUPPORTED", "Unsupported feature schema.")
    if request.get("preprocessingVersion") != PREPROCESSING_VERSION:
        raise AgentFailure("VALIDATION_REQUIRED", "MODEL_VERSION_UNSUPPORTED", "Unsupported preprocessing version.")
    payload = request.get("featurePayload")
    if not isinstance(payload, dict):
        raise AgentFailure("VALIDATION_REQUIRED", "FEATURE_REQUIRED_MISSING", "Feature payload is required.")
    if payload.get("featureSchemaVersion") != FEATURE_SCHEMA_VERSION:
        raise AgentFailure("VALIDATION_REQUIRED", "FEATURE_SCHEMA_VERSION_UNSUPPORTED", "Feature payload schema mismatch.")
    features = payload.get("features")
    if not isinstance(features, dict):
        raise AgentFailure("VALIDATION_REQUIRED", "FEATURE_REQUIRED_MISSING", "Feature values are required.")
    unknown = set(features) - set(FEATURE_ORDER)
    if unknown:
        raise AgentFailure("VALIDATION_REQUIRED", "FEATURE_UNKNOWN", "Feature payload contains an unknown feature.")
    missing = [name for name in FEATURE_ORDER if name not in features]
    if missing:
        raise AgentFailure("VALIDATION_REQUIRED", "FEATURE_REQUIRED_MISSING", "Feature payload is missing a required feature.")
    values = [_feature_value(name, features[name]) for name in FEATURE_ORDER]
    return PreparedFeatures(names=FEATURE_ORDER, values=np.asarray(values, dtype=np.float32).reshape(1, -1))


def _feature_value(name: str, value: Any) -> float:
    if name == "incomeDeclarationAvailable":
        if not isinstance(value, bool):
            raise AgentFailure("VALIDATION_REQUIRED", "FEATURE_TYPE_INVALID", "Feature type is invalid.")
        return 1.0 if value else 0.0
    integer_features = {"delinquencyCount", "platformSettlementMonths", "contractDurationMonths", "telecomPaymentDelinquencyCount"}
    if name in integer_features and (not isinstance(value, int) or isinstance(value, bool)):
        raise AgentFailure("VALIDATION_REQUIRED", "FEATURE_TYPE_INVALID", "Feature type is invalid.")
    if name not in integer_features and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        raise AgentFailure("VALIDATION_REQUIRED", "FEATURE_TYPE_INVALID", "Feature type is invalid.")
    number = float(value)
    bounds = {
        "annualIncome": (0.0, 1_000_000_000.0),
        "monthlyIncomeMean": (0.0, 100_000_000.0),
        "monthlyIncomeVolatility": (0.0, 10.0),
        "debtToIncomeRatio": (0.0, 5.0),
        "existingDebtAmount": (0.0, 1_000_000_000.0),
        "delinquencyCount": (0.0, 120.0),
        "platformSettlementMonths": (0.0, 240.0),
        "platformSettlementMean": (0.0, 100_000_000.0),
        "platformSettlementVolatility": (0.0, 10.0),
        "contractDurationMonths": (0.0, 240.0),
        "telecomPaymentDelinquencyCount": (0.0, 120.0),
    }[name]
    if number < bounds[0] or number > bounds[1]:
        raise AgentFailure("VALIDATION_REQUIRED", "FEATURE_VALUE_OUT_OF_RANGE", "Feature value is outside the allowed range.")
    return number
