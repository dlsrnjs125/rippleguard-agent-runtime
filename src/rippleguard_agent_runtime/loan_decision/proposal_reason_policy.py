from __future__ import annotations

from typing import Any

from rippleguard_agent_runtime.ports.model import ModelPrediction


def reason_codes(
    *,
    prediction: ModelPrediction,
    features: dict[str, Any],
    explanation: list[dict[str, float | str]],
) -> list[str]:
    if prediction.score >= prediction.threshold:
        return _approval_reasons(features, explanation)
    return _decline_reasons(features, explanation)


def _approval_reasons(features: dict[str, Any], explanation: list[dict[str, float | str]]) -> list[str]:
    reasons: list[str] = []
    if float(features["debtToIncomeRatio"]) <= 0.4:
        reasons.append("LOW_DTI")
    if bool(features["incomeDeclarationAvailable"]) and float(features["monthlyIncomeVolatility"]) <= 0.3:
        reasons.append("STABLE_INCOME")
    for feature_name, code in _positive_feature_codes().items():
        if _has_positive_contribution(explanation, feature_name) and code not in reasons:
            reasons.append(code)
    return reasons[:2] or ["LOW_DTI"]


def _decline_reasons(features: dict[str, Any], explanation: list[dict[str, float | str]]) -> list[str]:
    candidates = [
        ("debtToIncomeRatio", "HIGH_DTI", float(features["debtToIncomeRatio"]) >= 0.45),
        ("delinquencyCount", "RECENT_DELINQUENCY", int(features["delinquencyCount"]) > 0),
        (
            "telecomPaymentDelinquencyCount",
            "RECENT_DELINQUENCY",
            int(features["telecomPaymentDelinquencyCount"]) > 0,
        ),
        ("platformSettlementMonths", "INSUFFICIENT_HISTORY", int(features["platformSettlementMonths"]) < 12),
        ("monthlyIncomeVolatility", "VOLATILE_SETTLEMENTS", float(features["monthlyIncomeVolatility"]) >= 0.4),
        ("platformSettlementVolatility", "VOLATILE_SETTLEMENTS", float(features["platformSettlementVolatility"]) >= 0.4),
    ]
    reasons: list[str] = []
    for feature_name, code, threshold_hit in candidates:
        if (threshold_hit or _has_adverse_contribution(explanation, feature_name)) and code not in reasons:
            reasons.append(code)
    return reasons[:3] or ["HIGH_DTI"]


def _positive_feature_codes() -> dict[str, str]:
    return {
        "debtToIncomeRatio": "LOW_DTI",
        "incomeDeclarationAvailable": "STABLE_INCOME",
        "monthlyIncomeVolatility": "STABLE_INCOME",
        "platformSettlementVolatility": "STABLE_INCOME",
    }


def _has_positive_contribution(explanation: list[dict[str, float | str]], feature_name: str) -> bool:
    contribution = _contribution(explanation, feature_name)
    return contribution is not None and contribution > 0


def _has_adverse_contribution(explanation: list[dict[str, float | str]], feature_name: str) -> bool:
    contribution = _contribution(explanation, feature_name)
    return contribution is not None and contribution < 0


def _contribution(explanation: list[dict[str, float | str]], feature_name: str) -> float | None:
    for item in explanation:
        if item.get("featureName") == feature_name:
            return float(item["contribution"])
    return None
