from __future__ import annotations

import pytest

from rippleguard_agent_runtime.domain.errors import AgentFailure
from rippleguard_agent_runtime.loan_decision.features import FEATURE_ORDER, validate_and_prepare_features


def test_feature_order_is_deterministic(valid_request: dict[str, object]) -> None:
    prepared = validate_and_prepare_features(valid_request)
    assert prepared.names == FEATURE_ORDER
    assert prepared.values.shape == (1, len(FEATURE_ORDER))


def test_unknown_feature_is_rejected(valid_request: dict[str, object]) -> None:
    features = valid_request["featurePayload"]["features"]  # type: ignore[index]
    features["promptInjectionScore"] = 1  # type: ignore[index]
    with pytest.raises(AgentFailure) as error:
        validate_and_prepare_features(valid_request)
    assert error.value.reason_code == "FEATURE_UNKNOWN"
