from __future__ import annotations

import json
from pathlib import Path

import pytest

from rippleguard_agent_runtime.domain.errors import AgentFailure
from rippleguard_agent_runtime.loan_decision.service import LoanDecisionAgentService


@pytest.mark.parametrize("value", [True, -1, 2])
def test_threshold_value_is_validated(tmp_path: Path, value: object) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    (tmp_path / "thresholds.v1.0.0.json").write_text(json.dumps({"threshold.v1.0.0": value}), encoding="utf-8")
    service = LoanDecisionAgentService(tmp_path, manifest, tmp_path)
    with pytest.raises(AgentFailure) as error:
        service._threshold("threshold.v1.0.0")
    assert error.value.classification == "VALIDATION_REQUIRED"
    assert error.value.reason_code == "CONTRACT_VALIDATION_FAILED"
