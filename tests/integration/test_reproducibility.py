from __future__ import annotations

from copy import deepcopy


def test_repeated_same_input_is_reproducible(service: object, valid_request: dict[str, object]) -> None:
    first = service.run(deepcopy(valid_request))  # type: ignore[attr-defined]
    second = service.run(deepcopy(valid_request))  # type: ignore[attr-defined]
    assert first["resultStatus"] == "COMPLETED"
    assert second["resultStatus"] == "COMPLETED"
    assert first["proposal"]["proposalOutcome"] == second["proposal"]["proposalOutcome"]
    assert abs(first["proposal"]["repaymentLikelihoodScore"] - second["proposal"]["repaymentLikelihoodScore"]) <= 1e-6
    assert first["explanationDigest"] == second["explanationDigest"]
