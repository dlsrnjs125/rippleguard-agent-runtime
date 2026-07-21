from __future__ import annotations

from pathlib import Path

from rippleguard_agent_runtime.adapters.digest import canonical_json, sha256_digest


def test_agent_result_digest_vector_matches_contracts() -> None:
    vector = Path("../rippleguard-contracts/examples/digest-vectors/agent-result-v1")
    value = (vector / "agent-result-input.json").read_text(encoding="utf-8")
    import json

    parsed = json.loads(value)
    assert canonical_json(parsed) == (vector / "canonical-agent-result.json").read_text(encoding="utf-8").rstrip("\n")
    assert sha256_digest(parsed) == (vector / "expected-sha256.txt").read_text(encoding="utf-8").strip()
