from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentFailure(Exception):
    classification: str
    reason_code: str
    safe_message: str


class RetryableRuntimeFailure(AgentFailure):
    ...
