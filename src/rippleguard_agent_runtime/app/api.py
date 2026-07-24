from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException

from rippleguard_agent_runtime.adapters.contracts import ContractValidationError
from rippleguard_agent_runtime.config.settings import Settings
from rippleguard_agent_runtime.loan_decision.service import LoanDecisionAgentService

app = FastAPI(title="RippleGuard Agent Runtime", version="0.1.0")


@lru_cache(maxsize=1)
def _service() -> LoanDecisionAgentService:
    settings = Settings.from_env()
    settings.validate()
    return LoanDecisionAgentService(settings.contracts_root, settings.model_manifest_path, settings.model_artifact_root)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    try:
        return _service().readiness()
    except Exception as error:
        raise HTTPException(status_code=503, detail="runtime is not ready") from error


@app.post("/internal/v1/loan-decision-agent/runs")
def run_loan_decision_agent(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return _service().run(payload)
    except ContractValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
