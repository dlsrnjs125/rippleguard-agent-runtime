# Phase 2 Runtime Scope

The runtime implements only the Loan Decision Agent path.

It accepts a schema-valid Agent Request from Governance and returns a schema-valid Agent Result to Governance. It never emits `governance.agent-result.validated.v1`; that event is Governance-owned.

Excluded:

- Local LLM and Ollama
- Prompt contracts
- OPA
- final Loan state changes
- Audit Replay direct publishing
- fallback model selection

## Architecture

```text
FastAPI adapter
  -> ContractValidator
  -> LoanDecisionAgentService
  -> Feature validation and deterministic ordering
  -> Agent Run input identity guard
  -> Manifest and artifact verification
  -> XGBoostModelAdapter
  -> SHAP explanation
  -> Result builder
  -> ContractValidator
```

## Idempotency Boundary

The runtime keeps a process-local, lock-protected Agent Run state repository.

It enforces:

- same `agentRunId` and same immutable input identity returns the existing completed Result
- same `agentRunId` and changed immutable input returns `BLOCKED / AGENT_RUN_INPUT_CONFLICT`
- duplicate delivery does not create a new Proposal after completion

The identity includes Decision Case, Evaluation Run, request idempotency key, Snapshot digest, Feature Payload digest, Feature Schema, Preprocessing, Model, Artifact digest, and Threshold.

This is not a distributed or durable idempotency store. Process restart, multiple Uvicorn workers, and scale-out replicas still require Governance-owned attempt identity or a persistent Runtime repository.
