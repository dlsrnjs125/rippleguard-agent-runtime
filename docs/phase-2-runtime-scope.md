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
  -> Manifest and artifact verification
  -> XGBoostModelAdapter
  -> SHAP explanation
  -> Result builder
  -> ContractValidator
```
