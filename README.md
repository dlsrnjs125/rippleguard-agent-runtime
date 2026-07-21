# RippleGuard Agent Runtime

Phase 2 Loan Decision Agent Runtime.

This service validates a Governance-issued Loan Decision Agent Request, verifies the fixed Model Manifest and artifact digest, runs the selected small XGBoost baseline model, generates SHAP explanation metadata, and returns a Phase 2 Loan Decision Agent Result.

It does not make the final loan decision, mutate Loan or Governance state, publish Governance audit events, or use Local LLM/Ollama.

## Scope

- Phase 2 contract validation from `rippleguard-contracts`
- Feature Payload validation and deterministic feature ordering
- Model Manifest validation
- model artifact SHA-256 verification
- XGBoost JSON model loading
- SHAP TreeExplainer output digest
- Loan Proposal generation
- completed and failed Agent Result generation
- REST adapter:
  - `POST /internal/v1/loan-decision-agent/runs`
  - `GET /health`
  - `GET /ready`

## Contracts

Set `CONTRACTS_ROOT` to a local checkout of `rippleguard-contracts`.

The implementation was built against contracts commit:

```text
f4012e8 feat: define phase 2 loan decision contracts (#4)
```

## Local Setup

```bash
python3 -m pip install -e ".[dev]"
make train-baseline
make verify
```

Run locally:

```bash
CONTRACTS_ROOT=../rippleguard-contracts \
MODEL_MANIFEST_PATH=artifacts/manifests/phase2-loan-xgboost.v1.0.0.json \
MODEL_ARTIFACT_ROOT=artifacts/models \
python3 -m uvicorn rippleguard_agent_runtime.app.api:app --host 127.0.0.1 --port 8080
```

## Model Artifact

The committed artifact is a small synthetic baseline:

```text
artifacts/models/phase2-loan-xgboost.v1.0.0.json
```

It is generated explicitly with:

```bash
make train-baseline
```

Runtime startup does not train models or choose fallback models.

## Verification

```bash
make verify
docker build -t rippleguard-agent-runtime:phase2-local .
```

## Known Limitations

- The model is a synthetic baseline for system reproducibility, not a real financial approval model.
- LightGBM is documented as an offline comparison candidate but is not a runtime fallback.
- Feature Payload digest canonicalization remains a contracts follow-up; runtime enforces schema and artifact provenance.
- Docker image expects contracts to be mounted at `/app/contracts` or supplied by deployment.
