# Model Reproducibility

The Phase 2 baseline uses synthetic data and a small XGBoost JSON artifact.

Fixed inputs:

- feature schema version: `phase-2-loan-features.v1.0.0`
- preprocessing version: `preprocess.v1.0.0`
- model version: `loan-model.v1.0.0`
- threshold version: `threshold.v1.0.0`
- seed: `42`
- thread count: `1`

Training and inference both use `preprocess.v1.0.0`:

- `log1p` scaling for monetary features
- bounded scaling for ratios, counts, and month features
- boolean conversion to `0/1`

The artifact digest is recorded in `artifacts/templates/phase2-loan-xgboost.v1.0.0.template.json` and rechecked against the Infra-materialized published manifest before every inference.
The dataset digests are computed from the preprocessed feature matrix, labels, dtype, and shape.

The committed manifest is a Contracts-valid template, not deployment evidence. `scripts/materialize_release_model_manifest.py` validates the template contract, injects the actual image digest and built Linux image platform after image build, verifies the model artifact digest, validates the published manifest against `rippleguard-contracts`, and writes the output atomically. The source commit must not be substituted for the image digest.

Release ownership is intentionally split:

- Template Manifest: model artifact provenance, training provenance, and runtime dependency constraints
- Infra Release Manifest: exact runtime image digest, model artifact digest, and model manifest digest

Published Model Manifest: Infra-materialized manifest with exact runtime image digest.

Runtime compatibility checks currently enforce the installed XGBoost version, SHAP version, and single-thread execution mode. The current contract mixes training and runtime environment fields, so Python/platform/image digest publication needs a follow-up split between model training manifest and runtime deployment manifest.
Readiness reports `provenanceStatus: MATERIALIZED` only after a published manifest has been loaded and verified.

Determinism tests compare semantic prediction fields: proposal outcome, score within tolerance, and SHAP explanation digest. Full Result payloads include execution metadata such as generated IDs and timestamps, so they are not byte-for-byte stable across attempts.

The synthetic baseline is not evidence of real credit performance, fairness, or regulatory suitability.
