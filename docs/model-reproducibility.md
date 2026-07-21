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

The artifact digest is recorded in `artifacts/manifests/phase2-loan-xgboost.v1.0.0.json` and rechecked before every inference.
The dataset digests are computed from the preprocessed feature matrix, labels, dtype, and shape.

The committed manifest still contains a schema-required candidate `runtimeImageDigest`. It is not deployment evidence until replaced by the actual image digest from the image build/release path.

Runtime compatibility checks currently enforce the installed XGBoost version, SHAP version, and single-thread execution mode. The current contract mixes training and runtime environment fields, so Python/platform/image digest publication needs a follow-up split between model training manifest and runtime deployment manifest.
Readiness reports `provenanceStatus: CANDIDATE` for this local baseline.

Determinism tests compare semantic prediction fields: proposal outcome, score within tolerance, and SHAP explanation digest. Full Result payloads include execution metadata such as generated IDs and timestamps, so they are not byte-for-byte stable across attempts.

The synthetic baseline is not evidence of real credit performance, fairness, or regulatory suitability.
