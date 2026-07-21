# Model Reproducibility

The Phase 2 baseline uses synthetic data and a small XGBoost JSON artifact.

Fixed inputs:

- feature schema version: `phase-2-loan-features.v1.0.0`
- preprocessing version: `preprocess.v1.0.0`
- model version: `loan-model.v1.0.0`
- threshold version: `threshold.v1.0.0`
- seed: `42`
- thread count: `1`

The artifact digest is recorded in `artifacts/manifests/phase2-loan-xgboost.v1.0.0.json` and rechecked before every inference.

The synthetic baseline is not evidence of real credit performance, fairness, or regulatory suitability.
