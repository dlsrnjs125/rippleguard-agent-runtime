# Failure Handling

Malformed requests fail transport validation and do not become Agent Results.

Schema-valid requests can produce contract-valid failed results:

- model manifest missing: `BLOCKED / MODEL_MANIFEST_NOT_FOUND`
- model artifact missing: `BLOCKED / MODEL_ARTIFACT_NOT_FOUND`
- model artifact digest mismatch: `BLOCKED / MODEL_ARTIFACT_DIGEST_MISMATCH`
- unsupported feature schema: `VALIDATION_REQUIRED / FEATURE_SCHEMA_VERSION_UNSUPPORTED`
- missing feature: `VALIDATION_REQUIRED / FEATURE_REQUIRED_MISSING`
- unknown feature: `VALIDATION_REQUIRED / FEATURE_UNKNOWN`
- invalid feature type: `VALIDATION_REQUIRED / FEATURE_TYPE_INVALID`
- out-of-range feature: `VALIDATION_REQUIRED / FEATURE_VALUE_OUT_OF_RANGE`
- timeout or expired deadline: `RETRYABLE / AGENT_TIMEOUT`
- SHAP failure: `VALIDATION_REQUIRED / SHAP_CALCULATION_FAILED`

Failed results never include a Proposal.
