# Failure Handling

Malformed requests fail transport validation and do not become Agent Results.

Schema-valid requests can produce contract-valid failed results:

- model manifest missing: `BLOCKED / MODEL_MANIFEST_NOT_FOUND`
- malformed manifest or threshold configuration: `NON_RETRYABLE / CONTRACT_VALIDATION_FAILED`
- model artifact missing: `BLOCKED / MODEL_ARTIFACT_NOT_FOUND`
- model artifact digest mismatch: `BLOCKED / MODEL_ARTIFACT_DIGEST_MISMATCH`
- corrupted or unsupported model artifact: `NON_RETRYABLE / MODEL_VERSION_UNSUPPORTED`
- feature payload digest mismatch: `BLOCKED / SNAPSHOT_DIGEST_MISMATCH`
- unsupported feature schema: `VALIDATION_REQUIRED / FEATURE_SCHEMA_VERSION_UNSUPPORTED`
- missing feature: `VALIDATION_REQUIRED / FEATURE_REQUIRED_MISSING`
- unknown feature: `VALIDATION_REQUIRED / FEATURE_UNKNOWN`
- invalid feature type: `VALIDATION_REQUIRED / FEATURE_TYPE_INVALID`
- out-of-range feature: `VALIDATION_REQUIRED / FEATURE_VALUE_OUT_OF_RANGE`
- timeout or expired deadline: `RETRYABLE / AGENT_TIMEOUT`
- SHAP failure: `VALIDATION_REQUIRED / SHAP_CALCULATION_FAILED`

Failed results never include a Proposal.

Runtime performs a single attempt for each accepted request. Retry scheduling, max attempts, backoff, and duplicate request idempotency remain Governance-owned unless the request contract is extended with explicit attempt metadata.
