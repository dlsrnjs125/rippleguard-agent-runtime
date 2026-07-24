from __future__ import annotations

import json
from pathlib import Path

import pytest

from rippleguard_agent_runtime.adapters.contracts import ContractValidator
from scripts.materialize_release_model_manifest import MaterializationError, materialize_manifest

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT.parent / "rippleguard-contracts"
TEMPLATE = ROOT / "artifacts" / "templates" / "phase2-loan-xgboost.v1.0.0.template.json"
ARTIFACT_ROOT = ROOT / "artifacts" / "models"
VALID_DIGEST = "sha256:abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
MODEL_DIGEST = "sha256:1780b376723b52ad04630a474c6bd2eeddab2e89caa13956e76b356595ed79df"


def test_materializes_valid_runtime_digest(tmp_path: Path) -> None:
    output = tmp_path / "model-manifest.json"
    manifest = materialize_manifest(
        template_path=TEMPLATE,
        runtime_image_digest=VALID_DIGEST,
        output_path=output,
        contracts_root=CONTRACTS,
        model_artifact_root=ARTIFACT_ROOT,
        expected_model_artifact_digest=MODEL_DIGEST,
        expected_model_version="loan-model.v1.0.0",
    )

    assert output.is_file()
    assert manifest["runtimeImageDigest"] == VALID_DIGEST
    assert "manifestPublicationState" not in manifest
    assert json.loads(output.read_text(encoding="utf-8")) == manifest
    ContractValidator(CONTRACTS).validate("agent-output/tabular-model-manifest.v1.0.0.schema.json", manifest)


@pytest.mark.parametrize(
    "digest",
    [
        "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "sha256:0000000000000000000000000000000000000000000000000000000000000000",
        "sha256:not-a-valid-digest",
        "sha256:99022f07ee439e309b515bccec8a3d351a070349000000000000000000000000",
    ],
)
def test_rejects_invalid_runtime_image_digest(tmp_path: Path, digest: str) -> None:
    with pytest.raises(MaterializationError):
        materialize_manifest(
            template_path=TEMPLATE,
            runtime_image_digest=digest,
            output_path=tmp_path / "model-manifest.json",
            contracts_root=CONTRACTS,
            model_artifact_root=ARTIFACT_ROOT,
        )


def test_rejects_model_artifact_digest_mismatch(tmp_path: Path) -> None:
    with pytest.raises(MaterializationError):
        materialize_manifest(
            template_path=TEMPLATE,
            runtime_image_digest=VALID_DIGEST,
            output_path=tmp_path / "model-manifest.json",
            contracts_root=CONTRACTS,
            model_artifact_root=ARTIFACT_ROOT,
            expected_model_artifact_digest="sha256:9999999999999999999999999999999999999999999999999999999999999999",
        )


def test_rejects_model_version_mismatch(tmp_path: Path) -> None:
    with pytest.raises(MaterializationError):
        materialize_manifest(
            template_path=TEMPLATE,
            runtime_image_digest=VALID_DIGEST,
            output_path=tmp_path / "model-manifest.json",
            contracts_root=CONTRACTS,
            model_artifact_root=ARTIFACT_ROOT,
            expected_model_version="loan-model.v9.9.9",
        )
