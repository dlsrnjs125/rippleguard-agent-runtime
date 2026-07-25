#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

contracts_module = importlib.import_module("rippleguard_agent_runtime.adapters.contracts")
ContractValidationError = contracts_module.ContractValidationError
ContractValidator = contracts_module.ContractValidator
file_sha256 = importlib.import_module("rippleguard_agent_runtime.adapters.digest").file_sha256

RUNTIME_IMAGE_TOKEN = "${RUNTIME_IMAGE_DIGEST}"
SHA256_RE = re.compile(r"^sha256:[a-f0-9]{64}$")
PLATFORM_ARCHITECTURES = frozenset(("linux/amd64", "linux/arm64"))


class MaterializationError(ValueError):
    pass


def materialize_manifest(
    *,
    template_path: Path,
    runtime_image_digest: str,
    platform_architecture: str,
    output_path: Path,
    contracts_root: Path,
    model_artifact_root: Path | None = None,
    expected_model_artifact_digest: str | None = None,
    expected_model_version: str | None = None,
) -> dict[str, Any]:
    template = _load_object(template_path)
    _validate_template_contract(contracts_root, template)
    _ensure_template_marker(template)
    _validate_runtime_image_digest(runtime_image_digest, template)
    _validate_platform_architecture(platform_architecture)

    manifest = dict(template)
    manifest["runtimeImageDigest"] = runtime_image_digest
    manifest["platformArchitecture"] = platform_architecture
    manifest.pop("manifestPublicationState", None)

    if expected_model_version is not None and manifest.get("modelVersion") != expected_model_version:
        raise MaterializationError(
            f"model version mismatch: expected {expected_model_version} actual {manifest.get('modelVersion')}"
        )

    artifact_root = model_artifact_root or _default_artifact_root(template_path)
    actual_artifact_digest = _verify_model_artifact_digest(manifest, artifact_root)
    if expected_model_artifact_digest is not None and actual_artifact_digest != expected_model_artifact_digest:
        raise MaterializationError(
            "model artifact digest mismatch: "
            f"expected {expected_model_artifact_digest} actual {actual_artifact_digest}"
        )

    _validate_contract(contracts_root, manifest)
    _write_atomic_json(output_path, manifest)
    return manifest


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise MaterializationError(f"template not found: {path}") from error
    except json.JSONDecodeError as error:
        raise MaterializationError(f"template is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise MaterializationError("template must be a JSON object")
    return value


def _validate_runtime_image_digest(value: str, template: dict[str, Any]) -> None:
    if not SHA256_RE.fullmatch(value):
        raise MaterializationError("runtime image digest must match sha256:<64 lowercase hex chars>")
    hex_value = value.removeprefix("sha256:")
    if len(set(hex_value)) == 1:
        raise MaterializationError("runtime image digest must not be a placeholder")
    source_commit = template.get("trainingCodeCommit")
    if isinstance(source_commit, str) and source_commit in hex_value:
        raise MaterializationError("runtime image digest must not be derived from a source commit")


def _ensure_template_marker(template: dict[str, Any]) -> None:
    if template.get("manifestPublicationState") != "TEMPLATE":
        raise MaterializationError("template must be marked as TEMPLATE")
    if template.get("runtimeImageDigest") == RUNTIME_IMAGE_TOKEN:
        raise MaterializationError("template must omit runtimeImageDigest; release materialization injects it")
    if "runtimeImageDigest" in template:
        raise MaterializationError("template runtimeImageDigest must not be source-controlled release evidence")
    if template.get("manifestPublicationState") == "PUBLISHED":
        raise MaterializationError("template must not be marked as published release evidence")


def _validate_platform_architecture(value: str) -> None:
    if value not in PLATFORM_ARCHITECTURES:
        raise MaterializationError("platform architecture must be the built Linux image platform")


def _default_artifact_root(template_path: Path) -> Path:
    parts = template_path.resolve().parts
    if "artifacts" in parts:
        artifacts_index = parts.index("artifacts")
        return Path(*parts[: artifacts_index + 1]) / "models"
    return ROOT / "artifacts" / "models"


def _verify_model_artifact_digest(manifest: dict[str, Any], artifact_root: Path) -> str:
    reference = str(manifest.get("modelBinaryArtifactReference", ""))
    if not reference.startswith("file://"):
        raise MaterializationError("only file:// model artifacts can be verified by this materializer")
    artifact = artifact_root / reference.removeprefix("file://")
    if not artifact.is_file():
        raise MaterializationError(f"model artifact not found: {artifact}")
    actual = file_sha256(str(artifact))
    if actual != manifest.get("modelBinaryArtifactDigest"):
        raise MaterializationError(
            f"manifest model artifact digest mismatch: expected {manifest.get('modelBinaryArtifactDigest')} actual {actual}"
        )
    return actual


def _validate_contract(contracts_root: Path, manifest: dict[str, Any]) -> None:
    try:
        ContractValidator(contracts_root).validate("agent-output/tabular-model-manifest.v1.0.0.schema.json", manifest)
    except ContractValidationError as error:
        raise MaterializationError(f"materialized manifest failed contract validation: {error}") from error


def _validate_template_contract(contracts_root: Path, template: dict[str, Any]) -> None:
    try:
        ContractValidator(contracts_root).validate(
            "agent-output/tabular-model-manifest-template.v1.0.0.schema.json",
            template,
        )
    except ContractValidationError as error:
        raise MaterializationError(f"template manifest failed contract validation: {error}") from error


def _write_atomic_json(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(encoded)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a Phase 2 release model manifest.")
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--runtime-image-digest", required=True)
    parser.add_argument("--platform-architecture", required=True, choices=sorted(PLATFORM_ARCHITECTURES))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--contracts-root", default=ROOT.parent / "rippleguard-contracts", type=Path)
    parser.add_argument("--model-artifact-root", type=Path)
    parser.add_argument("--expected-model-artifact-digest")
    parser.add_argument("--expected-model-version")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = materialize_manifest(
            template_path=args.template,
            runtime_image_digest=args.runtime_image_digest,
            platform_architecture=args.platform_architecture,
            output_path=args.output,
            contracts_root=args.contracts_root,
            model_artifact_root=args.model_artifact_root,
            expected_model_artifact_digest=args.expected_model_artifact_digest,
            expected_model_version=args.expected_model_version,
        )
    except MaterializationError as error:
        print(f"materialization failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output), "modelVersion": manifest["modelVersion"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
