from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_env: str
    contracts_root: Path
    model_manifest_path: Path
    model_artifact_root: Path
    log_level: str
    host: str
    port: int

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            app_env=_required("APP_ENV", "local"),
            contracts_root=Path(_required("CONTRACTS_ROOT")).expanduser(),
            model_manifest_path=Path(_required("MODEL_MANIFEST_PATH")).expanduser(),
            model_artifact_root=Path(_required("MODEL_ARTIFACT_ROOT")).expanduser(),
            log_level=_required("LOG_LEVEL", "INFO"),
            host=_required("HOST", "127.0.0.1"),
            port=_int_env("PORT", 8080, minimum=1),
        )

    def validate(self) -> None:
        if not self.contracts_root.exists():
            raise ValueError(f"CONTRACTS_ROOT does not exist: {self.contracts_root}")
        if not self.model_manifest_path.is_file():
            raise ValueError(f"MODEL_MANIFEST_PATH does not exist: {self.model_manifest_path}")
        if not self.model_artifact_root.is_dir():
            raise ValueError(f"MODEL_ARTIFACT_ROOT does not exist: {self.model_artifact_root}")
        if self.log_level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("LOG_LEVEL must be a standard Python log level")


def _required(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None or value == "":
        raise ValueError(f"{name} is required")
    if value in {"latest", "default", "auto"}:
        raise ValueError(f"{name} must not be {value!r}")
    return value


def _int_env(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value
