from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import FormatChecker, ValidationError, validators
from referencing import Registry, Resource


class ContractValidator:
    def __init__(self, contracts_root: Path) -> None:
        self.contracts_root = contracts_root
        self._registry = Registry()
        self._schemas: dict[str, dict[str, Any]] = {}
        self._load_schemas()

    def validate(self, contract: str, payload: dict[str, Any]) -> None:
        schema = self._schema(contract)
        validator_class = validators.validator_for(schema)
        validator = validator_class(schema, registry=self._registry, format_checker=FormatChecker())
        errors = sorted(validator.iter_errors(payload), key=lambda error: (list(error.absolute_path), error.message))
        if errors:
            raise ContractValidationError(_render_error(errors[0]))

    def _load_schemas(self) -> None:
        for path in sorted((self.contracts_root / "schemas").rglob("*.json")):
            with path.open(encoding="utf-8") as handle:
                schema = json.load(handle)
            if not isinstance(schema, dict) or "$id" not in schema:
                continue
            self._schemas[str(path.relative_to(self.contracts_root / "schemas"))] = schema
            resource = Resource.from_contents(schema)
            self._registry = self._registry.with_resource(path.resolve().as_uri(), resource)
            self._registry = self._registry.with_resource(schema["$id"], resource)

    def _schema(self, contract: str) -> dict[str, Any]:
        try:
            return self._schemas[contract]
        except KeyError as error:
            raise ContractValidationError(f"contract schema not found: {contract}") from error


class ContractValidationError(ValueError):
    ...


def _render_error(error: ValidationError) -> str:
    path = ".".join(str(part) for part in error.absolute_path)
    suffix = f" at {path}" if path else ""
    return f"{error.message}{suffix}"
