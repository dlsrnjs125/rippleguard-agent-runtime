.PHONY: install lint typecheck test contract-test integration-test reproducibility-test train-baseline build run verify docker-build docker-run-readiness

PYTHON ?= python3
CONTRACTS_ROOT ?= ../rippleguard-contracts
MODEL_MANIFEST_PATH ?= artifacts/manifests/phase2-loan-xgboost.v1.0.0.json
MODEL_ARTIFACT_ROOT ?= artifacts/models

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy src

test:
	CONTRACTS_ROOT=$(CONTRACTS_ROOT) MODEL_MANIFEST_PATH=$(MODEL_MANIFEST_PATH) MODEL_ARTIFACT_ROOT=$(MODEL_ARTIFACT_ROOT) $(PYTHON) -m pytest tests/unit

contract-test:
	CONTRACTS_ROOT=$(CONTRACTS_ROOT) MODEL_MANIFEST_PATH=$(MODEL_MANIFEST_PATH) MODEL_ARTIFACT_ROOT=$(MODEL_ARTIFACT_ROOT) $(PYTHON) -m pytest tests/contract

integration-test:
	CONTRACTS_ROOT=$(CONTRACTS_ROOT) MODEL_MANIFEST_PATH=$(MODEL_MANIFEST_PATH) MODEL_ARTIFACT_ROOT=$(MODEL_ARTIFACT_ROOT) $(PYTHON) -m pytest tests/integration

reproducibility-test:
	CONTRACTS_ROOT=$(CONTRACTS_ROOT) MODEL_MANIFEST_PATH=$(MODEL_MANIFEST_PATH) MODEL_ARTIFACT_ROOT=$(MODEL_ARTIFACT_ROOT) $(PYTHON) -m pytest tests/integration/test_reproducibility.py

train-baseline:
	$(PYTHON) scripts/train_baseline_model.py --seed 42 --output-dir artifacts

build:
	PYTHONPYCACHEPREFIX=.pycache $(PYTHON) -m compileall src

run:
	CONTRACTS_ROOT=$(CONTRACTS_ROOT) MODEL_MANIFEST_PATH=$(MODEL_MANIFEST_PATH) MODEL_ARTIFACT_ROOT=$(MODEL_ARTIFACT_ROOT) $(PYTHON) -m uvicorn rippleguard_agent_runtime.app.api:app --host 127.0.0.1 --port 8080

verify: lint typecheck test contract-test integration-test reproducibility-test build

docker-build:
	docker build -t rippleguard-agent-runtime:phase2-local .

docker-run-readiness:
	docker run --rm -v $(abspath $(CONTRACTS_ROOT)):/app/contracts:ro rippleguard-agent-runtime:phase2-local python -m rippleguard_agent_runtime.app.readiness
