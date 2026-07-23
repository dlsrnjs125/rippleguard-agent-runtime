.PHONY: install install-training lint typecheck test contract-test integration-test reproducibility-test train-baseline build run verify assert-clean build-image verify-image-provenance release-image-check docker-build docker-run-readiness

PYTHON ?= python3
CONTRACTS_ROOT ?= ../rippleguard-contracts
MODEL_MANIFEST_PATH ?= artifacts/manifests/phase2-loan-xgboost.v1.0.0.json
MODEL_ARTIFACT_ROOT ?= artifacts/models
OCI_SOURCE ?= https://github.com/dlsrnjs125/rippleguard-agent-runtime
OCI_REVISION := $(shell git rev-parse HEAD)
IMAGE_TAG := rippleguard-agent-runtime:$(shell git rev-parse --short=12 HEAD)

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-training:
	$(PYTHON) -m pip install -e ".[dev,training]"

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

assert-clean:
	@test -z "$$(git status --porcelain --untracked-files=all)" || \
		(echo "Refusing provenance build from dirty working tree" >&2; \
		 git status --short --untracked-files=all >&2; \
		 exit 1)

build-image: assert-clean
	@echo "$(OCI_REVISION)" | grep -Eq '^[0-9a-f]{40}$$'
	@test "$(OCI_SOURCE)" = "https://github.com/dlsrnjs125/rippleguard-agent-runtime"
	docker build \
		--build-arg "OCI_REVISION=$(OCI_REVISION)" \
		--build-arg "OCI_SOURCE=$(OCI_SOURCE)" \
		-t "$(IMAGE_TAG)" \
		.

verify-image-provenance:
	@actual_revision="$$(docker image inspect "$(IMAGE_TAG)" \
		--format '{{ index .Config.Labels "org.opencontainers.image.revision" }}')"; \
	actual_source="$$(docker image inspect "$(IMAGE_TAG)" \
		--format '{{ index .Config.Labels "org.opencontainers.image.source" }}')"; \
	test "$$actual_revision" = "$(OCI_REVISION)"; \
	test "$$actual_source" = "$(OCI_SOURCE)"; \
	case "$(IMAGE_TAG)" in *:latest) echo "Refusing latest tag" >&2; exit 1 ;; esac

release-image-check: verify build-image verify-image-provenance

docker-build: build-image

docker-run-readiness:
	docker run --rm -v $(abspath $(CONTRACTS_ROOT)):/app/contracts:ro "$(IMAGE_TAG)" python -m rippleguard_agent_runtime.app.readiness
