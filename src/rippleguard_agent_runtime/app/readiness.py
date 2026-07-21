from __future__ import annotations

import json

from rippleguard_agent_runtime.config.settings import Settings
from rippleguard_agent_runtime.loan_decision.service import LoanDecisionAgentService


def main() -> int:
    settings = Settings.from_env()
    settings.validate()
    service = LoanDecisionAgentService(settings.contracts_root, settings.model_manifest_path, settings.model_artifact_root)
    print(json.dumps(service.readiness(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
