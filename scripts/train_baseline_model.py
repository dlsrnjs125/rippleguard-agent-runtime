#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

FEATURE_ORDER = (
    "annualIncome",
    "monthlyIncomeMean",
    "monthlyIncomeVolatility",
    "debtToIncomeRatio",
    "existingDebtAmount",
    "delinquencyCount",
    "platformSettlementMonths",
    "platformSettlementMean",
    "platformSettlementVolatility",
    "contractDurationMonths",
    "incomeDeclarationAvailable",
    "telecomPaymentDelinquencyCount",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    models = args.output_dir / "models"
    manifests = args.output_dir / "manifests"
    reports = args.output_dir / "reports"
    for path in (models, manifests, reports):
        path.mkdir(parents=True, exist_ok=True)

    x, y = synthetic_dataset(args.seed, 640)
    train_x, eval_x, train_y, eval_y = train_test_split(x, y, test_size=0.25, random_state=args.seed, stratify=y)
    model = xgb.XGBClassifier(
        n_estimators=24,
        max_depth=3,
        learning_rate=0.08,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=args.seed,
        n_jobs=1,
        tree_method="hist",
    )
    model.fit(train_x, train_y)
    probabilities = model.predict_proba(eval_x)[:, 1]
    predicted = probabilities >= 0.72
    precision, recall, f1, _ = precision_recall_fscore_support(eval_y, predicted, average="binary", zero_division=0)
    report = {
        "dataset": "synthetic-loans.v1",
        "seed": args.seed,
        "modelCandidates": [
            {"framework": "xgboost", "status": "selected", "reason": "deterministic JSON artifact and SHAP TreeExplainer support"},
            {"framework": "lightgbm", "status": "not_selected", "reason": "not required for runtime fallback"},
        ],
        "metrics": {
            "roc_auc": round(float(roc_auc_score(eval_y, probabilities)), 6),
            "pr_auc": round(float(average_precision_score(eval_y, probabilities)), 6),
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1": round(float(f1), 6),
            "brier_score": round(float(brier_score_loss(eval_y, probabilities)), 6),
        },
        "limitations": [
            "synthetic baseline only",
            "not a real financial approval model",
            "not a fairness or regulatory suitability claim",
        ],
    }
    artifact = models / "phase2-loan-xgboost.v1.0.0.json"
    model.get_booster().save_model(str(artifact))
    artifact_digest = file_digest(artifact)
    dependency_digest = text_digest("\n".join(sorted(_dependency_versions())))
    manifest = {
        "schemaVersion": "1.0.0",
        "modelType": "TABULAR",
        "modelName": "phase2-loan-xgboost",
        "modelVersion": "loan-model.v1.0.0",
        "framework": "xgboost",
        "frameworkVersion": xgb.__version__,
        "featureSchemaVersion": "phase-2-loan-features.v1.0.0",
        "preprocessingVersion": "preprocess.v1.0.0",
        "trainingDatasetReference": "dataset://synthetic-loans/train/v1",
        "trainingDatasetDigest": text_digest(json.dumps({"seed": args.seed, "rows": int(train_x.shape[0])}, sort_keys=True)),
        "evaluationDatasetVersion": "synthetic-loans-eval.v1.0.0",
        "evaluationDatasetDigest": text_digest(json.dumps({"seed": args.seed, "rows": int(eval_x.shape[0])}, sort_keys=True)),
        "trainingCodeCommit": _git_commit(),
        "randomSeed": args.seed,
        "thresholdVersion": "threshold.v1.0.0",
        "modelBinaryArtifactReference": "file://phase2-loan-xgboost.v1.0.0.json",
        "artifactDigestAlgorithm": "sha256",
        "modelBinaryArtifactDigest": artifact_digest,
        "modelFormat": "xgboost-json",
        "targetDefinition": "repayment within agreed term",
        "scoreSemantics": "higher score means higher repayment likelihood",
        "selectedMetrics": {
            "primary": "PR_AUC",
            "guardrail": "RECALL_AT_FIXED_APPROVAL_RISK_THRESHOLD",
            "calibration": "BRIER_SCORE",
        },
        "shapExplainerVersion": "shap.v0.47.2",
        "shapExplainerConfig": {"algorithm": "tree", "checkAdditivity": True},
        "pythonVersion": platform.python_version(),
        "runtimeImageDigest": "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "platformArchitecture": _platform_architecture(),
        "threadCount": 1,
        "deterministicConfig": "single-threaded-xgboost-hist",
        "dependencyLockDigest": dependency_digest,
        "license": "Apache-2.0",
        "source": "https://github.com/dlsrnjs125/rippleguard-agent-runtime",
        "createdAt": "2026-07-21T00:00:00Z",
    }
    (manifests / "phase2-loan-xgboost.v1.0.0.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (manifests / "thresholds.v1.0.0.json").write_text(json.dumps({"threshold.v1.0.0": 0.72}, indent=2, sort_keys=True) + "\n")
    (reports / "model-selection.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"artifact": str(artifact), "digest": artifact_digest}, sort_keys=True))
    return 0


def synthetic_dataset(seed: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    annual_income = rng.normal(72_000_000, 18_000_000, rows).clip(12_000_000, 180_000_000)
    monthly_mean = annual_income / 12
    income_vol = rng.uniform(0.02, 0.6, rows)
    dti = rng.uniform(0.05, 0.75, rows)
    existing_debt = annual_income * dti * rng.uniform(0.5, 1.8, rows)
    delinquency = rng.poisson(0.35, rows).clip(0, 6)
    settlement_months = rng.integers(1, 84, rows)
    settlement_mean = monthly_mean * rng.uniform(0.55, 1.25, rows)
    settlement_vol = rng.uniform(0.02, 0.7, rows)
    duration = rng.choice([12, 24, 36, 48, 60], rows)
    income_declared = rng.choice([0, 1], rows, p=[0.08, 0.92])
    telecom = rng.poisson(0.18, rows).clip(0, 5)
    features = np.column_stack([
        annual_income,
        monthly_mean,
        income_vol,
        dti,
        existing_debt,
        delinquency,
        settlement_months,
        settlement_mean,
        settlement_vol,
        duration,
        income_declared,
        telecom,
    ]).astype(np.float32)
    logit = 2.2 - dti * 4.0 - delinquency * 0.45 - telecom * 0.35 + income_declared * 0.55
    logit += (settlement_months > 24) * 0.35 - income_vol * 0.8 - settlement_vol * 0.6
    probability = 1.0 / (1.0 + np.exp(-logit))
    labels = (rng.random(rows) < probability).astype(int)
    return features, labels


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def text_digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _dependency_versions() -> list[str]:
    return [f"python=={platform.python_version()}", f"xgboost=={xgb.__version__}", "shap==0.47.2"]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        return "0000000000000000000000000000000000000000"


def _platform_architecture() -> str:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "darwin/arm64" if sys.platform == "darwin" else "linux/arm64"
    return "linux/amd64"


if __name__ == "__main__":
    raise SystemExit(main())
