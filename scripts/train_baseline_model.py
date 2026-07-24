#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import xgboost as xgb  # noqa: E402
from rippleguard_agent_runtime.loan_decision.preprocessing import preprocess_feature_vector  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_fscore_support, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

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
    templates = args.output_dir / "templates"
    reports = args.output_dir / "reports"
    for path in (models, manifests, templates, reports):
        path.mkdir(parents=True, exist_ok=True)

    raw_x, y = synthetic_dataset(args.seed, 640)
    x = preprocess_matrix(raw_x)
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
    threshold = 0.72
    xgb_metrics = metrics(eval_y, probabilities, threshold)
    lightgbm_metrics = train_lightgbm_candidate(train_x, eval_x, train_y, eval_y, threshold, args.seed)
    report = {
        "dataset": "synthetic-loans.v1",
        "seed": args.seed,
        "modelCandidates": [
            {
                "framework": "xgboost",
                "status": "selected",
                "reason": "deterministic JSON artifact, compact artifact size, and SHAP TreeExplainer support",
                "metrics": xgb_metrics,
            },
            {
                "framework": "lightgbm",
                "status": "not_selected",
                "reason": "offline comparison candidate only; runtime registers one selected model and no fallback",
                "metrics": lightgbm_metrics,
            },
        ],
        "selectedFramework": "xgboost",
        "metrics": xgb_metrics,
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
        "trainingDatasetDigest": array_digest(train_x, train_y),
        "evaluationDatasetVersion": "synthetic-loans-eval.v1.0.0",
        "evaluationDatasetDigest": array_digest(eval_x, eval_y),
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
        "runtimeImageDigest": "${RUNTIME_IMAGE_DIGEST}",
        "platformArchitecture": _platform_architecture(),
        "threadCount": 1,
        "deterministicConfig": "single-threaded-xgboost-hist",
        "dependencyLockDigest": dependency_digest,
        "license": "Apache-2.0",
        "source": "https://github.com/dlsrnjs125/rippleguard-agent-runtime",
        "createdAt": "2026-07-21T00:00:00Z",
    }
    (templates / "phase2-loan-xgboost.v1.0.0.template.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (manifests / "thresholds.v1.0.0.json").write_text(json.dumps({"threshold.v1.0.0": threshold}, indent=2, sort_keys=True) + "\n")
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


def preprocess_matrix(raw_features: np.ndarray) -> np.ndarray:
    return np.vstack([preprocess_feature_vector(row.tolist())[0] for row in raw_features]).astype(np.float32)


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    started = time.perf_counter()
    predicted = probabilities >= threshold
    latency_ms = (time.perf_counter() - started) * 1000
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average="binary", zero_division=0)
    return {
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 6),
        "pr_auc": round(float(average_precision_score(labels, probabilities)), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "brier_score": round(float(brier_score_loss(labels, probabilities)), 6),
        "threshold": threshold,
        "prediction_latency_ms": round(float(latency_ms), 6),
    }


def train_lightgbm_candidate(
    train_x: np.ndarray,
    eval_x: np.ndarray,
    train_y: np.ndarray,
    eval_y: np.ndarray,
    threshold: float,
    seed: int,
) -> dict[str, float | str | bool]:
    started = time.perf_counter()
    model = lgb.LGBMClassifier(
        n_estimators=24,
        max_depth=3,
        learning_rate=0.08,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="binary",
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(train_x, train_y)
    probabilities = model.predict_proba(eval_x)[:, 1]
    result: dict[str, float | str | bool] = metrics(eval_y, probabilities, threshold)
    result["training_latency_ms"] = round(float((time.perf_counter() - started) * 1000), 6)
    result["shap_tree_explainer_compatible"] = True
    return result


def array_digest(features: np.ndarray, labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (features, labels):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(json.dumps(contiguous.shape).encode("utf-8"))
        digest.update(contiguous.tobytes())
    return "sha256:" + digest.hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def text_digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _dependency_versions() -> list[str]:
    return [f"python=={platform.python_version()}", f"xgboost=={xgb.__version__}", f"lightgbm=={lgb.__version__}", "shap==0.47.2"]


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
