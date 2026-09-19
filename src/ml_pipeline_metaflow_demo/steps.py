"""Pipeline steps: train → validate → register (stub) with run lineage."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_pipeline_metaflow_demo.dataset import load_tiny_split


class ValidationError(RuntimeError):
    """Raised when validation metrics fall below the configured threshold."""


def _dataset_hash_stub(X_train: np.ndarray, y_train: np.ndarray, random_state: int) -> str:
    """Stable offline fingerprint of the training split (teaching stand-in for data lineage)."""
    h = hashlib.sha256()
    h.update(f"random_state={random_state}".encode())
    h.update(np.ascontiguousarray(X_train).tobytes())
    h.update(np.ascontiguousarray(y_train).tobytes())
    return h.hexdigest()[:16]


def train_step(ctx: dict[str, Any]) -> dict[str, Any]:
    """Fit a tiny sklearn classifier and stash it on the context."""
    random_state = int(ctx.get("random_state", 42))
    X_train, X_test, y_train, y_test = load_tiny_split(random_state=random_state)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=200, random_state=random_state),
            ),
        ]
    )
    model.fit(X_train, y_train)

    ctx["model"] = model
    ctx["X_test"] = X_test
    ctx["y_test"] = y_test
    ctx["train_samples"] = int(len(X_train))
    ctx["test_samples"] = int(len(X_test))
    ctx["dataset_hash"] = _dataset_hash_stub(X_train, y_train, random_state)
    ctx["random_state"] = random_state
    return ctx


def validate_step(ctx: dict[str, Any]) -> dict[str, Any]:
    """Score the model; fail the DAG if accuracy is below threshold."""
    model = ctx["model"]
    X_test = ctx["X_test"]
    y_test = ctx["y_test"]
    threshold = float(ctx.get("accuracy_threshold", 0.85))

    # Optional test hook: force a bad metric without retraining
    if ctx.get("force_bad_metrics"):
        y_pred = 1 - y_test  # invert labels → near-zero accuracy
    else:
        y_pred = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    metrics = {"accuracy": accuracy, "f1": f1, "threshold": threshold}
    ctx["metrics"] = metrics

    if accuracy < threshold:
        raise ValidationError(
            f"accuracy {accuracy:.4f} < threshold {threshold:.4f}; refusing to register"
        )
    return ctx


def _jsonable_context(ctx: dict[str, Any], *, run_id: str, registered_at: str) -> dict[str, Any]:
    """Serialize lineage-friendly context (drop non-JSON model/arrays)."""
    history = list(ctx.get("_history", []))
    # register has not appended itself yet when we snapshot inside register_step
    steps = history + (["register"] if history and history[-1] != "register" else [])
    if "register" not in steps:
        steps = history + ["register"]
    return {
        "run_id": run_id,
        "steps": steps,
        "history": history,
        "metrics": ctx.get("metrics"),
        "accuracy_threshold": float(ctx.get("accuracy_threshold", 0.85)),
        "dataset_hash": ctx.get("dataset_hash"),
        "random_state": ctx.get("random_state"),
        "train_samples": ctx.get("train_samples"),
        "test_samples": ctx.get("test_samples"),
        "timestamp": registered_at,
        "notes": (
            "OSS learning stub — immutable run folder is a teaching stand-in for "
            "Metaflow/MLflow lineage, not a production artifact store."
        ),
    }


def register_step(ctx: dict[str, Any]) -> dict[str, Any]:
    """Write immutable run folder + model + registry with lineage."""
    out_dir = Path(ctx.get("artifact_dir", "artifacts"))
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = str(ctx.get("run_id") or uuid.uuid4().hex[:12])
    registered_at = datetime.now(timezone.utc).isoformat()
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.joblib"
    run_model_path = run_dir / "model.joblib"
    joblib.dump(ctx["model"], model_path)
    joblib.dump(ctx["model"], run_model_path)

    lineage_steps = list(ctx.get("_history", [])) + ["register"]
    context_doc = _jsonable_context(ctx, run_id=run_id, registered_at=registered_at)
    context_doc["steps"] = lineage_steps
    (run_dir / "context.json").write_text(
        json.dumps(context_doc, indent=2) + "\n", encoding="utf-8"
    )

    card = {
        "name": ctx.get("model_name", "iris-binary-logreg"),
        "version": ctx.get("model_version", "0.1.0"),
        "framework": "sklearn",
        "algorithm": "LogisticRegression + StandardScaler",
        "metrics": ctx["metrics"],
        "train_samples": ctx.get("train_samples"),
        "test_samples": ctx.get("test_samples"),
        "artifact": str(model_path),
        "run_artifact": str(run_model_path),
        "run_id": run_id,
        "lineage": {
            "steps": lineage_steps,
            "dataset_hash": ctx.get("dataset_hash"),
            "run_dir": str(run_dir),
            "context": str(run_dir / "context.json"),
        },
        "registered_at": registered_at,
        "status": "registered_stub",
        "notes": "OSS learning stub — not a production model registry.",
    }

    registry_path = out_dir / "registry.json"
    run_registry_path = run_dir / "registry.json"
    payload = json.dumps(card, indent=2) + "\n"
    registry_path.write_text(payload, encoding="utf-8")
    run_registry_path.write_text(payload, encoding="utf-8")

    ctx["run_id"] = run_id
    ctx["run_dir"] = str(run_dir)
    ctx["model_path"] = str(model_path)
    ctx["registry_path"] = str(registry_path)
    ctx["model_card"] = card
    return ctx
