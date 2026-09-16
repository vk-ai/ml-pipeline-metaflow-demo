"""Pipeline steps: train → validate → register (stub)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_pipeline_metaflow_demo.dataset import load_tiny_split


class ValidationError(RuntimeError):
    """Raised when validation metrics fall below the configured threshold."""


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


def register_step(ctx: dict[str, Any]) -> dict[str, Any]:
    """Write a model artifact + registry/model-card JSON (stub registry)."""
    out_dir = Path(ctx.get("artifact_dir", "artifacts"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.joblib"
    joblib.dump(ctx["model"], model_path)

    card = {
        "name": ctx.get("model_name", "iris-binary-logreg"),
        "version": ctx.get("model_version", "0.1.0"),
        "framework": "sklearn",
        "algorithm": "LogisticRegression + StandardScaler",
        "metrics": ctx["metrics"],
        "train_samples": ctx.get("train_samples"),
        "test_samples": ctx.get("test_samples"),
        "artifact": str(model_path),
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "status": "registered_stub",
        "notes": "OSS learning stub — not a production model registry.",
    }

    registry_path = out_dir / "registry.json"
    registry_path.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")

    ctx["model_path"] = str(model_path)
    ctx["registry_path"] = str(registry_path)
    ctx["model_card"] = card
    return ctx
