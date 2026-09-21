"""Pipeline steps: train → validate → register (stub) with run lineage + gates."""

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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_pipeline_metaflow_demo.dataset import load_tiny_split

# Metric name → callable(y_true, y_pred) -> float
_METRIC_FNS = {
    "accuracy": lambda y_true, y_pred: float(accuracy_score(y_true, y_pred)),
    "f1": lambda y_true, y_pred: float(f1_score(y_true, y_pred, zero_division=0)),
    "precision": lambda y_true, y_pred: float(
        precision_score(y_true, y_pred, zero_division=0)
    ),
    "recall": lambda y_true, y_pred: float(recall_score(y_true, y_pred, zero_division=0)),
}


class ValidationError(RuntimeError):
    """Raised when validation metrics fall below the configured threshold(s)."""


class GateFailed(ValidationError):
    """Raised when any multi-metric quality gate fails (alias of ValidationError)."""


def _dataset_hash_stub(X_train: np.ndarray, y_train: np.ndarray, random_state: int) -> str:
    """Stable offline fingerprint of the training split (teaching stand-in for data lineage)."""
    h = hashlib.sha256()
    h.update(f"random_state={random_state}".encode())
    h.update(np.ascontiguousarray(X_train).tobytes())
    h.update(np.ascontiguousarray(y_train).tobytes())
    return h.hexdigest()[:16]


def resolve_gates(ctx: dict[str, Any]) -> dict[str, float]:
    """Resolve multi-metric gates from ctx / YAML path / legacy accuracy_threshold.

    Preferred: ``ctx["gates"]`` dict, e.g. ``{"accuracy": 0.90, "f1": 0.85}``.
    Or ``ctx["gates_path"]`` pointing at a YAML/JSON file with a top-level ``gates:`` map.
    Legacy: ``accuracy_threshold`` alone → ``{"accuracy": threshold}``.
    """
    gates: dict[str, float] | None = None
    if isinstance(ctx.get("gates"), dict) and ctx["gates"]:
        gates = {str(k): float(v) for k, v in ctx["gates"].items()}
    elif ctx.get("gates_path"):
        gates = _load_gates_file(Path(ctx["gates_path"]))
    if gates is None:
        threshold = float(ctx.get("accuracy_threshold", 0.85))
        gates = {"accuracy": threshold}
    for name in gates:
        if name not in _METRIC_FNS:
            raise ValueError(
                f"unknown gate metric {name!r}; expected one of {sorted(_METRIC_FNS)}"
            )
    return gates


def _parse_simple_gates_yaml(text: str) -> dict[str, Any]:
    """Tiny YAML subset for ``gates:`` maps — no PyYAML dependency.

    Supports::

        gates:
          accuracy: 0.90
          f1: 0.85

    or a bare flat map of metric → float. Teaching stub only.
    """
    root: dict[str, Any] = {}
    current_map: dict[str, Any] | None = None
    current_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"unsupported YAML line: {raw_line!r}")
        key, _, rest = stripped.partition(":")
        key = key.strip()
        rest = rest.strip()
        if indent == 0:
            if rest == "":
                current_key = key
                current_map = {}
                root[key] = current_map
            else:
                root[key] = _yaml_scalar(rest)
                current_map = None
                current_key = None
        else:
            if current_map is None or current_key is None:
                raise ValueError(f"unexpected indented key {key!r}")
            current_map[key] = _yaml_scalar(rest)
    return root


def _yaml_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            return value[1:-1]
        return value


def _load_gates_file(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    data: dict[str, Any]
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = _parse_simple_gates_yaml(text)
    else:
        data = json.loads(text)
    raw = data.get("gates", data)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"no gates map found in {path}")
    return {str(k): float(v) for k, v in raw.items()}


def evaluate_gates(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gates: dict[str, float],
) -> dict[str, Any]:
    """Compute metrics and per-gate pass/fail. Teaching stand-in for MLflow MetricThreshold."""
    metrics: dict[str, float] = {}
    results: list[dict[str, Any]] = []
    all_passed = True
    for name, threshold in gates.items():
        value = _METRIC_FNS[name](y_true, y_pred)
        metrics[name] = value
        passed = bool(value >= threshold)
        if not passed:
            all_passed = False
        results.append(
            {
                "metric": name,
                "value": value,
                "threshold": float(threshold),
                "passed": passed,
            }
        )
    # Always include accuracy/f1 for lineage even if not gated
    if "accuracy" not in metrics:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    if "f1" not in metrics:
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    return {
        "gates": {k: float(v) for k, v in gates.items()},
        "metrics": metrics,
        "results": results,
        "passed": all_passed,
        "status": "passed" if all_passed else "failed",
        "notes": (
            "OSS learning stub — multi-metric gates teach MLflow-style "
            "MetricThreshold / validate_evaluation_results ideas; not MLflow."
        ),
    }


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
    """Score the model against pluggable multi-metric gates; fail DAG if any miss."""
    model = ctx["model"]
    X_test = ctx["X_test"]
    y_test = ctx["y_test"]
    gates = resolve_gates(ctx)

    # Optional test hook: force a bad metric without retraining
    if ctx.get("force_bad_metrics"):
        y_pred = 1 - y_test  # invert labels → near-zero accuracy
    else:
        y_pred = model.predict(X_test)

    gate_doc = evaluate_gates(y_test, y_pred, gates)
    # Keep legacy metrics shape (+ thresholds map)
    metrics = dict(gate_doc["metrics"])
    metrics["threshold"] = float(gates.get("accuracy", next(iter(gates.values()))))
    metrics["gates"] = dict(gates)
    ctx["metrics"] = metrics
    ctx["gates"] = gates
    ctx["gates_result"] = gate_doc

    if not gate_doc["passed"]:
        failed = [r for r in gate_doc["results"] if not r["passed"]]
        detail = ", ".join(
            f"{r['metric']} {r['value']:.4f} < {r['threshold']:.4f}" for r in failed
        )
        raise GateFailed(f"quality gates failed: {detail}; refusing to register")
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
        "gates": ctx.get("gates"),
        "accuracy_threshold": float(
            (ctx.get("gates") or {}).get(
                "accuracy", ctx.get("accuracy_threshold", 0.85)
            )
        ),
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
    """Write immutable run folder + model + registry with lineage + gates.json."""
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

    gate_doc = ctx.get("gates_result") or {
        "gates": ctx.get("gates") or {},
        "metrics": ctx.get("metrics") or {},
        "results": [],
        "passed": True,
        "status": "passed",
        "notes": "OSS learning stub — multi-metric gates.",
    }
    gates_path = run_dir / "gates.json"
    gates_path.write_text(json.dumps(gate_doc, indent=2) + "\n", encoding="utf-8")

    promotion = ctx.get("promotion", "candidate")
    card = {
        "name": ctx.get("model_name", "iris-binary-logreg"),
        "version": ctx.get("model_version", "0.1.0"),
        "framework": "sklearn",
        "algorithm": "LogisticRegression + StandardScaler",
        "metrics": ctx["metrics"],
        "gates": ctx.get("gates"),
        "promotion": promotion,
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
            "gates": str(gates_path),
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
    ctx["gates_path"] = str(gates_path)
    ctx["model_card"] = card
    return ctx
