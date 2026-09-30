"""Champion vs challenger relative gate (OSS learning stub).

Absolute gates (``gates.json``) answer "is this run good enough?". This module
answers the next question practitioners ask: **"does it beat what's live?"**

Both the challenger run and the current ``production`` run (the *champion*) are
re-scored on the **same frozen eval split**, and per metric the challenger must
satisfy::

    challenger >= champion + max(min_absolute_change,
                                 min_relative_change * |champion|)

``min_absolute_change = 0`` means "no regression"; set it > 0 to require a
real improvement (a margin / delta). The comparison is written to
``runs/<challenger>/champion_compare.json``.

Teaching stand-in for MLflow ``validate_evaluation_results(...)`` with
``MetricThreshold(min_absolute_change=..., min_relative_change=...)`` against a
baseline model + registry aliases — **not** MLflow. No statistical significance
test: on a 45-row eval split one sample flips accuracy by ~0.022, so small
deltas are noise-sensitive (see README caveat).

Sources:
- https://mlflow.org/docs/latest/ml/evaluation/
- https://github.com/tkgo1599-max/model-promotion-gate
- https://github.com/AyushPatel94/modelgate
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

from ml_pipeline_metaflow_demo.dataset import load_tiny_split
from ml_pipeline_metaflow_demo.tags import latest_with_tag, list_tags, load_tags

DEFAULT_CHAMPION_METRICS = ("accuracy", "f1")
DEFAULT_EVAL_RANDOM_STATE = 42
_EPS = 1e-12

ChampionGate = dict[str, dict[str, float]]


def frozen_eval_split(
    random_state: int = DEFAULT_EVAL_RANDOM_STATE,
    test_size: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return ``(X_eval, y_eval, fingerprint)`` for the frozen eval split.

    The split is fixed by ``random_state`` (default 42, the pipeline default), so
    champion and challenger are always scored on identical rows. The fingerprint
    is a short sha256 of the rows, recorded in ``champion_compare.json``.
    """
    _, X_eval, _, y_eval = load_tiny_split(test_size=test_size, random_state=random_state)
    h = hashlib.sha256()
    h.update(f"eval_random_state={random_state};test_size={test_size}".encode())
    h.update(np.ascontiguousarray(X_eval).tobytes())
    h.update(np.ascontiguousarray(y_eval).tobytes())
    return X_eval, y_eval, h.hexdigest()[:16]


def _normalize_rule(metric: str, rule: Any) -> dict[str, float]:
    if isinstance(rule, Mapping):
        unknown = set(rule) - {"min_absolute_change", "min_relative_change"}
        if unknown:
            raise ValueError(f"unknown champion rule keys for {metric!r}: {sorted(unknown)}")
        return {
            "min_absolute_change": float(rule.get("min_absolute_change", 0.0)),
            "min_relative_change": float(rule.get("min_relative_change", 0.0)),
        }
    return {"min_absolute_change": float(rule), "min_relative_change": 0.0}


def load_champion_gate_file(path: str | Path) -> ChampionGate:
    """Load the ``champion:`` section of a gates YAML/JSON file.

    YAML shape (tiny parser, no PyYAML)::

        champion:
          accuracy: 0.01   # min_absolute_change
          f1: 0.0
    """
    from ml_pipeline_metaflow_demo.steps import _parse_simple_gates_yaml

    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        data = _parse_simple_gates_yaml(text)
    else:
        data = json.loads(text)
    raw = data.get("champion")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"no champion: map found in {p}")
    return resolve_champion_gate(dict(raw))


def resolve_champion_gate(spec: Any = None) -> ChampionGate:
    """Normalize a champion gate spec to ``{metric: {min_absolute_change, min_relative_change}}``.

    Accepts:
    - ``None`` → accuracy + f1 with delta 0.0 (no-regression)
    - a number ``d`` → accuracy + f1 with ``min_absolute_change = d``
    - ``{"accuracy": 0.01}`` or ``{"f1": {"min_relative_change": 0.02}}``
    - a path to a gates YAML/JSON with a ``champion:`` section
    """
    from ml_pipeline_metaflow_demo.steps import _METRIC_FNS

    if spec is None:
        gate = {m: _normalize_rule(m, 0.0) for m in DEFAULT_CHAMPION_METRICS}
    elif isinstance(spec, (str, Path)):
        return load_champion_gate_file(spec)
    elif isinstance(spec, (int, float)) and not isinstance(spec, bool):
        gate = {m: _normalize_rule(m, spec) for m in DEFAULT_CHAMPION_METRICS}
    elif isinstance(spec, Mapping):
        if not spec:
            raise ValueError("champion gate must name at least one metric")
        gate = {str(m): _normalize_rule(str(m), r) for m, r in spec.items()}
    else:
        raise TypeError(f"unsupported champion gate spec: {spec!r}")
    for name in gate:
        if name not in _METRIC_FNS:
            raise ValueError(
                f"unknown champion metric {name!r}; expected one of {sorted(_METRIC_FNS)}"
            )
    return gate


def production_stack(artifact_dir: str | Path) -> list[str]:
    """Ordered production history (oldest → current) used by ``rollback``."""
    return list(load_tags(artifact_dir).get("production_stack") or [])


def current_champion(artifact_dir: str | Path) -> str | None:
    """Run currently tagged ``production`` (top of the production stack when set)."""
    stack = production_stack(artifact_dir)
    if stack:
        top = stack[-1]
        if "production" in (list_tags(artifact_dir, top).get("tags") or []):
            return top
    return latest_with_tag(artifact_dir, "production")


def _load_run_model(artifact_dir: str | Path, run_id: str):
    path = Path(artifact_dir) / "runs" / run_id / "model.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"no model.joblib for run {run_id!r} ({path})")
    return joblib.load(path)


def _run_context(artifact_dir: str | Path, run_id: str) -> dict[str, Any]:
    path = Path(artifact_dir) / "runs" / run_id / "context.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def score_run(
    artifact_dir: str | Path,
    run_id: str,
    metrics: list[str] | tuple[str, ...],
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Re-score a registered run's model on ``(X, y)`` (does not trust stored metrics)."""
    from ml_pipeline_metaflow_demo.steps import _METRIC_FNS

    y_pred = _load_run_model(artifact_dir, run_id).predict(X)
    return {m: _METRIC_FNS[m](y, y_pred) for m in metrics}


def compare_to_champion(
    artifact_dir: str | Path,
    challenger_id: str,
    *,
    champion_id: str | None = None,
    gate: Any = None,
    eval_random_state: int = DEFAULT_EVAL_RANDOM_STATE,
    write: bool = True,
) -> dict[str, Any]:
    """Compare ``challenger_id`` against the current champion on the frozen split.

    Returns a doc with ``status`` in ``{"passed", "blocked", "no_champion", "self"}``
    and ``passed`` (True for everything except ``blocked``). ``no_champion`` means
    there is no production run yet → only the absolute gates apply.
    """
    rules = resolve_champion_gate(gate)
    champion = champion_id if champion_id is not None else current_champion(artifact_dir)
    X, y, fingerprint = frozen_eval_split(eval_random_state)
    metrics = list(rules)
    doc: dict[str, Any] = {
        "challenger": challenger_id,
        "champion": champion,
        "eval_split": {
            "dataset": "iris-binary (setosa vs rest)",
            "random_state": int(eval_random_state),
            "n_rows": int(len(y)),
            "fingerprint": fingerprint,
        },
        "rule": "challenger >= champion + max(min_absolute_change, "
        "min_relative_change * |champion|)",
        "gate": rules,
        "results": [],
        "warnings": [],
        "compared_at": datetime.now(timezone.utc).isoformat(),
        "notes": (
            "OSS learning stub — champion/challenger relative gate (MLflow "
            "validate_evaluation_results-style); not MLflow, no significance test."
        ),
    }

    if champion is None or champion == challenger_id:
        doc["status"] = "no_champion" if champion is None else "self"
        doc["passed"] = True
        doc["challenger_metrics"] = score_run(artifact_dir, challenger_id, metrics, X, y)
    else:
        ch = score_run(artifact_dir, challenger_id, metrics, X, y)
        cp = score_run(artifact_dir, champion, metrics, X, y)
        doc["challenger_metrics"] = ch
        doc["champion_metrics"] = cp
        all_ok = True
        for m, rule in rules.items():
            margin = max(rule["min_absolute_change"], rule["min_relative_change"] * abs(cp[m]))
            required = cp[m] + margin
            ok = bool(ch[m] + _EPS >= required)
            all_ok &= ok
            doc["results"].append(
                {
                    "metric": m,
                    "challenger": ch[m],
                    "champion": cp[m],
                    "delta": ch[m] - cp[m],
                    "required_margin": margin,
                    "required": required,
                    "passed": ok,
                }
            )
        doc["passed"] = all_ok
        doc["status"] = "passed" if all_ok else "blocked"
        # Leakage / comparability hints: runs trained on a different split seed may
        # have seen frozen-eval rows during training.
        for rid in (challenger_id, champion):
            rs = _run_context(artifact_dir, rid).get("random_state")
            if rs is not None and int(rs) != int(eval_random_state):
                doc["warnings"].append(
                    f"run {rid!r} trained with random_state={rs} != eval_random_state="
                    f"{eval_random_state}; its train split may overlap the frozen eval rows"
                )
        n = len(y)
        if any(0 < r["required_margin"] < 1.0 / n for r in doc["results"]):
            doc["warnings"].append(
                f"required margin < 1/{n} (one eval row); comparison is noise-sensitive"
            )

    if write:
        run_dir = Path(artifact_dir) / "runs" / challenger_id
        if run_dir.is_dir():
            out = run_dir / "champion_compare.json"
            out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
            doc["path"] = str(out)
    return doc


def format_failures(doc: dict[str, Any]) -> str:
    """Human-readable reason naming each failing metric."""
    parts = [
        f"{r['metric']} {r['challenger']:.4f} < required {r['required']:.4f} "
        f"(champion {r['champion']:.4f} + margin {r['required_margin']:.4f})"
        for r in doc.get("results", [])
        if not r["passed"]
    ]
    return "; ".join(parts)
