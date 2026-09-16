"""Wire train → validate → register into a runnable DAG."""

from __future__ import annotations

from typing import Any

from ml_pipeline_metaflow_demo.dag import DAG
from ml_pipeline_metaflow_demo.steps import register_step, train_step, validate_step


def build_pipeline(name: str = "ml-train-validate-register") -> DAG:
    """Build the canonical three-step ML DAG."""
    dag = DAG(name=name)
    dag.add_step("train", train_step)
    dag.add_step("validate", validate_step)
    dag.add_step("register", register_step)
    dag.connect("train", "validate")
    dag.connect("validate", "register")
    return dag


def run_pipeline(**overrides: Any) -> dict[str, Any]:
    """Convenience: build + run with optional context overrides."""
    dag = build_pipeline()
    return dag.run(dict(overrides))
