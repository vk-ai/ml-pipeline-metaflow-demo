"""Wire train → validate → register into a runnable DAG.

Default path is the thin custom ``[metaflow-style]`` DAG. Optional real Metaflow
is opt-in via ``ML_PIPELINE_USE_METAFLOW`` + ``pip install '.[metaflow]'`` and is
exercised through ``flows/train_validate_register.py`` — ``run_pipeline`` always
stays on the toy DAG for CI reliability.
"""

from __future__ import annotations

from typing import Any

from ml_pipeline_metaflow_demo.backend import (
    backend_label,
    fallback_reason,
    metaflow_importable,
    wants_real_metaflow,
)
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
    """Convenience: build + run toy DAG with optional context overrides.

    Always executes the ``[metaflow-style]`` custom DAG. Honesty fields
    ``pipeline_mode`` / ``backend_label`` reflect what ran. When the real
    Metaflow flag is set, ``messages`` explain fallback or the FlowSpec CLI.
    """
    messages: list[str] = []
    reason = fallback_reason()
    if reason:
        messages.append(reason)
    elif wants_real_metaflow() and metaflow_importable():
        messages.append(
            "[metaflow] package importable and ML_PIPELINE_USE_METAFLOW set; "
            "run_pipeline still uses the toy [metaflow-style] DAG — execute "
            "`python flows/train_validate_register.py run` for a real FlowSpec."
        )

    dag = build_pipeline()
    result = dag.run(dict(overrides))
    # Library API always ran the toy path — label honestly.
    result["pipeline_mode"] = "metaflow-style"
    result["backend_label"] = backend_label("metaflow-style")
    if messages:
        result["messages"] = messages
    return result
