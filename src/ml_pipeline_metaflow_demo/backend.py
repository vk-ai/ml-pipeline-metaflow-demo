"""Backend selection: default Metaflow-*style* toy DAG vs optional real Metaflow.

Default path stays offline with the thin custom DAG (no Metaflow/Airflow required).
Real Metaflow is opt-in via ``ML_PIPELINE_USE_METAFLOW`` *and* ``pip install '.[metaflow]'``.
The library ``run_pipeline`` API always executes the toy DAG for reliability; learners
run the FlowSpec CLI for a real Metaflow local run.
"""

from __future__ import annotations

import os
from typing import Literal

PipelineMode = Literal["metaflow-style", "metaflow"]

_TRUTHY = {"1", "true", "yes", "on"}

ENV_FLAG = "ML_PIPELINE_USE_METAFLOW"


def wants_real_metaflow() -> bool:
    """Return True when ``ML_PIPELINE_USE_METAFLOW`` opts into the real package."""
    raw = os.environ.get(ENV_FLAG, "")
    return raw.strip().lower() in _TRUTHY


def metaflow_importable() -> bool:
    """True if real Metaflow FlowSpec APIs are importable."""
    try:
        from metaflow import FlowSpec, step  # noqa: F401
    except ImportError:
        return False
    return True


def pipeline_mode(*, force_style: bool = False) -> PipelineMode:
    """Return ``metaflow`` only when opted in and importable; else ``metaflow-style``."""
    if force_style:
        return "metaflow-style"
    if wants_real_metaflow() and metaflow_importable():
        return "metaflow"
    return "metaflow-style"


def backend_label(mode: PipelineMode | None = None) -> str:
    """Bracket tag for logs: ``[metaflow-style]`` or ``[metaflow]``."""
    m = mode if mode is not None else pipeline_mode()
    if m == "metaflow":
        return "[metaflow]"
    return "[metaflow-style]"


def fallback_reason() -> str | None:
    """Human-readable reason when real path was requested but cannot run."""
    if not wants_real_metaflow():
        return None
    if metaflow_importable():
        return None
    return (
        f"{ENV_FLAG}=true but metaflow is not installed "
        "(pip install '.[metaflow]'). Falling back to [metaflow-style]."
    )
