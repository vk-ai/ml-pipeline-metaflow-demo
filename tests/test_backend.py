"""Backend selection / honesty labels — all offline, no metaflow/airflow required."""

from __future__ import annotations

import builtins
import sys

from ml_pipeline_metaflow_demo.backend import (
    backend_label,
    fallback_reason,
    metaflow_importable,
    pipeline_mode,
    wants_real_metaflow,
)
from ml_pipeline_metaflow_demo.pipeline import run_pipeline


def test_default_run_pipeline_tagged_style(monkeypatch, tmp_path):
    monkeypatch.delenv("ML_PIPELINE_USE_METAFLOW", raising=False)
    out = run_pipeline(artifact_dir=str(tmp_path), accuracy_threshold=0.85)
    assert out["_history"] == ["train", "validate", "register"]
    assert out["pipeline_mode"] == "metaflow-style"
    assert out["backend_label"] == "[metaflow-style]"


def test_flag_false_never_requires_metaflow(monkeypatch):
    """With flag unset/false, selection stays style even if metaflow were present."""
    monkeypatch.delenv("ML_PIPELINE_USE_METAFLOW", raising=False)
    assert wants_real_metaflow() is False
    assert pipeline_mode() == "metaflow-style"
    assert backend_label() == "[metaflow-style]"
    assert fallback_reason() is None

    monkeypatch.setenv("ML_PIPELINE_USE_METAFLOW", "false")
    assert wants_real_metaflow() is False
    assert pipeline_mode() == "metaflow-style"


def test_flag_true_missing_metaflow_falls_back(monkeypatch, tmp_path):
    """Opt-in without the package → style run + clear fallback reason."""
    monkeypatch.setenv("ML_PIPELINE_USE_METAFLOW", "true")

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "metaflow" or name.startswith("metaflow."):
            raise ImportError("metaflow deliberately unavailable in offline test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    for key in list(sys.modules):
        if key == "metaflow" or key.startswith("metaflow."):
            sys.modules.pop(key, None)

    assert metaflow_importable() is False
    assert pipeline_mode() == "metaflow-style"
    reason = fallback_reason()
    assert reason is not None
    assert "metaflow" in reason.lower()
    assert "ML_PIPELINE_USE_METAFLOW" in reason

    out = run_pipeline(artifact_dir=str(tmp_path), accuracy_threshold=0.85)
    assert out["pipeline_mode"] == "metaflow-style"
    assert out["backend_label"] == "[metaflow-style]"
    assert out["_history"] == ["train", "validate", "register"]
    assert any("ML_PIPELINE_USE_METAFLOW" in m for m in out.get("messages", []))


def test_selection_helpers_when_importable(monkeypatch):
    """Unit-test real-path selection without requiring the dep: stub importable()."""
    monkeypatch.setenv("ML_PIPELINE_USE_METAFLOW", "true")
    monkeypatch.setattr(
        "ml_pipeline_metaflow_demo.backend.metaflow_importable",
        lambda: True,
    )
    assert pipeline_mode() == "metaflow"
    assert backend_label() == "[metaflow]"
    assert fallback_reason() is None


def test_run_pipeline_notes_flowspec_when_importable(monkeypatch, tmp_path):
    """Library API stays toy even when metaflow is importable; message points to FlowSpec."""
    monkeypatch.setenv("ML_PIPELINE_USE_METAFLOW", "true")
    monkeypatch.setattr(
        "ml_pipeline_metaflow_demo.pipeline.metaflow_importable",
        lambda: True,
    )
    monkeypatch.setattr(
        "ml_pipeline_metaflow_demo.pipeline.wants_real_metaflow",
        lambda: True,
    )
    monkeypatch.setattr(
        "ml_pipeline_metaflow_demo.pipeline.fallback_reason",
        lambda: None,
    )
    out = run_pipeline(artifact_dir=str(tmp_path), accuracy_threshold=0.85)
    assert out["pipeline_mode"] == "metaflow-style"
    assert out["backend_label"] == "[metaflow-style]"
    assert any("train_validate_register.py" in m for m in out.get("messages", []))
