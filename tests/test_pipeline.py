from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_pipeline_metaflow_demo.pipeline import build_pipeline, run_pipeline
from ml_pipeline_metaflow_demo.steps import ValidationError


def test_happy_path_registers_artifacts(tmp_path: Path):
    out = run_pipeline(
        artifact_dir=str(tmp_path),
        accuracy_threshold=0.85,
        model_name="demo-iris",
        model_version="0.1.0",
    )

    assert out["_history"] == ["train", "validate", "register"]
    assert out["metrics"]["accuracy"] >= 0.85
    assert Path(out["model_path"]).is_file()
    assert Path(out["registry_path"]).is_file()

    card = json.loads(Path(out["registry_path"]).read_text(encoding="utf-8"))
    assert card["name"] == "demo-iris"
    assert card["status"] == "registered_stub"
    assert card["metrics"]["accuracy"] == out["metrics"]["accuracy"]
    assert "registered_at" in card


def test_fail_on_bad_metrics_skips_register(tmp_path: Path):
    dag = build_pipeline()
    with pytest.raises(ValidationError, match="accuracy"):
        dag.run(
            {
                "artifact_dir": str(tmp_path),
                "accuracy_threshold": 0.99,
                "force_bad_metrics": True,
            }
        )

    # Register must not have run
    assert not (tmp_path / "registry.json").exists()
    assert not (tmp_path / "model.joblib").exists()


def test_fail_on_impossible_threshold(tmp_path: Path):
    """Even a good model fails when threshold is set impossibly high."""
    with pytest.raises(ValidationError):
        run_pipeline(
            artifact_dir=str(tmp_path),
            accuracy_threshold=1.01,  # impossible
        )
    assert not (tmp_path / "registry.json").exists()


def test_pipeline_graph_shape():
    dag = build_pipeline()
    assert dag.topo_order() == ["train", "validate", "register"]
    assert dag.graph_text() == "ml-train-validate-register: train → validate → register"
