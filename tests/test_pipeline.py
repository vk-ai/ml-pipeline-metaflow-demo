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


def test_immutable_run_lineage_artifacts(tmp_path: Path):
    out = run_pipeline(
        artifact_dir=str(tmp_path),
        accuracy_threshold=0.85,
        model_name="demo-iris",
        model_version="0.1.0",
        run_id="testrun01",
    )

    run_dir = tmp_path / "runs" / "testrun01"
    assert run_dir.is_dir()
    context_path = run_dir / "context.json"
    assert context_path.is_file()
    assert (run_dir / "model.joblib").is_file()
    assert (run_dir / "registry.json").is_file()

    ctx_doc = json.loads(context_path.read_text(encoding="utf-8"))
    assert ctx_doc["run_id"] == "testrun01"
    assert ctx_doc["steps"] == ["train", "validate", "register"]
    assert ctx_doc["metrics"]["accuracy"] >= 0.85
    assert ctx_doc["accuracy_threshold"] == 0.85
    assert ctx_doc["dataset_hash"]
    assert "timestamp" in ctx_doc
    assert "OSS learning stub" in ctx_doc["notes"]

    card = json.loads(Path(out["registry_path"]).read_text(encoding="utf-8"))
    assert card["run_id"] == "testrun01"
    assert card["lineage"]["steps"] == ["train", "validate", "register"]
    assert card["lineage"]["dataset_hash"] == ctx_doc["dataset_hash"]
    assert card["lineage"]["run_dir"] == str(run_dir)


def test_metric_gate_skips_run_folder(tmp_path: Path):
    with pytest.raises(ValidationError):
        run_pipeline(
            artifact_dir=str(tmp_path),
            accuracy_threshold=0.99,
            force_bad_metrics=True,
            run_id="should-not-exist",
        )
    assert not (tmp_path / "runs" / "should-not-exist").exists()
    assert not (tmp_path / "registry.json").exists()


def test_multi_metric_gates_pass_writes_gates_json(tmp_path: Path):
    out = run_pipeline(
        artifact_dir=str(tmp_path),
        gates={"accuracy": 0.85, "f1": 0.80},
        model_name="demo-iris",
        model_version="0.2.0",
        run_id="gatespass01",
    )
    assert out["_history"] == ["train", "validate", "register"]
    gates_path = tmp_path / "runs" / "gatespass01" / "gates.json"
    assert gates_path.is_file()
    gate_doc = json.loads(gates_path.read_text(encoding="utf-8"))
    assert gate_doc["passed"] is True
    assert gate_doc["status"] == "passed"
    assert "accuracy" in gate_doc["gates"]
    assert "f1" in gate_doc["gates"]
    assert all(r["passed"] for r in gate_doc["results"])

    card = json.loads(Path(out["registry_path"]).read_text(encoding="utf-8"))
    assert card["promotion"] == "candidate"
    assert card["gates"]["accuracy"] == 0.85
    assert card["lineage"]["gates"] == str(gates_path)
    # Lineage run folder still present
    assert (tmp_path / "runs" / "gatespass01" / "context.json").is_file()


def test_multi_metric_gates_fail_one(tmp_path: Path):
    """Fail-one: impossible f1 gate blocks register (no artifacts / no gates.json)."""
    with pytest.raises((ValidationError, Exception), match="f1|quality gates"):
        run_pipeline(
            artifact_dir=str(tmp_path),
            gates={"accuracy": 0.50, "f1": 1.01},  # f1 impossible
            run_id="gatesfail01",
        )
    assert not (tmp_path / "runs" / "gatesfail01").exists()
    assert not (tmp_path / "registry.json").exists()


def test_gates_yaml_file(tmp_path: Path):
    yaml_path = tmp_path / "gates.yaml"
    yaml_path.write_text("gates:\n  accuracy: 0.85\n  f1: 0.80\n", encoding="utf-8")
    out = run_pipeline(
        artifact_dir=str(tmp_path / "art"),
        gates_path=str(yaml_path),
        run_id="gatesyaml01",
    )
    assert out["gates"]["accuracy"] == 0.85
    assert out["gates"]["f1"] == 0.80
    assert (tmp_path / "art" / "runs" / "gatesyaml01" / "gates.json").is_file()


def test_legacy_accuracy_threshold_still_works(tmp_path: Path):
    out = run_pipeline(
        artifact_dir=str(tmp_path),
        accuracy_threshold=0.85,
        run_id="legacy01",
    )
    assert out["gates"] == {"accuracy": 0.85}
    card = json.loads(Path(out["registry_path"]).read_text(encoding="utf-8"))
    assert card["promotion"] == "candidate"
