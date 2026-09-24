"""Tests for mutable promotion tags + promote gates + Airflow stub export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_pipeline_metaflow_demo.airflow_stub import airflow_available, export_airflow_stub
from ml_pipeline_metaflow_demo.pipeline import run_pipeline
from ml_pipeline_metaflow_demo.promote import PromotionError, mint_or_load_token, promote
from ml_pipeline_metaflow_demo.tags import (
    add_tag,
    latest_with_tag,
    list_tags,
    remove_tag,
    runs_with_tag,
)


def test_register_seeds_candidate_tag(tmp_path: Path):
    out = run_pipeline(
        artifact_dir=str(tmp_path),
        gates={"accuracy": 0.85, "f1": 0.80},
        run_id="tagseed01",
    )
    assert "candidate" in out["tags"]
    store = list_tags(tmp_path)
    assert "candidate" in store["runs"]["tagseed01"]["tags"]
    snap = tmp_path / "runs" / "tagseed01" / "tags.json"
    assert snap.is_file()
    assert "candidate" in json.loads(snap.read_text())["tags"]


def test_tag_add_remove_list(tmp_path: Path):
    run_pipeline(artifact_dir=str(tmp_path), run_id="tagops01", accuracy_threshold=0.85)
    add_tag(tmp_path, "tagops01", "staging")
    tags = list_tags(tmp_path, "tagops01")["tags"]
    assert "candidate" in tags and "staging" in tags
    remove_tag(tmp_path, "tagops01", "staging")
    assert "staging" not in list_tags(tmp_path, "tagops01")["tags"]
    assert runs_with_tag(tmp_path, "candidate") == ["tagops01"]


def test_promote_to_staging_then_production(tmp_path: Path):
    run_pipeline(
        artifact_dir=str(tmp_path),
        gates={"accuracy": 0.85},
        run_id="prom01",
    )
    st = promote(tmp_path, "prom01", to="staging")
    assert "staging" in st["tags"]
    assert st["gates_passed"] is True

    # First production promote mints token
    prod = promote(tmp_path, "prom01", to="production")
    assert "production" in prod["tags"]
    assert "staging" not in prod["tags"]  # stripped
    assert "gate:passed" in prod["tags"]
    token = Path(prod["authorize_token_path"]).read_text().strip()

    # Second production on another run needs authorize
    run_pipeline(
        artifact_dir=str(tmp_path),
        gates={"accuracy": 0.85},
        run_id="prom02",
    )
    with pytest.raises(PromotionError, match="authorize"):
        promote(tmp_path, "prom02", to="production")
    ok = promote(tmp_path, "prom02", to="production", authorize=token)
    assert "production" in ok["tags"]
    assert latest_with_tag(tmp_path, "production") in {"prom01", "prom02"}


def test_promote_refuses_failed_gates(tmp_path: Path):
    # Manually create a run folder with failed gates (register never ran)
    run_dir = tmp_path / "runs" / "bad01"
    run_dir.mkdir(parents=True)
    (run_dir / "gates.json").write_text(
        json.dumps({"passed": False, "status": "failed", "results": []}) + "\n"
    )
    with pytest.raises(PromotionError, match="gates did not pass"):
        promote(tmp_path, "bad01", to="production")


def test_export_airflow_stub_no_airflow_required(tmp_path: Path):
    path = tmp_path / "dags" / "ml_stub.py"
    out = export_airflow_stub(path)
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "train" in text and "validate" in text and "register" in text
    assert "_AIRFLOW_AVAILABLE" in text
    assert "from airflow import DAG" in text
    # Demo itself must not require airflow
    assert airflow_available() is False or isinstance(airflow_available(), bool)
    # Stub module must import cleanly even without airflow
    ns: dict = {}
    exec(compile(text, str(path), "exec"), ns)
    assert "dag" in ns  # may be None when airflow missing
