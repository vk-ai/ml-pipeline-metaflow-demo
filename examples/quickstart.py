#!/usr/bin/env python3
"""Run the train → validate → register DAG with multi-metric gates."""

from __future__ import annotations

import json
from pathlib import Path

from ml_pipeline_metaflow_demo.airflow_stub import export_airflow_stub
from ml_pipeline_metaflow_demo.backend import backend_label, pipeline_mode
from ml_pipeline_metaflow_demo.pipeline import build_pipeline, run_pipeline
from ml_pipeline_metaflow_demo.promote import promote
from ml_pipeline_metaflow_demo.tags import list_tags


def main() -> None:
    mode = pipeline_mode()
    print(f"backend: {backend_label(mode)}  (mode={mode})")
    print()

    dag = build_pipeline()
    print(dag.graph_text())
    print()

    root = Path(__file__).resolve().parents[1]
    artifacts = root / "artifacts"
    gates_path = root / "examples" / "gates.yaml"
    result = run_pipeline(
        artifact_dir=str(artifacts),
        gates_path=str(gates_path),
        model_name="iris-binary-logreg",
        model_version="0.1.0",
    )

    print("pipeline_mode:", result.get("pipeline_mode"))
    print("backend_label:", result.get("backend_label"))
    if result.get("messages"):
        for m in result["messages"]:
            print("note:", m)
    print("history:", result["_history"])
    print("gates:", result.get("gates"))
    print("metrics:", json.dumps(result["metrics"], indent=2))
    print("registry:", result["registry_path"])
    print("gates.json:", result.get("gates_path"))
    print()
    print(Path(result["registry_path"]).read_text(encoding="utf-8"))
    print("tags:", list_tags(artifacts, result["run_id"]))
    # Optional: promote to staging (gates already passed) then export Airflow stub
    promote(artifacts, result["run_id"], to="staging")
    stub = export_airflow_stub(artifacts / "airflow_dag_stub.py")
    print("airflow stub:", stub)
    print("tags after staging:", list_tags(artifacts, result["run_id"]))


if __name__ == "__main__":
    main()
