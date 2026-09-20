#!/usr/bin/env python3
"""Run the train → validate → register DAG with multi-metric gates."""

from __future__ import annotations

import json
from pathlib import Path

from ml_pipeline_metaflow_demo.pipeline import build_pipeline, run_pipeline


def main() -> None:
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

    print("history:", result["_history"])
    print("gates:", result.get("gates"))
    print("metrics:", json.dumps(result["metrics"], indent=2))
    print("registry:", result["registry_path"])
    print("gates.json:", result.get("gates_path"))
    print()
    print(Path(result["registry_path"]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
