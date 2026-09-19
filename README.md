# ml-pipeline-metaflow-demo

> **OSS / learning demo only.** This is a personal open-source teaching project by [vk-ai](https://github.com/vk-ai). It is **not** employer production software, is **not** affiliated with any employer, and must not be described as a production ML platform or Metaflow/Airflow deployment.

Tiny **Metaflow / Airflow-style ML pipeline**: a thin custom DAG runner that executes **train → validate → register** on a toy sklearn model. No Airflow cluster, no Metaflow service — just a clear step graph you can read and pytest.

## Why

Production orchestrators are excellent — and heavy. For learning and CI you often want:

- A **visible step graph** (`train → validate → register`)
- A **metric gate** that blocks registration on bad scores
- A **stub registry** (model card JSON) instead of a real model store
- Tests that cover the **happy path** and **fail-on-bad-metrics**

This repo is that slice.

## What’s inside

| Piece | Role |
|---|---|
| `src/ml_pipeline_metaflow_demo/dag.py` | Thin DAG: `add_step` / `connect` / `run` (topo order) |
| `src/ml_pipeline_metaflow_demo/steps.py` | `train` → `validate` → `register` (+ immutable run folder / lineage) |
| `src/ml_pipeline_metaflow_demo/pipeline.py` | Wires the three-step ML DAG |
| `src/ml_pipeline_metaflow_demo/dataset.py` | Tiny Iris binary split (offline, deterministic) |
| `examples/quickstart.py` | End-to-end run + prints model card |
| `tests/` | Happy path + fail-on-bad-metrics + DAG unit tests |
| `ci/github-actions.yml` | GitHub Actions workflow mirror (copy to `.github/workflows/ci.yml` to enable) |

## Pipeline shape

```text
   ┌───────┐     ┌──────────┐     ┌──────────┐
   │ train │ ──▶ │ validate │ ──▶ │ register │
   └───────┘     └──────────┘     └──────────┘
        │              │                 │
   fit logreg    accuracy ≥ τ      model.joblib
   on Iris       else raise        + registry.json
```

## Quickstart

```bash
git clone https://github.com/vk-ai/ml-pipeline-metaflow-demo.git
cd ml-pipeline-metaflow-demo
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python examples/quickstart.py
```

Example output:

```text
ml-train-validate-register: train → validate → register

history: ['train', 'validate', 'register']
metrics: { "accuracy": 1.0, "f1": 1.0, "threshold": 0.85 }
```

## Design notes

- **Thin custom DAG, not Airflow/Metaflow runtime.** Airflow needs a scheduler/DB; Metaflow is great but heavier to install cleanly for a learning repo. The runner here is ~100 lines with the same mental model (named steps + edges + shared context).
- **Validate is a hard gate.** If accuracy &lt; threshold (or tests force bad metrics), `ValidationError` stops the DAG before register writes artifacts.
- **Register is a stub with run lineage.** Writes `artifacts/model.joblib`, `artifacts/registry.json` (model card with `run_id` + `lineage.steps`), and an **immutable** `artifacts/runs/<run_id>/` folder (`context.json`, model copy, registry copy). Teaching stand-in for Metaflow/MLflow lineage — **not** Metaflow Client API, **not** MLflow Model Registry, **not** Airflow.
- **Deterministic & offline.** Iris from sklearn; no network, no cloud credentials.

## Tests

```bash
pytest -q
```

- **Happy path** — full DAG runs; registry JSON + model artifact exist; accuracy ≥ threshold.
- **Run lineage** — `artifacts/runs/<run_id>/context.json` + registry `run_id` / `lineage.steps`.
- **Fail on bad metrics** — forced inverted predictions (or impossible threshold) raise `ValidationError` and leave no registry files.
- **DAG unit tests** — topo order, cycle detection, graph text.

## License

MIT © vk-ai
