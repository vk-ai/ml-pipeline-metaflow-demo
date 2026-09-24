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
| `src/ml_pipeline_metaflow_demo/steps.py` | `train` → `validate` → `register` (+ lineage + multi-metric `gates.json`) |
| `src/ml_pipeline_metaflow_demo/pipeline.py` | Wires the three-step ML DAG |
| `src/ml_pipeline_metaflow_demo/dataset.py` | Tiny Iris binary split (offline, deterministic) |
| `examples/quickstart.py` | End-to-end run + prints model card |
| `examples/gates.yaml` | Pluggable multi-metric quality gates (`accuracy` / `f1` / …) |
| `src/ml_pipeline_metaflow_demo/tags.py` | Mutable promotion tags (`candidate` / `staging` / `production` / `gate:passed`) on immutable runs |
| `src/ml_pipeline_metaflow_demo/promote.py` | `promote(...)` — only when `gates.json` passed; toy `--authorize` token for production |
| `src/ml_pipeline_metaflow_demo/airflow_stub.py` | Optional Airflow DAG stub export (import-guarded; Airflow **not** required) |
| `tests/` | Happy path + fail-on-bad-metrics + DAG unit tests + promotion tags |
| `ci/github-actions.yml` | GitHub Actions workflow mirror (copy to `.github/workflows/ci.yml` to enable) |

## Pipeline shape

```text
   ┌───────┐     ┌──────────┐     ┌──────────┐
   │ train │ ──▶ │ validate │ ──▶ │ register │
   └───────┘     └──────────┘     └──────────┘
        │              │                 │
   fit logreg    multi-metric      model.joblib
   on Iris       gates (YAML)      + registry.json
                 else GateFailed   + runs/<id>/gates.json
                                   promotion: candidate
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


## Promotion tags (round 3)

Run folders under `artifacts/runs/<run_id>/` stay **immutable**. Mutable labels live in `artifacts/tags.json` (Metaflow-style: tags organize interpretation of immutable facts — they are **not** namespace isolation).

```python
from ml_pipeline_metaflow_demo import run_pipeline, promote, add_tag, list_tags, export_airflow_stub

out = run_pipeline(artifact_dir="artifacts", gates={"accuracy": 0.85})
# register seeds tag "candidate"
promote("artifacts", out["run_id"], to="staging")          # requires gates passed
token = open("artifacts/.promote_token").read().strip()    # minted on first prod promote
promote("artifacts", out["run_id"], to="production")       # first prod mints token
# later: promote(..., to="production", authorize=token)
export_airflow_stub("artifacts/airflow_dag_stub.py")       # illustrative only
```

- **Sources:** [Metaflow tagging](https://docs.metaflow.org/scaling/tagging), [Metaflow + Airflow](https://docs.metaflow.org/production/scheduling-metaflow-flows/scheduling-with-airflow), [coordinating projects](https://docs.metaflow.org/production/coordinating-larger-metaflow-projects).
- Airflow stub is **optional and illustrative** — the happy path never imports Airflow.

## Design notes

- **Thin custom DAG, not Airflow/Metaflow runtime.** Airflow needs a scheduler/DB; Metaflow is great but heavier to install cleanly for a learning repo. The runner here is ~100 lines with the same mental model (named steps + edges + shared context).
- **Validate is a hard multi-metric gate.** Pass `gates={accuracy: 0.90, f1: 0.85}` or `gates_path=examples/gates.yaml`. Any miss raises `GateFailed` (subclass of `ValidationError`) and skips register. Legacy `accuracy_threshold` still maps to a single accuracy gate. Teaching stand-in for MLflow `MetricThreshold` / `validate_evaluation_results` — **not** MLflow.
- **Register is a stub with run lineage + gates.json.** Writes `artifacts/model.joblib`, `artifacts/registry.json` (model card with `run_id`, `lineage.steps`, **`promotion: "candidate"`**), and an **immutable** `artifacts/runs/<run_id>/` folder (`context.json`, `gates.json`, model copy, registry copy). Teaching stand-in for Metaflow/MLflow lineage — **not** Metaflow Client API, **not** MLflow Model Registry, **not** Airflow.
- **Promotion tags + optional Airflow stub.** Mutable `candidate`/`staging`/`production`/`gate:passed` tags beside immutable lineage; `promote` refuses when gates fail. Airflow DAG file export is import-guarded and never required.
- **Deterministic & offline.** Iris from sklearn; no network, no cloud credentials.

## Tests

```bash
pytest -q
```

- **Happy path** — full DAG runs; registry JSON + model artifact exist; accuracy ≥ threshold.
- **Multi-metric gates** — pass-all writes `gates.json` + `promotion: candidate`; fail-one raises `GateFailed` with no artifacts.
- **Run lineage** — `artifacts/runs/<run_id>/context.json` + `gates.json` + registry `run_id` / `lineage.steps`.
- **Fail on bad metrics** — forced inverted predictions (or impossible threshold) raise `ValidationError`/`GateFailed` and leave no registry files.
- **DAG unit tests** — topo order, cycle detection, graph text.
- **Promotion tags** — candidate seed, staging→production, authorize token, refuse on failed gates, Airflow stub writes without Airflow installed.

## License

MIT © vk-ai
