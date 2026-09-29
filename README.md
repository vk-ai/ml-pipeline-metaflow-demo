# ml-pipeline-metaflow-demo

> **OSS / learning demo only.** This is a personal open-source teaching project by [vk-ai](https://github.com/vk-ai). It is **not** employer production software, is **not** affiliated with any employer, and must not be described as a production ML platform or Metaflow/Airflow deployment.

```text
╔══════════════════════════════════════════════════════════════════════════╗
║  HONESTY BANNER                                                          ║
║  Default backend: [metaflow-style] — a thin *custom* DAG runner          ║
║  (train → validate → register). This is NOT Metaflow Client API, NOT     ║
║  Airflow, NOT MLflow Model Registry, and NOT employer production.        ║
║  Optional real Metaflow / Airflow are opt-in extras only.                ║
╚══════════════════════════════════════════════════════════════════════════╝
```

Tiny **Metaflow / Airflow-style ML pipeline**: a thin custom DAG runner that executes **train → validate → register** on a toy sklearn model. Default CI path needs **no** Metaflow service and **no** Airflow cluster — just a clear step graph you can read and pytest. Optional real Metaflow (`FlowSpec`) and Airflow stub import sit behind extras + an env flag.

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
| `src/ml_pipeline_metaflow_demo/dag.py` | Thin DAG: `add_step` / `connect` / `run` (topo order) — tagged `[metaflow-style]` |
| `src/ml_pipeline_metaflow_demo/backend.py` | Mode / label helpers (`pipeline_mode`, `backend_label`, `fallback_reason`) |
| `src/ml_pipeline_metaflow_demo/metaflow_backend.py` | Optional real Metaflow `FlowSpec` builder — tagged `[metaflow]` |
| `flows/train_validate_register.py` | Learner CLI for `python … run` with real Metaflow |
| `src/ml_pipeline_metaflow_demo/steps.py` | `train` → `validate` → `register` (+ lineage + multi-metric `gates.json`) |
| `src/ml_pipeline_metaflow_demo/pipeline.py` | Wires the three-step ML DAG (`run_pipeline` always toy path) |
| `src/ml_pipeline_metaflow_demo/dataset.py` | Tiny Iris binary split (offline, deterministic) |
| `examples/quickstart.py` | End-to-end run + prints model card |
| `examples/gates.yaml` | Pluggable multi-metric quality gates (`accuracy` / `f1` / …) |
| `src/ml_pipeline_metaflow_demo/tags.py` | Mutable promotion tags (`candidate` / `staging` / `production` / `gate:passed`) on immutable runs |
| `src/ml_pipeline_metaflow_demo/promote.py` | `promote(...)` — only when `gates.json` passed; toy `--authorize` token for production |
| `src/ml_pipeline_metaflow_demo/champion.py` | Champion/challenger relative gate on a frozen eval split → `champion_compare.json` (round 4) |
| `examples/champion_challenger.py` | Promote → blocked challenger → better challenger → `rollback()` walkthrough |
| `src/ml_pipeline_metaflow_demo/airflow_stub.py` | Optional Airflow DAG stub export (import-guarded; Airflow **not** required) |
| `tests/` | Happy path + fail-on-bad-metrics + DAG unit tests + promotion tags + backend selection |
| `.github/workflows/ci.yml` | GitHub Actions CI (`pip install -e ".[dev]"` + pytest; never installs metaflow/airflow) |

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
backend: [metaflow-style]  (mode=metaflow-style)

ml-train-validate-register: train → validate → register

history: ['train', 'validate', 'register']
metrics: { "accuracy": 1.0, "f1": 1.0, "threshold": 0.85 }
```

### Optional real Metaflow (opt-in)

The **default** path is offline and clearly labeled `[metaflow-style]` — a thin custom DAG. CI and `pytest` never require Metaflow.

To attempt a real Metaflow `FlowSpec`:

1. Install the optional extra: `pip install '.[metaflow]'` (pins `metaflow>=2.12`)
2. Set **`ML_PIPELINE_USE_METAFLOW=true`** (default is unset / false)
3. Run the FlowSpec CLI (Metaflow local datastore; first run may prompt / set up locally):

```bash
pip install -e '.[metaflow]'
ML_PIPELINE_USE_METAFLOW=true python flows/train_validate_register.py run
```

`run_pipeline(...)` **always** stays on the toy `[metaflow-style]` DAG for reliability (so CI never shells out to `metaflow run`). When the flag is set and Metaflow is missing, results include a clear `messages` fallback reason. When Metaflow is installed, `messages` points at the FlowSpec CLI above. Selection helpers (`pipeline_mode()`, `backend_label()`, `metaflow_importable()`, `fallback_reason()`) mirror the langgraph-eval-demo pattern.

This optional path is for learning how the same train→validate→register shape looks on a real `FlowSpec`. It is **not** a production Metaflow deployment and makes no claims about any employer's systems.

### Optional Airflow (stub import only)

`export_airflow_stub(path)` writes an **illustrative**, import-guarded DAG skeleton. The happy path never imports Airflow.

- Check availability: `from ml_pipeline_metaflow_demo import airflow_available` → `airflow_available()`
- To import the generated stub with real Airflow (heavy): `pip install '.[airflow]'` (`apache-airflow>=2.8`)
- Convenience combo: `pip install '.[orchestrators]'` (metaflow + airflow)
- **Do not** install Airflow in default CI — it is large and unused by pytest

## Promotion tags (round 3)

Run folders under `artifacts/runs/<run_id>/` stay **immutable**. Mutable labels live in `artifacts/tags.json` (Metaflow-style: tags organize interpretation of immutable facts — they are **not** namespace isolation).

```python
from ml_pipeline_metaflow_demo import run_pipeline, promote, add_tag, list_tags, export_airflow_stub

out = run_pipeline(artifact_dir="artifacts", gates={"accuracy": 0.85})
# out["backend_label"] == "[metaflow-style]"
# register seeds tag "candidate"
promote("artifacts", out["run_id"], to="staging")          # requires gates passed
token = open("artifacts/.promote_token").read().strip()    # minted on first prod promote
promote("artifacts", out["run_id"], to="production")       # first prod mints token
# later: promote(..., to="production", authorize=token)
export_airflow_stub("artifacts/airflow_dag_stub.py")       # illustrative only
```

- **Sources:** [Metaflow tagging](https://docs.metaflow.org/scaling/tagging), [Metaflow + Airflow](https://docs.metaflow.org/production/scheduling-metaflow-flows/scheduling-with-airflow), [coordinating projects](https://docs.metaflow.org/production/coordinating-larger-metaflow-projects).
- Airflow stub is **optional and illustrative** — the happy path never imports Airflow.

## Champion vs challenger gate + rollback (round 4)

Absolute gates answer *"is this run good enough?"*. Before a run replaces what is live, `promote(..., to="production")` now also asks *"does it beat the current `production` run?"*:

1. The **champion** is the run currently tagged `production` (top of `production_stack` in `tags.json`).
2. The challenger's and champion's `runs/<id>/model.joblib` are both **re-scored on the same frozen eval split** (`frozen_eval_split()`, Iris `random_state=42`, 45 rows, sha256 fingerprint recorded). Stored metrics are not trusted.
3. Per metric: `challenger >= champion + max(min_absolute_change, min_relative_change * |champion|)`. The default is accuracy + f1 with delta `0.0` (no regression). Set a delta > 0 to require a real improvement.
4. The result goes to `runs/<challenger>/champion_compare.json` (`status`: `passed` / `blocked` / `no_champion` / `self`). A block raises `ChampionGateFailed` and names each failing metric. No tags move.
5. On success, `production` **moves** to the challenger (`exclusive_production=True`). The previous champion is kept in `production_stack`, so `rollback()` can restore it.

```python
from ml_pipeline_metaflow_demo import promote, rollback, run_pipeline, ChampionGateFailed

run_pipeline(artifact_dir="artifacts", run_id="v1", C=0.01)       # weaker model (acc≈0.978)
promote("artifacts", "v1", to="production")                      # no champion → absolute gates only
token = open("artifacts/.promote_token").read().strip()

run_pipeline(artifact_dir="artifacts", run_id="v2", C=1.0)        # acc 1.0 on the frozen split
promote("artifacts", "v2", to="production", authorize=token,
        champion_gate={"accuracy": 0.01})                         # or champion_gate="examples/gates.yaml"
rollback("artifacts", authorize=token, reason="toy incident")    # production → v1, v2 tagged rolled_back
```

Try it: `python examples/champion_challenger.py` (uses a temp dir). `examples/gates.yaml` has a `champion:` section (`accuracy: 0.01`, `f1: 0.0`). `compare_champion=False` skips the check, and the override is recorded in the tag history.

**Caveats (read these):** there is **no statistical significance test**. On 45 eval rows one flipped prediction moves accuracy by about 0.022, so deltas below `1/n` are flagged as noise-sensitive in `warnings`. Runs trained with a different `random_state` may have seen frozen-eval rows during training, and that is flagged as well. Rollback does not re-run gates: it restores a run that already passed them. This is a teaching stand-in for MLflow [`validate_evaluation_results` / `MetricThreshold(min_absolute_change=…)`](https://mlflow.org/docs/latest/ml/evaluation/) plus registry aliases. It is **not** MLflow and not shadow/canary deployment. Related OSS demos: [model-promotion-gate](https://github.com/tkgo1599-max/model-promotion-gate) and [modelgate](https://github.com/AyushPatel94/modelgate).

## Design notes

- **Thin custom DAG by default (`[metaflow-style]`).** Airflow needs a scheduler/DB; Metaflow is great but heavier for a learning repo CI matrix. The runner here is ~100 lines with the same mental model (named steps + edges + shared context). Real Metaflow is an **optional** extra behind `ML_PIPELINE_USE_METAFLOW`.
- **Validate is a hard multi-metric gate.** Pass `gates={accuracy: 0.90, f1: 0.85}` or `gates_path=examples/gates.yaml`. Any miss raises `GateFailed` (subclass of `ValidationError`) and skips register. Legacy `accuracy_threshold` still maps to a single accuracy gate. Teaching stand-in for MLflow `MetricThreshold` / `validate_evaluation_results` — **not** MLflow.
- **Register is a stub with run lineage + gates.json.** Writes `artifacts/model.joblib`, `artifacts/registry.json` (model card with `run_id`, `lineage.steps`, **`promotion: "candidate"`**), and an **immutable** `artifacts/runs/<run_id>/` folder (`context.json`, `gates.json`, model copy, registry copy). Teaching stand-in for Metaflow/MLflow lineage — **not** Metaflow Client API, **not** MLflow Model Registry, **not** Airflow.
- **Promotion tags + optional Airflow stub.** Mutable `candidate`/`staging`/`production`/`gate:passed` tags beside immutable lineage; `promote` refuses when gates fail. Airflow DAG file export is import-guarded and never required.
- **Deterministic & offline.** Iris from sklearn; no network, no cloud credentials on the default path.

## Tests

```bash
pytest -q
```

- **Happy path** — full DAG runs; registry JSON + model artifact exist; accuracy ≥ threshold; `backend_label == "[metaflow-style]"`.
- **Multi-metric gates** — pass-all writes `gates.json` + `promotion: candidate`; fail-one raises `GateFailed` with no artifacts.
- **Run lineage** — `artifacts/runs/<run_id>/context.json` + `gates.json` + registry `run_id` / `lineage.steps`.
- **Fail on bad metrics** — forced inverted predictions (or impossible threshold) raise `ValidationError`/`GateFailed` and leave no registry files.
- **DAG unit tests** — topo order, cycle detection, graph text.
- **Promotion tags** — candidate seed, staging→production, authorize token, refuse on failed gates, Airflow stub writes without Airflow installed.
- **Champion/challenger + rollback** — first production has no champion; a better challenger promotes and moves the tag; a too-large delta, an equal model with delta > 0, and a worse model are all blocked with the failing metric named; relative-change rule; rollback restores the previous run (authorize required); `champion:` YAML section.
- **Backend selection** — default style; flag false; flag true + missing metaflow → fallback; selection when importable stubbed True (no metaflow/airflow install in CI).

## License

MIT © vk-ai
