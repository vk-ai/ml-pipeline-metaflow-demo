"""OSS learning demo: thin Metaflow/Airflow-style ML DAG (train → validate → register).

Default backend is ``[metaflow-style]`` (custom DAG). Optional real Metaflow behind
``ML_PIPELINE_USE_METAFLOW`` + ``pip install '.[metaflow]'``. Airflow is optional
via ``pip install '.[airflow]'`` for importing the generated stub only.
"""

__version__ = "0.1.0"

from ml_pipeline_metaflow_demo.airflow_stub import airflow_available, export_airflow_stub
from ml_pipeline_metaflow_demo.backend import (
    backend_label,
    fallback_reason,
    metaflow_importable,
    pipeline_mode,
    wants_real_metaflow,
)
from ml_pipeline_metaflow_demo.significance import mcnemar_exact, paired_significance
from ml_pipeline_metaflow_demo.champion import (
    compare_to_champion,
    current_champion,
    frozen_eval_split,
    production_stack,
)
from ml_pipeline_metaflow_demo.pipeline import build_pipeline, run_pipeline
from ml_pipeline_metaflow_demo.promote import ChampionGateFailed, promote, rollback
from ml_pipeline_metaflow_demo.tags import add_tag, list_tags, remove_tag, runs_with_tag

__all__ = [
    "build_pipeline",
    "run_pipeline",
    "pipeline_mode",
    "backend_label",
    "fallback_reason",
    "metaflow_importable",
    "wants_real_metaflow",
    "add_tag",
    "remove_tag",
    "list_tags",
    "runs_with_tag",
    "promote",
    "rollback",
    "ChampionGateFailed",
    "compare_to_champion",
    "mcnemar_exact",
    "paired_significance",
    "current_champion",
    "frozen_eval_split",
    "production_stack",
    "export_airflow_stub",
    "airflow_available",
    "__version__",
]
