"""OSS learning demo: thin Metaflow/Airflow-style ML DAG (train → validate → register)."""

__version__ = "0.1.0"

from ml_pipeline_metaflow_demo.airflow_stub import export_airflow_stub
from ml_pipeline_metaflow_demo.pipeline import build_pipeline, run_pipeline
from ml_pipeline_metaflow_demo.promote import promote
from ml_pipeline_metaflow_demo.tags import add_tag, list_tags, remove_tag, runs_with_tag

__all__ = [
    "build_pipeline",
    "run_pipeline",
    "add_tag",
    "remove_tag",
    "list_tags",
    "runs_with_tag",
    "promote",
    "export_airflow_stub",
    "__version__",
]
