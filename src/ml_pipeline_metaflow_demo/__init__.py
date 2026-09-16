"""OSS learning demo: thin Metaflow/Airflow-style ML DAG (train → validate → register)."""

__version__ = "0.1.0"

from ml_pipeline_metaflow_demo.pipeline import build_pipeline, run_pipeline

__all__ = ["build_pipeline", "run_pipeline", "__version__"]
