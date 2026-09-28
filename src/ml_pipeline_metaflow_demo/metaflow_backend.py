"""Optional real Metaflow FlowSpec (opt-in via ML_PIPELINE_USE_METAFLOW).

Defines a thin FlowSpec mirroring train → validate → register using the same
step functions as the toy DAG. Not imported on the default offline path.

Learners run::

    pip install -e '.[metaflow]'
    ML_PIPELINE_USE_METAFLOW=true python flows/train_validate_register.py run

``run_pipeline`` stays on the toy DAG for CI reliability — it never shells out
to ``metaflow run``.
"""

from __future__ import annotations

from typing import Any


def build_train_validate_register_flow():
    """Construct the FlowSpec *class* using real Metaflow (ImportError if absent)."""
    from metaflow import FlowSpec, Parameter, step

    from ml_pipeline_metaflow_demo.steps import (
        register_step,
        train_step,
        validate_step,
    )

    class TrainValidateRegisterFlow(FlowSpec):
        """OSS learning FlowSpec — train → validate → register (not employer prod)."""

        artifact_dir = Parameter(
            "artifact_dir",
            help="Directory for model/registry/run artifacts",
            default="artifacts",
        )
        accuracy_threshold = Parameter(
            "accuracy_threshold",
            help="Legacy single accuracy gate (maps to gates.accuracy)",
            default=0.85,
            type=float,
        )

        @step
        def start(self):
            """Entry: seed context and run train_step."""
            ctx: dict[str, Any] = {
                "artifact_dir": str(self.artifact_dir),
                "accuracy_threshold": float(self.accuracy_threshold),
                "_history": [],
            }
            ctx = train_step(ctx)
            ctx["_history"] = list(ctx.get("_history", [])) + ["train"]
            self.ctx = _strip_unpicklable(ctx)
            # Keep model on self for next steps (joblib path after register)
            self.model = ctx["model"]
            self.X_test = ctx["X_test"]
            self.y_test = ctx["y_test"]
            self.next(self.validate)

        @step
        def validate(self):
            """Score multi-metric gates; fail the flow if any miss."""
            ctx = dict(self.ctx)
            ctx["model"] = self.model
            ctx["X_test"] = self.X_test
            ctx["y_test"] = self.y_test
            ctx = validate_step(ctx)
            ctx["_history"] = list(ctx.get("_history", [])) + ["validate"]
            self.ctx = _strip_unpicklable(ctx)
            self.model = ctx["model"]
            self.X_test = ctx["X_test"]
            self.y_test = ctx["y_test"]
            self.next(self.register)

        @step
        def register(self):
            """Write stub registry + immutable run folder + candidate tag."""
            ctx = dict(self.ctx)
            ctx["model"] = self.model
            ctx["X_test"] = self.X_test
            ctx["y_test"] = self.y_test
            ctx = register_step(ctx)
            ctx["_history"] = list(ctx.get("_history", [])) + ["register"]
            self.ctx = _strip_unpicklable(ctx)
            self.run_id = ctx.get("run_id")
            self.registry_path = ctx.get("registry_path")
            self.next(self.end)

        @step
        def end(self):
            """Terminal step — prints honesty label."""
            print("[metaflow] TrainValidateRegisterFlow finished (OSS learning demo)")
            print(f"  run_id={getattr(self, 'run_id', None)}")
            print(f"  registry={getattr(self, 'registry_path', None)}")

    return TrainValidateRegisterFlow


def _strip_unpicklable(ctx: dict[str, Any]) -> dict[str, Any]:
    """Drop numpy/sklearn objects that Metaflow would need to serialize between steps.

    Model/arrays travel as attributes on the flow instance instead.
    """
    skip = {"model", "X_test", "y_test", "X_train", "y_train"}
    return {k: v for k, v in ctx.items() if k not in skip}


def get_flow_class():
    """Return FlowSpec class when Metaflow is importable; raise ImportError otherwise."""
    return build_train_validate_register_flow()


def metaflow_flow_constructible() -> bool:
    """True if the FlowSpec class can be built (package present + class defines)."""
    try:
        cls = build_train_validate_register_flow()
        return cls is not None and getattr(cls, "__name__", "") == "TrainValidateRegisterFlow"
    except ImportError:
        return False
