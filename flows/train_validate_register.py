#!/usr/bin/env python3
"""Learner entrypoint: real Metaflow FlowSpec for train → validate → register.

Honesty: this is an OSS/learning FlowSpec. It is NOT employer production, NOT a
hosted Metaflow deployment, and NOT required for the default toy DAG path.

Usage::

    pip install -e '.[metaflow]'
    ML_PIPELINE_USE_METAFLOW=true python flows/train_validate_register.py run

Optional::

    python flows/train_validate_register.py run --artifact_dir /tmp/mf-art \\
        --accuracy_threshold 0.85
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from a clone without editable install (src layout).
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main() -> None:
    try:
        from ml_pipeline_metaflow_demo.metaflow_backend import get_flow_class
    except ImportError as exc:  # pragma: no cover - CLI path
        print(
            "metaflow is required for this entrypoint.\n"
            "  pip install -e '.[metaflow]'\n"
            "  ML_PIPELINE_USE_METAFLOW=true python flows/train_validate_register.py run\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    Flow = get_flow_class()
    # Metaflow CLI: instantiating FlowSpec under __main__ parses argv (run, …).
    Flow()


if __name__ == "__main__":
    main()
