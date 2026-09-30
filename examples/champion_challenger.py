#!/usr/bin/env python3
"""Round 4 demo: champion/challenger relative gate + rollback (temp dir, offline)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ml_pipeline_metaflow_demo import (
    ChampionGateFailed,
    current_champion,
    promote,
    rollback,
    run_pipeline,
)

GATES = {"accuracy": 0.90, "f1": 0.85}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    gate_file = root / "examples" / "gates.yaml"  # champion: accuracy +0.01
    with tempfile.TemporaryDirectory() as tmp:
        art = Path(tmp)
        # 1) weaker champion (heavy regularization) → first production, no compare
        run_pipeline(artifact_dir=str(art), gates=GATES, run_id="champion", C=0.01)
        first = promote(art, "champion", to="production")
        token = (art / ".promote_token").read_text().strip()
        print("first production:", first["champion_compare"]["status"])

        # 2) a challenger that is *not* better is blocked
        run_pipeline(artifact_dir=str(art), gates=GATES, run_id="same", C=0.01)
        try:
            promote(art, "same", to="production", authorize=token, champion_gate=gate_file)
        except ChampionGateFailed as exc:
            print("blocked:", exc)

        # 3) a better challenger beats the champion by >= delta → promoted
        run_pipeline(artifact_dir=str(art), gates=GATES, run_id="better", C=1.0)
        out = promote(art, "better", to="production", authorize=token, champion_gate=gate_file)
        print("promoted:", json.dumps(out["champion_compare"]["results"], indent=2))
        print("champion now:", current_champion(art))

        # 4) rollback restores the previous production run
        rb = rollback(art, authorize=token, reason="toy incident")
        print("rollback:", rb["rolled_back"], "→", rb["production"])
        print("champion now:", current_champion(art))


if __name__ == "__main__":
    main()
