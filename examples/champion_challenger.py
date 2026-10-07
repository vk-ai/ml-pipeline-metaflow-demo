#!/usr/bin/env python3
"""Round 4/5 demo: champion/challenger gate + paired significance + rollback (temp dir, offline)."""

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

        # 3) a "better" challenger (+1 row of 45) clears the margin...
        run_pipeline(artifact_dir=str(art), gates=GATES, run_id="better", C=1.0)
        # ...but with significance required it is blocked as underpowered (round 5)
        try:
            promote(
                art, "better", to="production", authorize=token,
                champion_gate=gate_file, significance={"require": True},
            )
        except ChampionGateFailed as exc:
            print("blocked (significance required):", exc)

        # 3b) report-only (gates.yaml default): promoted, verdict recorded
        out = promote(art, "better", to="production", authorize=token, champion_gate=gate_file)
        print("promoted:", json.dumps(out["champion_compare"]["results"], indent=2))
        sig = out["champion_compare"]["significance"]
        print(
            "significance:",
            json.dumps(
                {k: sig[k] for k in ("delta", "challenger_only_correct", "champion_only_correct",
                                     "mcnemar_p", "bootstrap_ci", "min_significant_delta", "verdict")},
                indent=2,
            ),
        )
        print("champion now:", current_champion(art))

        # 4) rollback restores the previous production run
        rb = rollback(art, authorize=token, reason="toy incident")
        print("rollback:", rb["rolled_back"], "→", rb["production"])
        print("champion now:", current_champion(art))


if __name__ == "__main__":
    main()
