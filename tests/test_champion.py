"""Round 4: champion/challenger relative gate + rollback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_pipeline_metaflow_demo.champion import (
    compare_to_champion,
    current_champion,
    frozen_eval_split,
    load_champion_gate_file,
    production_stack,
    resolve_champion_gate,
)
from ml_pipeline_metaflow_demo.pipeline import run_pipeline
from ml_pipeline_metaflow_demo.promote import (
    ChampionGateFailed,
    PromotionError,
    promote,
    rollback,
)
from ml_pipeline_metaflow_demo.tags import list_tags, runs_with_tag

ROOT = Path(__file__).resolve().parents[1]
GATES = {"accuracy": 0.90, "f1": 0.85}
# On the frozen split (45 rows): C=0.01 → acc≈0.978 / f1≈0.966; C=1.0 → 1.0 / 1.0.
WEAK_C = 0.01
STRONG_C = 1.0


def _run(tmp: Path, run_id: str, C: float) -> str:
    out = run_pipeline(artifact_dir=str(tmp), gates=GATES, run_id=run_id, C=C)
    return out["run_id"]


def _token(tmp: Path) -> str:
    return (tmp / ".promote_token").read_text().strip()


def test_frozen_split_is_deterministic():
    X1, y1, f1 = frozen_eval_split()
    X2, y2, f2 = frozen_eval_split()
    assert f1 == f2 and len(y1) == 45
    assert (X1 == X2).all() and (y1 == y2).all()
    assert frozen_eval_split(random_state=7)[2] != f1


def test_first_production_has_no_champion(tmp_path: Path):
    _run(tmp_path, "champ01", WEAK_C)
    out = promote(tmp_path, "champ01", to="production", champion_gate={"accuracy": 0.05})
    assert out["previous_champion"] is None
    assert out["champion_compare"]["status"] == "no_champion"
    assert current_champion(tmp_path) == "champ01"
    assert production_stack(tmp_path) == ["champ01"]
    doc = json.loads((tmp_path / "runs" / "champ01" / "champion_compare.json").read_text())
    assert doc["status"] == "no_champion" and doc["passed"] is True


def test_better_challenger_promotes_and_moves_tag(tmp_path: Path):
    _run(tmp_path, "champ01", WEAK_C)
    promote(tmp_path, "champ01", to="production")
    _run(tmp_path, "chall01", STRONG_C)
    out = promote(
        tmp_path,
        "chall01",
        to="production",
        authorize=_token(tmp_path),
        champion_gate={"accuracy": 0.02},
    )
    cmp = out["champion_compare"]
    assert cmp["status"] == "passed" and cmp["champion"] == "champ01"
    acc = next(r for r in cmp["results"] if r["metric"] == "accuracy")
    assert acc["delta"] == pytest.approx(1 / 45)
    # production tag moved (exclusive), stack recorded for rollback
    assert runs_with_tag(tmp_path, "production") == ["chall01"]
    assert production_stack(tmp_path) == ["champ01", "chall01"]
    doc = json.loads((tmp_path / "runs" / "chall01" / "champion_compare.json").read_text())
    assert doc["eval_split"]["n_rows"] == 45 and doc["eval_split"]["fingerprint"]
    assert doc["champion_metrics"]["accuracy"] < doc["challenger_metrics"]["accuracy"]


def test_margin_too_large_blocks_and_names_metric(tmp_path: Path):
    _run(tmp_path, "champ01", WEAK_C)
    promote(tmp_path, "champ01", to="production")
    _run(tmp_path, "chall01", STRONG_C)
    with pytest.raises(ChampionGateFailed, match=r"accuracy .* < required") as ei:
        promote(
            tmp_path,
            "chall01",
            to="production",
            authorize=_token(tmp_path),
            champion_gate={"accuracy": 0.05, "f1": 0.0},
        )
    doc = ei.value.compare
    assert doc["status"] == "blocked" and doc["passed"] is False
    failed = [r["metric"] for r in doc["results"] if not r["passed"]]
    assert failed == ["accuracy"]
    # nothing moved
    assert runs_with_tag(tmp_path, "production") == ["champ01"]
    assert "production" not in list_tags(tmp_path, "chall01")["tags"]
    assert production_stack(tmp_path) == ["champ01"]


def test_equal_challenger_blocked_when_delta_positive(tmp_path: Path):
    _run(tmp_path, "champ01", STRONG_C)
    promote(tmp_path, "champ01", to="production")
    _run(tmp_path, "chall01", STRONG_C)
    tok = _token(tmp_path)
    with pytest.raises(ChampionGateFailed, match="accuracy"):
        promote(tmp_path, "chall01", to="production", authorize=tok, champion_gate=0.01)
    # delta 0 (default) = "no regression" → an equal model may replace the champion
    ok = promote(tmp_path, "chall01", to="production", authorize=tok)
    assert ok["champion_compare"]["status"] == "passed"


def test_worse_challenger_blocked_even_at_zero_delta(tmp_path: Path):
    _run(tmp_path, "champ01", STRONG_C)
    promote(tmp_path, "champ01", to="production")
    _run(tmp_path, "chall01", WEAK_C)  # passes absolute gates, but regresses
    with pytest.raises(ChampionGateFailed, match=r"accuracy.*f1"):
        promote(tmp_path, "chall01", to="production", authorize=_token(tmp_path))
    # staging is unaffected by the champion gate
    st = promote(tmp_path, "chall01", to="staging")
    assert "staging" in st["tags"]


def test_relative_change_rule(tmp_path: Path):
    _run(tmp_path, "champ01", WEAK_C)
    _run(tmp_path, "chall01", STRONG_C)
    # acc 0.978 → 1.0 is +2.3% relative: passes 2%, fails 3%
    ok = compare_to_champion(
        tmp_path, "chall01", champion_id="champ01",
        gate={"accuracy": {"min_relative_change": 0.02}}, write=False,
    )
    bad = compare_to_champion(
        tmp_path, "chall01", champion_id="champ01",
        gate={"accuracy": {"min_relative_change": 0.03}}, write=False,
    )
    assert ok["passed"] is True and bad["passed"] is False


def test_rollback_restores_previous_production(tmp_path: Path):
    _run(tmp_path, "champ01", WEAK_C)
    promote(tmp_path, "champ01", to="production")
    tok = _token(tmp_path)
    _run(tmp_path, "chall01", STRONG_C)
    promote(tmp_path, "chall01", to="production", authorize=tok)
    assert current_champion(tmp_path) == "chall01"

    with pytest.raises(PromotionError, match="authorize"):
        rollback(tmp_path)
    rb = rollback(tmp_path, authorize=tok, reason="drift alert (toy)")
    assert rb == {**rb, "rolled_back": "chall01", "production": "champ01"}
    assert current_champion(tmp_path) == "champ01"
    assert runs_with_tag(tmp_path, "production") == ["champ01"]
    assert "rolled_back" in list_tags(tmp_path, "chall01")["tags"]
    assert production_stack(tmp_path) == ["champ01"]
    history = json.loads((tmp_path / "tags.json").read_text())["history"]
    assert history[-1]["action"] == "rollback" and history[-1]["to"] == "champ01"
    # nothing further back
    with pytest.raises(PromotionError, match="no previous production"):
        rollback(tmp_path, authorize=tok)


def test_champion_gate_from_yaml():
    gate = load_champion_gate_file(ROOT / "examples" / "gates.yaml")
    assert gate["accuracy"]["min_absolute_change"] == pytest.approx(0.01)
    assert resolve_champion_gate(str(ROOT / "examples" / "gates.yaml")) == gate
    assert resolve_champion_gate(None)["f1"] == {
        "min_absolute_change": 0.0,
        "min_relative_change": 0.0,
    }
    with pytest.raises(ValueError, match="unknown champion metric"):
        resolve_champion_gate({"auc": 0.1})
