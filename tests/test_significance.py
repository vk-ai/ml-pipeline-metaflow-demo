"""Round 5: paired significance (McNemar exact + paired bootstrap CI) on the champion gate."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from ml_pipeline_metaflow_demo.champion import compare_to_champion
from ml_pipeline_metaflow_demo.pipeline import run_pipeline
from ml_pipeline_metaflow_demo.promote import ChampionGateFailed, promote
from ml_pipeline_metaflow_demo.significance import (
    mcnemar_exact,
    min_significant_delta,
    paired_bootstrap_ci,
    paired_significance,
    resolve_significance,
)
from ml_pipeline_metaflow_demo.tags import runs_with_tag

ROOT = Path(__file__).resolve().parents[1]
GATES = {"accuracy": 0.90, "f1": 0.85}
WEAK_C, STRONG_C, MAJORITY_C = 0.01, 1.0, 1e-3  # 44/45, 45/45, 30/45 on the frozen split


# --- McNemar ----------------------------------------------------------------------


def test_mcnemar_exact_hand_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(1, 0) == 1.0  # 2 * 0.5
    assert mcnemar_exact(5, 0) == pytest.approx(2 * 0.5**5)  # 0.0625 > 0.05
    assert mcnemar_exact(6, 0) == pytest.approx(2 * 0.5**6)  # 0.03125 < 0.05
    # symmetric, and matches the binomial tail sum
    assert mcnemar_exact(3, 8) == mcnemar_exact(8, 3)
    tail = sum(math.comb(11, i) for i in range(4)) / 2**11
    assert mcnemar_exact(8, 3) == pytest.approx(2 * tail)
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


def test_min_significant_delta():
    assert min_significant_delta(45) == pytest.approx(6 / 45)
    assert min_significant_delta(45, alpha=0.01) == pytest.approx(8 / 45)  # 2*0.5**8 < 0.01
    assert min_significant_delta(3) is None  # even 3/3 fixes gives p = 0.25


# --- bootstrap -----------------------------------------------------------------------


def test_bootstrap_ci_is_seeded_and_brackets_delta():
    a = np.array([1] * 40 + [0] * 5)
    b = np.array([1] * 30 + [0] * 15)
    d1 = paired_bootstrap_ci(a, b, seed=3)
    d2 = paired_bootstrap_ci(a, b, seed=3)
    assert d1 == d2
    delta, lo, hi = d1
    assert delta == pytest.approx(10 / 45)
    assert lo <= delta <= hi
    assert lo > 0


def test_bootstrap_ci_identical_models_is_zero():
    a = np.array([1, 0, 1, 1])
    assert paired_bootstrap_ci(a, a) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        paired_bootstrap_ci(a, a[:2])


# --- verdicts --------------------------------------------------------------------------


def _preds(n: int, fixes: int, breaks: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.ones(n, dtype=int)
    champ = y.copy()
    chall = y.copy()
    champ[:fixes] = 0  # challenger fixes these
    chall[fixes : fixes + breaks] = 0  # challenger breaks these
    return y, chall, champ


def test_one_row_gain_is_underpowered():
    y, ch, cp = _preds(45, fixes=1, breaks=0)
    sig = paired_significance(y, ch, cp)
    assert sig["delta"] == pytest.approx(1 / 45)
    assert sig["challenger_only_correct"] == 1 and sig["champion_only_correct"] == 0
    assert sig["mcnemar_p"] == 1.0
    assert sig["verdict"] == "underpowered" and sig["significant"] is False
    assert sig["min_significant_delta"] == pytest.approx(6 / 45)


def test_large_gain_is_significant_and_large_loss_is_regression():
    y, ch, cp = _preds(45, fixes=10, breaks=1)
    assert paired_significance(y, ch, cp)["verdict"] == "significant_improvement"
    assert paired_significance(y, cp, ch)["verdict"] == "significant_regression"


def test_rules_can_disagree_at_small_counts():
    """4 fixes / 0 breaks: bootstrap CI excludes 0 but McNemar p = 0.125."""
    y, ch, cp = _preds(45, fixes=4, breaks=0)
    assert paired_significance(y, ch, cp, {"rule": "bootstrap"})["verdict"] == "significant_improvement"
    assert paired_significance(y, ch, cp, {"rule": "mcnemar"})["verdict"] == "underpowered"
    assert paired_significance(y, ch, cp, {"rule": "both"})["verdict"] == "underpowered"
    assert paired_significance(y, ch, cp, {"rule": "either"})["verdict"] == "significant_improvement"


def test_resolve_significance_specs(tmp_path: Path):
    assert resolve_significance(None)["require"] is False
    assert resolve_significance(True)["require"] is True
    assert resolve_significance({"alpha": 0.1})["alpha"] == 0.1
    cfg = resolve_significance(ROOT / "examples" / "gates.yaml")
    assert cfg == {"alpha": 0.05, "require": False, "rule": "both", "n_boot": 2000, "seed": 0}
    no_section = tmp_path / "g.json"
    no_section.write_text(json.dumps({"champion": {"accuracy": 0.0}}))
    assert resolve_significance(no_section)["require"] is False
    for bad in ({"alpha": 1.5}, {"rule": "vibes"}, {"n_boot": 5}, {"nope": 1}):
        with pytest.raises(ValueError):
            resolve_significance(bad)
    with pytest.raises(TypeError):
        resolve_significance(3.0)


# --- gate integration ----------------------------------------------------------------------


def _run(tmp: Path, run_id: str, C: float, gates=GATES) -> None:
    run_pipeline(artifact_dir=str(tmp), gates=gates, run_id=run_id, C=C)


def _token(tmp: Path) -> str:
    return (tmp / ".promote_token").read_text().strip()


def test_compare_always_reports_significance(tmp_path: Path):
    _run(tmp_path, "champ", WEAK_C)
    _run(tmp_path, "chall", STRONG_C)
    doc = compare_to_champion(tmp_path, "chall", champion_id="champ", gate={"accuracy": 0.01})
    assert doc["status"] == "passed"  # report-only by default (round-4 behaviour kept)
    sig = doc["significance"]
    assert sig["verdict"] == "underpowered"
    assert sig["challenger_only_correct"] == 1 and sig["champion_only_correct"] == 0
    assert any("underpowered" in w and "+0.133" in w for w in doc["warnings"])
    saved = json.loads((tmp_path / "runs" / "chall" / "champion_compare.json").read_text())
    assert saved["significance"]["mcnemar_p"] == 1.0


def test_require_significance_blocks_one_row_win(tmp_path: Path):
    _run(tmp_path, "champ", WEAK_C)
    promote(tmp_path, "champ", to="production")
    _run(tmp_path, "chall", STRONG_C)
    with pytest.raises(ChampionGateFailed) as exc:
        promote(
            tmp_path,
            "chall",
            to="production",
            authorize=_token(tmp_path),
            champion_gate={"accuracy": 0.01},
            significance={"require": True},
        )
    assert "not significant" in str(exc.value)
    assert exc.value.compare["status"] == "underpowered"
    assert runs_with_tag(tmp_path, "production") == ["champ"]  # nothing moved


def test_require_significance_passes_a_real_improvement(tmp_path: Path):
    _run(tmp_path, "majority", MAJORITY_C, gates={"accuracy": 0.5})  # 30/45
    promote(tmp_path, "majority", to="production")
    _run(tmp_path, "strong", STRONG_C)  # 45/45 → 15 fixes, 0 breaks
    out = promote(
        tmp_path,
        "strong",
        to="production",
        authorize=_token(tmp_path),
        champion_gate={"accuracy": 0.05},
        significance=True,
    )
    cmp = out["champion_compare"]
    assert cmp["status"] == "passed"
    assert cmp["significance"]["verdict"] == "significant_improvement"
    assert cmp["significance"]["mcnemar_p"] < 1e-3
    assert cmp["significance"]["bootstrap_ci"][0] > 0


def test_gates_yaml_significance_section_is_used(tmp_path: Path):
    gate_file = tmp_path / "gates.yaml"
    gate_file.write_text(
        "champion:\n  accuracy: 0.01\nsignificance:\n  require: true\n  alpha: 0.05\n"
    )
    _run(tmp_path, "champ", WEAK_C)
    _run(tmp_path, "chall", STRONG_C)
    doc = compare_to_champion(tmp_path, "chall", champion_id="champ", gate=gate_file)
    assert doc["status"] == "underpowered" and doc["passed"] is False


def test_margin_failure_still_reports_blocked_not_underpowered(tmp_path: Path):
    _run(tmp_path, "champ", WEAK_C)
    _run(tmp_path, "chall", STRONG_C)
    doc = compare_to_champion(
        tmp_path, "chall", champion_id="champ", gate={"accuracy": 0.05}, significance=True
    )
    assert doc["status"] == "blocked"
