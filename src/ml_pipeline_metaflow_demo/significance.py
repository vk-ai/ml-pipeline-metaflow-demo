"""Paired significance for the champion/challenger gate (stdlib + numpy only).

Champion and challenger are scored on the **same** frozen eval rows, so the
right question is paired: on how many rows does one model fix what the other
gets wrong?

- ``b`` = rows the challenger gets right and the champion gets wrong
- ``c`` = rows the champion gets right and the challenger gets wrong
- accuracy delta = ``(b - c) / n``. Rows both models agree on carry no information.

Two complementary checks:

1. **McNemar's exact test**: under H0 (no difference) the ``b + c`` discordant
   rows split 50/50, so ``p = min(1, 2 * P[Binom(b+c, 0.5) <= min(b, c)])``.
   It uses ``math.comb`` only.
2. **Paired bootstrap CI** on the accuracy delta: resample row indices with
   replacement (seeded numpy RNG) and take percentile bounds of the mean
   per-row difference.

``min_significant_delta(n, alpha)`` is the smallest accuracy gain that *could*
reach ``p < alpha`` on ``n`` rows (best case: the challenger fixes ``k`` rows and
breaks none). On 45 rows at alpha = 0.05 that is 6 rows, i.e. +0.133 accuracy.
So a "+0.022" promotion (one row) can never be significant on this split.

The two checks can disagree at tiny counts (the percentile bootstrap is
anti-conservative), so the default rule requires **both**. Teaching code only,
not a stats library. See ``paired-evalkit`` / ``evalkit`` for fuller versions.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

RULES = ("both", "either", "mcnemar", "bootstrap")
DEFAULT_SIGNIFICANCE: dict[str, Any] = {
    "alpha": 0.05,
    "require": False,  # report-only by default; True turns it into a promotion gate
    "rule": "both",
    "n_boot": 2000,
    "seed": 0,
}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts ``b`` and ``c``."""
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be >= 0")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def paired_bootstrap_ci(
    correct_challenger: np.ndarray,
    correct_champion: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """``(delta, lo, hi)``: accuracy delta and percentile CI from a paired bootstrap."""
    a = np.asarray(correct_challenger, dtype=float)
    b = np.asarray(correct_champion, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("need two non-empty 1-D correctness vectors of equal length")
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(int(n_boot), diff.size))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(diff.mean()), float(lo), float(hi)


def min_significant_delta(n: int, alpha: float = 0.05) -> float | None:
    """Smallest accuracy delta that can reach ``p < alpha`` on ``n`` paired rows.

    Best case for the challenger: ``k`` fixes and zero regressions, so
    ``p = 2 * 0.5**k``. Returns ``k_min / n``, or ``None`` if even ``k = n`` is
    not enough.
    """
    for k in range(1, n + 1):
        if mcnemar_exact(k, 0) < alpha:
            return k / n
    return None


def resolve_significance(spec: Any = None) -> dict[str, Any]:
    """Normalize a significance spec.

    - ``None`` → report-only defaults (``require: False``)
    - ``True`` / ``False`` → defaults with ``require`` set
    - a mapping → merged over defaults (keys: alpha, require, rule, n_boot, seed)
    - a path to a gates YAML/JSON → its ``significance:`` section (defaults if absent)
    """
    cfg = dict(DEFAULT_SIGNIFICANCE)
    if spec is None:
        pass
    elif isinstance(spec, bool):
        cfg["require"] = spec
    elif isinstance(spec, (str, Path)):
        return resolve_significance(_load_section(spec))
    elif isinstance(spec, Mapping):
        unknown = set(spec) - set(DEFAULT_SIGNIFICANCE)
        if unknown:
            raise ValueError(f"unknown significance keys: {sorted(unknown)}")
        cfg.update(spec)
    else:
        raise TypeError(f"unsupported significance spec: {spec!r}")
    cfg["alpha"] = float(cfg["alpha"])
    cfg["require"] = bool(cfg["require"])
    cfg["n_boot"] = int(cfg["n_boot"])
    cfg["seed"] = int(cfg["seed"])
    if not 0.0 < cfg["alpha"] < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if cfg["rule"] not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    if cfg["n_boot"] < 100:
        raise ValueError("n_boot must be >= 100")
    return cfg


def _load_section(path: str | Path) -> dict[str, Any] | None:
    import json

    from ml_pipeline_metaflow_demo.steps import _parse_simple_gates_yaml

    p = Path(path)
    text = p.read_text(encoding="utf-8")
    data = _parse_simple_gates_yaml(text) if p.suffix.lower() in {".yaml", ".yml"} else json.loads(text)
    section = data.get("significance")
    return dict(section) if isinstance(section, Mapping) else None


def paired_significance(
    y_true: np.ndarray,
    pred_challenger: np.ndarray,
    pred_champion: np.ndarray,
    spec: Any = None,
) -> dict[str, Any]:
    """Paired accuracy comparison → JSON-friendly doc with a verdict.

    ``verdict`` is one of:
    - ``significant_improvement``: delta > 0 and the configured ``rule`` holds
    - ``significant_regression``:  delta < 0 and the mirrored rule holds
    - ``underpowered``: anything else (difference too small for n to tell)
    """
    cfg = resolve_significance(spec)
    y = np.asarray(y_true)
    ok_ch = np.asarray(pred_challenger) == y
    ok_cp = np.asarray(pred_champion) == y
    n = int(y.size)
    b = int(np.sum(ok_ch & ~ok_cp))
    c = int(np.sum(~ok_ch & ok_cp))
    p = mcnemar_exact(b, c)
    delta, lo, hi = paired_bootstrap_ci(
        ok_ch, ok_cp, n_boot=cfg["n_boot"], alpha=cfg["alpha"], seed=cfg["seed"]
    )
    alpha = cfg["alpha"]
    mc = p < alpha
    up = lo > 0
    down = hi < 0
    rule = cfg["rule"]
    combine = {
        "both": lambda m, bs: m and bs,
        "either": lambda m, bs: m or bs,
        "mcnemar": lambda m, bs: m,
        "bootstrap": lambda m, bs: bs,
    }[rule]
    if delta > 0 and combine(mc, up):
        verdict = "significant_improvement"
    elif delta < 0 and combine(mc, down):
        verdict = "significant_regression"
    else:
        verdict = "underpowered"
    mde = min_significant_delta(n, alpha)
    return {
        "metric": "accuracy",
        "n": n,
        "challenger_only_correct": b,
        "champion_only_correct": c,
        "delta": delta,
        "mcnemar_p": p,
        "bootstrap_ci": [lo, hi],
        "alpha": alpha,
        "rule": rule,
        "n_boot": cfg["n_boot"],
        "seed": cfg["seed"],
        "min_significant_delta": mde,
        "verdict": verdict,
        "significant": verdict == "significant_improvement",
        "required": cfg["require"],
    }
