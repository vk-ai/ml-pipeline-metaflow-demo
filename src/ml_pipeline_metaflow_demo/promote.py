"""Promote a run only when quality gates pass; record lineage in the tag store.

Teaching stand-in for Metaflow staging→production promotion and production
tokens (``--authorize``). See:
https://docs.metaflow.org/scaling/tagging
https://docs.metaflow.org/production/coordinating-larger-metaflow-projects
"""

from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Any

from ml_pipeline_metaflow_demo.champion import (
    DEFAULT_EVAL_RANDOM_STATE,
    compare_to_champion,
    current_champion,
    format_failures,
)
from ml_pipeline_metaflow_demo.tags import (
    add_tag,
    list_tags,
    load_tags,
    remove_tag,
    runs_with_tag,
    save_tags,
)


class PromotionError(RuntimeError):
    """Raised when promotion is refused (gates / authorize / missing run)."""


class ChampionGateFailed(PromotionError):
    """Raised when a challenger does not beat the current production champion."""

    def __init__(self, message: str, compare: dict[str, Any]):
        super().__init__(message)
        self.compare = compare


def _run_dir(artifact_dir: str | Path, run_id: str) -> Path:
    return Path(artifact_dir) / "runs" / run_id


def _load_gates(artifact_dir: str | Path, run_id: str) -> dict[str, Any]:
    path = _run_dir(artifact_dir, run_id) / "gates.json"
    if not path.is_file():
        raise PromotionError(f"no gates.json for run {run_id!r}; refuse promote")
    return json.loads(path.read_text(encoding="utf-8"))


def auth_token_path(artifact_dir: str | Path) -> Path:
    return Path(artifact_dir) / ".promote_token"


def mint_or_load_token(artifact_dir: str | Path) -> str:
    """First production promote mints a local authorize token (toy)."""
    path = auth_token_path(artifact_dir)
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    token = secrets.token_hex(16)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token + "\n", encoding="utf-8")
    return token


def promote(
    artifact_dir: str | Path,
    run_id: str,
    *,
    to: str = "production",
    authorize: str | None = None,
    strip_staging: bool = True,
    require_gates: bool = True,
    champion_gate: Any = None,
    compare_champion: bool = True,
    eval_random_state: int = DEFAULT_EVAL_RANDOM_STATE,
    exclusive_production: bool = True,
) -> dict[str, Any]:
    """
    Promote ``run_id`` to ``to`` (default ``production``) iff gates passed.

    - Always requires ``gates.json`` with ``passed: true`` when ``require_gates``.
    - Adds tag ``gate:passed`` when gates are green.
    - First ``production`` promote mints ``.promote_token``; later ones need
      ``authorize=<token>`` (or env-style pass-through via caller).
    - Optionally strips ``staging`` when promoting to production.
    - Updates registry.json ``promotion`` field when present (mutable card only).
    - **Champion/challenger (round 4):** when promoting to ``production`` and a
      *different* run currently holds ``production``, both models are re-scored
      on the same frozen eval split and the challenger must beat the champion by
      ``champion_gate`` (default: no regression on accuracy + f1). Otherwise
      :class:`ChampionGateFailed` is raised naming the failing metric(s). The
      comparison is written to ``runs/<run_id>/champion_compare.json``.
      ``compare_champion=False`` skips it (explicit override, recorded in history).
    - With ``exclusive_production`` (default) the previous champion loses the
      ``production`` tag and is pushed onto ``production_stack`` for
      :func:`rollback`.
    """
    to = str(to).strip()
    if not to:
        raise ValueError("promotion target tag must be non-empty")

    rd = _run_dir(artifact_dir, run_id)
    if not rd.is_dir():
        raise PromotionError(f"unknown run_id {run_id!r} under {artifact_dir}")

    gate_doc: dict[str, Any] | None = None
    if require_gates:
        gate_doc = _load_gates(artifact_dir, run_id)
        if not gate_doc.get("passed"):
            raise PromotionError(
                f"gates did not pass for run {run_id!r}; refuse promote to {to!r}"
            )
        add_tag(artifact_dir, run_id, "gate:passed", actor="promote")

    if to == "production":
        token_path = auth_token_path(artifact_dir)
        if token_path.is_file():
            expected = token_path.read_text(encoding="utf-8").strip()
            if not authorize or authorize != expected:
                raise PromotionError(
                    "production promote requires --authorize <token> "
                    f"(token file: {token_path})"
                )
        else:
            minted = mint_or_load_token(artifact_dir)
            # First-time mint: authorize may be omitted; return token to caller
            authorize = authorize or minted

    compare_doc: dict[str, Any] | None = None
    previous_champion: str | None = None
    if to == "production":
        previous_champion = current_champion(artifact_dir)
        if compare_champion:
            try:
                compare_doc = compare_to_champion(
                    artifact_dir,
                    run_id,
                    champion_id=previous_champion,
                    gate=champion_gate,
                    eval_random_state=eval_random_state,
                )
            except FileNotFoundError as exc:
                raise PromotionError(f"champion compare impossible: {exc}") from exc
            if not compare_doc["passed"]:
                raise ChampionGateFailed(
                    f"challenger {run_id!r} does not beat champion "
                    f"{previous_champion!r}: {format_failures(compare_doc)}; "
                    "refuse promote to 'production'",
                    compare_doc,
                )

    result = add_tag(artifact_dir, run_id, to, actor="promote")

    if to == "production" and strip_staging:
        remove_tag(artifact_dir, run_id, "staging", actor="promote")

    if to == "production":
        remove_tag(artifact_dir, run_id, "rolled_back", actor="promote")

    if to == "production" and exclusive_production:
        for other in runs_with_tag(artifact_dir, "production"):
            if other != run_id:
                remove_tag(artifact_dir, other, "production", actor="promote")

    # Update mutable registry card promotion field (run folder registry stays as snapshot)
    registry_path = Path(artifact_dir) / "registry.json"
    if registry_path.is_file():
        card = json.loads(registry_path.read_text(encoding="utf-8"))
        if card.get("run_id") == run_id:
            card["promotion"] = to
            card["promotion_tags"] = list_tags(artifact_dir, run_id).get("tags", [])
            registry_path.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")

    # Append promote event into tag store history with gate pointer
    store = load_tags(artifact_dir)
    store.setdefault("history", []).append(
        {
            "action": "promote",
            "run_id": run_id,
            "to": to,
            "gates_passed": None if gate_doc is None else bool(gate_doc.get("passed")),
            "at": result.get("tags") and list_tags(artifact_dir, run_id).get("updated_at"),
            "actor": "promote",
            **(
                {
                    "previous_champion": previous_champion,
                    "champion_compare": (
                        compare_doc["status"] if compare_doc else "skipped"
                    ),
                }
                if to == "production"
                else {}
            ),
        }
    )
    if to == "production":
        stack = list(store.get("production_stack") or [])
        if not stack and previous_champion and previous_champion != run_id:
            stack.append(previous_champion)  # pre-round-4 stores: seed with champion
        if not stack or stack[-1] != run_id:
            stack.append(run_id)
        store["production_stack"] = stack
    save_tags(artifact_dir, store)

    out = {
        "run_id": run_id,
        "promotion": to,
        "tags": list_tags(artifact_dir, run_id).get("tags", []),
        "gates_passed": None if gate_doc is None else bool(gate_doc.get("passed")),
    }
    if to == "production":
        out["previous_champion"] = previous_champion
        if compare_doc is not None:
            out["champion_compare"] = {
                k: compare_doc.get(k) for k in ("status", "passed", "champion", "results", "path")
            }
        out["authorize_token_path"] = str(auth_token_path(artifact_dir))
        if authorize:
            out["authorize_hint"] = "keep .promote_token private (toy local auth)"
    return out


def rollback(
    artifact_dir: str | Path,
    *,
    authorize: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Move ``production`` back to the previous production run.

    Uses ``production_stack`` in ``tags.json`` (appended by each production
    promote). The current champion loses ``production`` and gains
    ``rolled_back``; the previous run regains ``production``. No gates are
    re-run: rollback restores a run that already passed them. Repeated calls
    walk further back. Requires ``authorize`` when a ``.promote_token`` exists.
    """
    store = load_tags(artifact_dir)
    stack = list(store.get("production_stack") or [])
    if len(stack) < 2:
        raise PromotionError("no previous production run to roll back to")

    token_path = auth_token_path(artifact_dir)
    if token_path.is_file():
        expected = token_path.read_text(encoding="utf-8").strip()
        if not authorize or authorize != expected:
            raise PromotionError(
                f"rollback requires --authorize <token> (token file: {token_path})"
            )

    current = stack[-1]
    previous = stack[-2]
    if not _run_dir(artifact_dir, previous).is_dir():
        raise PromotionError(f"previous production run {previous!r} is missing on disk")

    remove_tag(artifact_dir, current, "production", actor="rollback")
    add_tag(artifact_dir, current, "rolled_back", actor="rollback")
    add_tag(artifact_dir, previous, "production", actor="rollback")

    store = load_tags(artifact_dir)
    store["production_stack"] = stack[:-1]
    store.setdefault("history", []).append(
        {
            "action": "rollback",
            "from": current,
            "to": previous,
            "reason": reason,
            "at": list_tags(artifact_dir, previous).get("updated_at"),
            "actor": "rollback",
        }
    )
    save_tags(artifact_dir, store)

    registry_path = Path(artifact_dir) / "registry.json"
    if registry_path.is_file():
        card = json.loads(registry_path.read_text(encoding="utf-8"))
        if card.get("run_id") in {current, previous}:
            card["promotion"] = "production" if card["run_id"] == previous else "rolled_back"
            card["promotion_tags"] = list_tags(artifact_dir, card["run_id"]).get("tags", [])
            registry_path.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")

    return {
        "rolled_back": current,
        "production": previous,
        "production_stack": stack[:-1],
        "tags": {
            current: list_tags(artifact_dir, current).get("tags", []),
            previous: list_tags(artifact_dir, previous).get("tags", []),
        },
        "reason": reason,
    }
