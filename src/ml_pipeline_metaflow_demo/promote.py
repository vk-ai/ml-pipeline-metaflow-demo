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

from ml_pipeline_metaflow_demo.tags import (
    add_tag,
    list_tags,
    load_tags,
    remove_tag,
    save_tags,
)


class PromotionError(RuntimeError):
    """Raised when promotion is refused (gates / authorize / missing run)."""


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
) -> dict[str, Any]:
    """
    Promote ``run_id`` to ``to`` (default ``production``) iff gates passed.

    - Always requires ``gates.json`` with ``passed: true`` when ``require_gates``.
    - Adds tag ``gate:passed`` when gates are green.
    - First ``production`` promote mints ``.promote_token``; later ones need
      ``authorize=<token>`` (or env-style pass-through via caller).
    - Optionally strips ``staging`` when promoting to production.
    - Updates registry.json ``promotion`` field when present (mutable card only).
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

    result = add_tag(artifact_dir, run_id, to, actor="promote")

    if to == "production" and strip_staging:
        remove_tag(artifact_dir, run_id, "staging", actor="promote")

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
        }
    )
    save_tags(artifact_dir, store)

    out = {
        "run_id": run_id,
        "promotion": to,
        "tags": list_tags(artifact_dir, run_id).get("tags", []),
        "gates_passed": None if gate_doc is None else bool(gate_doc.get("passed")),
    }
    if to == "production":
        out["authorize_token_path"] = str(auth_token_path(artifact_dir))
        if authorize:
            out["authorize_hint"] = "keep .promote_token private (toy local auth)"
    return out
