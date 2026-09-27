"""Mutable promotion tags on immutable run artifacts (Metaflow-style teaching stub).

Run folders under ``artifacts/runs/<run_id>/`` stay immutable. Tags live in a
separate ``artifacts/tags.json`` (and per-run ``tags.json`` snapshot) so labels
like ``candidate``, ``staging``, ``production``, ``gate:passed`` can change
without rewriting lineage digests.

Honesty: tags organize interpretation of immutable facts — they are **not**
namespace isolation. Metaflow docs are explicit: namespaces/production tokens
isolate; tags organize. This demo teaches that distinction.
See: https://docs.metaflow.org/scaling/tagging
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_TAGS = ("candidate", "staging", "production", "gate:passed")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tags_store_path(artifact_dir: str | Path) -> Path:
    return Path(artifact_dir) / "tags.json"


def run_dir(artifact_dir: str | Path, run_id: str) -> Path:
    return Path(artifact_dir) / "runs" / run_id


def load_tags(artifact_dir: str | Path) -> dict[str, Any]:
    """Load the mutable tag store (run_id → sorted tag list + history)."""
    path = tags_store_path(artifact_dir)
    if not path.is_file():
        return {
            "runs": {},
            "history": [],
            "notes": (
                "OSS learning stub — mutable tags on immutable runs "
                "(Metaflow tagging pattern); not Metaflow Client API."
            ),
        }
    return json.loads(path.read_text(encoding="utf-8"))


def save_tags(artifact_dir: str | Path, store: dict[str, Any]) -> Path:
    path = tags_store_path(artifact_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2) + "\n", encoding="utf-8")
    return path


def _ensure_run(store: dict[str, Any], run_id: str) -> dict[str, Any]:
    runs = store.setdefault("runs", {})
    if run_id not in runs:
        runs[run_id] = {"tags": [], "updated_at": None}
    return runs[run_id]


def list_tags(artifact_dir: str | Path, run_id: str | None = None) -> dict[str, Any]:
    """List tags for one run or the whole store."""
    store = load_tags(artifact_dir)
    if run_id is None:
        return store
    entry = store.get("runs", {}).get(run_id, {"tags": [], "updated_at": None})
    return {"run_id": run_id, **entry}


def add_tag(
    artifact_dir: str | Path,
    run_id: str,
    tag: str,
    *,
    actor: str = "local",
) -> dict[str, Any]:
    """Attach a mutable tag to an immutable run. Idempotent."""
    tag = str(tag).strip()
    if not tag:
        raise ValueError("tag must be non-empty")
    store = load_tags(artifact_dir)
    entry = _ensure_run(store, run_id)
    tags = list(entry.get("tags") or [])
    if tag not in tags:
        tags.append(tag)
        tags.sort()
        entry["tags"] = tags
        entry["updated_at"] = _now()
        store.setdefault("history", []).append(
            {
                "action": "add",
                "run_id": run_id,
                "tag": tag,
                "at": entry["updated_at"],
                "actor": actor,
            }
        )
        save_tags(artifact_dir, store)
        _sync_run_snapshot(artifact_dir, run_id, tags)
    return {"run_id": run_id, "tags": list(entry["tags"]), "added": tag}


def remove_tag(
    artifact_dir: str | Path,
    run_id: str,
    tag: str,
    *,
    actor: str = "local",
) -> dict[str, Any]:
    """Remove a tag from a run (no-op if absent)."""
    tag = str(tag).strip()
    store = load_tags(artifact_dir)
    entry = _ensure_run(store, run_id)
    tags = [t for t in (entry.get("tags") or []) if t != tag]
    removed = tag in (entry.get("tags") or [])
    if removed:
        entry["tags"] = tags
        entry["updated_at"] = _now()
        store.setdefault("history", []).append(
            {
                "action": "remove",
                "run_id": run_id,
                "tag": tag,
                "at": entry["updated_at"],
                "actor": actor,
            }
        )
        save_tags(artifact_dir, store)
        _sync_run_snapshot(artifact_dir, run_id, tags)
    return {"run_id": run_id, "tags": list(entry["tags"]), "removed": tag if removed else None}


def runs_with_tag(artifact_dir: str | Path, tag: str) -> list[str]:
    """Return run_ids that currently carry ``tag`` (sorted)."""
    store = load_tags(artifact_dir)
    out = [
        rid
        for rid, entry in store.get("runs", {}).items()
        if tag in (entry.get("tags") or [])
    ]
    return sorted(out)


def latest_with_tag(artifact_dir: str | Path, tag: str) -> str | None:
    """Best-effort 'latest' run with tag: prefer newest history add, else lex max id."""
    store = load_tags(artifact_dir)
    tagged = runs_with_tag(artifact_dir, tag)
    if not tagged:
        return None
    for event in reversed(store.get("history") or []):
        if event.get("action") == "add" and event.get("tag") == tag:
            rid = event.get("run_id")
            if rid in tagged:
                return rid
    return tagged[-1]


def _sync_run_snapshot(artifact_dir: str | Path, run_id: str, tags: list[str]) -> None:
    """Write a mutable tags snapshot beside immutable lineage (if run folder exists)."""
    rd = run_dir(artifact_dir, run_id)
    if not rd.is_dir():
        return
    doc = {
        "run_id": run_id,
        "tags": list(tags),
        "updated_at": _now(),
        "notes": (
            "Mutable tag snapshot — run context.json / gates.json stay immutable. "
            "OSS learning stub (Metaflow tagging pattern)."
        ),
    }
    (rd / "tags.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def seed_candidate_tag(artifact_dir: str | Path, run_id: str) -> dict[str, Any]:
    """After register, seed the ``candidate`` tag (matches registry promotion field)."""
    return add_tag(artifact_dir, run_id, "candidate", actor="register")
