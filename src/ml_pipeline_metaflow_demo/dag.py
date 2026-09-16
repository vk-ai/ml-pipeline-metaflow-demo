"""Thin DAG runner — Metaflow/Airflow-style step graph without a cluster.

Nodes are callables ``(ctx: dict) -> dict`` that mutate/return a shared context.
Edges define a linear or lightly branched order; this demo uses a linear chain.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


StepFn = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class DAG:
    """Minimal directed acyclic graph of named steps."""

    name: str
    steps: dict[str, StepFn] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)
    start: str | None = None

    def add_step(self, name: str, fn: StepFn) -> DAG:
        if name in self.steps:
            raise ValueError(f"duplicate step: {name}")
        self.steps[name] = fn
        if self.start is None:
            self.start = name
        return self

    def connect(self, upstream: str, downstream: str) -> DAG:
        if upstream not in self.steps or downstream not in self.steps:
            raise KeyError(f"unknown step in edge {upstream!r} → {downstream!r}")
        self.edges.append((upstream, downstream))
        return self

    def topo_order(self) -> list[str]:
        """Kahn topological sort; raises on cycles or disconnected graphs."""
        if not self.steps:
            return []
        if self.start is None:
            raise ValueError("DAG has no start step")

        indegree = {n: 0 for n in self.steps}
        children: dict[str, list[str]] = {n: [] for n in self.steps}
        for u, v in self.edges:
            children[u].append(v)
            indegree[v] += 1

        # Prefer start first among zero-indegree nodes
        ready = sorted(n for n, d in indegree.items() if d == 0)
        if self.start not in ready:
            raise ValueError(f"start step {self.start!r} has incoming edges")
        ready.remove(self.start)
        queue = [self.start] + ready

        order: list[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in children[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.steps):
            raise ValueError("cycle or unreachable steps in DAG")
        return order

    def run(self, initial: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute steps in topological order, threading a shared context."""
        ctx: dict[str, Any] = dict(initial or {})
        ctx.setdefault("_history", [])
        for name in self.topo_order():
            ctx = self.steps[name](ctx)
            ctx["_history"] = list(ctx.get("_history", [])) + [name]
        return ctx

    def graph_text(self) -> str:
        order = self.topo_order()
        arrows = " → ".join(order)
        return f"{self.name}: {arrows}"
