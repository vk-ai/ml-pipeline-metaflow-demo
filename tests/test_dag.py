from __future__ import annotations

import pytest

from ml_pipeline_metaflow_demo.dag import DAG


def test_linear_topo_and_history():
    dag = DAG(name="toy")
    dag.add_step("a", lambda ctx: {**ctx, "a": 1})
    dag.add_step("b", lambda ctx: {**ctx, "b": ctx["a"] + 1})
    dag.add_step("c", lambda ctx: {**ctx, "c": ctx["b"] + 1})
    dag.connect("a", "b")
    dag.connect("b", "c")

    assert dag.topo_order() == ["a", "b", "c"]
    out = dag.run({})
    assert out["c"] == 3
    assert out["_history"] == ["a", "b", "c"]
    assert "toy: a → b → c" == dag.graph_text()


def test_cycle_raises():
    dag = DAG(name="cycle")
    dag.add_step("a", lambda ctx: ctx)
    dag.add_step("b", lambda ctx: ctx)
    dag.connect("a", "b")
    dag.connect("b", "a")
    with pytest.raises(ValueError, match="incoming edges|cycle"):
        dag.topo_order()


def test_duplicate_step_raises():
    dag = DAG(name="dup")
    dag.add_step("a", lambda ctx: ctx)
    with pytest.raises(ValueError, match="duplicate"):
        dag.add_step("a", lambda ctx: ctx)
