"""Tests for bilevel_planning_graph.py."""

import pickle
from pathlib import Path

import numpy as np

from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph


def test_bilevel_planning_graph():
    """Tests for BilevelPlanningGraph()."""

    # Make sure that this works with non-hashable states and actions.
    state1 = ["foo"]
    action1 = [1]
    state2 = ["bar"]
    abstract_state1 = ("hello", "world")
    abstract_state2 = ("hi", "there")
    abstract_action1 = 123
    bpg = BilevelPlanningGraph()
    bpg.add_state_node(state1)
    bpg.add_state_node(state2)
    bpg.add_action_edge(state1, action1, state2)
    bpg.add_abstract_state_node(abstract_state1)
    bpg.add_abstract_state_node(abstract_state2)
    bpg.add_abstract_action_edge(abstract_state1, abstract_action1, abstract_state2)
    bpg.add_state_abstractor_edge(state1, abstract_state1)
    bpg.add_state_abstractor_edge(state2, abstract_state2)
    assert len(bpg.states) == 2
    assert len(bpg.action_edges) == 1
    assert len(bpg.abstract_states) == 2
    assert len(bpg.abstract_action_edges) == 1
    assert len(bpg.state_abstractor_edges) == 2

    # Calling the methods with the same objects shouldn't change the counts.
    bpg.add_state_node(state1)
    bpg.add_state_node(state2)
    bpg.add_action_edge(state1, action1, state2)
    bpg.add_abstract_state_node(abstract_state1)
    bpg.add_abstract_state_node(abstract_state2)
    bpg.add_abstract_action_edge(abstract_state1, abstract_action1, abstract_state2)
    bpg.add_state_abstractor_edge(state1, abstract_state1)
    bpg.add_state_abstractor_edge(state2, abstract_state2)
    assert len(bpg.states) == 2
    assert len(bpg.action_edges) == 1
    assert len(bpg.abstract_states) == 2
    assert len(bpg.abstract_action_edges) == 1
    assert len(bpg.state_abstractor_edges) == 2

    # Test sample_state_from_abstract_state().
    rng = np.random.default_rng(123)
    assert bpg.sample_state_from_abstract_state(abstract_state1, rng) == state1
    assert bpg.sample_state_from_abstract_state(abstract_state2, rng) == state2

    # Test extract_plan().
    plan = bpg.extract_plan(state2)
    assert plan.states == [state1, state2]
    assert plan.actions == [action1]
    plan = bpg.extract_plan(state1)
    assert plan.states == [state1]
    assert not plan.actions

    # Uncomment to make GIF.
    # bpg.render_gif(save_path="debug.gif")


def _make_small_bpg() -> BilevelPlanningGraph:
    bpg: BilevelPlanningGraph = BilevelPlanningGraph()
    states = [np.array([i], dtype=np.int64) for i in range(4)]
    for s in states:
        bpg.add_state_node(s)
    bpg.add_abstract_state_node("start")
    bpg.add_abstract_state_node("end")
    bpg.add_state_abstractor_edge(states[0], "start")
    bpg.add_state_abstractor_edge(states[3], "end")
    bpg.add_abstract_action_edge("start", "go", "end")
    for a, b in zip(states[:-1], states[1:]):
        bpg.add_action_edge(a, "step", b)
    return bpg


def test_export_roundtrip(tmp_path: Path):
    """``export`` writes a single pickle with a ``graph`` and ``states`` half.

    Every concrete node referenced in the topology half should be keyed in the states
    half, and the stored states should be the originals (not string reprs).
    """
    bpg = _make_small_bpg()
    path = tmp_path / "bundle.pkl"
    bpg.export(path, final_state=bpg.states[-1])

    with open(path, "rb") as f:
        bundle = pickle.load(f)

    assert set(bundle.keys()) == {"graph", "states"}
    graph = bundle["graph"]
    states = bundle["states"]

    assert set(graph.keys()) >= {"nodes", "edges", "plan", "config", "state_data"}

    # The exporter prunes degree-1 chain nodes; at minimum the start and goal
    # (both of which have abstract-state mappings) should survive.
    concrete_ids = {n["id"] for n in graph["nodes"] if n["type"] == "concrete"}
    assert len(concrete_ids) >= 2
    assert concrete_ids.issubset(set(states.keys()))
    for state in states.values():
        assert isinstance(state, np.ndarray)

    plan_nodes = graph["plan"]["nodes"]
    assert plan_nodes, "plan should be non-empty when final_state is provided"
    assert graph["plan"]["start"] == plan_nodes[0]
    assert graph["plan"]["goal"] == plan_nodes[-1]

    # Abstract states are emitted as nodes on the z_top plane, with a
    # state-abstractor edge from each kept concrete node to its abstract
    # state and an abstract_action edge between the two abstract states.
    abstract_nodes = [n for n in graph["nodes"] if n["type"] == "abstract"]
    assert len(abstract_nodes) == 2
    z_top = graph["config"]["z_top"]
    for n in abstract_nodes:
        assert n["position"][2] == z_top

    edge_types = {e["type"] for e in graph["edges"]}
    assert "action" in edge_types
    assert "abstractor" in edge_types
    assert "abstract_action" in edge_types

    # Hierarchical layout: the plan walks a single linear chain of
    # concrete nodes, so we should see strictly-decreasing y as we step
    # through the plan (roots at top, descendants below).
    id_to_y = {n["id"]: n["position"][1] for n in graph["nodes"]}
    plan_ys = [id_to_y[nid] for nid in plan_nodes]
    assert plan_ys == sorted(plan_ys, reverse=True)
    assert plan_ys[0] != plan_ys[-1]

    # Short node ids: every concrete node id should parse to an integer
    # within [0, len(self.states)), not the full 80-digit content hash.
    n_states = len(states)
    for n in graph["nodes"]:
        if n["type"] == "concrete":
            idx = int(n["id"].split(":")[1])
            assert 0 <= idx < n_states
    # The states-dict keys must match the node ids used elsewhere so
    # the backend can look each rendered concrete node up at render time.
    assert concrete_ids.issubset(set(states.keys()))
