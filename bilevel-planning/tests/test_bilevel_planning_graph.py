"""Tests for bilevel_planning_graph.py."""

import json
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


def test_export_graph_for_web():
    """export_graph_for_web returns a JSON-serializable dict with expected keys."""
    bpg = _make_small_bpg()
    final_state = bpg.states[-1]
    graph_data = bpg.export_graph_for_web(final_state=final_state)

    assert set(graph_data.keys()) >= {"nodes", "edges", "plan", "config", "state_data"}
    # The exporter prunes degree-1 chain nodes; at minimum the start and goal
    # (both of which have abstract-state mappings) should survive.
    concrete = [n for n in graph_data["nodes"] if n["type"] == "concrete"]
    assert len(concrete) >= 2
    # Round-trip through json to confirm serializability.
    json.loads(json.dumps(graph_data))

    plan_nodes = graph_data["plan"]["nodes"]
    assert plan_nodes, "plan should be non-empty when final_state is provided"
    assert graph_data["plan"]["start"] == plan_nodes[0]
    assert graph_data["plan"]["goal"] == plan_nodes[-1]


def test_export_graph_with_pickle(tmp_path: Path):
    """export_graph_with_pickle writes a JSON + pickle pair with consistent IDs."""
    bpg = _make_small_bpg()
    json_path = tmp_path / "graph.json"
    pickle_path = tmp_path / "states.pkl"
    bpg.export_graph_with_pickle(
        json_path=json_path,
        pickle_path=pickle_path,
        final_state=bpg.states[-1],
    )

    with open(json_path, encoding="utf-8") as f:
        graph_data = json.load(f)
    with open(pickle_path, "rb") as f:
        state_data = pickle.load(f)

    # Every concrete node in the JSON should have a matching state object in
    # the pickle, and the state objects should be the originals (not strings).
    concrete_ids = {n["id"] for n in graph_data["nodes"] if n["type"] == "concrete"}
    assert concrete_ids
    assert concrete_ids.issubset(set(state_data.keys()))
    for state in state_data.values():
        assert isinstance(state, np.ndarray)
