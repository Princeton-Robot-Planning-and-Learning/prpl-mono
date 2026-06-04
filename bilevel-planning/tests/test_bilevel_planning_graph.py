"""Tests for bilevel_planning_graph.py."""

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bilevel_planning.bilevel_planning_graph import (
    BilevelPlanningGraph,
    _abstract_state_atom_strs,
)


@dataclass(frozen=True)
class _AtomsAbstractState:
    """Minimal abstract state exposing ``.atoms``, like RelationalAbstractState."""

    atoms: frozenset = field(default_factory=frozenset)


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

    # The exporter renders every concrete state; at minimum the start and
    # goal nodes are present.
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

    # Abstract nodes are depth-stamped (s:<abstract_id>_<depth>) and laid out
    # top-down, so "start" (abstract id 0, depth 0) sits above "end" (abstract
    # id 1, reached at depth 1).
    abs_y = {n["id"]: n["position"][1] for n in abstract_nodes}
    assert abs_y["s:0_0"] > abs_y["s:1_1"]

    # The abstract plane is fit into the concrete plane's xy bounds, so it
    # never spreads wider or taller than the concrete plane.
    concrete_nodes = [n for n in graph["nodes"] if n["type"] == "concrete"]
    cxs = [n["position"][0] for n in concrete_nodes]
    cys = [n["position"][1] for n in concrete_nodes]
    axs = [n["position"][0] for n in abstract_nodes]
    ays = [n["position"][1] for n in abstract_nodes]
    eps = 1e-6
    assert min(cxs) - eps <= min(axs) and max(axs) <= max(cxs) + eps
    assert min(cys) - eps <= min(ays) and max(ays) <= max(cys) + eps

    edge_types = {e["type"] for e in graph["edges"]}
    assert "action" in edge_types
    assert "abstractor" in edge_types
    assert "abstract_action" in edge_types

    # Abstract-action edges carry the action's short name for click inspection;
    # other edge types carry no name.
    for e in graph["edges"]:
        if e["type"] == "abstract_action":
            assert e["name"] == "go"
        else:
            assert "name" not in e

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

    # These abstract states are plain strings with no ``.atoms``, so the
    # exporter attaches no atoms list to them.
    abstract_nodes = [n for n in graph["nodes"] if n["type"] == "abstract"]
    assert abstract_nodes
    for n in abstract_nodes:
        assert "atoms" not in n


def test_abstract_state_atom_strs():
    """``_abstract_state_atom_strs`` extracts sorted atoms, else None."""
    assert _abstract_state_atom_strs("no-atoms-attr") is None
    state = _AtomsAbstractState(frozenset({"Holding(b)", "On(a, b)", "Clear(a)"}))
    assert _abstract_state_atom_strs(state) == [
        "Clear(a)",
        "Holding(b)",
        "On(a, b)",
    ]
    assert _abstract_state_atom_strs(_AtomsAbstractState()) == []


def test_export_abstract_atoms(tmp_path: Path):
    """Abstract nodes carry their sorted atom strings when available."""
    bpg: BilevelPlanningGraph = BilevelPlanningGraph()
    states = [np.array([i], dtype=np.int64) for i in range(2)]
    for s in states:
        bpg.add_state_node(s)
    start = _AtomsAbstractState(frozenset({"On(a, b)", "Clear(a)"}))
    end = _AtomsAbstractState(frozenset({"Holding(a)"}))
    bpg.add_abstract_state_node(start)
    bpg.add_abstract_state_node(end)
    bpg.add_state_abstractor_edge(states[0], start)
    bpg.add_state_abstractor_edge(states[1], end)
    bpg.add_abstract_action_edge(start, "pick", end)
    bpg.add_action_edge(states[0], "step", states[1])

    path = tmp_path / "bundle.pkl"
    bpg.export(path, final_state=states[-1])
    with open(path, "rb") as f:
        graph = pickle.load(f)["graph"]

    atoms_by_id = {
        n["id"]: n.get("atoms") for n in graph["nodes"] if n["type"] == "abstract"
    }
    assert atoms_by_id == {
        "s:0_0": ["Clear(a)", "On(a, b)"],
        "s:1_1": ["Holding(a)"],
    }


def test_export_abstract_depth_unrolling(tmp_path: Path):
    """Revisiting an abstract state yields a deeper node, not a back-edge."""
    bpg: BilevelPlanningGraph = BilevelPlanningGraph()
    states = [np.array([i], dtype=np.int64) for i in range(3)]
    for s in states:
        bpg.add_state_node(s)
    # The concrete chain visits abstract states s0 -> s1 -> s0 (back to root).
    s0 = _AtomsAbstractState(frozenset({"At(home)"}))
    s1 = _AtomsAbstractState(frozenset({"At(away)"}))
    bpg.add_abstract_state_node(s0)
    bpg.add_abstract_state_node(s1)
    bpg.add_state_abstractor_edge(states[0], s0)
    bpg.add_state_abstractor_edge(states[1], s1)
    bpg.add_state_abstractor_edge(states[2], s0)
    bpg.add_abstract_action_edge(s0, "Go(away)", s1)
    bpg.add_abstract_action_edge(s1, "Go(home)", s0)
    bpg.add_action_edge(states[0], "step", states[1])
    bpg.add_action_edge(states[1], "step", states[2])

    path = tmp_path / "bundle.pkl"
    bpg.export(path, final_state=states[-1])
    with open(path, "rb") as f:
        graph = pickle.load(f)["graph"]

    # s0 (abstract id 0) appears at depth 0 and again at depth 2; s1 at depth 1.
    abstract_ids = sorted(n["id"] for n in graph["nodes"] if n["type"] == "abstract")
    assert abstract_ids == ["s:0_0", "s:0_2", "s:1_1"]

    aa = {
        (e["source"], e["target"]): e.get("name")
        for e in graph["edges"]
        if e["type"] == "abstract_action"
    }
    assert aa == {
        ("s:0_0", "s:1_1"): "Go(away)",
        ("s:1_1", "s:0_2"): "Go(home)",
    }
    # No abstract-action edge points back to the depth-0 root.
    assert all(target != "s:0_0" for _, target in aa)
