"""Tests for bilevel_planning_graph.py."""

import tempfile
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


def test_render_image():
    """Test that render_image() produces a PNG file."""
    bpg = BilevelPlanningGraph()
    states = [["s0"], ["s1"], ["s2"], ["s3"]]
    for s in states:
        bpg.add_state_node(s)
    bpg.add_action_edge(states[0], [0], states[1])
    bpg.add_action_edge(states[1], [1], states[2])
    bpg.add_action_edge(states[2], [2], states[3])

    abs1, abs2 = ("a",), ("b",)
    bpg.add_abstract_state_node(abs1)
    bpg.add_abstract_state_node(abs2)
    bpg.add_abstract_action_edge(abs1, "go", abs2)
    bpg.add_state_abstractor_edge(states[0], abs1)
    bpg.add_state_abstractor_edge(states[1], abs1)
    bpg.add_state_abstractor_edge(states[2], abs2)
    bpg.add_state_abstractor_edge(states[3], abs2)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.png"
        bpg.render_image(save_path=path, final_state=states[3])
        assert path.exists()
        assert path.stat().st_size > 0

        # Test with max_concrete_states subsampling.
        path2 = Path(tmpdir) / "test_subsampled.png"
        bpg.render_image(save_path=path2, final_state=states[3], max_concrete_states=2)
        assert path2.exists()
        assert path2.stat().st_size > 0
