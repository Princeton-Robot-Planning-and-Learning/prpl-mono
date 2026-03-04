"""Tests for bilevel_planning_graph.py."""

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
