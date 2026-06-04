"""Build a minimal two-abstract-state bilevel planning graph.

Topology:

  abstract:  "before" ---transition---> "after"
                 |                         |
                 v                         v
  concrete:  c0 -> c1 -> c2 -> c3 -> c4

The first four concrete states abstract to "before"; the fifth abstracts
to "after". The concrete chain is a single path of ``step`` actions. The
abstract layer has a single ``transition`` action from ``before`` to
``after``.

Running this script writes a visualizer bundle to
``bilevel-planning/examples/data/simple_two_state.pkl`` which you can
load in the ``bilevel_planning.visualizer``.
"""

from pathlib import Path

from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph


def build_simple_two_state_bpg() -> BilevelPlanningGraph:
    """Construct the example BPG described in the module docstring."""
    bpg: BilevelPlanningGraph = BilevelPlanningGraph()

    # Abstract layer: two states and one action connecting them.
    bpg.add_abstract_state_node("before")
    bpg.add_abstract_state_node("after")
    bpg.add_abstract_action_edge("before", "transition", "after")

    # Concrete layer: five states on a line, indexed by a single int.
    concrete_states = [(i,) for i in range(5)]
    for cs in concrete_states:
        bpg.add_state_node(cs)

    # Ground the first four to "before" and the fifth to "after".
    for cs in concrete_states[:4]:
        bpg.add_state_abstractor_edge(cs, "before")
    bpg.add_state_abstractor_edge(concrete_states[4], "after")

    # Concrete action chain: c0 -> c1 -> c2 -> c3 -> c4.
    for src, dst in zip(concrete_states[:-1], concrete_states[1:]):
        bpg.add_action_edge(src, "step", dst)

    return bpg


def main() -> None:
    """Build the BPG, print a summary, and export a visualizer bundle."""
    bpg = build_simple_two_state_bpg()
    goal = (4,)

    print("Simple two-state BPG:")
    print(f"  concrete states:        {len(bpg.states)}")
    print(f"  abstract states:        {len(bpg.abstract_states)}")
    print(f"  concrete action edges:  {len(bpg.action_edges)}")
    print(f"  abstract action edges:  {len(bpg.abstract_action_edges)}")
    print(f"  state-abstractor edges: {len(bpg.state_abstractor_edges)}")

    plan = bpg.extract_plan(goal)
    print(f"  extracted plan:         {len(plan.actions)} actions")

    out_path = Path(__file__).parent / "data" / "simple_two_state.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpg.export(out_path, final_state=goal)
    print(f"\nWrote visualizer bundle to {out_path}")
    renderer_path = Path(__file__).parent / "renderers" / "gradient.py"
    print(
        "\nView it with:\n"
        f"  python -m bilevel_planning.visualizer \\\n"
        f"      --bundle {out_path} \\\n"
        f"      --renderer {renderer_path}"
    )


if __name__ == "__main__":
    main()
