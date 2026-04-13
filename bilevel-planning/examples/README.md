# Bilevel Planning Graph Examples

Runnable Python scripts that build small `BilevelPlanningGraph` (BPG)
objects and export them for use with the
[visualizer](../src/bilevel_planning/visualizer/). They exist to show
what a BPG actually *is*, unclouded by the planner and environment
machinery you'd normally need to produce one.

## What's a bilevel planning graph?

A BPG is a record of the state space a bilevel planner searches through
as it produces a plan. "Bilevel" means two layers:

- **Concrete states** are low-level world states (continuous values,
  object poses, joint angles). The planner's trajectory sampler moves
  between them via **concrete action edges**.
- **Abstract states** are high-level descriptions of those world states
  (typically a set of true predicates like `(OnTable block)`,
  `(HandEmpty robot)`). The planner's abstract plan generator moves
  between them via **abstract action edges** (often PDDL operators).

A **state-abstractor edge** connects a concrete state to the abstract
state it satisfies — each concrete state groups into one abstract
state. A concrete action edge represents a low-level transition; an
abstract action edge represents a high-level skill or operator.

A `BilevelPlanningGraph` holds all five pieces:

| method                       | adds                                  |
| ---                          | ---                                   |
| `add_state_node`             | a concrete state                      |
| `add_abstract_state_node`    | an abstract state                     |
| `add_action_edge`            | a concrete transition (state, u, s')  |
| `add_abstract_action_edge`   | an abstract transition (S, A, S')     |
| `add_state_abstractor_edge`  | concrete → abstract grouping          |

Real planners (e.g. `SesamePlanner`) build this graph incrementally as
they search. Once built, you can:

- Call `bpg.extract_plan(final_state)` to walk back from a goal state
  and recover the concrete state/action sequence.
- Call `bpg.export(path, final_state=...)` to dump the whole graph plus
  the concrete state objects to a single pickle. The
  `bilevel_planning.visualizer` serves that pickle to a browser-based
  3D viewer.

The examples in this directory build the graph by hand, so you can see
exactly what goes in and what comes out without a planner in the loop.

## Running the examples

From the `bilevel-planning/` directory:

```bash
python examples/simple_two_state.py
```

Each example prints a summary of the graph it built and writes a
pickle bundle to `examples/data/`. To view one in the visualizer:

```bash
python -m bilevel_planning.visualizer
```

Open http://localhost:5001, paste the snippet below into the **Python
renderer** pane, click **Apply renderer**, then upload the `.pkl` from
`examples/data/`. The snippet maps each concrete state `(i,)` to a
solid color along a red-to-green gradient, so clicking through the
chain `c0 -> c1 -> c2 -> c3 -> c4` walks visibly from red to green:

```python
import numpy as np

def render_state(state):
    # The example concrete states are single-int tuples like (0,), (1,),
    # ..., (4,). Map the index to a color along a red -> green gradient.
    (index,) = state
    t = index / 4.0
    color = np.array([int(255 * (1 - t)), int(255 * t), 0], dtype=np.uint8)
    return np.broadcast_to(color, (256, 256, 3)).astype(np.uint8)
```

The point of these examples is the graph structure, not the imagery —
the render above exists just to make it visually obvious which node
you're clicking.

## Examples

- **`simple_two_state.py`** — the smallest interesting BPG. Two
  abstract states (`before`, `after`) connected by one abstract action
  (`transition`), and a five-step concrete chain where the first four
  states ground out `before` and the fifth grounds out `after`. A good
  first read for what each `add_*` method does.
