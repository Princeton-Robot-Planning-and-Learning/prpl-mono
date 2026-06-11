"""Bilevel planning graphs: primarily for visualization, analysis, debugging."""

import heapq
import pickle
from collections import deque
from pathlib import Path
from typing import Any, Callable, Generic, Hashable, TypeVar

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from prpl_utils.utils import consistent_hash
from relational_structs import ObjectCentricState

from bilevel_planning.structs import Plan

_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action
_S = TypeVar("_S", bound=Hashable)  # abstract state
_A = TypeVar("_A", bound=Hashable)  # abstract action


def _hierarchical_layout(graph: nx.DiGraph) -> dict[str, tuple[float, float]]:
    """Assign each node an (x, y) in a top-down tree arrangement.

    The y coordinate is the node's layer (depth): its longest-path distance
    from any root (node with in_degree 0) when the graph is a DAG, or its
    shortest-path distance via BFS otherwise. Roots sit at y=0 and descendants
    descend (y decreases).

    The x coordinate comes from a tree layout over a spanning forest of the
    graph: leaves are spaced evenly left to right and every internal node is
    centered over its children. Sibling subtrees occupy disjoint x ranges, so
    distinct branches fan out instead of stacking onto the same per-layer
    slots.

    Coordinates are unscaled; the caller is expected to center and rescale them
    to its preferred display box.
    """
    if not graph.nodes:
        return {}

    roots = sorted(n for n in graph.nodes if graph.in_degree(n) == 0)
    if not roots:
        # Pure cycle (or no identifiable root): pick any node to start.
        roots = [min(graph.nodes)]

    # --- y: layer (depth) ---------------------------------------------------
    layer_of: dict[str, int] = {}
    if nx.is_directed_acyclic_graph(graph):
        # Longest-path layering: each node sits one layer below its
        # deepest predecessor. Produces the cleanest top-down read.
        for node in nx.topological_sort(graph):
            preds = list(graph.predecessors(node))
            layer_of[node] = max((layer_of[p] for p in preds), default=-1) + 1
    else:
        # Cycle fallback: BFS shortest-path from the roots.
        for root in roots:
            layer_of.setdefault(root, 0)
        queue: deque[str] = deque(roots)
        while queue:
            u = queue.popleft()
            for v in graph.successors(u):
                if v not in layer_of:
                    layer_of[v] = layer_of[u] + 1
                    queue.append(v)
        # Nodes unreachable from any root (possible inside a cycle
        # component with no root) fall to layer 0.
        for node in graph.nodes:
            layer_of.setdefault(node, 0)

    # --- x: subtree-centroid layout over a spanning forest ------------------
    # Claim one tree-parent per node via BFS from the roots, so a node with
    # several predecessors (a merge) is pulled toward only one of them rather
    # than averaged between two columns. Every tree edge is a real graph edge,
    # and longest-path layering keeps a child's layer below its tree-parent's,
    # so edges still read downward. Children are listed in sorted order to fix
    # a deterministic left-to-right ordering of subtrees.
    tree_children: dict[str, list[str]] = {n: [] for n in graph.nodes}
    visited: set[str] = set()
    forest_roots: list[str] = []
    # Real roots first, then any node not yet placed (a cycle component with
    # no in_degree-0 node), so every node lands in exactly one spanning tree.
    seeds = roots + [n for n in sorted(graph.nodes) if n not in roots]
    for seed in seeds:
        if seed in visited:
            continue
        forest_roots.append(seed)
        visited.add(seed)
        queue = deque([seed])
        while queue:
            u = queue.popleft()
            for v in sorted(graph.successors(u)):
                if v not in visited:
                    visited.add(v)
                    tree_children[u].append(v)
                    queue.append(v)

    # Iterative post-order: leaves take the next x slot left to right, and
    # every internal node is placed at the mean x of its children.
    x_of: dict[str, float] = {}
    next_leaf_x = 0.0
    for root in forest_roots:
        stack: list[tuple[str, bool]] = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                kids = tree_children[node]
                if kids:
                    x_of[node] = sum(x_of[k] for k in kids) / len(kids)
                else:
                    x_of[node] = next_leaf_x
                    next_leaf_x += 1.0
            else:
                stack.append((node, True))
                # Push children reversed so the leftmost is popped first and
                # therefore claims the smallest leaf x.
                for child in reversed(tree_children[node]):
                    stack.append((child, False))

    return {node: (x_of[node], float(-layer_of[node])) for node in graph.nodes}


def _center_and_scale(
    positions: dict[str, tuple[float, float]], box: float = 20.0
) -> dict[str, tuple[float, float]]:
    """Center positions on the origin and uniformly scale them into a box.

    The larger of the layout's width/height is scaled to span ``box``,
    preserving aspect. Each plane is scaled independently so a small graph (the
    abstract plane) still fills the box rather than being squished to match a
    larger graph's node spacing. Empty input returns an empty dict.
    """
    if not positions:
        return {}
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    max_dim = max(max_x - min_x, max_y - min_y, 1.0)
    scale = box / max_dim
    return {
        nid: ((x - center_x) * scale, (y - center_y) * scale)
        for nid, (x, y) in positions.items()
    }


def _fit_into_bounds(
    positions: dict[str, tuple[float, float]],
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
) -> dict[str, tuple[float, float]]:
    """Remap positions to span the given x/y bounds, per axis.

    Used so the abstract plane occupies the same xy bounding box as the concrete plane
    rather than filling a square box of its own (which would stretch its dominant axis
    past the concrete extent). Each axis is mapped independently; an axis with no extent
    collapses to the bound midpoint.
    """
    if not positions:
        return {}
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    def remap(v: float, lo: float, hi: float, tgt_lo: float, tgt_hi: float) -> float:
        if hi - lo < 1e-9:
            return (tgt_lo + tgt_hi) / 2.0
        return tgt_lo + (v - lo) / (hi - lo) * (tgt_hi - tgt_lo)

    return {
        nid: (
            remap(x, min_x, max_x, x_lo, x_hi),
            remap(y, min_y, max_y, y_lo, y_hi),
        )
        for nid, (x, y) in positions.items()
    }


def _abstract_state_atom_strs(abstract_state: Any) -> list[str] | None:
    """Sorted atom strings for an abstract state, or None if it has none.

    Abstract states are generic; relational ones expose their ground atoms as
    a ``.atoms`` set. When that attribute is absent there is nothing to list,
    so the abstract node carries no atoms.
    """
    atoms = getattr(abstract_state, "atoms", None)
    if atoms is None:
        return None
    return sorted(str(atom) for atom in atoms)


def _abstract_action_name(abstract_action: Any) -> str:
    """Short display name for an abstract action.

    Relational abstract actions (ground operators) expose a ``.short_str`` like
    ``Pick(robot, block)``; fall back to ``str`` for anything else.
    """
    short = getattr(abstract_action, "short_str", None)
    if isinstance(short, str):
        return short
    return str(abstract_action)


class BilevelPlanningGraph(Generic[_X, _U, _S, _A]):
    """Bilevel planning graphs: primarily for visualization, analysis, debugging.

    Can also be convenient for extracting plans.
    """

    def __init__(self) -> None:
        self.states: list[_X] = []
        self._state_ids: set[int] = set()  # prevent duplicates
        self._state_id_to_state: dict[int, _X] = (
            {}
        )  # reverse lookup: id -> state, todo: remove _state_ids
        self.abstract_states: list[_S] = []
        self.action_edges: list[tuple[_X, _U, _X]] = []
        self._action_edge_ids: set[int] = set()  # prevent duplicates
        self.abstract_action_edges: list[tuple[_S, _A, _S]] = []
        self.state_abstractor_edges: list[tuple[_X, _S]] = []
        self._state_abstractor_edge_ids: set[int] = set()  # prevent_duplicates
        self._abstract_state_to_states: dict[_S, list[_X]] = {}
        self._abstract_state_to_state_ids: dict[_S, set[int]] = {}  # prevent dups
        self._state_id_to_parent: dict[int, tuple[_X, _U]] = {}
        self._get_state_pos: Callable[[_X], tuple[float, float]] | None = None
        self._get_abstract_state_pos: Callable[[_S], tuple[float, float]] | None = None

    def add_state_node(self, state: _X) -> None:
        """Add a state to the graph."""
        state_id = self._state_to_id(state)
        if state_id in self._state_ids:
            return
        self.states.append(state)
        self._state_ids.add(state_id)
        self._state_id_to_state[state_id] = state
        return

    def add_action_edge(self, state: _X, action: _U, next_state: _X) -> None:
        """Add an action to the graph."""
        transition = (state, action, next_state)
        transition_id = consistent_hash(transition)
        if transition_id in self._action_edge_ids:
            return
        self.action_edges.append(transition)
        self._action_edge_ids.add(transition_id)
        next_state_id = self._state_to_id(next_state)
        # Only set parent if not already set (preserve first path found)
        if next_state_id not in self._state_id_to_parent:
            self._state_id_to_parent[next_state_id] = (state, action)
        return

    def add_abstract_state_node(self, abstract_state: _S) -> None:
        """Add an abstract state to the graph."""
        if abstract_state in self.abstract_states:
            return
        self.abstract_states.append(abstract_state)

        return

    def add_abstract_action_edge(
        self, abstract_state: _S, abstract_action: _A, next_abstract_state: _S
    ) -> None:
        """Add an abstract action to the graph."""
        transition = (abstract_state, abstract_action, next_abstract_state)
        if transition in self.abstract_action_edges:
            return
        self.abstract_action_edges.append(transition)
        return

    def add_state_abstractor_edge(self, state: _X, abstract_state: _S) -> None:
        """Add a state abstractor edge to the graph."""
        edge = (state, abstract_state)
        edge_id = consistent_hash(edge)
        if edge_id in self._state_abstractor_edge_ids:
            return
        self.state_abstractor_edges.append(edge)
        self._state_abstractor_edge_ids.add(edge_id)
        if abstract_state not in self._abstract_state_to_states:
            self._abstract_state_to_states[abstract_state] = []
            self._abstract_state_to_state_ids[abstract_state] = set()
        state_id = self._state_to_id(state)
        if state_id not in self._abstract_state_to_state_ids[abstract_state]:
            self._abstract_state_to_state_ids[abstract_state].add(state_id)
            self._abstract_state_to_states[abstract_state].append(state)
        return

    def sample_state_from_abstract_state(
        self, abstract_state: _S, rng: np.random.Generator
    ) -> _X:
        """Randomly sample one of the states in the graph for the abstract state."""
        assert (
            abstract_state in self._abstract_state_to_states
        ), "No states found for abstract state"
        states = self._abstract_state_to_states[abstract_state]
        assert states, "No states found for abstract state"
        idx = rng.choice(len(states))
        state = states[idx]
        return state

    # Could be made more efficient with .append() followed by .reverse().
    def extract_plan(self, final_state: _X) -> Plan:
        """Follow backpointers from final state and create a plan."""
        x_plan: list[_X] = [final_state]
        u_plan: list[_U] = []
        x = final_state
        x_id = self._state_to_id(x)
        while x_id in self._state_id_to_parent:
            x, u = self._state_id_to_parent[x_id]
            x_plan = [x] + x_plan
            u_plan = [u] + u_plan
            x_id = self._state_to_id(x)
        return Plan(x_plan, u_plan)

    def _state_to_id(self, state: _X) -> int:
        """Get an integer ID for a state that is in the graph."""
        return consistent_hash(state)

    def _id_to_state(self, state_id: int) -> _X | None:
        """Get a state from its integer ID.

        Returns None if not found.
        """
        return self._state_id_to_state.get(state_id)

    def _abstract_state_to_id(self, abstract_state: _S) -> int:
        """Get an integer ID for an abstract state that is in the graph."""
        return self.abstract_states.index(abstract_state)

    def set_state_position_function(
        self, fn: Callable[[_X], tuple[float, float]]
    ) -> None:
        """Allow users to determine xy position of nodes in the graph for rendering."""
        self._get_state_pos = fn

    def set_abstract_state_position_function(
        self, fn: Callable[[_S], tuple[float, float]]
    ) -> None:
        """Allow users to determine xy position of nodes in the graph for rendering."""
        self._get_abstract_state_pos = fn

    def render_gif(
        self,
        save_path: Path,
        final_state: _X | None = None,
        title: str | None = None,
        figsize: tuple[int, int] = (4, 3),
        customize_fig_ax: Callable[[Figure, Axes], None] | None = None,
        abstract_state_color: str = "tab:purple",
        state_color: str = "tab:blue",
        plan_color: str = "tab:green",
        text_size: int = 10,
        node_size: int = 50,
        state_abstractor_edge_color: str = "gray",
        action_edge_color: str = "black",
        abstract_action_edge_color: str = "black",
        node_alpha: float = 0.7,
        edge_alpha: float = 0.7,
        frame_skip: int = 5,
        anim_interval: int = 50,
        view_elevation: int = 20,
    ) -> None:
        """Visualize the bilevel planning graph in 3D with animation.

        Abstract states/actions are on z=1, concrete states/actions on z=0.
        """
        G: nx.DiGraph = nx.DiGraph()
        pos = {}
        z_top = 1
        z_bottom = 0

        # Place abstract states on top plane.
        for abstract_state in self.abstract_states:
            i = self._abstract_state_to_id(abstract_state)
            G.add_node(f"s:{i}")
            if self._get_abstract_state_pos is None:
                node_x: float = i
                node_y = 0.0
            else:
                node_x, node_y = self._get_abstract_state_pos(abstract_state)
            pos[f"s:{i}"] = (node_x, node_y, z_top)

        # Place concrete states on bottom plane.
        for state in self.states:
            i = self._state_to_id(state)
            G.add_node(f"x:{i}")
            if self._get_state_pos is None:
                node_x = i
                node_y = 0.0
            else:
                node_x, node_y = self._get_state_pos(state)
            pos[f"x:{i}"] = (node_x, node_y, z_bottom)

        # Add abstract action edges (top plane).
        for abstract_state1, _, abstract_state2 in self.abstract_action_edges:
            i = self._abstract_state_to_id(abstract_state1)
            j = self._abstract_state_to_id(abstract_state2)
            G.add_edge(f"s:{i}", f"s:{j}")

        # Add concrete action edges (bottom plane).
        for state1, _, state2 in self.action_edges:
            i = self._state_to_id(state1)
            j = self._state_to_id(state2)
            G.add_edge(f"x:{i}", f"x:{j}")

        # Add state-abstractor edges (vertical).
        for state1, abstract_state1 in self.state_abstractor_edges:
            i = self._state_to_id(state1)
            j = self._abstract_state_to_id(abstract_state1)
            G.add_edge(f"x:{i}", f"s:{j}")

        # Find the plan nodes to color differently.
        if final_state is not None:
            plan = self.extract_plan(final_state)
            plan_nodes = []
            for state in plan.states:
                i = self._state_to_id(state)
                plan_nodes.append(f"x:{i}")
            start_node = plan_nodes[0]
            goal_node = plan_nodes[-1]
        else:
            start_node = None
            goal_node = None
            plan_nodes = []

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Draw nodes.
        for node, (x, y, z) in pos.items():
            if node in plan_nodes:
                color = plan_color
                alpha = 1.0
            elif z == z_bottom:
                color = state_color
                alpha = node_alpha
            else:
                color = abstract_state_color
                alpha = node_alpha
            ax.scatter(x, y, z, s=node_size, c=color, alpha=alpha)  # type: ignore

        # Label start and goal if given.
        if start_node is not None:
            x, y, z = pos[start_node]
            pad = (z_top - z_bottom) / 5
            ax.text(x, y, z - pad, "x0", None, fontsize=text_size)  # type: ignore
        if goal_node is not None:
            x, y, z = pos[goal_node]
            pad = (z_top - z_bottom) / 5
            ax.text(x, y, z_bottom - pad, "g", None, fontsize=text_size)  # type: ignore

        # Draw edges.
        for u, v in G.edges():
            x0, y0, z0 = pos[u]
            x1, y1, z1 = pos[v]
            if z0 != z1:
                color = state_abstractor_edge_color
            elif z0 == z_bottom:
                color = action_edge_color
            else:
                assert z1 == z_top
                color = abstract_action_edge_color
            ax.plot([x0, x1], [y0, y1], [z0, z1], c=color, alpha=edge_alpha)

        ax.set_axis_off()
        if title is not None:
            ax.set_title(title)

        if customize_fig_ax is not None:
            customize_fig_ax(fig, ax)

        ax.view_init(elev=view_elevation, azim=30)  # type: ignore

        def update(frame):
            ax.view_init(elev=view_elevation, azim=frame)
            return (fig,)

        anim = FuncAnimation(
            fig,
            update,
            frames=range(0, 360, frame_skip),
            interval=anim_interval,
            blit=False,
        )
        anim.save(save_path, writer="pillow")

    def _abstract_depths(
        self, state_to_abstract: dict[int, int]
    ) -> tuple[dict[int, int], dict[int, int | None]]:
        """Depth and nearest mapped ancestor for each concrete state.

        Walks the concrete search graph (self-loops excluded) in topological
        order. ``incoming_depth[cid]`` is the depth an abstract node at concrete
        state ``cid`` takes -- the number of abstract-state changes along the
        path from the root -- and ``last_mapped[cid]`` is its nearest
        abstract-mapped concrete ancestor (or None). A state reached by two
        paths at conflicting depths is assumed not to occur and raises.
        """
        graph: nx.DiGraph = nx.DiGraph()
        for state in self.states:
            graph.add_node(self._state_to_id(state))
        for source_state, _, target_state in self.action_edges:
            uid = self._state_to_id(source_state)
            vid = self._state_to_id(target_state)
            if uid != vid:
                graph.add_edge(uid, vid)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError(
                "Concrete search graph has a cycle (beyond self-loops); cannot "
                "assign abstract depths for the visualizer."
            )
        incoming_depth: dict[int, int] = {}
        last_mapped: dict[int, int | None] = {}
        for cid in nx.topological_sort(graph):
            parents = list(graph.predecessors(cid))
            if not parents:
                resolved: tuple[int, int | None] = (0, None)
            else:
                candidates: set[tuple[int, int | None]] = set()
                for p in parents:
                    if p in state_to_abstract:
                        candidates.add((incoming_depth[p] + 1, p))
                    else:
                        candidates.add((incoming_depth[p], last_mapped[p]))
                if len(candidates) != 1:
                    raise ValueError(
                        "A concrete state is reached at conflicting abstract "
                        "depths; depth-stamping assumes this cannot happen."
                    )
                resolved = next(iter(candidates))
            incoming_depth[cid], last_mapped[cid] = resolved
        return incoming_depth, last_mapped

    def _build_graph_payload(self, final_state: _X | None = None) -> dict:
        """Build the frontend-facing topology dict for the visualizer.

        Returns a dict with ``nodes``, ``edges``, ``plan``, ``config``, and
        ``state_data``. The payload carries graph topology and plan
        membership only — the frontend chooses colors, sizes, and alphas
        from ``node.type`` (``"concrete"`` / ``"abstract"``),
        ``node.in_plan``, and ``edge.type`` (``"action"`` / ``"abstractor"``
        / ``"abstract_action"``).

        Concrete node ids in the emitted payload are short insertion-order
        integers (``x:0``, ``x:1``, ...) rather than the content-hash ints
        used for internal deduplication. Abstract nodes are depth-stamped:
        ``s:<abstract_id>_<depth>``, where ``depth`` counts abstract-state
        changes along the concrete path from the root, so the same abstract
        state reached at different depths becomes distinct nodes (see
        ``_abstract_depths``).

        The abstract plane renders the entire abstract search graph -- every
        abstract state and abstract-action edge the planner generated -- not
        only the transitions the refiner instantiated with concrete states.
        ``node.in_plan`` distinguishes the refined plan from the rest.
        """
        # ------------------------------------------------------------------
        # Phase 1: gather concrete node ids and the concrete->abstract map.
        # ------------------------------------------------------------------
        state_to_abstract: dict[int, int] = {}
        for x, s in self.state_abstractor_edges:
            state_to_abstract[self._state_to_id(x)] = self._abstract_state_to_id(s)

        # Render every concrete state the planner produced.
        kept_state_ids: set[int] = {self._state_to_id(state) for state in self.states}

        # Short display ids for concrete nodes. Content hashes are
        # internally unambiguous but 80-digit integers are useless in a
        # UI; remap to insertion-order indices for every payload string.
        display_id_of: dict[int, int] = {
            self._state_to_id(state): idx for idx, state in enumerate(self.states)
        }

        def concrete_nid(content_hash: int) -> str:
            return f"x:{display_id_of[content_hash]}"

        # ------------------------------------------------------------------
        # Phase 2: one layout edge per action edge between concrete nodes.
        # ------------------------------------------------------------------
        # Self-loops (a no-op step that lands back on the same deduplicated
        # state) carry no information and are dropped everywhere they'd render.
        G_layout: nx.DiGraph = nx.DiGraph()
        for sid in kept_state_ids:
            G_layout.add_node(concrete_nid(sid))
        for source_state, _, target_state in self.action_edges:
            uid = self._state_to_id(source_state)
            vid = self._state_to_id(target_state)
            if uid != vid:
                G_layout.add_edge(concrete_nid(uid), concrete_nid(vid))

        # ------------------------------------------------------------------
        # Phase 3: lay out the kept concrete nodes, scale into [-10, 10].
        # ------------------------------------------------------------------
        # Hierarchical layer layout: assign each node a layer via
        # longest-path depth from any root (or shortest-path BFS when the
        # graph has cycles), then spread nodes evenly within each layer.
        # Roots sit at top (y=0), descendants descend. Produces a clean
        # top-down DAG read for BPGs produced by a planner.
        layout_pos = _center_and_scale(_hierarchical_layout(G_layout))

        # ------------------------------------------------------------------
        # Phase 4: build the (G, pos) triple we'll emit.
        # ------------------------------------------------------------------
        G: nx.DiGraph = nx.DiGraph()
        pos: dict[str, tuple[float, float, float]] = {}

        z_bottom = 0.0
        z_top = 1.0

        # Kept concrete nodes on the z_bottom plane.
        for sid in kept_state_ids:
            nid = concrete_nid(sid)
            if nid in layout_pos:
                pos[nid] = (layout_pos[nid][0], layout_pos[nid][1], z_bottom)
                G.add_node(nid, type="concrete")

        # Action edges between kept concrete nodes.
        for u, v in G_layout.edges():
            G.add_edge(u, v, type="action")

        # Abstract states on the z_top plane, depth-stamped by the concrete
        # search. Each abstract state is split into one node per depth at which
        # the concrete search reaches it, where depth counts abstract-state
        # changes along the concrete path from the root. This unrolls the
        # abstract graph into a DAG with no back-edges: revisiting an abstract
        # state (e.g. a plan that returns to the root abstract state) shows up
        # as a fresh node one layer deeper rather than an edge pointing back.
        #
        # The loop below renders only the abstract transitions the refiner
        # actually instantiated with concrete states; the grafting pass that
        # follows extends this to the entire abstract search graph.
        incoming_depth, last_mapped = self._abstract_depths(state_to_abstract)

        def abstract_nid(aid: int, depth: int) -> str:
            return f"s:{aid}_{depth}"

        # Short name for each realized abstract transition (src abstract id ->
        # dst abstract id), used to label abstract-action edges.
        abstract_action_name_by_pair: dict[tuple[int, int], str] = {}
        for src_abs, action, dst_abs in self.abstract_action_edges:
            pair = (
                self._abstract_state_to_id(src_abs),
                self._abstract_state_to_id(dst_abs),
            )
            abstract_action_name_by_pair[pair] = _abstract_action_name(action)

        # Build the depth-stamped abstract nodes and the abstractor / abstract-
        # action edges from the mapped concrete states.
        abstract_node_to_aid: dict[str, int] = {}
        abstract_node_depth: dict[str, int] = {}
        G_abstract: nx.DiGraph = nx.DiGraph()
        abstractor_edges: list[tuple[str, str]] = []
        abstract_action_specs: list[tuple[str, str, str | None]] = []
        rendered_action_edges: set[tuple[str, str]] = set()
        for cid, aid in state_to_abstract.items():
            node_id = abstract_nid(aid, incoming_depth[cid])
            abstract_node_to_aid[node_id] = aid
            abstract_node_depth[node_id] = incoming_depth[cid]
            G_abstract.add_node(node_id)
            abstractor_edges.append((concrete_nid(cid), node_id))
            parent_cid = last_mapped[cid]
            if parent_cid is not None:
                parent_aid = state_to_abstract[parent_cid]
                parent_node_id = abstract_nid(parent_aid, incoming_depth[parent_cid])
                if (parent_node_id, node_id) not in rendered_action_edges:
                    rendered_action_edges.add((parent_node_id, node_id))
                    G_abstract.add_edge(parent_node_id, node_id)
                    name = abstract_action_name_by_pair.get((parent_aid, aid))
                    abstract_action_specs.append((parent_node_id, node_id, name))

        # Grafting pass: extend the abstract plane from the refiner-instantiated
        # transitions above to the ENTIRE abstract search graph -- every abstract
        # state in ``self.abstract_states`` and every edge in
        # ``self.abstract_action_edges``, including branches the refiner never
        # sampled a concrete state for. Nothing new is generated here: this only
        # re-renders abstract states and actions the planner already explored.
        #
        # The same depth-unrolling rule applies: an edge into an already-seen
        # abstract state spawns a fresh node one layer deeper rather than a
        # back-edge. To keep this finite on cyclic abstract graphs, each abstract
        # state is expanded (its outgoing edges followed) only at its shallowest
        # rendered depth; deeper duplicates render as leaves. Processing nodes in
        # nondecreasing depth order guarantees the shallowest occurrence of each
        # abstract state is the one that gets expanded.
        abstract_adj: dict[int, list[tuple[int, str | None]]] = {}
        for src_abs, action, dst_abs in self.abstract_action_edges:
            s_aid = self._abstract_state_to_id(src_abs)
            d_aid = self._abstract_state_to_id(dst_abs)
            abstract_adj.setdefault(s_aid, []).append(
                (d_aid, _abstract_action_name(action))
            )

        # Min-heap of (depth, tiebreak, node_id); tiebreak keeps ordering stable
        # and avoids comparing node-id strings.
        heap: list[tuple[int, int, str]] = []
        tiebreak = 0
        for node_id, depth in abstract_node_depth.items():
            heapq.heappush(heap, (depth, tiebreak, node_id))
            tiebreak += 1
        expanded_aids: set[int] = set()
        while heap:
            depth, _, node_id = heapq.heappop(heap)
            aid = abstract_node_to_aid[node_id]
            if aid in expanded_aids:
                continue  # a shallower duplicate was already expanded
            expanded_aids.add(aid)
            for dst_aid, name in abstract_adj.get(aid, []):
                dst_node = abstract_nid(dst_aid, depth + 1)
                if (node_id, dst_node) in rendered_action_edges:
                    continue
                rendered_action_edges.add((node_id, dst_node))
                if dst_node not in abstract_node_to_aid:
                    abstract_node_to_aid[dst_node] = dst_aid
                    abstract_node_depth[dst_node] = depth + 1
                    G_abstract.add_node(dst_node)
                    heapq.heappush(heap, (depth + 1, tiebreak, dst_node))
                    tiebreak += 1
                G_abstract.add_edge(node_id, dst_node)
                abstract_action_specs.append((node_id, dst_node, name))

        # Any abstract state disconnected from the instantiated roots (no path
        # via abstract-action edges, e.g. a never-instantiated isolated state)
        # still gets a node so the plane shows every abstract state.
        rendered_aids = set(abstract_node_to_aid.values())
        for aid in range(len(self.abstract_states)):
            if aid not in rendered_aids:
                node_id = abstract_nid(aid, 0)
                abstract_node_to_aid[node_id] = aid
                abstract_node_depth[node_id] = 0
                G_abstract.add_node(node_id)

        # Lay out the abstract DAG (depth -> layer) and fit it into the concrete
        # plane's xy bounds so the two planes line up. When the concrete plane is
        # degenerate on an axis -- e.g. a single linear trajectory has zero
        # x-extent -- fitting would squash the whole abstract plane onto that
        # line and stack its branches on top of each other, so fall back to the
        # abstract plane's own box (it now carries the full, possibly branching,
        # search graph rather than a single refined path).
        abstract_layout = _hierarchical_layout(G_abstract)
        concrete_xs = [p[0] for p in layout_pos.values()]
        concrete_ys = [p[1] for p in layout_pos.values()]
        concrete_has_extent = (
            concrete_xs
            and concrete_ys
            and max(concrete_xs) - min(concrete_xs) > 1e-9
            and max(concrete_ys) - min(concrete_ys) > 1e-9
        )
        if concrete_has_extent:
            abstract_pos = _fit_into_bounds(
                abstract_layout,
                min(concrete_xs),
                max(concrete_xs),
                min(concrete_ys),
                max(concrete_ys),
            )
        else:
            abstract_pos = _center_and_scale(abstract_layout)
        for node_id in abstract_node_to_aid:
            ax, ay = abstract_pos.get(node_id, (0.0, 0.0))
            pos[node_id] = (ax, ay, z_top)
            G.add_node(node_id, type="abstract")

        # State-abstractor edges (concrete -> abstract, cross the z planes).
        for concrete_node, abstract_node in abstractor_edges:
            G.add_edge(concrete_node, abstract_node, type="abstractor")
        # Abstract-action edges (abstract -> abstract on z_top).
        for src_node, dst_node, name in abstract_action_specs:
            attrs: dict = {"type": "abstract_action"}
            if name is not None:
                attrs["name"] = name
            G.add_edge(src_node, dst_node, **attrs)

        # ------------------------------------------------------------------
        # Phase 5: plan membership + time index.
        # ------------------------------------------------------------------
        plan_node_ids: list[str] = []
        abstract_plan_node_ids: list[str] = []
        start_node: str | None = None
        goal_node: str | None = None
        if final_state is not None:
            seen_abstract: set[str] = set()
            for s in self.extract_plan(final_state).states:
                sid = self._state_to_id(s)
                if sid in kept_state_ids:
                    plan_node_ids.append(concrete_nid(sid))
                if sid in state_to_abstract:
                    s_node = abstract_nid(state_to_abstract[sid], incoming_depth[sid])
                    if s_node not in seen_abstract:
                        seen_abstract.add(s_node)
                        abstract_plan_node_ids.append(s_node)
            if plan_node_ids:
                start_node = plan_node_ids[0]
                goal_node = plan_node_ids[-1]
        plan_nodes_set = set(plan_node_ids)
        abstract_plan_nodes_set = set(abstract_plan_node_ids)

        # Time indices: one per rendered concrete node, in graph insertion order.
        rendered_concrete_ids: set[str] = {
            concrete_nid(sid) for sid in kept_state_ids if concrete_nid(sid) in pos
        }
        node_time_index: dict[str, int] = {}
        current_index = 1
        for state in self.states:
            nid = concrete_nid(self._state_to_id(state))
            if nid in rendered_concrete_ids:
                node_time_index[nid] = current_index
                current_index += 1

        # ------------------------------------------------------------------
        # Phase 6: emit the payload.
        # ------------------------------------------------------------------
        nodes: list[dict] = []
        for node_id, (px, py, pz) in pos.items():
            is_abstract = node_id.startswith("s:")
            in_plan = node_id in plan_nodes_set or node_id in abstract_plan_nodes_set
            node_dict: dict = {
                "id": node_id,
                "type": "abstract" if is_abstract else "concrete",
                "position": [px, py, pz],
                "in_plan": in_plan,
            }
            if not is_abstract:
                if node_id in node_time_index:
                    node_dict["time_index"] = node_time_index[node_id]
            else:
                aid = abstract_node_to_aid[node_id]
                atom_strs = _abstract_state_atom_strs(self.abstract_states[aid])
                if atom_strs is not None:
                    node_dict["atoms"] = atom_strs
            nodes.append(node_dict)

        edges: list[dict] = []
        for u, v in G.edges():
            edge_data = G.edges[u, v]
            edge_dict: dict = {"source": u, "target": v, "type": edge_data["type"]}
            if "name" in edge_data:
                edge_dict["name"] = edge_data["name"]
            edges.append(edge_dict)

        plan_info = {
            "nodes": plan_node_ids,
            "start": start_node,
            "goal": goal_node,
        }

        config: dict = {"z_top": z_top, "z_bottom": z_bottom}
        if node_time_index:
            config["min_time"] = 1
            config["max_time"] = current_index - 1

        state_data = {f"x:{idx}": str(state) for idx, state in enumerate(self.states)}

        return {
            "nodes": nodes,
            "edges": edges,
            "plan": plan_info,
            "config": config,
            "state_data": state_data,
        }

    def export(
        self,
        path: Path,
        final_state: _X | None = None,
        constant_state: ObjectCentricState | None = None,
        **kwargs,
    ) -> None:
        """Write a single pickle bundling graph topology and state objects.

        The resulting pickle is a dict with three keys:

          * ``"graph"``: the frontend-facing topology dict (nodes, edges,
            plan info, config) produced by ``_build_graph_payload``. The
            backend serves this through ``GET /api/graph``.
          * ``"states"``: ``{node_id: state}``, mapping node ids of the form
            ``"x:<display_id>"`` to the original state objects. Display
            ids are insertion-order indices into ``self.states``. The
            backend indexes into this dict when rendering a specific node.
          * ``"constant_state"``: optional static objects (e.g. walls, a table)
            that the environment keeps separate from the per-step state. It is
            stored once here; the visualizer backend merges it into a state at
            render time (mirroring how the env merges constants only when it
            needs the full scene), so the renderer draws the full picture
            without duplicating the static objects into every state.
        """
        graph_payload = self._build_graph_payload(final_state=final_state, **kwargs)
        states_payload: dict[str, _X] = {
            f"x:{idx}": state for idx, state in enumerate(self.states)
        }
        bundle = {
            "graph": graph_payload,
            "states": states_payload,
            "constant_state": constant_state,
        }
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
