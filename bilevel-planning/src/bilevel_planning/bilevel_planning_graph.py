"""Bilevel planning graphs: primarily for visualization, analysis, debugging."""

import pickle
from collections import deque
from pathlib import Path
from typing import Callable, Generic, Hashable, TypeVar

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from prpl_utils.utils import consistent_hash

from bilevel_planning.structs import Plan

_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action
_S = TypeVar("_S", bound=Hashable)  # abstract state
_A = TypeVar("_A", bound=Hashable)  # abstract action


def _hierarchical_layout(graph: nx.DiGraph) -> dict[str, tuple[float, float]]:
    """Assign each node an (x, y) in a top-down layered arrangement.

    Each node's layer is its longest-path distance from any root (node with in_degree 0)
    when the graph is a DAG, or its shortest-path distance via BFS otherwise. Nodes
    within a layer are spread evenly along x. Roots sit at y=0 and descendants descend
    (y decreases).

    Coordinates are unscaled; the caller is expected to center and rescale them to its
    preferred display box.
    """
    if not graph.nodes:
        return {}

    roots = [n for n in graph.nodes if graph.in_degree(n) == 0]
    if not roots:
        # Pure cycle (or no identifiable root): pick any node to start.
        roots = [next(iter(graph.nodes))]

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

    # Group by layer and spread evenly along x. Sort within a layer for
    # deterministic output across runs.
    layers: dict[int, list[str]] = {}
    for node, layer in layer_of.items():
        layers.setdefault(layer, []).append(node)

    positions: dict[str, tuple[float, float]] = {}
    for layer, nodes_in_layer in layers.items():
        nodes_in_layer.sort()
        count = len(nodes_in_layer)
        for i, node in enumerate(nodes_in_layer):
            x = 0.0 if count == 1 else (i / (count - 1) - 0.5) * 2.0
            positions[node] = (x, float(-layer))
    return positions


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

    def _build_graph_payload(self, final_state: _X | None = None) -> dict:
        """Build the frontend-facing topology dict for the visualizer.

        Returns a dict with ``nodes``, ``edges``, ``plan``, ``config``, and
        ``state_data``. The payload carries graph topology and plan
        membership only — the frontend chooses colors, sizes, and alphas
        from ``node.type`` (``"concrete"`` / ``"abstract"``),
        ``node.in_plan``, and ``edge.type`` (``"action"`` / ``"abstractor"``
        / ``"abstract_action"``).
        """
        # ------------------------------------------------------------------
        # Phase 1: analyze topology and pick which concrete nodes to render.
        # ------------------------------------------------------------------
        adj: dict[int, set[int]] = {}  # state_id -> next state_ids
        rev_adj: dict[int, set[int]] = {}  # state_id -> previous state_ids
        for source_state, _, target_state in self.action_edges:
            uid = self._state_to_id(source_state)
            vid = self._state_to_id(target_state)
            adj.setdefault(uid, set()).add(vid)
            rev_adj.setdefault(vid, set()).add(uid)

        state_to_abstract: dict[int, int] = {}
        for x, s in self.state_abstractor_edges:
            state_to_abstract[self._state_to_id(x)] = self._abstract_state_to_id(s)

        # Keep nodes that are not part of a degree-1 chain (roots, leaves,
        # branches, merges, or any node with an abstract mapping).
        kept_state_ids: set[int] = set()
        for state in self.states:
            sid = self._state_to_id(state)
            in_d = len(rev_adj.get(sid, set()))
            out_d = len(adj.get(sid, set()))
            is_critical = (
                in_d == 0
                or out_d == 0
                or out_d > 1
                or in_d > 1
                or sid in state_to_abstract
            )
            if is_critical:
                kept_state_ids.add(sid)

        # ------------------------------------------------------------------
        # Phase 2: stitch edges between kept nodes, skipping pruned chains.
        # ------------------------------------------------------------------
        G_layout: nx.DiGraph = nx.DiGraph()
        for sid in kept_state_ids:
            G_layout.add_node(f"x:{sid}")
        for sid in kept_state_ids:
            for child_id in adj.get(sid, set()):
                curr = child_id
                visited: set[int] = set()
                while (
                    curr not in kept_state_ids and curr in adj and curr not in visited
                ):
                    visited.add(curr)
                    next_nodes = adj[curr]
                    if not next_nodes:
                        break
                    curr = next(iter(next_nodes))
                if curr in kept_state_ids:
                    G_layout.add_edge(f"x:{sid}", f"x:{curr}")

        # ------------------------------------------------------------------
        # Phase 3: lay out the kept concrete nodes, scale into [-10, 10].
        # ------------------------------------------------------------------
        # Hierarchical layer layout: assign each node a layer via
        # longest-path depth from any root (or shortest-path BFS when the
        # graph has cycles), then spread nodes evenly within each layer.
        # Roots sit at top (y=0), descendants descend. Produces a clean
        # top-down DAG read for BPGs produced by a planner.
        layout_pos = _hierarchical_layout(G_layout)

        # Uniform scale into a [-10, 10] box, preserving aspect.
        if layout_pos:
            xs = [p[0] for p in layout_pos.values()]
            ys = [p[1] for p in layout_pos.values()]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            width = max_x - min_x
            height = max_y - min_y
            center_x = (min_x + max_x) / 2.0
            center_y = (min_y + max_y) / 2.0
            max_dim = max(width, height, 1.0)
            scale_factor = 20.0 / max_dim
            for nid, (layout_x, layout_y) in layout_pos.items():
                layout_pos[nid] = (
                    (layout_x - center_x) * scale_factor,
                    (layout_y - center_y) * scale_factor,
                )

        # ------------------------------------------------------------------
        # Phase 4: build the (G, pos) triple we'll emit.
        # ------------------------------------------------------------------
        G: nx.DiGraph = nx.DiGraph()
        pos: dict[str, tuple[float, float, float]] = {}

        z_bottom = 0.0
        z_top = 1.0

        # Kept concrete nodes on the z_bottom plane.
        for sid in kept_state_ids:
            nid = f"x:{sid}"
            if nid in layout_pos:
                pos[nid] = (layout_pos[nid][0], layout_pos[nid][1], z_bottom)
                G.add_node(nid, type="concrete")

        # Action edges between kept concrete nodes.
        for u, v in G_layout.edges():
            G.add_edge(u, v, type="action")

        # Abstract state nodes on the z_top plane, positioned at the xy
        # centroid of their kept concrete members.
        abstract_members: dict[int, list[str]] = {}
        for cid, aid in state_to_abstract.items():
            if cid in kept_state_ids:
                abstract_members.setdefault(aid, []).append(f"x:{cid}")
        for abstract_idx in range(len(self.abstract_states)):
            abs_nid = f"s:{abstract_idx}"
            members = abstract_members.get(abstract_idx, [])
            if members:
                cx = sum(pos[m][0] for m in members) / len(members)
                cy = sum(pos[m][1] for m in members) / len(members)
            else:
                cx, cy = 0.0, 0.0
            pos[abs_nid] = (cx, cy, z_top)
            G.add_node(abs_nid, type="abstract")

        # State-abstractor edges (concrete -> abstract, cross the z planes)
        # and abstract-action edges (abstract -> abstract on z_top).
        for cid, aid in state_to_abstract.items():
            if cid in kept_state_ids:
                G.add_edge(f"x:{cid}", f"s:{aid}", type="abstractor")
        for src_abs, _action, dst_abs in self.abstract_action_edges:
            src_idx = self._abstract_state_to_id(src_abs)
            dst_idx = self._abstract_state_to_id(dst_abs)
            G.add_edge(f"s:{src_idx}", f"s:{dst_idx}", type="abstract_action")

        # ------------------------------------------------------------------
        # Phase 5: plan membership + time index.
        # ------------------------------------------------------------------
        plan_node_ids: list[str] = []
        abstract_plan_node_ids: list[str] = []
        start_node: str | None = None
        goal_node: str | None = None
        if final_state is not None:
            for s in self.extract_plan(final_state).states:
                sid = self._state_to_id(s)
                if sid in kept_state_ids:
                    plan_node_ids.append(f"x:{sid}")
            if plan_node_ids:
                start_node = plan_node_ids[0]
                goal_node = plan_node_ids[-1]
            seen_abstract: set[str] = set()
            for x_node in plan_node_ids:
                x_id = int(x_node.split(":")[1])
                if x_id in state_to_abstract:
                    s_node = f"s:{state_to_abstract[x_id]}"
                    if s_node not in seen_abstract:
                        seen_abstract.add(s_node)
                        abstract_plan_node_ids.append(s_node)
        plan_nodes_set = set(plan_node_ids)
        abstract_plan_nodes_set = set(abstract_plan_node_ids)

        # Time indices: one per rendered concrete node, in graph insertion order.
        rendered_concrete_ids: set[str] = {
            f"x:{sid}" for sid in kept_state_ids if f"x:{sid}" in pos
        }
        node_time_index: dict[str, int] = {}
        current_index = 1
        for state in self.states:
            nid = f"x:{self._state_to_id(state)}"
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
                state_id = int(node_id.split(":")[1])
                if state_id in state_to_abstract:
                    node_dict["abstract_state_id"] = state_to_abstract[state_id]
            nodes.append(node_dict)

        edges: list[dict] = [
            {"source": u, "target": v, "type": G.edges[u, v]["type"]}
            for u, v in G.edges()
        ]

        plan_info = {
            "nodes": plan_node_ids,
            "start": start_node,
            "goal": goal_node,
        }

        config: dict = {"z_top": z_top, "z_bottom": z_bottom}
        if node_time_index:
            config["min_time"] = 1
            config["max_time"] = current_index - 1

        state_data = {f"x:{self._state_to_id(s)}": str(s) for s in self.states}

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
        **kwargs,
    ) -> None:
        """Write a single pickle bundling graph topology and state objects.

        The resulting pickle is a dict with two keys:

          * ``"graph"``: the frontend-facing topology dict (nodes, edges,
            plan info, config) produced by ``_build_graph_payload``. The
            backend serves this through ``GET /api/graph``.
          * ``"states"``: ``{node_id: state}``, mapping node ids of the form
            ``"x:<state_id>"`` to the original state objects. The backend
            indexes into this when rendering a specific node.

        The topology half has string reprs of states under ``state_data`` for
        display; the real objects live in ``"states"`` so that rendering can
        reconstruct them without needing to unpickle topology data.
        """
        graph_payload = self._build_graph_payload(final_state=final_state, **kwargs)
        states_payload: dict[str, _X] = {}
        for state in self.states:
            state_id = self._state_to_id(state)
            states_payload[f"x:{state_id}"] = state

        bundle = {"graph": graph_payload, "states": states_payload}
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
