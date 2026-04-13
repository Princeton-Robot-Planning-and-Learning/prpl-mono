"""Bilevel planning graphs: primarily for visualization, analysis, debugging."""

import pickle
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

    def _build_graph_structure(
        self,
        final_state: _X | None = None,
        n_intermediate_per_side: int = 0,
    ) -> tuple[nx.DiGraph, dict[str, tuple[float, float, float]], dict]:
        """Build the graph structure for visualization.

        Returns:
            G: NetworkX DiGraph with nodes and edges
            pos: Dictionary mapping node IDs to (x, y, z) positions
            metadata: Dictionary with plan_nodes, start_node, goal_node, z_top, z_bottom

        Planned follow-ups (tracked on this helper because this is the only
        call site; see also the comments at each individual concern):
          * Fold this method into ``_build_graph_payload``. The split exists
            for no reason — nothing else calls ``_build_graph_structure``.
          * Remove the ``n_intermediate_per_side`` knob. It defaults to 0, so
            the interpolation branch below is dead code today, and deciding
            how many intermediate nodes to show should happen in the viewer.
          * Replace the force-directed layout with a hierarchical layer
            assignment (BFS from roots) for DAG-shaped graphs.
        """

        G: nx.DiGraph = nx.DiGraph()
        pos: dict[str, tuple[float, float, float]] = {}
        metadata: dict = {}

        # Analyze graph topology to identify critical nodes to keep
        # for visualization purposes
        adj: dict[int, set[int]] = {}  # state_id -> list of next state_ids
        rev_adj: dict[int, set[int]] = {}  # state_id -> list of prev state_ids

        for source_state, _, target_state in self.action_edges:
            uid = self._state_to_id(source_state)
            vid = self._state_to_id(target_state)

            adj.setdefault(uid, set()).add(vid)
            rev_adj.setdefault(vid, set()).add(uid)

        # Identify Plan Nodes
        plan_state_ids = set()
        if final_state is not None:
            plan = self.extract_plan(final_state)
            for state in plan.states:
                plan_state_ids.add(self._state_to_id(state))

        # Track which concrete nodes have mapping to abstract nodes
        state_to_abstract = {}
        for x, s in self.state_abstractor_edges:
            state_to_abstract[self._state_to_id(x)] = self._abstract_state_to_id(s)

        # Determine Kept Nodes
        kept_state_ids = set()

        for state in self.states:
            sid = self._state_to_id(state)
            in_d = len(rev_adj.get(sid, set()))
            out_d = len(adj.get(sid, set()))

            # Non-critical nodes: indegree = 1 AND outdegree = 1
            # i.e., intermediate nodes that represent motion planning steps
            is_critical = (
                in_d == 0  # Root
                or out_d == 0  # Leaf
                or out_d > 1  # Branch
                or in_d > 1  # Merge
                or sid in state_to_abstract  # Has abstract mapping
            )

            if is_critical:
                kept_state_ids.add(sid)

        # Build layout graph with only critical nodes
        # Then we'll add back some pruned nodes later by interpolating along edges
        G_layout: nx.DiGraph = nx.DiGraph()

        # Record pruned segments for interpolation later
        # Ex: if critical nodes A, D pass through intermediate pruned nodes B, C
        # then forward_segments[(A, D)] = [B, C]
        forward_segments: dict[tuple[int, int], list[int]] = {}

        for sid in kept_state_ids:
            G_layout.add_node(f"x:{sid}")

        # Stitch edges of the kept nodes together, and record the
        # intermediate pruned nodes along the way for later interpolation
        for sid in kept_state_ids:
            children: set[int] = adj.get(sid, set())
            for child_id in children:
                curr = child_id
                path: list[int] = []
                visited: set[int] = set()
                # Traverse down pruned nodes (degree-1 chains)
                while (
                    curr not in kept_state_ids and curr in adj and curr not in visited
                ):
                    visited.add(curr)
                    path.append(curr)
                    next_nodes = adj[curr]
                    if not next_nodes:
                        break
                    # curr = next_nodes[0]
                    curr = next(iter(next_nodes))  # Move to the next node

                if curr in kept_state_ids:
                    G_layout.add_edge(f"x:{sid}", f"x:{curr}")
                    forward_segments[(sid, curr)] = path

        # Run layout on concrete nodes. spring_layout is a force-directed
        # placement: good enough to draw the graph without pulling in the
        # graphviz C library, but less readable than a hierarchical top-down
        # DAG layout. A follow-up can replace this with a BFS-based layer
        # assignment for DAG-shaped graphs.
        layout_pos: dict[str, tuple[float, float]] = {
            n: (float(xy[0]), float(xy[1]))
            for n, xy in nx.spring_layout(G_layout, seed=0).items()
        }

        # Post-process layout: center and scale into a [-10, 10] box.
        #
        # Planned follow-up: the aspect-ratio stretching below is a hack
        # calibrated against graphviz "dot" output and is only loosely
        # meaningful for a force-directed layout. When we switch to a
        # hierarchical layer layout, this whole block should be replaced
        # with a single uniform scale-to-box, and the frontend should own
        # any display-time aspect handling.
        if layout_pos:
            xs = [p[0] for p in layout_pos.values()]
            ys = [p[1] for p in layout_pos.values()]

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            width = max_x - min_x if max_x != min_x else 1.0
            height = max_y - min_y if max_y != min_y else 1.0

            center_x = (min_x + max_x) / 2.0
            center_y = (min_y + max_y) / 2.0

            target_aspect = 1.5
            current_aspect = width / height

            y_scale = 1.0
            if current_aspect > target_aspect:
                y_scale = current_aspect / target_aspect

            max_dim = max(width, height * y_scale)
            scale_factor = 20.0 / max_dim if max_dim > 0 else 1.0

            for nid, (layout_x, layout_y) in layout_pos.items():
                new_x = (layout_x - center_x) * scale_factor
                new_y = (layout_y - center_y) * y_scale * scale_factor
                layout_pos[nid] = (new_x, new_y)

        # Construct Final G and pos
        # First, add all critical (kept) concrete nodes with their layout positions
        for sid in kept_state_ids:
            nid = f"x:{sid}"
            if nid in layout_pos:
                pos[nid] = (layout_pos[nid][0], layout_pos[nid][1], 0.0)
                G.add_node(nid, type="concrete")

        # For each stitched edge between critical nodes, optionally add
        # a bounded number of intermediate nodes that are evenly spread along
        # the original chain between the endpoints, and interpolate positions
        # along the segment.
        #
        # Planned follow-up: this loop is gated on ``n_intermediate_per_side``
        # which defaults to 0, so in practice the interpolation branch never
        # runs. Either wire the knob through as an actual feature or delete
        # it. The viewer is a better place to decide how much of the pruned
        # chain to surface, since the user can pan/zoom.
        for u, v in G_layout.edges():
            u_sid = int(u.split(":")[1])
            v_sid = int(v.split(":")[1])

            full_path = forward_segments.get((u_sid, v_sid), [])

            # Select up to n_intermediate_per_side nodes, evenly distributed
            # along the chain from u_sid to v_sid.
            selected: list[int] = []
            if full_path:
                k = min(n_intermediate_per_side, len(full_path))
                if k == len(full_path):
                    selected = full_path[:]
                else:
                    step = len(full_path) / (k + 1)
                    used: set[int] = set()
                    for i in range(1, k + 1):
                        idx = int(round(i * step)) - 1
                        idx = max(0, min(idx, len(full_path) - 1))
                        mid_sid = full_path[idx]
                        if mid_sid not in used:
                            used.add(mid_sid)
                            selected.append(mid_sid)

            # Build the visualization sequence: [u, selected..., v]
            seq: list[int] = [u_sid] + selected + [v_sid]

            # Interpolate positions along the line from u to v
            ux, uy = layout_pos[u]
            vx, vy = layout_pos[v]
            z = 0.0
            segments = max(len(seq) - 1, 1)

            for i, sid in enumerate(seq):
                nid = f"x:{sid}"
                if nid not in pos:
                    t = i / segments
                    interp_x = ux + t * (vx - ux)
                    interp_y = uy + t * (vy - uy)
                    pos[nid] = (interp_x, interp_y, z)
                    G.add_node(nid, type="concrete")

            # Add edges along the expanded sequence
            for a, b in zip(seq[:-1], seq[1:]):
                G.add_edge(f"x:{a}", f"x:{b}", type="action")

        # Add abstract state nodes on the z_top plane. Each abstract node
        # is placed at the xy centroid of the kept concrete nodes that
        # group under it; abstract states with no kept concrete members
        # fall back to the origin.
        z_top_value = 1.0
        abstract_members: dict[int, list[str]] = {}
        for cid, aid in state_to_abstract.items():
            if cid in kept_state_ids:
                abstract_members.setdefault(aid, []).append(f"x:{cid}")

        for abstract_idx in range(len(self.abstract_states)):
            abs_nid = f"s:{abstract_idx}"
            members = abstract_members.get(abstract_idx, [])
            if members:
                xs = [pos[m][0] for m in members]
                ys = [pos[m][1] for m in members]
                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)
            else:
                cx, cy = 0.0, 0.0
            pos[abs_nid] = (cx, cy, z_top_value)
            G.add_node(abs_nid, type="abstract")

        # State-abstractor edges: concrete -> abstract, crossing the z planes.
        for cid, aid in state_to_abstract.items():
            if cid in kept_state_ids:
                G.add_edge(f"x:{cid}", f"s:{aid}", type="abstractor")

        # Abstract action edges: abstract -> abstract on the z_top plane.
        for src_abs, _action, dst_abs in self.abstract_action_edges:
            src_idx = self._abstract_state_to_id(src_abs)
            dst_idx = self._abstract_state_to_id(dst_abs)
            G.add_edge(f"s:{src_idx}", f"s:{dst_idx}", type="abstract_action")

        # Compute time indices for rendered nodes only
        # Map underlying state_id -> rendered node id ("x:{state_id}")
        rendered_state_to_node: dict[int, str] = {}
        for nid in G.nodes:
            if isinstance(nid, str) and nid.startswith("x:"):
                try:
                    sid = int(nid.split(":")[1])
                except (IndexError, ValueError):
                    continue
                rendered_state_to_node[sid] = nid

        node_time_index: dict[str, int] = {}
        current_index = 1
        # Iterate over states in insertion order; assign indices only to rendered ones
        for state in self.states:
            sid = self._state_to_id(state)
            rendered_nid = rendered_state_to_node.get(sid)
            if rendered_nid is None:
                continue
            node_time_index[rendered_nid] = current_index
            current_index += 1

        if node_time_index:
            min_time = 1
            max_time = current_index - 1
        else:
            min_time = None
            max_time = None

        # Metadata

        # z_top holds the abstract-state plane; z_bottom the concrete plane.
        # The frontend's plotly layout reads these for the zaxis range.
        metadata["z_top"] = z_top_value
        metadata["z_bottom"] = 0.0

        # node information
        metadata["plan_nodes"] = []
        metadata["abstract_plan_nodes"] = []
        metadata["start_node"] = None
        metadata["goal_node"] = None
        metadata["state_to_abstract_id"] = state_to_abstract
        # order in which filtered nodes were added to bpg
        metadata["node_time_index"] = node_time_index

        # timeline index range (used for slider range in plotly)
        metadata["min_time"] = min_time
        metadata["max_time"] = max_time

        if final_state is not None:
            # Only include plan nodes that were kept
            plan_nodes = []
            for s in self.extract_plan(final_state).states:
                sid = self._state_to_id(s)
                if sid in kept_state_ids:
                    plan_nodes.append(f"x:{sid}")

            metadata["plan_nodes"] = plan_nodes
            if plan_nodes:
                metadata["start_node"] = plan_nodes[0]
                metadata["goal_node"] = plan_nodes[-1]

            # Identify abstract plan nodes
            abstract_plan_nodes = []
            for x_node in plan_nodes:
                x_id = int(x_node.split(":")[1])
                if x_id in state_to_abstract:
                    s_id = state_to_abstract[x_id]
                    s_node = f"s:{s_id}"
                    if s_node not in abstract_plan_nodes:
                        abstract_plan_nodes.append(s_node)
            metadata["abstract_plan_nodes"] = abstract_plan_nodes

        return G, pos, metadata

    def _build_graph_payload(self, final_state: _X | None = None) -> dict:
        """Build the frontend-facing topology dict (nodes, edges, plan, config).

        The payload carries graph topology and plan membership only. The
        frontend chooses colors, sizes, and alphas from ``node.type`` (the
        ``"concrete"`` / ``"abstract"`` distinction), ``node.in_plan``, and
        ``edge.type`` (``"action"`` / ``"abstractor"`` / ``"abstract_action"``).

        Planned follow-up: fold ``_build_graph_structure`` into this method;
        it has no other caller.
        """
        G, pos, metadata = self._build_graph_structure(final_state=final_state)

        plan_nodes_set = set(metadata["plan_nodes"])
        abstract_plan_nodes_set = set(metadata["abstract_plan_nodes"])
        node_time_index: dict[str, int] = metadata.get("node_time_index", {})

        nodes = []
        for node_id, (x, y, z) in pos.items():
            is_abstract = node_id.startswith("s:")
            in_plan = node_id in plan_nodes_set or node_id in abstract_plan_nodes_set

            node_dict: dict = {
                "id": node_id,
                "type": "abstract" if is_abstract else "concrete",
                "position": [x, y, z],
                "in_plan": in_plan,
            }

            # Attach time index for rendered concrete nodes.
            if not is_abstract and node_id in node_time_index:
                node_dict["time_index"] = node_time_index[node_id]

            # Attach the owning abstract state id for concrete nodes that
            # have a state-abstractor edge, so the frontend can highlight
            # abstract groupings.
            if not is_abstract and node_id.startswith("x:"):
                state_id = int(node_id.split(":")[1])
                if state_id in metadata["state_to_abstract_id"]:
                    node_dict["abstract_state_id"] = metadata["state_to_abstract_id"][
                        state_id
                    ]

            nodes.append(node_dict)

        # Build edges list. The type lets the frontend pick styling.
        edges = []
        for source, target in G.edges():
            source_pos = pos[source]
            target_pos = pos[target]
            if source_pos[2] != target_pos[2]:
                edge_type = "abstractor"
            elif source_pos[2] == metadata["z_bottom"]:
                edge_type = "action"
            else:
                edge_type = "abstract_action"

            edges.append({"source": source, "target": target, "type": edge_type})

        # Build plan info
        plan_info = {
            "nodes": metadata["plan_nodes"],
            "start": metadata["start_node"],
            "goal": metadata["goal_node"],
        }

        # Build config
        config = {
            "z_top": metadata["z_top"],
            "z_bottom": metadata["z_bottom"],
        }

        # Include time index range in config if available
        if (
            metadata.get("min_time") is not None
            and metadata.get("max_time") is not None
        ):
            config["min_time"] = metadata["min_time"]
            config["max_time"] = metadata["max_time"]

        # Build state data dictionary (id -> state repr)
        # Convert state objects to their string representation for JSON serialization
        state_data = {}
        for state in self.states:
            state_id = self._state_to_id(state)
            node_id = f"x:{state_id}"
            # Convert state to string representation for JSON
            state_data[node_id] = str(state)

        graph_data = {
            "nodes": nodes,
            "edges": edges,
            "plan": plan_info,
            "config": config,
            "state_data": state_data,
        }

        return graph_data

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
