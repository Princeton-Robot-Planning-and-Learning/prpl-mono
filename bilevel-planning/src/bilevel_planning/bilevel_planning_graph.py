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
from networkx.drawing.nx_agraph import graphviz_layout

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
        self._state_id_to_state: dict[int, _X] = {}  # reverse lookup: id -> state, todo: remove _state_ids
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

    # TODO: make more efficient, use .append() and .reverse()
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
        """Get a state from its integer ID. Returns None if not found."""
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
        abstract_state_color: str = "tab:purple",
        state_color: str = "tab:blue",
        plan_color: str = "tab:green",
        node_size: int = 50,
        state_abstractor_edge_color: str = "gray",
        action_edge_color: str = "black",
        abstract_action_edge_color: str = "black",
        node_alpha: float = 0.7,
        edge_alpha: float = 0.7,
        n_intermediate_per_side: int = 0,
    ) -> tuple[nx.DiGraph, dict[str, tuple[float, float, float]], dict]:
        """Build the graph structure for visualization.

        Returns:
            G: NetworkX DiGraph with nodes and edges
            pos: Dictionary mapping node IDs to (x, y, z) positions
            metadata: Dictionary with plan_nodes, start_node, goal_node, z_top, z_bottom
        """

        G = nx.DiGraph()
        pos: dict[str, tuple[float, float, float]] = {}
        metadata: dict = {}

        # Analyze graph topology to identify critical nodes to keep
        # for visualization purposes
        adj: dict[int, set[int]] = {}  # state_id -> list of next state_ids
        rev_adj: dict[int, set[int]] = {}  # state_id -> list of prev state_ids
        
        for u, _, v in self.action_edges:
            uid = self._state_to_id(u)
            vid = self._state_to_id(v)
            
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
                in_d == 0 or  # Root
                out_d == 0 or  # Leaf
                out_d > 1 or  # Branch
                in_d > 1  or # Merge
                sid in state_to_abstract # Has abstract mapping
            )
            
            if is_critical:
                kept_state_ids.add(sid)

        # Build layout graph with only critical nodes
        # Then we'll add back some pruned nodes later by interpolating along edges
        G_layout = nx.DiGraph()

        # Record pruned segments for interpolation later
        # Ex: if critical nodes A, D pass through intermediate pruned nodes B, C
        # then forward_segments[(A, D)] = [B, C]
        forward_segments: dict[tuple[int, int], list[int]] = {}

        for sid in kept_state_ids:
            G_layout.add_node(f"x:{sid}")

        # Stitch edges of the kept nodes together, and record the 
        # intermediate pruned nodes along the way for later interpolation
        for sid in kept_state_ids:
            children = adj.get(sid, [])
            for child_id in children:
                curr = child_id
                path: list[int] = []
                visited: set[int] = set()
                # Traverse down pruned nodes (degree-1 chains)
                while curr not in kept_state_ids and curr in adj and curr not in visited:
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

        # Run Layout on Concrete Nodes
        layout_pos = graphviz_layout(G_layout, prog="dot", args="-Granksep=2.0 -Gnodesep=0.5")

        # Post-process layout: Center and Scale
        if layout_pos:
            xs = [p[0] for p in layout_pos.values()]
            ys = [p[1] for p in layout_pos.values()]
            
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            width = max_x - min_x if max_x != min_x else 1.0
            height = max_y - min_y if max_y != min_y else 1.0
            
            center_x = (min_x + max_x) / 2.0
            center_y = (min_y + max_y) / 2.0
            
            # Fix Aspect Ratio: If width >> height, stretch Y (depth)
            target_aspect = 1.5 # Allow some width, but not too much
            current_aspect = width / height
            
            y_scale = 1.0
            if current_aspect > target_aspect:
                # Stretch Y to match target aspect
                y_scale = current_aspect / target_aspect
                
            # Normalize to fit in a box of size [-10, 10]
            max_dim = max(width, height * y_scale)
            scale_factor = 20.0 / max_dim if max_dim > 0 else 1.0
            
            # Apply transformation (flip Y so roots appear at top)
            for nid, (x, y) in layout_pos.items():
                new_x = (x - center_x) * scale_factor
                new_y = (y - center_y) * y_scale * scale_factor
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
                    x = ux + t * (vx - ux)
                    y = uy + t * (vy - uy)
                    pos[nid] = (x, y, z)
                    G.add_node(nid, type="concrete")

            # Add edges along the expanded sequence
            for a, b in zip(seq[:-1], seq[1:]):
                G.add_edge(f"x:{a}", f"x:{b}", type="action")

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
            nid = rendered_state_to_node.get(sid)
            if nid is None:
                continue
            node_time_index[nid] = current_index
            current_index += 1

        if node_time_index:
            min_time = 1
            max_time = current_index - 1
        else:
            min_time = None
            max_time = None

        # Metadata

        # z_top, z_bottom previously used to visualize abstract nodes on a separate plane
        # currently only used to set zaxis range of plotly, but keep in case we want to add abstract nodes back in later
        metadata["z_top"] = 1.0
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
        
        # Pass through colors/sizes for export
        metadata["node_size"] = node_size
        metadata["edge_alpha"] = edge_alpha

        # coloring/visualization data
        metadata["plan_color"] = plan_color
        metadata["state_color"] = state_color
        metadata["abstract_state_color"] = abstract_state_color
        metadata["state_abstractor_edge_color"] = state_abstractor_edge_color
        metadata["action_edge_color"] = action_edge_color
        metadata["abstract_action_edge_color"] = abstract_action_edge_color
        metadata["node_alpha"] = node_alpha

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

    def export_graph_for_web(
        self,
        final_state: _X | None = None,
        abstract_state_color: str = "tab:purple",
        state_color: str = "tab:blue",
        plan_color: str = "tab:green",
        node_size: int = 50,
        state_abstractor_edge_color: str = "gray",
        action_edge_color: str = "black",
        abstract_action_edge_color: str = "black",
        node_alpha: float = 0.7,
        edge_alpha: float = 0.7,
    ) -> dict:
        """
        Export graph structure as JSON-serializable dictionary for web frontend.
        Returns a dictionary with nodes, edges, plan info, and configuration.
        """
        # Build the graph structure
        G, pos, metadata = self._build_graph_structure(
            final_state=final_state,
            abstract_state_color=abstract_state_color,
            state_color=state_color,
            plan_color=plan_color,
            node_size=node_size,
            state_abstractor_edge_color=state_abstractor_edge_color,
            action_edge_color=action_edge_color,
            abstract_action_edge_color=abstract_action_edge_color,
            node_alpha=node_alpha,
            edge_alpha=edge_alpha
        )

        # Color mapping from matplotlib to RGB
        color_map = {
            "tab:purple": "rgb(148, 103, 189)",
            "tab:blue": "rgb(31, 119, 180)",
            "tab:green": "rgb(44, 160, 44)",
            "gray": "rgb(128, 128, 128)",
            "black": "rgb(0, 0, 0)",
        }

        def convert_color(mpl_color: str) -> str:
            return color_map.get(mpl_color, mpl_color)

        # Build nodes list
        nodes = []
        plan_nodes_set = set(metadata["plan_nodes"])
        abstract_plan_nodes_set = set(metadata["abstract_plan_nodes"])
        node_time_index: dict[str, int] = metadata.get("node_time_index", {})

        for node_id, (x, y, z) in pos.items():
            is_abstract = node_id.startswith("s:")
            in_plan = node_id in plan_nodes_set
            in_abstract_plan = node_id in abstract_plan_nodes_set

            # Create node dict first to check for abstract association
            node_dict = {
                "id": node_id,
                "type": "abstract" if is_abstract else "concrete",
                "position": [x, y, z],
                "size": metadata["node_size"],
                "in_plan": in_plan or in_abstract_plan,
            }

            # Attach time index for rendered concrete nodes
            if not is_abstract and node_id in node_time_index:
                node_dict["time_index"] = node_time_index[node_id]
            
            # Add abstract state ID for concrete nodes (if associated)
            if not is_abstract and node_id.startswith("x:"):
                state_id = int(node_id.split(":")[1])
                if state_id in metadata["state_to_abstract_id"]:
                    node_dict["abstract_state_id"] = metadata["state_to_abstract_id"][state_id]


            # Color based on node type, plan membership, and abstract association
            if in_plan:
                # Plan coloring
                color = "rgb(255, 165, 0)" if "abstract_state_id" in node_dict else convert_color(metadata["plan_color"])
                alpha = 1.0
            else:
                # Non-plan coloring...
                if node_dict["type"] == "concrete":
                    color = convert_color(metadata["abstract_state_color"]) if "abstract_state_id" in node_dict else convert_color(metadata["state_color"])
                else:
                    color = convert_color(metadata["abstract_state_color"])
                alpha = metadata["node_alpha"]

            node_dict["color"] = color
            node_dict["alpha"] = alpha
            
            nodes.append(node_dict)

        # Build edges list
        edges = []
        for source, target in G.edges():
            source_pos = pos[source]
            target_pos = pos[target]
            if source_pos[2] != target_pos[2]:
                edge_type = "abstractor"
                color = convert_color(metadata["state_abstractor_edge_color"])
            elif source_pos[2] == metadata["z_bottom"]:
                edge_type = "action"
                color = convert_color(metadata["action_edge_color"])
            else:
                edge_type = "abstract_action"
                color = convert_color(metadata["abstract_action_edge_color"])

            edges.append({
                "source": source,
                "target": target,
                "type": edge_type,
                "color": color,
                "alpha": metadata["edge_alpha"],
            })

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
        if metadata.get("min_time") is not None and metadata.get("max_time") is not None:
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
    
    
    def export_state_data_pickle(self, pickle_path: Path) -> None:
        """
        Export pickled state data dictionary for backend use.  
        This saves a dictionary mapping node IDs to actual state objects (not strings).
        The backend can load this pickle to access the original state representations.
        """
        # Build state data dictionary with actual state objects
        state_data_pickle = {}
        for state in self.states:
            state_id = self._state_to_id(state)
            node_id = f"x:{state_id}"
            # Store the actual state object, not string representation
            state_data_pickle[node_id] = state
        
        # Save as pickle
        with open(pickle_path, 'wb') as f:
            pickle.dump(state_data_pickle, f)
    
    def export_graph_with_pickle(
        self,
        json_path: Path,
        pickle_path: Path,
        final_state: _X | None = None,
        **kwargs
    ) -> dict:
        """
        Export graph to JSON and pickle state data.
        """
        import json
        
        # Export JSON graph data
        graph_data = self.export_graph_for_web(final_state=final_state, **kwargs)
        
        with open(json_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        # Export pickled state data
        self.export_state_data_pickle(pickle_path)
        
        return graph_data