"""Bilevel planning graphs: primarily for visualization, analysis, debugging."""

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
