"""Motion planning utilities."""

from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Iterable, Optional, TypeVar

import numpy as np

_RRTState = TypeVar("_RRTState")


@dataclass
class MotionPlanningMetrics:
    """Metrics recorded and returned by motion planning algorithms."""

    num_nodes_extended: int = 0  # number of collision-free nodes added to the tree(s)
    num_collision_checks: int = 0  # number of calls to collision_fn


class RRT(Generic[_RRTState]):
    """Rapidly-exploring random tree."""

    def __init__(
        self,
        sample_fn: Callable[[_RRTState], _RRTState],
        extend_fn: Callable[[_RRTState, _RRTState], Iterable[_RRTState]],
        collision_fn: Callable[[_RRTState], bool],
        distance_fn: Callable[[_RRTState, _RRTState], float],
        rng: np.random.Generator,
        num_attempts: int,
        num_iters: int,
        smooth_amt: int,
    ):
        self._sample_fn = sample_fn
        self._extend_fn = extend_fn
        self._collision_fn = collision_fn
        self._distance_fn = distance_fn
        self._rng = rng
        self._num_attempts = num_attempts
        self._num_iters = num_iters
        self._smooth_amt = smooth_amt

    def query(
        self, pt1: _RRTState, pt2: _RRTState, sample_goal_eps: float = 0.0
    ) -> tuple[Optional[list[_RRTState]], MotionPlanningMetrics]:
        """Query the RRT, to get a collision-free path from pt1 to pt2.

        If no path is found, returns None. Always returns metrics.
        """
        metrics = MotionPlanningMetrics()
        result: Optional[list[_RRTState]] = None
        if not self._checked_collision(pt1, metrics) and not self._checked_collision(
            pt2, metrics
        ):
            direct_path = self.try_direct_path(pt1, pt2, metrics)
            if direct_path is not None:
                result = direct_path
            else:
                for _ in range(self._num_attempts):
                    path = self._rrt_connect(
                        pt1,
                        metrics,
                        goal_sampler=lambda: pt2,
                        sample_goal_eps=sample_goal_eps,
                    )
                    if path is not None:
                        result = self._smooth_path(path, metrics)
                        break
        if metrics_path := os.environ.get("birrt_metrics_path"):
            dump_path = Path(metrics_path)
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            with dump_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "num_nodes_extended": metrics.num_nodes_extended,
                            "num_collision_checks": metrics.num_collision_checks,
                        }
                    )
                    + "\n"
                )
        return result, metrics

    def query_to_goal_fn(
        self,
        start: _RRTState,
        goal_fn: Callable[[_RRTState], bool],
        goal_sampler: Callable[[], _RRTState] | None = None,
        sample_goal_eps: float = 0.0,
    ) -> tuple[Optional[list[_RRTState]], MotionPlanningMetrics]:
        """Query the RRT, to get a collision-free path from start to a point such that
        goal_fn(point) is True. Uses goal_sampler to sample a target for a direct path
        or with probability sample_goal_eps.

        If no path is found, returns None. Always returns metrics.
        """
        assert sample_goal_eps == 0.0 or goal_sampler is not None
        metrics = MotionPlanningMetrics()
        if self._checked_collision(start, metrics):
            return None, metrics
        if goal_sampler:
            direct_path = self.try_direct_path(start, goal_sampler(), metrics)
            if direct_path is not None:
                return direct_path, metrics
        for _ in range(self._num_attempts):
            path = self._rrt_connect(
                start, metrics, goal_sampler, goal_fn, sample_goal_eps=sample_goal_eps
            )
            if path is not None:
                return self._smooth_path(path, metrics), metrics
        return None, metrics

    def _checked_collision(self, pt: _RRTState, metrics: MotionPlanningMetrics) -> bool:
        """Call collision_fn and increment the counter."""
        metrics.num_collision_checks += 1
        return self._collision_fn(pt)

    def try_direct_path(
        self,
        pt1: _RRTState,
        pt2: _RRTState,
        metrics: Optional[MotionPlanningMetrics] = None,
    ) -> Optional[list[_RRTState]]:
        """Attempt to plan a direct path from pt1 to pt2, returning None if no
        collision-free path can be found."""
        if metrics is None:
            metrics = MotionPlanningMetrics()
        path = [pt1]
        for newpt in self._extend_fn(pt1, pt2):
            if self._checked_collision(newpt, metrics):
                return None
            path.append(newpt)
        return path

    def _rrt_connect(
        self,
        pt1: _RRTState,
        metrics: MotionPlanningMetrics,
        goal_sampler: Callable[[], _RRTState] | None = None,
        goal_fn: Callable[[_RRTState], bool] | None = None,
        sample_goal_eps: float = 0.0,
    ) -> Optional[list[_RRTState]]:
        root = _RRTNode(pt1)
        nodes = [root]
        metrics.num_nodes_extended += 1

        for _ in range(self._num_iters):
            # Sample the goal with a small probability, otherwise randomly
            # choose a point.
            sample_goal = self._rng.random() < sample_goal_eps
            if sample_goal:
                assert goal_sampler is not None
                samp = goal_sampler()
            else:
                samp = self._sample_fn(pt1)
            min_key = functools.partial(self._get_pt_dist_to_node, samp)
            nearest = min(nodes, key=min_key)
            reached_goal = False
            for newpt in self._extend_fn(nearest.data, samp):
                if self._checked_collision(newpt, metrics):
                    break
                nearest = _RRTNode(newpt, parent=nearest)
                nodes.append(nearest)
                metrics.num_nodes_extended += 1
            else:
                reached_goal = sample_goal
            # Check goal_fn if defined
            if reached_goal or goal_fn is not None and goal_fn(nearest.data):
                path = nearest.path_from_root()
                return [node.data for node in path]
        return None

    def _get_pt_dist_to_node(self, pt: _RRTState, node: _RRTNode[_RRTState]) -> float:
        return self._distance_fn(pt, node.data)

    def _smooth_path(
        self, path: list[_RRTState], metrics: MotionPlanningMetrics
    ) -> list[_RRTState]:
        assert len(path) > 2
        for _ in range(self._smooth_amt):
            i = self._rng.integers(0, len(path) - 1)
            j = self._rng.integers(0, len(path) - 1)
            if abs(i - j) <= 1:
                continue
            if j < i:
                i, j = j, i
            shortcut = list(self._extend_fn(path[i], path[j]))
            if len(shortcut) < j - i and all(
                not self._checked_collision(pt, metrics) for pt in shortcut
            ):
                path = path[: i + 1] + shortcut + path[j + 1 :]
        return path


class BiRRT(RRT[_RRTState]):
    """Bidirectional rapidly-exploring random tree."""

    def query_to_goal_fn(
        self,
        start: _RRTState,
        goal_fn: Callable[[_RRTState], bool],
        goal_sampler: Callable[[], _RRTState] | None = None,
        sample_goal_eps: float = 0.0,
    ) -> tuple[Optional[list[_RRTState]], MotionPlanningMetrics]:
        raise NotImplementedError("Can't query to goal function using BiRRT")

    def _rrt_connect(
        self,
        pt1: _RRTState,
        metrics: MotionPlanningMetrics,
        goal_sampler: Callable[[], _RRTState] | None = None,
        goal_fn: Callable[[_RRTState], bool] | None = None,
        sample_goal_eps: float = 0.0,
    ) -> Optional[list[_RRTState]]:
        # goal_fn and sample_goal_eps are unused
        assert goal_sampler is not None
        pt2 = goal_sampler()
        root1, root2 = _RRTNode(pt1), _RRTNode(pt2)
        nodes1, nodes2 = [root1], [root2]
        metrics.num_nodes_extended += 2

        for _ in range(self._num_iters):
            if len(nodes1) > len(nodes2):
                nodes1, nodes2 = nodes2, nodes1
            samp = self._sample_fn(pt1)
            min_key1 = functools.partial(self._get_pt_dist_to_node, samp)
            nearest1 = min(nodes1, key=min_key1)
            for newpt in self._extend_fn(nearest1.data, samp):
                if self._checked_collision(newpt, metrics):
                    break
                nearest1 = _RRTNode(newpt, parent=nearest1)
                nodes1.append(nearest1)
                metrics.num_nodes_extended += 1
            min_key2 = functools.partial(self._get_pt_dist_to_node, nearest1.data)
            nearest2 = min(nodes2, key=min_key2)
            for newpt in self._extend_fn(nearest2.data, nearest1.data):
                if self._checked_collision(newpt, metrics):
                    break
                nearest2 = _RRTNode(newpt, parent=nearest2)
                nodes2.append(nearest2)
                metrics.num_nodes_extended += 1
            else:
                path1 = nearest1.path_from_root()
                path2 = nearest2.path_from_root()
                # This is a tricky case to cover.
                if path1[0] != root1:  # pragma: no cover
                    path1, path2 = path2, path1
                assert path1[0] == root1
                path = path1[:-1] + path2[::-1]
                return [node.data for node in path]
        return None


class _RRTNode(Generic[_RRTState]):
    """A node for RRT."""

    def __init__(
        self, data: _RRTState, parent: Optional[_RRTNode[_RRTState]] = None
    ) -> None:
        self.data = data
        self.parent = parent

    def path_from_root(self) -> list[_RRTNode[_RRTState]]:
        """Return the path from the root to this node."""
        sequence = []
        node: Optional[_RRTNode[_RRTState]] = self
        while node is not None:
            sequence.append(node)
            node = node.parent
        return sequence[::-1]
