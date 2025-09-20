"""Utilities and classes for continuous trajectories."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Generic, Iterator, Sequence, TypeVar

import numpy as np

TrajectoryPoint = TypeVar("TrajectoryPoint")


class Trajectory(Generic[TrajectoryPoint]):
    """A continuous-time trajectory."""

    @property
    @abc.abstractmethod
    def duration(self) -> float:
        """The length of the trajectory in time."""

    @property
    @abc.abstractmethod
    def distance(self) -> float:
        """The length of the trajectory in distance."""

    @abc.abstractmethod
    def __call__(self, time: float) -> TrajectoryPoint:
        """Get the point at the given time."""

    def __getitem__(self, key: float | slice):
        """Shorthand for indexing or sub-trajectory creation."""
        if isinstance(key, float):
            return self(key)
        assert isinstance(key, slice)
        assert key.step is None
        start = key.start or 0
        end = key.stop or self.duration
        return self.get_sub_trajectory(start, end)

    @abc.abstractmethod
    def get_sub_trajectory(
        self, start_time: float, end_time: float
    ) -> Trajectory[TrajectoryPoint]:
        """Create a new trajectory with time re-indexed."""

    @abc.abstractmethod
    def reverse(self) -> Trajectory[TrajectoryPoint]:
        """Create a new trajectory with time re-indexed."""


class TrajectorySegment(Trajectory[TrajectoryPoint]):
    """A trajectory defined by a single start and end point."""

    def __init__(
        self,
        start: TrajectoryPoint,
        end: TrajectoryPoint,
        duration: float,
        interpolate_fn: Callable[
            [TrajectoryPoint, TrajectoryPoint, float], TrajectoryPoint
        ],
        distance_fn: Callable[[TrajectoryPoint, TrajectoryPoint], float],
    ) -> None:
        self.start = start
        self.end = end
        self._duration = duration
        self._interpolate_fn = interpolate_fn
        self._distance_fn = distance_fn

    @cached_property
    def duration(self) -> float:
        return self._duration

    @cached_property
    def distance(self) -> float:
        return self._distance_fn(self.start, self.end)

    def __call__(self, time: float) -> TrajectoryPoint:
        # Avoid numerical issues.
        time = np.clip(time, 0, self.duration)
        s = time / self.duration
        return self._interpolate_fn(self.start, self.end, s)

    def get_sub_trajectory(
        self, start_time: float, end_time: float
    ) -> Trajectory[TrajectoryPoint]:
        elapsed_time = end_time - start_time
        frac = elapsed_time / self.duration
        new_duration = frac * self.duration
        return self.__class__(
            self(start_time),
            self(end_time),
            new_duration,
            self._interpolate_fn,
            self._distance_fn,
        )

    def reverse(self) -> Trajectory[TrajectoryPoint]:
        return self.__class__(
            self.end, self.start, self.duration, self._interpolate_fn, self._distance_fn
        )


@dataclass(frozen=True)
class ConcatTrajectory(Trajectory[TrajectoryPoint]):
    """A trajectory that concatenates other trajectories."""

    trajs: Sequence[Trajectory[TrajectoryPoint]]

    @cached_property
    def duration(self) -> float:
        return sum(t.duration for t in self.trajs)

    @cached_property
    def distance(self) -> float:
        return sum(t.distance for t in self.trajs)

    def __call__(self, time: float) -> TrajectoryPoint:
        # Avoid numerical issues.
        time = np.clip(time, 0, self.duration)
        start_time = 0.0
        for traj in self.trajs:
            end_time = start_time + traj.duration
            if time <= end_time:
                assert time >= start_time
                return traj(time - start_time)
            start_time = end_time
        raise ValueError(f"Time {time} exceeds duration {self.duration}")

    def get_sub_trajectory(
        self, start_time: float, end_time: float
    ) -> Trajectory[TrajectoryPoint]:
        new_trajs = []
        st = 0.0
        keep_traj = False
        for traj in self.trajs:
            et = st + traj.duration
            # Start keeping trajectories.
            if st <= start_time < et:
                keep_traj = True
                # Shorten the current trajectory so it starts at start_time.
                traj = traj.get_sub_trajectory(start_time - st, traj.duration)
                st = start_time
            # Stop keeping trajectories.
            if st < end_time <= et:
                # Shorten the current trajectory so it ends at end_time.
                traj = traj.get_sub_trajectory(0, end_time - st)
                # Finish.
                assert keep_traj
                new_trajs.append(traj)
                break
            if keep_traj:
                new_trajs.append(traj)
            st = et
        return concatenate_trajectories(new_trajs)

    def reverse(self) -> Trajectory[TrajectoryPoint]:
        return self.__class__([t.reverse() for t in self.trajs][::-1])


def concatenate_trajectories(
    trajectories: Sequence[Trajectory[TrajectoryPoint]],
) -> Trajectory[TrajectoryPoint]:
    """Concatenate one or more trajectories."""
    inner_trajs: list[Trajectory[TrajectoryPoint]] = []
    for traj in trajectories:
        if isinstance(traj, ConcatTrajectory):
            inner_trajs.extend(traj.trajs)
        else:
            inner_trajs.append(traj)
    return ConcatTrajectory(inner_trajs)


def iter_traj_with_max_distance(
    traj: Trajectory[TrajectoryPoint],
    max_distance: float,
    include_start: bool = True,
    include_end: bool = True,
) -> Iterator[TrajectoryPoint]:
    """Iterate through the trajectory while guaranteeing that the distance in
    each step is no more than the given max distance."""
    num_steps = int(np.ceil(traj.distance / max_distance)) + 1
    ts = np.linspace(0, traj.duration, num=num_steps, endpoint=True)
    if not include_start:
        ts = ts[1:]
    if not include_end:
        ts = ts[:-1]
    for t in ts:
        yield traj(t)
