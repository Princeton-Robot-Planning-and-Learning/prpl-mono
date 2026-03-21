"""Continuous-time trajectory representations."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


class Trajectory:
    """A continuous-time trajectory over NDArrays."""

    @property
    @abc.abstractmethod
    def duration(self) -> float:
        """The length of the trajectory in time."""

    @abc.abstractmethod
    def __call__(self, time: float) -> NDArray[np.floating]:
        """Get the point at the given time."""

    def __getitem__(self, key: float | slice) -> NDArray[np.floating] | Trajectory:
        """Shorthand for indexing or sub-trajectory creation."""
        if isinstance(key, float):
            return self(key)
        assert isinstance(key, slice)
        assert key.step is None
        start = key.start or 0
        end = key.stop or self.duration
        return self.get_sub_trajectory(start, end)

    @abc.abstractmethod
    def get_sub_trajectory(self, start_time: float, end_time: float) -> Trajectory:
        """Create a new trajectory with time re-indexed."""


@dataclass(frozen=True)
class _TrajectorySegment(Trajectory):
    """A trajectory defined by a single start and end point."""

    start: NDArray[np.floating]
    end: NDArray[np.floating]
    _duration: float = 1.0

    @cached_property
    def duration(self) -> float:
        return self._duration

    def __call__(self, time: float) -> NDArray[np.floating]:
        time = np.clip(time, 0, self.duration)
        s = time / self.duration
        return self.start + (self.end - self.start) * s

    def get_sub_trajectory(self, start_time: float, end_time: float) -> Trajectory:
        elapsed_time = end_time - start_time
        frac = elapsed_time / self.duration
        new_duration = frac * self.duration
        return _TrajectorySegment(self(start_time), self(end_time), new_duration)


@dataclass(frozen=True)
class _ConcatTrajectory(Trajectory):
    """A trajectory that concatenates other trajectories."""

    trajs: Sequence[Trajectory]

    @cached_property
    def duration(self) -> float:
        return sum(t.duration for t in self.trajs)

    def __call__(self, time: float) -> NDArray[np.floating]:
        time = np.clip(time, 0, self.duration)
        start_time = 0.0
        for traj in self.trajs:
            end_time = start_time + traj.duration
            if time <= end_time:
                assert time >= start_time
                return traj(time - start_time)
            start_time = end_time
        raise ValueError(f"Time {time} exceeds duration {self.duration}")

    def get_sub_trajectory(self, start_time: float, end_time: float) -> Trajectory:
        new_trajs: list[Trajectory] = []
        st = 0.0
        keep_traj = False
        for traj in self.trajs:
            et = st + traj.duration
            if st <= start_time < et:
                keep_traj = True
                traj = traj.get_sub_trajectory(start_time - st, traj.duration)
                st = start_time
            if st < end_time <= et:
                traj = traj.get_sub_trajectory(0, end_time - st)
                assert keep_traj
                new_trajs.append(traj)
                break
            if keep_traj:
                new_trajs.append(traj)
            st = et
        return concatenate_trajectories(new_trajs)


def concatenate_trajectories(
    trajectories: Sequence[Trajectory],
) -> Trajectory:
    """Concatenate one or more trajectories."""
    inner_trajs: list[Trajectory] = []
    for traj in trajectories:
        if isinstance(traj, _ConcatTrajectory):
            inner_trajs.extend(traj.trajs)
        else:
            inner_trajs.append(traj)
    return _ConcatTrajectory(inner_trajs)


def point_sequence_to_trajectory(
    point_sequence: list[NDArray[np.floating]],
    dt: float,
) -> Trajectory:
    """Convert a sequence of points to a trajectory."""
    segments = []
    for t in range(len(point_sequence) - 1):
        start = point_sequence[t]
        end = point_sequence[t + 1]
        seg = _TrajectorySegment(start, end, dt)
        segments.append(seg)
    return concatenate_trajectories(segments)
