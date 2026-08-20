"""Regression test: _ConcatTrajectory must be callable at exactly t == duration.

See pybullet-helpers/tests/test_trajectory_boundary.py for the mechanism
(CPython 3.12+ compensated sum() vs the naive segment fold, one-ULP mismatch).
"""

import numpy as np

from prpl_utils.trajopt.trajectory import _ConcatTrajectory, _TrajectorySegment

DURATIONS = [
    0.34475140273571014, 0.1414589475579653, 0.14145881566227003,
    0.14145866239931015, 0.1414591006327488, 0.14145870251896164,
    0.14145887962159054, 0.1414587834763923, 0.1414589024844985,
    0.14145877032865228, 0.14145886964314117, 1.0386589765548706,
    0.19324145689126673,
]


def test_concat_trajectory_callable_at_total_duration() -> None:
    """A query at exactly the total duration returns the final endpoint."""
    segs = [_TrajectorySegment(np.zeros(2), np.ones(2), d) for d in DURATIONS]
    traj = _ConcatTrajectory(segs)
    out = traj(traj.duration)
    assert np.allclose(np.asarray(out), np.ones(2))
