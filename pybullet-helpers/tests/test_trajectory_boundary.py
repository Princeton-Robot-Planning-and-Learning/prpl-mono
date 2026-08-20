"""Regression test: ConcatTrajectory must be callable at exactly t == duration.

On CPython 3.12+, built-in sum() uses Neumaier-compensated summation for
floats, so the cached `duration` (computed with sum()) can exceed the naive
running fold used when walking the segments by one ULP. A query at the total
duration then falls through the walk and raises "Time X exceeds duration X"
with both numbers printing identically. The durations below are captured from
a real failing planning episode and reproduce the mismatch deterministically.
"""

import numpy as np

from pybullet_helpers.trajectory import ConcatTrajectory, TrajectorySegment

DURATIONS = [
    0.34475140273571014, 0.1414589475579653, 0.14145881566227003,
    0.14145866239931015, 0.1414591006327488, 0.14145870251896164,
    0.14145887962159054, 0.1414587834763923, 0.1414589024844985,
    0.14145877032865228, 0.14145886964314117, 1.0386589765548706,
    0.19324145689126673,
]


def test_concat_trajectory_callable_at_total_duration() -> None:
    """A query at exactly the total duration returns the final endpoint."""
    segs = [
        TrajectorySegment(
            np.zeros(2),
            np.ones(2),
            d,
            lambda a, b, t: a + t * (b - a),
            lambda a, b: float(np.linalg.norm(b - a)),
        )
        for d in DURATIONS
    ]
    traj = ConcatTrajectory(segs)
    out = traj(traj.duration)
    assert np.allclose(np.asarray(out), np.ones(2))
