"""Tests for motion_planning.py."""

import json
import os

import numpy as np
import pytest

from prpl_utils.motion_planning import BiRRT, MotionPlanningMetrics


def _make_birrt(collision_fn):
    return BiRRT(
        sample_fn=lambda x: x,
        extend_fn=lambda x, y: [x, y],
        collision_fn=collision_fn,
        distance_fn=lambda x, y: 0.0,
        rng=np.random.default_rng(0),
        num_attempts=1,
        num_iters=1,
        smooth_amt=0,
    )


def test_motion_planning():
    """Basic BiRRT query works and metrics are tracked."""
    birrt = _make_birrt(lambda _: False)

    # query_to_goal_fn raises NotImplementedError
    with pytest.raises(NotImplementedError):
        birrt.query_to_goal_fn(0, lambda _: False, lambda: 1)

    # Successful query returns a path and populated metrics.
    path, metrics = birrt.query(0, 1)
    assert path is not None
    assert metrics.num_collision_checks > 0
    assert isinstance(metrics, MotionPlanningMetrics)

    # When start is in collision, path is None but metrics still returned.
    birrt_blocked = _make_birrt(lambda x: x == 0)
    path, metrics = birrt_blocked.query(0, 1)
    assert path is None
    assert metrics.num_collision_checks >= 1


def test_motion_planning_metrics_dump(tmp_path):
    """birrt_metrics_path env var causes query to append a JSON line."""
    birrt = _make_birrt(lambda _: False)
    dump_file = tmp_path / "metrics.jsonl"

    old = os.environ.get("birrt_metrics_path")
    try:
        os.environ["birrt_metrics_path"] = str(dump_file)
        birrt.query(0, 1)
        birrt.query(0, 1)
    finally:
        if old is None:
            del os.environ["birrt_metrics_path"]
        else:
            os.environ["birrt_metrics_path"] = old

    lines = dump_file.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        entry = json.loads(line)
        assert "num_nodes_extended" in entry
        assert "num_collision_checks" in entry
