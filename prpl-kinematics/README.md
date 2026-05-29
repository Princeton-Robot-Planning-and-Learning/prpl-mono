# prpl-kinematics

Kinematics-only robot modeling, IK, motion planning, and manipulation primitives
built on a general `KinematicTree`. A ground-up successor to `pybullet-helpers`:
engine-agnostic (PyBullet is a pluggable collision/render backend, not the source
of truth), with poses represented as `spatialmath` `SE3`/`SE2`.

See [`DESIGN.md`](DESIGN.md) for the architecture and roadmap.

## Status

Early. The built core is the geometry layer and the `KinematicTree` (joints,
forward kinematics, grasp via re-parenting, state snapshots). Loading, backends,
IK, planning, robots, and manipulation are planned milestones — each lands with
unit tests.

## Requirements

- Python 3.10+

## Installation

From the monorepo root, with [uv](https://docs.astral.sh/uv/):

```
uv pip install -e "./prpl-kinematics[develop]"
```

## Development

```
./run_ci_checks.sh   # autoformat, mypy, pylint, pytest
```
