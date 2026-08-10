# prpl-kinematics

Kinematics-only robot modeling, IK, motion planning, and manipulation primitives
built on a general `KinematicTree`. A ground-up successor to `pybullet-helpers`:
engine-agnostic (PyBullet is a pluggable collision/render backend, not the source
of truth), with poses represented as `spatialmath` `SE3`/`SE2`.

See [`DESIGN.md`](DESIGN.md) for the architecture and roadmap.

## Status

Early, and the API may still change. The geometry layer and the `KinematicTree`
(joints, forward kinematics, grasp via re-parenting, state snapshots) are the
stable core. Built on top of them: URDF loading, PyBullet and Blender backends,
numerical and analytic IK (IKFast, EAIK, optionally SSIK), BiRRT and OMPL motion
planning, manipulation primitives, and robot models for the Franka Panda, Kinova
Gen3, TidyBot, and the bimanual Dexmate Vega.

Pin a version if you need stable behavior.

## Requirements

- Python 3.10+

## Installation

```
pip install prpl_kinematics
```

IKFast solvers are compiled from the bundled C++ sources on first use, which needs
a C++ toolchain and LAPACK/BLAS. The other IK backends have no such requirement.

The optional `ssik` extra adds the SSIK analytic IK backend:

```
pip install "prpl_kinematics[ssik]"
```

For development, from the monorepo root, with [uv](https://docs.astral.sh/uv/):

```
uv pip install -e "./prpl-kinematics[develop]"
```

### Optional extras

`OMPLPlanner` and `seed_ompl` need the `planning` extra, because `ompl` publishes
wheels for far fewer platforms than everything else here (none for Windows, and
macOS 15 or newer only):

```
uv pip install -e "./prpl-kinematics[planning]"
```

Without it, importing those two names raises an `ImportError` saying so; everything
else, including `BiRRTPlanner`, works untouched.

## Development

```
./run_ci_checks.sh   # autoformat, mypy, pylint, pytest
```
