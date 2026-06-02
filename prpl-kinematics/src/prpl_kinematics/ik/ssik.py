"""Analytic inverse kinematics via the ``ssik`` library (optional dependency).

``ssik`` (https://pypi.org/project/ssik/) solves 6R and 7R revolute arms
analytically, including the non-spherical-wrist (non-SRS) 7R class that Vega's
arm falls in -- the same geometry :class:`~prpl_kinematics.robots.vega_ik.VegaArmIK`
handles with a lock-and-grid-search over EAIK. ssik's ``jointlock`` solver locks
one joint, sweeps it, and solves the residual 6R in closed form; with its Newton
refinement enabled it polishes each branch to machine precision (the lock sweep
only samples the locked joint, so a generic 7R needs the polish to FK-close).

``ssik`` is an optional dependency (it requires ``numpy>=2``), so it is imported
lazily: importing this module never needs ssik, only constructing
:class:`SSIKSolver` does. Conforms to the
:class:`~prpl_kinematics.ik.interface.InverseKinematics` protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from spatialmath import SE3

from prpl_kinematics.tree.kinematic_tree import Configuration, KinematicTree


class SSIKSolver:
    """Analytic IK via ssik for one revolute arm of a URDF-loaded robot.

    The arm is the chain from the joint group's base link to ``ee_link``; ssik
    parses it from ``urdf_path`` (its forward kinematics matches the tree's to
    machine precision, so targets computed in the tree's frames map straight in).
    """

    def __init__(
        self,
        tree: KinematicTree,
        arm_joints: Sequence[str],
        ee_link: str,
        urdf_path: str | Path,
        tool_frame: str | None = None,
        allow_refinement: bool = True,
    ) -> None:
        try:
            # pylint: disable=import-outside-toplevel
            from ssik import Manipulator
        except ImportError as exc:  # pragma: no cover - exercised only without ssik
            raise ImportError(
                "SSIKSolver needs the optional 'ssik' package: `pip install "
                '"ssik[urdf]"` (requires numpy>=2).'
            ) from exc

        self._tree = tree
        self._arm_joints = list(arm_joints)
        self._allow_refinement = allow_refinement

        edges = [e for e in tree.path_from_root(ee_link) if e.joint.name in arm_joints]
        self._base_frame = edges[0].parent
        self._manipulator = Manipulator.from_urdf(
            str(urdf_path), base=self._base_frame, ee=ee_link
        )
        tool = tool_frame if tool_frame is not None else ee_link
        self._ee_from_tool = tree.relative_pose(ee_link, tool, {})

    def solve(self, target_pose: SE3, seed: Configuration) -> Configuration | None:
        """A configuration reaching ``target_pose`` (the tool frame), or ``None``.

        ssik returns every analytic branch sorted by wrap-to-pi distance from the seed,
        so the first solution is the seed-closest one -- the natural, away-from-limits
        branch rather than an arbitrary contorted one.
        """
        ee_target = target_pose * self._ee_from_tool.inv()
        in_base = (
            self._tree.forward_kinematics(self._base_frame, seed).inv() * ee_target
        )
        seed_q = np.array([float(seed[name][0]) for name in self._arm_joints])
        # max_solutions=1 with q_seed is ssik's trajectory-tracking idiom: it
        # returns just the seed-closest branch and, on jointlock-7R arms,
        # short-circuits the lock sweep -- the single config this interface needs.
        solutions = self._manipulator.solve(
            np.asarray(in_base.A),
            q_seed=seed_q,
            max_solutions=1,
            respect_limits=True,
            allow_refinement=self._allow_refinement,
        )
        if not solutions:
            return None
        best_q = np.asarray(solutions[0].q, dtype=float)
        values = {name: [float(v)] for name, v in zip(self._arm_joints, best_q)}
        return {**dict(seed), **values}
