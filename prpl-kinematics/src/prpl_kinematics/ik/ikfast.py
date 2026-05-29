"""IKFast-backed analytic inverse kinematics.

IKFast solvers are generated per robot as C++ that closed-form-solves a 6-DOF
chain (extra joints are "free" and sampled). ``IKFastSolver`` wraps such a
compiled module: it uses the :class:`~prpl_kinematics.tree.kinematic_tree.\
KinematicTree` as the forward-kinematics source to express the target in the
solver's base frame, calls the module for candidate solutions, discards those
out of joint limits, and returns the candidate closest to the seed.

The compiled module is built on demand from the committed C++ source the first
time it is needed (it is git-ignored), so a C++ toolchain plus LAPACK/BLAS must
be available -- the same requirement as the upstream IKFast build.
"""

from __future__ import annotations

import glob
import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Sequence

import numpy as np
from spatialmath import SE3

from prpl_kinematics.planning.joint_space import JointSpace
from prpl_kinematics.tree.kinematic_tree import Configuration, KinematicTree

_IKFAST_DIR = Path(__file__).resolve().parent.parent / "third_party" / "ikfast"


@dataclass(frozen=True)
class IKFastInfo:
    """Locates a robot's IKFast module and its chain endpoints.

    ``module_dir`` is the subdirectory under ``third_party/ikfast`` holding the
    C++ source; ``module_name`` is the compiled module's name. ``base_link`` and
    ``ee_link`` are the frames the solver was generated for. ``free_joints`` are
    the chain joints not solved analytically (sampled by the solver).
    """

    module_dir: str
    module_name: str
    base_link: str
    ee_link: str
    free_joints: Sequence[str]


def _import_ikfast(info: IKFastInfo) -> ModuleType:
    """Import the robot's IKFast module, compiling it on first use."""
    if info.module_name in sys.modules:
        return sys.modules[info.module_name]
    module_dir = _IKFAST_DIR / info.module_dir
    pattern = str(module_dir / f"{info.module_name}*.so")
    matches = glob.glob(pattern)
    if not matches:
        result = subprocess.run(
            [sys.executable, "setup.py"], cwd=module_dir, check=False
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"IKFast compilation failed for {info.module_name} (exit "
                f"{result.returncode}). A C++ toolchain and LAPACK/BLAS are "
                "required."
            )
        matches = glob.glob(pattern)
    spec = importlib.util.spec_from_file_location(info.module_name, matches[0])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[info.module_name] = module
    spec.loader.exec_module(module)
    return module


class IKFastSolver:
    """Analytic IK for one robot via its compiled IKFast module."""

    def __init__(
        self,
        tree: KinematicTree,
        info: IKFastInfo,
        ik_joints: Sequence[str],
        rng: np.random.Generator,
        max_samples: int = 100,
        tool_frame: str | None = None,
    ) -> None:
        self._tree = tree
        self._info = info
        self._ik_joints = list(ik_joints)
        self._space = JointSpace(tree, ik_joints)
        self._free_space = JointSpace(tree, list(info.free_joints))
        self._rng = rng
        self._max_samples = max_samples
        self._module = _import_ikfast(info)
        # Targets are given for ``tool_frame``; IKFast solves for ``ee_link``.
        # Their relative transform is fixed (both past the last actuated joint).
        tool = tool_frame if tool_frame is not None else info.ee_link
        self._ee_from_tool = tree.relative_pose(info.ee_link, tool, {})

    def solve(self, target_pose: SE3, seed: Configuration) -> Configuration | None:
        """The reachable configuration closest to ``seed``, or ``None``.

        ``target_pose`` is the desired world pose of the tool frame (the
        ``ee_link`` when no ``tool_frame`` was given).
        """
        ee_target = target_pose * self._ee_from_tool.inv()
        base_from_ee = (
            self._tree.forward_kinematics(self._info.base_link, seed).inv() * ee_target
        )
        rotation = np.asarray(base_from_ee.R).tolist()
        position = list(np.asarray(base_from_ee.t))
        seed_vector = self._space.to_vector(seed)

        best: Configuration | None = None
        best_distance = np.inf
        for free_values in self._free_samples(seed):
            candidates = self._module.get_ik(rotation, position, list(free_values))
            if not candidates:
                continue
            for candidate in candidates:
                vector = np.asarray(candidate, dtype=float)
                if not np.allclose(vector, self._space.clamp(vector)):
                    continue  # out of joint limits
                # Wrap-aware so continuous joints aren't judged a full turn away.
                distance = self._space.distance(vector, seed_vector)
                if distance < best_distance:
                    best_distance = distance
                    best = {**dict(seed), **self._space.to_configuration(vector)}
        return best

    def _free_samples(self, seed: Configuration) -> list[np.ndarray]:
        """The seed's free-joint values first, then random samples within limits."""
        samples = [self._free_space.to_vector(seed)]
        samples.extend(
            self._free_space.sample(self._rng) for _ in range(self._max_samples)
        )
        return samples
