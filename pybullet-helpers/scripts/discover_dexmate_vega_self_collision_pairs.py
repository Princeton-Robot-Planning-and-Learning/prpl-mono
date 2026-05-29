"""Discover candidate self-collision link pairs for the bimanual Dexmate Vega 1U.

The bimanual robot's two arms can cross in front of the torso and reach the head, so
per-arm motion planning needs to check those link pairs. Enumerating them by hand is
error prone, so we discover them by sampling random whole-body configurations and
recording which cross-group link pairs come within a margin of each other (excluding
pairs that already touch at the home configuration, i.e. adjacent links).

The printed list is frozen into dexmate_vega._SELF_COLLISION_LINK_PAIRS. Re-run this
script (and widen the margin / add samples) if the URDF changes. A conservative margin
is used so the frozen list is a superset of what tighter checks would find.

Usage: python scripts/discover_dexmate_vega_self_collision_pairs.py
"""

import numpy as np
import pybullet as p

from pybullet_helpers.joint import get_joint_infos, get_joints
from pybullet_helpers.robots.dexmate_vega import DexmateVega1UPyBulletRobot

_MARGIN = 0.02
_NUM_SAMPLES = 8000
_SEED = 0


def _group(link_name: str) -> str:
    if link_name.startswith(("L_arm", "L_ee", "L_gripper")):
        return "L"
    if link_name.startswith(("R_arm", "R_ee", "R_gripper")):
        return "R"
    if link_name.startswith("head") or "zed" in link_name:
        return "head"
    if link_name in (
        "lift_link",
        "arm_center",
        "torso_flip_link",
        "base_link",
        "torso_link",
    ):
        return "torso"
    return "other"


# Cross-group pairs we care about for per-arm planning: arm-vs-other-arm, arm-vs-torso,
# arm-vs-head.
_CROSS_GROUPS = {
    frozenset(("L", "R")),
    frozenset(("L", "torso")),
    frozenset(("L", "head")),
    frozenset(("R", "torso")),
    frozenset(("R", "head")),
}


def main() -> None:
    """Sample whole-body configurations and print the discovered self-collision
    pairs."""
    physics_client_id = p.connect(p.DIRECT)
    robot = DexmateVega1UPyBulletRobot(physics_client_id)
    robot_id = robot.robot_id
    infos = get_joint_infos(
        robot_id, get_joints(robot_id, physics_client_id), physics_client_id
    )
    name_of = {info.jointIndex: info.linkName for info in infos}
    movable = [i for i, n in name_of.items() if _group(n) != "other"]

    def touching(margin: float) -> set[tuple[int, int]]:
        found: set[tuple[int, int]] = set()
        for idx, link_a in enumerate(movable):
            for link_b in movable[idx + 1 :]:
                if p.getClosestPoints(
                    robot_id,
                    robot_id,
                    distance=margin,
                    linkIndexA=link_a,
                    linkIndexB=link_b,
                    physicsClientId=physics_client_id,
                ):
                    found.add((link_a, link_b))
        return found

    home_touching = touching(_MARGIN / 2)

    left, right = robot.left_arm, robot.right_arm
    l_lo, l_hi = np.array(left.joint_lower_limits), np.array(left.joint_upper_limits)
    r_lo, r_hi = np.array(right.joint_lower_limits), np.array(right.joint_upper_limits)
    rng = np.random.default_rng(_SEED)

    discovered: set[tuple[str, str]] = set()
    for _ in range(_NUM_SAMPLES):
        left.set_joints(rng.uniform(l_lo, l_hi).tolist())
        right.set_joints(rng.uniform(r_lo, r_hi).tolist())
        robot.set_torso_joints([rng.uniform(0.0, 0.2), rng.uniform(-0.5, 0.5)])
        for idx, link_a in enumerate(movable):
            for link_b in movable[idx + 1 :]:
                if (link_a, link_b) in home_touching:
                    continue
                groups = frozenset((_group(name_of[link_a]), _group(name_of[link_b])))
                if groups not in _CROSS_GROUPS:
                    continue
                if p.getClosestPoints(
                    robot_id,
                    robot_id,
                    distance=_MARGIN,
                    linkIndexA=link_a,
                    linkIndexB=link_b,
                    physicsClientId=physics_client_id,
                ):
                    discovered.add((name_of[link_a], name_of[link_b]))
    p.disconnect(physics_client_id)

    print(
        f"# {len(discovered)} pairs "
        f"(margin={_MARGIN}, samples={_NUM_SAMPLES}, seed={_SEED})"
    )
    for pair in sorted(discovered):
        print(f"    {pair!r},")


if __name__ == "__main__":
    main()
