"""Execute Blockly programs in KinDER environments."""

from collections.abc import Iterator
from typing import Any

import kinder
import numpy as np
from kinder.envs.kinematic3d.base_motion3d import (
    BaseMotion3DObjectCentricState,
    ObjectCentricBaseMotion3DEnv,
)
from numpy.typing import NDArray
from pybullet_helpers.geometry import SE2Pose
from pybullet_helpers.motion_planning import (
    run_single_arm_mobile_base_motion_planning,
)

kinder.register_all_environments()

MAX_STEPS = 500
FRAME_SKIP = 5


def render_initial_frame(seed: int = 0) -> NDArray[np.uint8]:
    """Reset the environment and return the first rendered frame."""
    env = kinder.make(
        "kinder/BaseMotion3D-v0",
        render_mode="rgb_array",
        use_gui=False,
    )
    try:
        env.reset(seed=seed)
        frame: NDArray[np.uint8] = env.render()  # type: ignore[assignment]
        return frame
    finally:
        env.close()  # type: ignore[no-untyped-call]


def execute_program(
    program: dict[str, Any], seed: int = 0
) -> Iterator[NDArray[np.uint8]]:
    """Execute a Blockly program and yield rendered frames.

    Yields an initial frame after reset, then intermediate frames during skill
    execution.
    """
    blocks: list[dict[str, Any]] = program.get("blocks", [])
    if not blocks:
        return

    env = kinder.make(
        "kinder/BaseMotion3D-v0",
        render_mode="rgb_array",
        use_gui=False,
    )
    try:
        obs, _ = env.reset(seed=seed)
        state = env.observation_space.devectorize(obs)  # type: ignore[attr-defined]

        sim = ObjectCentricBaseMotion3DEnv(allow_state_access=True)

        frame: NDArray[np.uint8] = env.render()  # type: ignore[assignment]
        yield frame

        for block in blocks:
            block_type = block["type"]
            if block_type == "move_base_to_target":
                target_x = float(block.get("x", 0.0))
                target_y = float(block.get("y", 0.0))
                state, frames = _run_move_base_to(env, state, sim, target_x, target_y)
                yield from frames
    finally:
        env.close()  # type: ignore[no-untyped-call]


def _run_move_base_to(
    env: Any,
    state: Any,
    sim: ObjectCentricBaseMotion3DEnv,
    target_x: float,
    target_y: float,
) -> tuple[Any, list[NDArray[np.uint8]]]:
    """Move the robot base to (target_x, target_y), return updated state and frames."""
    assert isinstance(state, BaseMotion3DObjectCentricState)
    sim.set_state(state)  # type: ignore[no-untyped-call]

    goal_pose = SE2Pose(target_x, target_y, 0.0)
    base_plan = run_single_arm_mobile_base_motion_planning(
        sim.robot,
        sim.robot.base.get_pose(),
        goal_pose,
        collision_bodies=set(),
        seed=0,
    )
    if base_plan is None:
        raise RuntimeError(f"Motion planning to ({target_x}, {target_y}) failed")

    plan = base_plan[1:]
    frames: list[NDArray[np.uint8]] = []

    for step_i, waypoint in enumerate(plan):
        current_base_pose = state.base_pose
        delta = waypoint - current_base_pose
        action_lst = [delta.x, delta.y, delta.rot] + [0.0] * 8
        action = np.array(action_lst, dtype=np.float32)

        obs, _, _, _, _ = env.step(action)
        state = env.observation_space.devectorize(obs)  # type: ignore[attr-defined]

        if step_i % FRAME_SKIP == 0 or step_i == len(plan) - 1:
            frame: NDArray[np.uint8] = env.render()  # type: ignore[assignment]
            frames.append(frame)

    return state, frames
