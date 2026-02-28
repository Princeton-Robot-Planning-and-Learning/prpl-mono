"""Execute Blockly programs in KinDER environments."""

from collections.abc import Iterator
from typing import Any

import kinder
import numpy as np
from kinder.envs.kinematic3d.base_motion3d import ObjectCentricBaseMotion3DEnv
from kinder_models.kinematic3d.base_motion3d.parameterized_skills import (
    create_lifted_controllers,
)
from numpy.typing import NDArray
from relational_structs.spaces import ObjectCentricBoxSpace

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
    blocks: list[dict[str, str]] = program.get("blocks", [])
    if not blocks:
        return

    env = kinder.make(
        "kinder/BaseMotion3D-v0",
        render_mode="rgb_array",
        use_gui=False,
    )
    try:
        obs, _ = env.reset(seed=seed)
        assert isinstance(env.observation_space, ObjectCentricBoxSpace)
        state = env.observation_space.devectorize(obs)

        sim = ObjectCentricBaseMotion3DEnv()
        controllers = create_lifted_controllers(
            env.action_space, sim  # type: ignore[arg-type]
        )

        rng = np.random.default_rng(seed)

        frame: NDArray[np.uint8] = env.render()  # type: ignore[assignment]
        yield frame

        for block in blocks:
            block_type = block["type"]
            if block_type == "move_base_to_target":
                yield from _run_move_base_to_target(env, state, controllers, rng)
    finally:
        env.close()  # type: ignore[no-untyped-call]


def _run_move_base_to_target(
    env: Any,
    state: Any,
    controllers: dict[str, Any],
    rng: np.random.Generator,
) -> Iterator[NDArray[np.uint8]]:
    """Run the move_base_to_target controller, yielding frames."""
    lifted_controller = controllers["move_base_to_target"]
    robot = state.get_object_from_name("robot")
    target = state.get_object_from_name("target")
    controller = lifted_controller.ground((robot, target))

    params = controller.sample_parameters(state, rng)
    controller.reset(state, params)

    for step_i in range(MAX_STEPS):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state

        if step_i % FRAME_SKIP == 0 or controller.terminated():
            frame: NDArray[np.uint8] = env.render()  # type: ignore[assignment]
            yield frame

        if controller.terminated():
            break
