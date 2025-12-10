"""Tests for cupboard real state_abstractions.py."""

import numpy as np
import prbench
from conftest import MAKE_VIDEOS  # pylint: disable=import-error
from gymnasium.wrappers import RecordVideo
from prbench.envs.dynamic3d.tidybot3d import ObjectCentricTidyBot3DEnv
from relational_structs import ObjectCentricState

from prbench_models.dynamic3d.cupboard_real.state_abstractions import (
    CupboardRealStateAbstractor,
)
from prbench_models.dynamic3d.ground.parameterized_skills import (
    PyBulletSim,
    create_lifted_controllers,
)


def test_cupboard_real_state_abstraction():
    """Tests for CupboardRealStateAbstractor()."""
    prbench.register_all_environments()
    num_objects = 1
    env = prbench.make(
        f"prbench/TidyBot3D-cupboard_real-o{num_objects}-v0", render_mode="rgb_array"
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env,
            "unit_test_videos",
            name_prefix="TidyBot3D-cupboard-real-state-abstraction",
        )
    sim = ObjectCentricTidyBot3DEnv(
        scene_type="cupboard_real", num_objects=num_objects, render_images=False
    )
    abstractor = CupboardRealStateAbstractor(sim)

    # Check state abstraction in the initial state. The robot's hand should be empty
    # and the object should be on the ground.
    obs, _ = env.reset(seed=123)
    state = env.observation_space.devectorize(obs)
    assert isinstance(state, ObjectCentricState)
    abstract_state = abstractor.state_abstractor(state)
    assert str(sorted(abstract_state.atoms)) == "[(HandEmpty robot), (OnGround cube1)]"

    pybullet_sim = PyBulletSim(state, rendering=False)
    # Create controllers.
    controllers = create_lifted_controllers(env.action_space, pybullet_sim=pybullet_sim)

    # Pick up the cube.
    lifted_controller = controllers["pick_ground"]
    robot = state.get_object_from_name("robot")
    cube = state.get_object_from_name("cube1")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    target_distance = 0.5
    target_rotation = 0.0
    params = np.array([target_distance, target_rotation])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(400):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # Check updated state abstraction: the robot should be Holding the cube.
    abstract_state = abstractor.state_abstractor(state)
    assert str(sorted(abstract_state.atoms)) == "[(Holding robot cube1)]"

    # Plce the cube.
    lifted_controller = controllers["place_ground"]
    robot = state.get_object_from_name("robot")
    cube = state.get_object_from_name("cube1")
    cupboard = state.get_object_from_name("cupboard_1")
    object_parameters = (robot, cube, cupboard)
    controller = lifted_controller.ground(object_parameters)
    target_distance = 0.9
    offset = 0.0
    target_rotation = -np.pi / 2
    params = np.array([target_distance, offset, target_rotation])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(400):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    abstract_state = abstractor.state_abstractor(state)
    assert (
        str(sorted(abstract_state.atoms))
        == "[(HandEmpty robot), (OnFixture cube1 cupboard_1)]"
    )

    env.close()
