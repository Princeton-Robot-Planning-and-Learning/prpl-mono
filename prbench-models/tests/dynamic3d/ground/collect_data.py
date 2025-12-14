"""Tests for ground parameterized skills."""

import numpy as np
import prbench
from gymnasium.wrappers import RecordVideo
from prpl_tidybot.interfaces.interface import FakeInterface
from prpl_tidybot.perceivers.prbench_ground_perceiver import PRBenchGroundPerceiver
from relational_structs.spaces import ObjectCentricBoxSpace
from spatialmath import SE2
from prbench_models.dynamic3d.fk_solver import TidybotFKSolver
from prbench_models.dynamic3d.ground.parameterized_skills import (
    PyBulletSim,
    create_lifted_controllers,
    get_target_robot_pose_from_parameters,
)

prbench.register_all_environments()

def collect_data():
    """Test pick and place skill in ground environment with 1 cube."""

    # Create the environment.
    num_cubes = 2
    env = prbench.make(
        f"prbench/TidyBot3D-cupboard_real-o{num_cubes}-v0", render_mode="rgb_array"
    )
    
    # Reset the environment and get the initial state.
    obs, _ , raw_obs = env.reset_with_images(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    assert state is not None
    pybullet_sim = PyBulletSim(state, rendering=False)

    controllers = create_lifted_controllers(env.action_space, pybullet_sim=pybullet_sim)

    fk_solver = TidybotFKSolver(ee_offset=0.0)
    
    # create the pick ground controller.
    lifted_controller = controllers["pick_ground"]
    robot = state.get_object_from_name("robot")
    cube = state.get_object_from_name("cube1")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    params = controller.sample_parameters(state, np.random.default_rng(123))

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(400):
        action = controller.step()  
        robot = state.get_object_from_name("robot")
        current_joints = [state.get(robot, "pos_arm_joint1"), state.get(robot, "pos_arm_joint2"), state.get(robot, "pos_arm_joint3"), state.get(robot, "pos_arm_joint4"), state.get(robot, "pos_arm_joint5"), state.get(robot, "pos_arm_joint6"), state.get(robot, "pos_arm_joint7")]
        current_pose = fk_solver.forward_kinematics(np.array(current_joints))
        obs, _, _, _, _, raw_obs = env.step_with_images(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # create the place ground controller.
    lifted_controller = controllers["place_ground"]
    robot = state.get_object_from_name("robot")
    cube = state.get_object_from_name("cube1")
    cupboard = state.get_object_from_name("cupboard_1")
    object_parameters = (robot, cube, cupboard)
    controller = lifted_controller.ground(object_parameters)
    params = controller.sample_parameters(state, np.random.default_rng(123))

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(400):
        action = controller.step()
        obs, _, _, _, _, raw_obs = env.step_with_images(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    env.close()

def main():
    collect_data()

if __name__ == "__main__":
    main()