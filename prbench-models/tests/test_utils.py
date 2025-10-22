"""Tests for utils.py."""

import prbench
from prbench.envs.dynamic3d.tidybot3d import ObjectCentricTidyBot3DEnv
from prbench_bilevel_planning.env_models.tidybot3d.tidybot3d_base_motion import (
    create_bilevel_planning_models,
)

from prbench_models.utils import (
    ParameterizedSkillReference,
    PRBenchParameterizedSkillEnv,
)


def test_prbench_parameterized_skill_env():
    """Tests for PRBenchParameterizedSkillEnv()."""

    # Set up the environment.
    prbench.register_all_environments()
    prbench_env = prbench.make("prbench/TidyBot3D-base_motion-o1-v0")
    sim = prbench_env.unwrapped._object_centric_env  # pylint: disable=protected-access
    assert isinstance(sim, ObjectCentricTidyBot3DEnv)
    env_models = create_bilevel_planning_models(
        prbench_env.observation_space,
        prbench_env.action_space,
    )
    lifted_skills = env_models.skills
    parameterized_skills = [s.controller for s in lifted_skills]
    assert len(parameterized_skills) == 1
    move_to_target_skill = parameterized_skills[0]
    assert move_to_target_skill.name == "MoveToTargetGroundController"

    # Reset the environment.
    env = PRBenchParameterizedSkillEnv(sim, parameterized_skills)
    obs, _ = env.reset(seed=123)

    # Make a plan.
    robot = obs.get_object_from_name("robot")
    cube = obs.get_object_from_name("cube1")
    move_to_cube = ParameterizedSkillReference(
        "MoveToTargetGroundController", objects=[robot, cube], params={}
    )
    plan = [move_to_cube]

    # Execute the plan.
    for action in plan:
        obs, _, _, _, _ = env.step(action)
        robot_x = obs.get(robot, "pos_base_x")
        robot_y = obs.get(robot, "pos_base_y")
        cube_x = obs.get(cube, "x")
        cube_y = obs.get(cube, "y")
        assert abs(robot_x - cube_x) < 1e-1
        assert abs(robot_y - cube_y) < 1e-1
