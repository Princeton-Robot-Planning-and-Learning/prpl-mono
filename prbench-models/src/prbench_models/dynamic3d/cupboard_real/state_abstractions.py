"""State abstractions for the TidyBot3D cupboard real environment."""

from bilevel_planning.structs import (
    RelationalAbstractGoal,
    RelationalAbstractState,
)
from prbench.envs.dynamic3d.object_types import (
    MujocoObjectType,
    MujocoTidyBotRobotObjectType,
    MujocoFixtureObjectType,
    MujocoMovableObjectType,
)
from prbench_models.dynamic3d.ground.parameterized_skills import PyBulletSim
from prbench.envs.dynamic3d.tidybot_rewards import BaseMotionRewardCalculator
from relational_structs import (
    GroundAtom,
    ObjectCentricState,
    Predicate,
)
from prbench.envs.dynamic3d.tidybot3d import ObjectCentricTidyBot3DEnv
import numpy as np

# Predicates.
AtTarget = Predicate("AtTarget", [MujocoTidyBotRobotObjectType, MujocoObjectType])
OnFixture = Predicate("OnFixture", [MujocoObjectType, MujocoFixtureObjectType])
OnGround = Predicate("OnGround", [MujocoMovableObjectType])
Holding = Predicate("Holding", [MujocoTidyBotRobotObjectType, MujocoMovableObjectType])
HandEmpty = Predicate("HandEmpty", [MujocoTidyBotRobotObjectType])


class CupboardRealStateAbstractor:

    def __init__(self, sim: ObjectCentricTidyBot3DEnv) -> None:
        initial_state, _ = sim.reset()  # just need to access the objects
        self._pybullet_sim = PyBulletSim(initial_state)
    
    def state_abstractor(self, state: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state for the current state."""
        atoms: set[GroundAtom] = set()

        # Sync the pybullet simulator.
        self._pybullet_sim.set_state(state)

        # Uncomment to debug.
        # from pybullet_helpers.camera import capture_image
        # img = capture_image(
        #     self._pybullet_sim.physics_client_id,
        #     image_width=512,
        #     image_height=512,
        #     camera_yaw=90,
        #     camera_distance=2.5,
        #     camera_pitch=-20,
        #     camera_target=(0, 0, 0),
        # )
        # import imageio.v2 as iio
        # iio.imsave("pybullet_sim.png", img)
        # import ipdb; ipdb.set_trace()

        # Extract the relevant objects.
        robot = state.get_object_from_name("robot")
        fixtures = state.get_objects(MujocoFixtureObjectType)
        movables = state.get_objects(MujocoMovableObjectType)
        all_mujoco_objects = set(fixtures) | set(movables)

        # OnGround.
        on_ground_tol = 1e-2
        for target in movables:
            z = state.get(target, "z")
            bb_z = state.get(target, "bb_z")
            # Handle flipped cases later.
            assert np.isclose(state.get(target, "qx"), 0.0, atol=on_ground_tol)
            assert np.isclose(state.get(target, "qy"), 0.0, atol=on_ground_tol)
            if np.isclose(z - bb_z / 2, 0.0, atol=on_ground_tol):
                atoms.add(GroundAtom(OnGround, [target]))

        # AtTarget.
        for target in all_mujoco_objects:
            target_x = state.get(target, "x")
            target_y = state.get(target, "y")
            robot_x = state.get(robot, "pos_base_x")
            robot_y = state.get(robot, "pos_base_y")
            dx = target_x - robot_x
            dy = target_y - robot_y
            distance = (dx**2 + dy**2) ** 0.5
            # Divide threshold by 2 to avoid possible numerical issues.
            if distance <= BaseMotionRewardCalculator.dist_thresh / 2:
                atoms.add(GroundAtom(AtTarget, [robot, target]))

        # TODO: OnFixture.
        # for movable in movables:
        #     for fixture in fixtures:
        #         import ipdb; ipdb.set_trace()
        

        objects = {robot} | all_mujoco_objects
        return RelationalAbstractState(atoms, objects)


    def goal_deriver(self, state: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to have the robot on the target."""
        target = state.get_object_from_name("cube1")
        robot = state.get_object_from_name("robot")
        atoms = {GroundAtom(AtTarget, [robot, target])}
        return RelationalAbstractGoal(atoms, self.state_abstractor)
