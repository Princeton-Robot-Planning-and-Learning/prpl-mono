"""State abstractions for the TidyBot3D cupboard real environment."""

import numpy as np
from bilevel_planning.structs import (
    RelationalAbstractGoal,
    RelationalAbstractState,
)
from prbench.envs.dynamic3d.object_types import (
    MujocoFixtureObjectType,
    MujocoMovableObjectType,
    MujocoObjectType,
    MujocoTidyBotRobotObjectType,
)
from prbench.envs.dynamic3d.tidybot3d import ObjectCentricTidyBot3DEnv
from relational_structs import (
    GroundAtom,
    ObjectCentricState,
    Predicate,
)

from prbench_models.dynamic3d.ground.parameterized_skills import PyBulletSim

# Predicates.
OnFixture = Predicate("OnFixture", [MujocoObjectType, MujocoFixtureObjectType])
OnGround = Predicate("OnGround", [MujocoObjectType])
Holding = Predicate("Holding", [MujocoTidyBotRobotObjectType, MujocoMovableObjectType])
HandEmpty = Predicate("HandEmpty", [MujocoTidyBotRobotObjectType])


class CupboardRealStateAbstractor:
    """State abstractor for the TidyBot3D cupboard real environment."""

    def __init__(self, sim: ObjectCentricTidyBot3DEnv) -> None:
        """Initialize the state abstractor."""
        initial_state, _ = sim.reset()  # just need to access the objects
        self._pybullet_sim = PyBulletSim(initial_state, rendering=False)

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
            if (
                np.isclose(z - bb_z / 2, 0.0, atol=on_ground_tol)
                and np.isclose(state.get(target, "qx"), 0.0, atol=on_ground_tol)
                and np.isclose(state.get(target, "qy"), 0.0, atol=on_ground_tol)
            ):
                atoms.add(GroundAtom(OnGround, [target]))

        # HandEmpty.
        handempty_tol = 1e-3
        gripper_val = state.get(robot, "pos_gripper")
        if np.isclose(gripper_val, 0.0, atol=handempty_tol):
            atoms.add(GroundAtom(HandEmpty, [robot]))

        # Holding.
        # checking the ee pose and target pose.
        GraspThreshold = 0.1
        gripper_val = state.get(robot, "pos_gripper")
        if gripper_val > GraspThreshold:
            for target in movables:
                target_ee_pose = self._pybullet_sim.get_ee_pose()
                if state.get(target, "z") > 0.1:
                    if (
                        abs(target_ee_pose.position[0] - state.get(target, "x")) < 0.05
                        and abs(target_ee_pose.position[1] - state.get(target, "y"))
                        < 0.05
                        and abs(target_ee_pose.position[2] - state.get(target, "z"))
                        < 0.05
                    ):
                        atoms.add(GroundAtom(Holding, [robot, target]))

        # OnFixture.
        for movable in movables:
            for fixture in fixtures:
                if (
                    abs(state.get(movable, "x") - state.get(fixture, "x")) < 0.15
                    and abs(state.get(movable, "y") - state.get(fixture, "y")) < 0.25
                    and state.get(movable, "z") > 0.3
                ):
                    if GroundAtom(Holding, [robot, movable]) not in atoms:
                        atoms.add(GroundAtom(OnFixture, [movable, fixture]))

        objects = {robot} | all_mujoco_objects
        return RelationalAbstractState(atoms, objects)

    def goal_deriver_grasp(self, state: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to grasp the target."""
        target = state.get_object_from_name("cube1")
        robot = state.get_object_from_name("robot")
        atoms = {GroundAtom(Holding, [robot, target])}
        return RelationalAbstractGoal(atoms, self.state_abstractor)

    def goal_deriver_place(self, state: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to place the target in the cupboard."""
        target = state.get_object_from_name("cube1")
        cupboard = state.get_object_from_name("cupboard_1")
        robot = state.get_object_from_name("robot")
        atoms = {
            GroundAtom(HandEmpty, [robot]),
            GroundAtom(OnFixture, [target, cupboard]),
        }
        if "cube2" in state.get_object_names():
            atoms.add(GroundAtom(OnGround, [state.get_object_from_name("cube2")]))
        return RelationalAbstractGoal(atoms, self.state_abstractor)

    def goal_deriver_place_cube2(
        self, state: ObjectCentricState
    ) -> RelationalAbstractGoal:
        """The goal is to place the target in the cupboard."""
        target = state.get_object_from_name("cube2")
        cupboard = state.get_object_from_name("cupboard_1")
        robot = state.get_object_from_name("robot")
        atoms = {
            GroundAtom(HandEmpty, [robot]),
            GroundAtom(OnFixture, [target, cupboard]),
        }
        return RelationalAbstractGoal(atoms, self.state_abstractor)

    def goal_deriver_place_two_cubes(
        self, state: ObjectCentricState
    ) -> RelationalAbstractGoal:
        """The goal is to place the target in the cupboard."""
        target = state.get_object_from_name("cube1")
        target2 = state.get_object_from_name("cube2")
        cupboard = state.get_object_from_name("cupboard_1")
        robot = state.get_object_from_name("robot")
        atoms = {
            GroundAtom(HandEmpty, [robot]),
            GroundAtom(OnFixture, [target, cupboard]),
            GroundAtom(OnFixture, [target2, cupboard]),
        }
        return RelationalAbstractGoal(atoms, self.state_abstractor)

    def goal_deriver_place_cupboard(
        self, state: ObjectCentricState
    ) -> RelationalAbstractGoal:
        """The goal is to place the target in the cupboard."""
        cupboard = state.get_object_from_name("cupboard_1")
        robot = state.get_object_from_name("robot")
        atoms = {
            GroundAtom(HandEmpty, [robot]),
        }
        for object_name in state.get_object_names():
            if "cube" in object_name:
                atoms.add(
                    GroundAtom(
                        OnFixture, [state.get_object_from_name(object_name), cupboard]
                    )
                )
        return RelationalAbstractGoal(atoms, self.state_abstractor)
