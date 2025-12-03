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
AtPremanipulationTarget = Predicate(
    "AtPremanipulationTarget", [MujocoTidyBotRobotObjectType, MujocoObjectType]
)
OnFixture = Predicate("OnFixture", [MujocoObjectType, MujocoFixtureObjectType])
OnGround = Predicate("OnGround", [MujocoObjectType])
Holding = Predicate("Holding", [MujocoTidyBotRobotObjectType, MujocoMovableObjectType])
HandEmpty = Predicate("HandEmpty", [MujocoTidyBotRobotObjectType])
AtHome = Predicate("AtHome", [MujocoTidyBotRobotObjectType])


class CupboardRealStateAbstractor:
    """State abstractor for the TidyBot3D cupboard real environment."""

    def __init__(self, sim: ObjectCentricTidyBot3DEnv) -> None:
        """Initialize the state abstractor."""
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
                    abs(state.get(movable, "x") - state.get(fixture, "x")) < 0.1
                    and abs(state.get(movable, "y") - state.get(fixture, "y")) < 0.1
                ):
                    if GroundAtom(Holding, [robot, movable]) not in atoms:
                        atoms.add(GroundAtom(OnFixture, [movable, fixture]))

        # AtPremanipulationTarget.
        for target in fixtures + movables:
            if target in fixtures:
                premanipulation_distance_threshold = (
                    0.95  # should be within this cardinal dist
                )
                premanipulation_angle_threshold = (
                    3 * 1e-2
                )  # should be facing the target object
            else:
                premanipulation_distance_threshold = (
                    0.6  # should be within this cardinal dist
                )
                premanipulation_angle_threshold = (
                    1e-2  # should be facing the target object
                )
            target_x = state.get(target, "x")
            target_y = state.get(target, "y")
            robot_x = state.get(robot, "pos_base_x")
            robot_y = state.get(robot, "pos_base_y")
            robot_rot = state.get(robot, "pos_base_rot")
            dx = target_x - robot_x
            dy = target_y - robot_y
            dist = (dx**2 + dy**2) ** 0.5
            if dist > premanipulation_distance_threshold:
                continue  # too far away
            # Desired direction from robot -> target
            target_angle = np.arctan2(dy, dx)

            # Smallest signed angular difference
            angle_error = abs((target_angle - robot_rot + np.pi) % (2 * np.pi) - np.pi)
            if angle_error < premanipulation_angle_threshold:
                atoms.add(GroundAtom(AtPremanipulationTarget, [robot, target]))
                break  # only one target can be at the premanipulation target

        at_home = True
        for target in all_mujoco_objects:
            if GroundAtom(AtPremanipulationTarget, [robot, target]) in atoms:
                at_home = False
                break  # found a target, so we are not at home
            if target in movables and GroundAtom(Holding, [robot, target]) in atoms:
                at_home = False
                break  # found a target, so we are not at home
        if at_home:
            atoms.add(GroundAtom(AtHome, [robot]))
        objects = {robot} | all_mujoco_objects
        return RelationalAbstractState(atoms, objects)

    def goal_deriver(self, state: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to have the robot on the target."""
        target = state.get_object_from_name("cube1")
        robot = state.get_object_from_name("robot")
        atoms = {GroundAtom(AtPremanipulationTarget, [robot, target])}
        return RelationalAbstractGoal(atoms, self.state_abstractor)

    def goal_deriver_grasp(self, state: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to grasp the target."""
        target = state.get_object_from_name("cube1")
        robot = state.get_object_from_name("robot")
        atoms = {GroundAtom(Holding, [robot, target])}
        return RelationalAbstractGoal(atoms, self.state_abstractor)

    def goal_deriver_grasp_move(
        self, state: ObjectCentricState
    ) -> RelationalAbstractGoal:
        """The goal is to grasp the target and move to the target."""
        target = state.get_object_from_name("cube1")
        target_cupboard = state.get_object_from_name("cupboard_1")
        robot = state.get_object_from_name("robot")
        atoms = {
            GroundAtom(Holding, [robot, target]),
            GroundAtom(AtPremanipulationTarget, [robot, target_cupboard]),
        }
        return RelationalAbstractGoal(atoms, self.state_abstractor)

    def goal_deriver_place(self, state: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to place the target in the cupboard."""
        target = state.get_object_from_name("cube1")
        cupboard = state.get_object_from_name("cupboard_1")
        robot = state.get_object_from_name("robot")
        atoms = {
            GroundAtom(AtPremanipulationTarget, [robot, cupboard]),
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
            GroundAtom(AtPremanipulationTarget, [robot, cupboard]),
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
            GroundAtom(AtPremanipulationTarget, [robot, cupboard]),
            GroundAtom(HandEmpty, [robot]),
            GroundAtom(OnFixture, [target, cupboard]),
            GroundAtom(OnFixture, [target2, cupboard]),
        }
        return RelationalAbstractGoal(atoms, self.state_abstractor)
