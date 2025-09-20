"""Bilevel planning models for the stick button 2D environment."""

import numpy as np
from bilevel_planning.structs import (
    LiftedParameterizedController,
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from gymnasium.spaces import Space
from numpy.typing import NDArray
from prbench.envs.geom2d.object_types import CircleType, CRVRobotType, RectangleType
from prbench.envs.geom2d.stickbutton2d import (
    ObjectCentricStickButton2DEnv,
    StickButton2DEnvConfig,
)
from prbench.envs.geom2d.utils import (
    CRVRobotActionSpace,
    get_suctioned_objects,
    object_to_multibody2d,
)
from prbench_models.geom2d.envs.stickbutton2d.parameterized_skills import (
    GroundPickStickController,
    GroundPlaceStickController,
    GroundRobotPressButtonController,
    GroundStickPressButtonController,
)
from relational_structs import (
    GroundAtom,
    LiftedAtom,
    LiftedOperator,
    Object,
    ObjectCentricState,
    Predicate,
    Variable,
)
from relational_structs.spaces import ObjectCentricBoxSpace, ObjectCentricStateSpace
from tomsgeoms2d.structs import Geom2D


def create_bilevel_planning_models(
    observation_space: Space, action_space: Space, num_buttons: int
) -> SesameModels:
    """Create the env models for stick button 2D."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, CRVRobotActionSpace)

    sim = ObjectCentricStickButton2DEnv(num_buttons=num_buttons)

    # Convert observations into states. The important thing is that states are hashable.
    def observation_to_state(o: NDArray[np.float32]) -> ObjectCentricState:
        """Convert the vectors back into (hashable) object-centric states."""
        return observation_space.devectorize(o)

    # Create the transition function.
    def transition_fn(
        x: ObjectCentricState, u: NDArray[np.float32]
    ) -> ObjectCentricState:
        """Simulate the action."""
        state = x.copy()
        sim.reset(options={"init_state": state})
        obs, _, _, _, _ = sim.step(u)
        return obs

    # Types.
    types = {CRVRobotType, RectangleType, CircleType}

    # Create the state space.
    state_space = ObjectCentricStateSpace(types)

    # Predicates.
    Grasped = Predicate("Grasped", [CRVRobotType, RectangleType])
    HandEmpty = Predicate("HandEmpty", [CRVRobotType])
    Pressed = Predicate("Pressed", [CircleType])
    RobotAboveButton = Predicate("RobotAboveButton", [CRVRobotType, CircleType])
    StickAboveButton = Predicate("StickAboveButton", [RectangleType, CircleType])
    AboveNoButton = Predicate("AboveNoButton", [])
    predicates = {
        Grasped,
        HandEmpty,
        Pressed,
        RobotAboveButton,
        StickAboveButton,
        AboveNoButton,
    }

    # State abstractor.
    def state_abstractor(x: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state for the current state."""
        robot = x.get_objects(CRVRobotType)[0]
        stick = x.get_objects(RectangleType)[0]
        buttons = x.get_objects(CircleType)

        atoms: set[GroundAtom] = set()

        # Add grasped / handempty atoms.
        suctioned_objs = {o for o, _ in get_suctioned_objects(x, robot)}
        if stick in suctioned_objs:
            atoms.add(GroundAtom(Grasped, [robot, stick]))
        else:
            atoms.add(GroundAtom(HandEmpty, [robot]))

        # Get button colors from Config
        config = StickButton2DEnvConfig()

        # Add button status atoms based on color.
        for button in buttons:
            button_color = (
                x.get(button, "color_r"),
                x.get(button, "color_g"),
                x.get(button, "color_b"),
            )
            if np.allclose(button_color, config.button_pressed_rgb, atol=1e-3):
                atoms.add(GroundAtom(Pressed, [button]))

        # Add spatial relationship atoms using geometry
        robot_multi_body = object_to_multibody2d(robot, x, {})
        stick_multi_body = object_to_multibody2d(stick, x, {})
        assert len(stick_multi_body.bodies) == 1
        stick_geom = stick_multi_body.bodies[0].geom
        robot_geom: set[Geom2D] = set()
        for body in robot_multi_body.bodies:
            robot_geom.add(body.geom)

        robot_above_any_button = False
        stick_above_any_button = False

        for button in buttons:
            button_multi_body = object_to_multibody2d(button, x, {})
            assert len(button_multi_body.bodies) == 1
            button_geom = button_multi_body.bodies[0].geom

            # Check if robot is above button (geometrically intersects)
            if any(
                robot_geom_sub.intersects(button_geom) for robot_geom_sub in robot_geom
            ):
                atoms.add(GroundAtom(RobotAboveButton, [robot, button]))
                robot_above_any_button = True

            # Check if stick is above button (geometrically intersects)
            if stick_geom.intersects(button_geom):
                atoms.add(GroundAtom(StickAboveButton, [stick, button]))
                stick_above_any_button = True

        # Add AboveNoButton if neither robot nor stick is above any button
        if not robot_above_any_button and not stick_above_any_button:
            atoms.add(GroundAtom(AboveNoButton, []))

        objects = {robot, stick} | set(buttons)
        return RelationalAbstractState(atoms, objects)

    # Goal abstractor.
    def goal_deriver(x: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to press all buttons."""
        del x  # not needed
        atoms: set[GroundAtom] = set()
        for i in range(num_buttons):
            button = Object(f"button{i}", CircleType)
            atoms.add(GroundAtom(Pressed, [button]))
        return RelationalAbstractGoal(atoms, state_abstractor)

    # Operators.
    robot = Variable("?robot", CRVRobotType)
    stick = Variable("?stick", RectangleType)
    button = Variable("?button", CircleType)
    from_button = Variable("?from_button", CircleType)

    # RobotPressButtonFromNothing
    RobotPressButtonFromNothingOperator = LiftedOperator(
        "RobotPressButtonFromNothing",
        [robot, button],
        preconditions={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(AboveNoButton, []),
        },
        add_effects={
            LiftedAtom(Pressed, [button]),
            LiftedAtom(RobotAboveButton, [robot, button]),
        },
        delete_effects={LiftedAtom(AboveNoButton, [])},
    )

    # RobotPressButtonFromButton
    RobotPressButtonFromButtonOperator = LiftedOperator(
        "RobotPressButtonFromButton",
        [robot, button, from_button],
        preconditions={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(RobotAboveButton, [robot, from_button]),
        },
        add_effects={
            LiftedAtom(Pressed, [button]),
            LiftedAtom(RobotAboveButton, [robot, button]),
        },
        delete_effects={LiftedAtom(RobotAboveButton, [robot, from_button])},
    )

    # PickStickFromNothing
    PickStickFromNothingOperator = LiftedOperator(
        "PickStickFromNothing",
        [robot, stick],
        preconditions={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(AboveNoButton, []),
        },
        add_effects={
            LiftedAtom(Grasped, [robot, stick]),
        },
        delete_effects={LiftedAtom(HandEmpty, [robot])},
    )

    # PickStickFromButton
    PickStickFromButtonOperator = LiftedOperator(
        "PickStickFromButton",
        [robot, stick, from_button],
        preconditions={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(RobotAboveButton, [robot, from_button]),
        },
        add_effects={
            LiftedAtom(Grasped, [robot, stick]),
            LiftedAtom(AboveNoButton, []),
        },
        delete_effects={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(RobotAboveButton, [robot, from_button]),
        },
    )

    # StickPressButtonFromNothing
    StickPressButtonFromNothingOperator = LiftedOperator(
        "StickPressButtonFromNothing",
        [robot, stick, button],
        preconditions={
            LiftedAtom(Grasped, [robot, stick]),
            LiftedAtom(AboveNoButton, []),
        },
        add_effects={
            LiftedAtom(StickAboveButton, [stick, button]),
            LiftedAtom(Pressed, [button]),
        },
        delete_effects={LiftedAtom(AboveNoButton, [])},
    )

    # StickPressButtonFromButton
    StickPressButtonFromButtonOperator = LiftedOperator(
        "StickPressButtonFromButton",
        [robot, stick, button, from_button],
        preconditions={
            LiftedAtom(Grasped, [robot, stick]),
            LiftedAtom(StickAboveButton, [stick, from_button]),
        },
        add_effects={
            LiftedAtom(StickAboveButton, [stick, button]),
            LiftedAtom(Pressed, [button]),
        },
        delete_effects={LiftedAtom(StickAboveButton, [stick, from_button])},
    )

    # PlaceStick
    PlaceStickOperator = LiftedOperator(
        "PlaceStick",
        [robot, stick],
        preconditions={
            LiftedAtom(Grasped, [robot, stick]),
        },
        add_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(AboveNoButton, [])},
        delete_effects={LiftedAtom(Grasped, [robot, stick])},
    )

    class RobotPressButtonController(GroundRobotPressButtonController):
        """Controller for moving the robot to press a button."""

        def __init__(self, objects):
            super().__init__(objects, action_space, sim.initial_constant_state)

    class PickStickController(GroundPickStickController):
        """Controller for moving the robot to pick the stick."""

        def __init__(self, objects):
            super().__init__(objects, action_space, sim.initial_constant_state)

    class StickPressButtonController(GroundStickPressButtonController):
        """Controller for moving the robot to use the stick to press a button."""

        def __init__(self, objects):
            super().__init__(objects, action_space, sim.initial_constant_state)

    class PlaceStickController(GroundPlaceStickController):
        """Controller for moving the robot to place the stick down."""

        def __init__(self, objects):
            super().__init__(objects, action_space, sim.initial_constant_state)

    # Create lifted controllers that match operator parameters exactly

    RobotPressButtonFromNothingController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, button],
            RobotPressButtonController,
        )
    )

    RobotPressButtonFromButtonController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, button, from_button],
            RobotPressButtonController,
        )
    )

    PickStickFromNothingController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, stick],
            PickStickController,
        )
    )

    PickStickFromButtonController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, stick, from_button],
            PickStickController,
        )
    )

    StickPressButtonFromNothingController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, stick, button],
            StickPressButtonController,
        )
    )

    StickPressButtonFromButtonController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, stick, button, from_button],
            StickPressButtonController,
        )
    )

    RobotPlaceStickController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, stick],
            PlaceStickController,
        )
    )

    # Finalize the skills.
    skills = {
        LiftedSkill(PickStickFromNothingOperator, PickStickFromNothingController),
        LiftedSkill(PickStickFromButtonOperator, PickStickFromButtonController),
        LiftedSkill(PlaceStickOperator, RobotPlaceStickController),
        LiftedSkill(
            RobotPressButtonFromNothingOperator, RobotPressButtonFromNothingController
        ),
        LiftedSkill(
            RobotPressButtonFromButtonOperator, RobotPressButtonFromButtonController
        ),
        LiftedSkill(
            StickPressButtonFromNothingOperator, StickPressButtonFromNothingController
        ),
        LiftedSkill(
            StickPressButtonFromButtonOperator, StickPressButtonFromButtonController
        ),
    }

    # Finalize the models.
    return SesameModels(
        observation_space,
        state_space,
        action_space,
        transition_fn,
        types,
        predicates,
        observation_to_state,
        state_abstractor,
        goal_deriver,
        skills,
    )
