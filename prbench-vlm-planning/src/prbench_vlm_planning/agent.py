"""VLM planning agent for prbench environments."""

import logging
import os
from typing import (
    Any,
    Callable,
    Collection,
    Hashable,
    Optional,
    Sequence,
    TypeVar,
)

import numpy as np
import PIL.Image
from bilevel_planning.structs import (
    GroundParameterizedController,
    LiftedParameterizedController,
)
from PIL import ImageDraw
from prpl_utils.gym_agent import Agent
from relational_structs.objects import Type

from prbench_vlm_planning.utils import (
    controller_and_param_plan_to_policy,
    create_vlm_by_name,
    parse_model_output_into_option_plan,
)

_O = TypeVar("_O", bound=Hashable)
_U = TypeVar("_U", bound=Hashable)


class VLMPlanningAgentFailure(BaseException):
    """Raised when the VLM planning agent fails."""


class VLMPlanningAgent(Agent[_O, _U]):
    """VLM-based planning agent for prbench environments."""

    def __init__(
        self,
        observation_space: Any,
        env_controllers: dict[str, LiftedParameterizedController],
        vlm_model_name: str = "gpt-4o",
        temperature: float = 0.0,
        max_planning_horizon: int = 50,
        seed: int = 0,
        use_image: bool = True,
    ) -> None:
        """Initialize the VLM planning agent.

        Args:
            observation_space: Observation space with devectorize method
            vlm_model_name: Name of the VLM model to use
            temperature: Temperature for VLM sampling
            max_planning_horizon: Maximum steps in a plan
            seed: Random seed
            env_models: Optional environment models from prbench_models
            use_image: Whether to use image observations
        """
        super().__init__(seed)

        self._observation_space = observation_space
        self._vlm_model_name = vlm_model_name
        self._vlm = create_vlm_by_name(vlm_model_name)
        self._seed = seed
        self._temperature = temperature
        self._max_planning_horizon = max_planning_horizon
        self._controllers = env_controllers
        self._use_image = use_image

        # Current plan state
        self._current_policy: Optional[Callable[[_O], _U]] = None
        self._plan_step = 0
        self._last_obs: Optional[_O] = None
        self._next_action: Optional[_U] = None

        # Load base prompt from file
        self._base_prompt = self._load_base_prompt()

    def _load_base_prompt(self) -> str:
        """Load the base planning prompt from file."""
        # Get the path to the prompt file
        current_dir = os.path.dirname(__file__)
        prompt_path = os.path.join(current_dir, "prompts", "vlm_planning_prompt.txt")

        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def reset(self, obs: _O, info: dict[str, Any]) -> None:
        """Reset the agent for a new episode."""
        super().reset(obs, info)
        self._current_policy = None
        self._plan_step = 0

        try:
            self._current_policy = self._generate_plan(obs, info)
            self._next_action = self._current_policy(obs)
        except Exception as e:
            logging.exception("Failed to generate initial plan")
            raise VLMPlanningAgentFailure(f"Failed to generate initial plan: " f"{e}")

    def _get_action(self) -> _U:
        """Get the next action from the current plan."""
        if not self._current_policy:
            raise VLMPlanningAgentFailure("No current plan available")

        if self._plan_step >= self._max_planning_horizon:
            raise VLMPlanningAgentFailure("Plan exhausted")

        if self._next_action is None:
            raise VLMPlanningAgentFailure("No next action available")

        self._plan_step += 1
        return self._next_action

    def update(self, obs: _O, reward: float, done: bool, info: dict[str, Any]) -> None:
        """Update the agent with the latest observation and reward."""
        super().update(obs, reward, done, info)
        assert self._current_policy is not None
        self._next_action = self._current_policy(obs)

    def _generate_plan(self, obs: _O, info: dict[str, Any]) -> Callable[[_O], _U]:
        """Generate a plan using the VLM."""

        # Store observation for goal derivation
        self._last_obs = obs

        # Prepare images if available and using images
        images = None
        if (
            self._use_image
            and hasattr(obs, "get")
            and hasattr(obs, "__contains__")
            and "rgb" in obs
        ):
            rgb_obs = obs["rgb"]  # type: ignore
            if isinstance(rgb_obs, np.ndarray):
                pil_img = PIL.Image.fromarray(rgb_obs)
                # Add text overlay indicating this is the initial state
                draw = ImageDraw.Draw(pil_img)
                text = "Initial state for planning"
                # Simple text overlay at top-left
                draw.text((10, 10), text, fill="white")
                images = [pil_img]

        # Prepare prompt context
        state = self._observation_space.devectorize(obs)
        controller_str = self._get_controllers_str()
        goal_str = self._get_goal_str(info)

        prompt = self._base_prompt.format(
            controllers=controller_str,
            typed_objects=set(state.data),
            type_hierarchy=self.create_types_str(set(state.type_features)),
            init_state_str=state.pretty_str(),
            goal_str=goal_str,
        )

        # Query VLM
        try:
            # Prepare hyperparameters for prpl_llm_utils
            hyperparameters = {
                "temperature": self._temperature,
                "seed": self._seed,
            }

            # Query the VLM
            response = self._vlm.query(
                prompt=prompt, imgs=images, hyperparameters=hyperparameters
            )

            # Parse the plan
            plan_prediction_txt = response.text
            try:
                start_index = plan_prediction_txt.index("Plan:\n") + len("Plan:\n")
                parsable_plan_prediction = plan_prediction_txt[start_index:]
            except ValueError:
                raise ValueError("VLM output is badly formatted; cannot parse plan!")
            parsed_controller_plan = parse_model_output_into_option_plan(
                parsable_plan_prediction,
                set(state.data),
                set(state.type_features),
                self._controllers,
                parse_continuous_params=True,
            )
            controller_and_params_plan: list[
                tuple[GroundParameterizedController, Sequence[float]]
            ] = []
            for controller, objs, params in parsed_controller_plan:
                logging.info(
                    f"Parsed option: {controller} with objects "
                    f"{objs} and params {params}"
                )
                grounded_controller = controller.ground(objs)
                controller_and_params_plan.append((grounded_controller, params))

            policy = controller_and_param_plan_to_policy(
                controller_and_params_plan,
                self._max_planning_horizon,
                self._observation_space,
            )
            return policy

        except Exception as e:
            logging.exception("VLM query failed")
            raise VLMPlanningAgentFailure(f"VLM query failed: {e}")

    def _get_controllers_str(self) -> str:
        """Get string description of available actions."""
        controllers_str = "\n".join(
            f"{name}{controller.var_str}"
            for name, controller in self._controllers.items()
            if self._controllers
        )
        return controllers_str

    def _get_goal_str(
        self, info: dict[str, Any]  # pylint: disable=unused-argument
    ) -> str:
        """Get string description of the goal."""
        goal_description = info.get("description")
        assert isinstance(goal_description, str)
        return goal_description

    def create_types_str(self, types: Collection[Type]) -> str:
        """Create a PDDL-style types string that handles hierarchy correctly."""
        # Case 1: no type hierarchy.
        if all(t.parent is None for t in types):
            types_str = " ".join(t.name for t in sorted(types))
        # Case 2: type hierarchy.
        else:
            parent_to_children_types: dict[Type, list[Type]] = {t: [] for t in types}
            for t in sorted(types):
                if t.parent:
                    parent_to_children_types[t.parent].append(t)
            types_str = ""
            for parent_type in sorted(parent_to_children_types):
                child_types = parent_to_children_types[parent_type]
                if not child_types:
                    # Special case: type has no children and also does not appear
                    # as a child of another type.
                    is_child_type = any(
                        parent_type in children
                        for children in parent_to_children_types.values()
                    )
                    if not is_child_type:
                        types_str += f"\n    {parent_type.name}"
                    # Otherwise, the type will appear as a child elsewhere.
                else:
                    child_type_str = " ".join(t.name for t in child_types)
                    types_str += f"\n    {child_type_str} - {parent_type.name}"
        return types_str
