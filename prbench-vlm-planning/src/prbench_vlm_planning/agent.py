"""VLM planning agent for prbench environments."""

import os
import re
from pathlib import Path
from typing import Any, Hashable, List, Optional, TypeVar, cast

import numpy as np
import PIL.Image
from numpy.typing import NDArray
from PIL import ImageDraw
from prpl_llm_utils.cache import FilePretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel
from prpl_utils.gym_agent import Agent


def create_vlm_by_name(model_name: str):
    """Create a VLM instance using prpl_llm_utils."""
    # Create a cache directory in the current working directory
    cache_dir = Path("./vlm_cache")
    cache_dir.mkdir(exist_ok=True)
    cache = FilePretrainedLargeModelCache(cache_dir)

    try:
        return OpenAIModel(model_name, cache)
    except Exception as e:
        raise ValueError(f"Failed to create VLM model: {e}")


_O = TypeVar("_O", bound=Hashable)
_U = TypeVar("_U", bound=Hashable)


class VLMPlanningAgentFailure(BaseException):
    """Raised when the VLM planning agent fails."""


class VLMPlanningAgent(Agent[_O, _U]):
    """VLM-based planning agent for prbench environments."""

    def __init__(
        self,
        observation_space: Any,
        vlm_model_name: str = "gpt-4o",
        temperature: float = 0.0,
        max_planning_horizon: int = 50,
        seed: int = 0,
        env_controllers: Optional[Any] = None,
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
        self._current_plan: Optional[List[_U]] = None
        self._plan_step = 0
        self._last_obs: Optional[_O] = None

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
        self._current_plan = None
        self._plan_step = 0

        try:
            self._current_plan = self._generate_plan(obs, info)
        except Exception as e:
            raise VLMPlanningAgentFailure(f"Failed to generate initial plan: {e}")

    def _get_action(self) -> _U:
        """Get the next action from the current plan."""
        if not self._current_plan:
            raise VLMPlanningAgentFailure("No current plan available")

        if self._plan_step >= len(self._current_plan):
            raise VLMPlanningAgentFailure("Plan exhausted")

        action = self._current_plan[self._plan_step]
        self._plan_step += 1
        return action

    def _generate_plan(self, obs: _O, info: dict[str, Any]) -> List[_U]:
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
        objects_str = self._get_objects_str(obs, info)
        actions_str = self._get_available_actions_str()
        goal_str = self._get_goal_str(info)

        prompt = self._base_prompt.format(
            actions=actions_str, objects=objects_str, goal=goal_str
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

            # Extract text from response
            plan_text = response.text

            # Parse the plan
            return self._parse_plan_from_text(plan_text, obs, info)

        except Exception as e:
            raise VLMPlanningAgentFailure(f"VLM query failed: {e}")

    def _get_available_actions_str(self) -> str:
        """Get string description of available actions."""
        return "Actions: TODO"

    def _get_objects_str(
        self, obs: _O, info: dict[str, Any]  # pylint: disable=unused-argument
    ) -> str:
        """Get string description of objects in the scene."""

        def observation_to_state(o: NDArray[np.float32]):
            """Convert the vectors back into (hashable) object-centric states."""
            return self._observation_space.devectorize(o)

        # Convert observation to state using observation space
        # Cast obs to the expected NDArray type for observation_to_state
        state = observation_to_state(cast(NDArray[np.float32], obs))
        return state.pretty_str()

    def _get_goal_str(
        self, info: dict[str, Any]  # pylint: disable=unused-argument
    ) -> str:
        """Get string description of the goal."""
        goal_description = info.get("description")
        assert isinstance(goal_description, str)
        return goal_description

    def _parse_plan_from_text(
        self,
        plan_text: str,
        obs: _O,  # pylint: disable=unused-argument
        info: dict[str, Any],  # pylint: disable=unused-argument
    ) -> List[_U]:
        """Parse the VLM output into a list of actions."""
        lines = plan_text.split("\n")
        plan_actions = []

        # Find the "Plan:" section
        plan_start = -1
        for i, line in enumerate(lines):
            if line.strip().lower().startswith("plan:"):
                plan_start = i + 1
                break

        if plan_start == -1:
            raise VLMPlanningAgentFailure(
                "Could not find 'Plan:' section in VLM output"
            )

        # Parse each plan step
        for line in lines[plan_start:]:
            line = line.strip()
            if not line or not any(c.isalnum() for c in line):
                continue

            # Remove numbering (e.g., "1. ", "2. ", etc.)
            line = re.sub(r"^\d+\.\s*", "", line)

            # Try to parse action arrays from the text
            if line:
                action = self._parse_action_from_text(line)
                if action is not None:
                    plan_actions.append(cast(_U, action))

                if len(plan_actions) >= self._max_planning_horizon:
                    break

        if not plan_actions:
            raise VLMPlanningAgentFailure("No valid actions parsed from VLM output")

        return plan_actions

    def _parse_action_from_text(self, action_text: str) -> Optional[np.ndarray]:
        """Parse a single action from text."""
        # Look for array-like patterns [a, b, c, d, e]
        pattern = (
            r"\[([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
            r"(?:\s*,\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)*)\]"
        )
        array_match = re.search(pattern, action_text)
        if array_match:
            try:
                # Parse the numbers from the array
                numbers_str = array_match.group(1)
                numbers = [float(x.strip()) for x in numbers_str.split(",")]
                return np.array(numbers, dtype=np.float32)
            except (ValueError, TypeError):
                pass

        # No valid action found
        raise ValueError(f"Unable to parse action from text: {action_text}")
