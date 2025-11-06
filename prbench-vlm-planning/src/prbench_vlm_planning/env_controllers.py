"""Environment-specific controller loading utilities."""

import importlib
import logging
from typing import Any, Optional

from bilevel_planning.structs import LiftedParameterizedController


def get_controllers_for_environment(
    env_class_name: str, env_name: str, action_space: Optional[Any] = None
) -> Optional[dict[str, LiftedParameterizedController]]:
    """Automatically load LiftedParameterizedControllers for a given environment.

    Args:
        env_class_name: Class name of the environment (e.g., "geom2d")
        env_name: Name of the environment (e.g., "motion2d", "clutteredretrieval2d")
        action_space: Optional action space to pass to create_lifted_controllers

    Returns:
        Dictionary of LiftedParameterizedControllers, or None if not available
    """
    # Generate module path dynamically
    # e.g., prbench_models.geom2d.envs.clutteredretrieval2d.parameterized_skills
    module_path = (
        f"prbench_models.{env_class_name}.envs.{env_name}.parameterized_skills"
    )
    return _import_lifted_controllers(module_path, env_name, action_space)


def _import_lifted_controllers(
    module_path: str, env_type: str, action_space: Optional[Any] = None
) -> Optional[dict[str, LiftedParameterizedController]]:
    """Import LiftedParameterizedControllers using create_lifted_controllers method.

    Args:
        module_path: Python import path to the parameterized_skills module
        env_type: Environment type name for logging
        action_space: Optional action space to pass to create_lifted_controllers

    Returns:
        Dictionary of LiftedParameterizedControllers, or None if import fails
    """
    try:
        # Import the module
        module = importlib.import_module(module_path)

        # Check if create_lifted_controllers method exists
        if not hasattr(module, "create_lifted_controllers"):
            raise NotImplementedError(
                f"Module {module_path} does not have a create_lifted_controllers method"
            )

        # Get the create_lifted_controllers function
        create_lifted_controllers = getattr(module, "create_lifted_controllers")

        # Call create_lifted_controllers with action_space if provided
        lifted_controllers = create_lifted_controllers(
            action_space=action_space, init_constant_state=None
        )

        if lifted_controllers:
            logging.info(
                f"Loaded {len(lifted_controllers)} lifted controllers for {env_type}: "
                f"{list(lifted_controllers.keys())}"
            )
            return lifted_controllers

        logging.info(f"No lifted controllers found in {module_path}")
        return None

    except NotImplementedError as e:
        logging.error(f"{env_type}: {e}")
        raise
    except ImportError as e:
        logging.info(f"{env_type} controllers not available: {e}")
        return None
    except Exception as e:
        logging.error(f"Error loading controllers from {module_path}: {e}")
        return None
