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
        env_class_name: Class name of the environment
            (e.g., "kinematic2d", "kinematic3d")
        env_name: Name of the environment (e.g., "motion2d", "clutteredretrieval2d")
        action_space: Optional action space to pass to create_lifted_controllers

    Returns:
        Dictionary of LiftedParameterizedControllers, or None if not available
    """
    # Generate module path dynamically
    # e.g., kinder_models.kinematic2d.envs.clutteredretrieval2d.parameterized_skills
    if env_class_name == "kinematic2d":
        module_path = (
            f"kinder_models.{env_class_name}.envs.{env_name}.parameterized_skills"
        )
    else:
        module_path = f"kinder_models.{env_class_name}.{env_name}.parameterized_skills"

    return _import_lifted_controllers(
        module_path, env_name, env_class_name, action_space
    )


def _import_lifted_controllers(
    module_path: str,
    env_name: str,
    env_class_name: str,
    action_space: Optional[Any] = None,
) -> Optional[dict[str, LiftedParameterizedController]]:
    """Import LiftedParameterizedControllers using create_lifted_controllers method.

    Args:
        module_path: Python import path to the parameterized_skills module
        env_name: Environment name for logging
        env_class_name: Environment class name (e.g., "kinematic2d", "kinematic3d")
        action_space: Optional action space to pass to create_lifted_controllers

    Returns:
        Dictionary of LiftedParameterizedControllers, or None if import fails
    """
    try:
        # Import the module
        module = importlib.import_module(module_path)
        logging.info(f"Imported module: {module_path}")

        # Check if create_lifted_controllers method exists
        if not hasattr(module, "create_lifted_controllers"):
            raise NotImplementedError(
                f"Module {module_path} does not have a create_lifted_controllers method"
            )

        # Get the create_lifted_controllers function
        create_lifted_controllers = getattr(module, "create_lifted_controllers")

        # Call create_lifted_controllers with appropriate parameters
        # based on env_class_name
        if env_class_name == "kinematic3d":
            # For kinematic3d environments, we need to create a sim object
            # Import the environment class dynamically
            env_module_path = f"kinder.envs.{env_class_name}.{env_name}"
            env_module = importlib.import_module(env_module_path)

            # Find the environment class (typically ObjectCentric{EnvName}Env)
            # Convert env_name to CamelCase, handling special cases like "3d" -> "3D"
            words = env_name.split("_")
            camel_words = []
            for word in words:
                if word[-2:] in ["2d", "3d"]:
                    word = word[:-1].capitalize() + word[-1:].upper()
                    camel_words.append(word)
                else:
                    camel_words.append(word.capitalize())
            env_class_name_camel = "".join(camel_words)
            sim_class_name = f"ObjectCentric{env_class_name_camel}Env"

            if not hasattr(env_module, sim_class_name):
                raise ImportError(
                    f"Could not find class {sim_class_name} in {env_module_path}"
                )

            sim_class = getattr(env_module, sim_class_name)
            sim = sim_class()

            # Call with (action_space, sim) for kinematic3d
            lifted_controllers = create_lifted_controllers(action_space, sim)
        else:
            # For kinematic2d and other environments,
            # use (action_space, init_constant_state)
            lifted_controllers = create_lifted_controllers(
                action_space=action_space, init_constant_state=None
            )

        if lifted_controllers:
            logging.info(
                f"Loaded {len(lifted_controllers)} lifted controllers for {env_name}: "
                f"{list(lifted_controllers.keys())}"
            )
            return lifted_controllers

        logging.info(f"No lifted controllers found in {module_path}")
        return None

    except NotImplementedError as e:
        logging.error(f"{env_name}: {e}")
        raise
    except ImportError as e:
        logging.info(f"{env_name} controllers not available: {e}")
        return None
    except Exception as e:
        logging.error(f"Error loading controllers from {module_path}: {e}")
        return None
