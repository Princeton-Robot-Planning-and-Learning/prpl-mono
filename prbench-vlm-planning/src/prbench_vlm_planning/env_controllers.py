"""Environment-specific controller loading utilities."""

import importlib
import inspect
import logging
from typing import Any, Optional, Set

try:
    from prbench_models.geom2d.utils import Geom2dRobotController
except ImportError:
    Geom2dRobotController = None  # type: ignore


def get_controllers_for_environment(env_name: str) -> Optional[Set[Any]]:
    """Automatically load all controllers for a given environment from prbench_models.

    Args:
        env_name: Name of the environment (e.g., "Motion2D-p1", "StickButton2D-b3")

    Returns:
        Set with controller classes, or None if not available
    """

    # Environment type mapping
    env_types = [
        "Motion2D",
        "StickButton2D",
        "Obstruction2D",
        "ClutteredStorage2D",
        "ClutteredRetrieval2D",
    ]

    # Find matching environment type
    env_type = None
    for env in env_types:
        if env in env_name:
            env_type = env
            break

    if not env_type:
        logging.warning(
            f"No specific controllers available for environment: {env_name}"
        )
        return None

    # Generate module path dynamically
    env_name_lower = env_type.lower()
    module_path = f"prbench_models.geom2d.envs.{env_name_lower}." "parameterized_skills"
    return _import_all_controllers(module_path, env_type)


def _import_all_controllers(module_path: str, env_type: str) -> Optional[Set[Any]]:
    """Import all controller classes from a given module.

    Args:
        module_path: Python import path to the parameterized_skills module
        env_type: Environment type name for logging

    Returns:
        Set with all controller classes, or None if import fails
    """
    try:
        # Import the module
        module = importlib.import_module(module_path)

        # Check if base controller class is available
        if Geom2dRobotController is None:
            raise ImportError("Could not import Geom2dRobotController")

        # Find all controller classes that are subclasses of Geom2dRobotController
        controllers = set()
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if not name.startswith("_"):
                # Check if it's a subclass of Geom2dRobotController
                if (
                    Geom2dRobotController is not None
                    and issubclass(obj, Geom2dRobotController)
                    and obj != Geom2dRobotController
                ):
                    controllers.add(obj)

        if controllers:
            logging.info(
                f"Found {len(controllers)} controllers for {env_type}: "
                f"{[cls.__name__ for cls in controllers]}"
            )
            return controllers

        logging.info(f"No controllers found in {module_path}")
        return None

    except ImportError as e:
        logging.info(f"{env_type} controllers not available: {e}")
        return None
    except Exception as e:
        logging.error(f"Error loading controllers from {module_path}: {e}")
        return None
