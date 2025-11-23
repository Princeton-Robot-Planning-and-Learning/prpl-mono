"""Base robot class for dynamic3d environments."""

import abc
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from prbench.envs.dynamic3d.mujoco_utils import MjObs, MujocoEnv


class RobotEnv(MujocoEnv, abc.ABC):
    """Abstract base class for robots in dynamic3d environments."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the robot environment.

        Args:
            *args: Positional arguments passed to MujocoEnv.
            **kwargs: Keyword arguments passed to MujocoEnv.
        """
        super().__init__(*args, **kwargs)

        # Robot state/actuator references (initialized in _setup_robot_references)
        self.qpos: dict[str, NDArray[np.float64]] = {}
        self.qvel: dict[str, NDArray[np.float64]] = {}
        self.ctrl: dict[str, NDArray[np.float64]] = {}

    def _insert_robot_into_xml(
        self, xml_string: str, models_dir: str, robot_xml_name: str, assets_dir: str
    ) -> str:
        """Insert the robot model into the provided XML string."""
        # Parse the provided XML string
        input_tree = ET.ElementTree(ET.fromstring(xml_string))
        input_root = input_tree.getroot()

        # Read the scene XML content
        models_dir_path = Path(models_dir)
        robot_path = models_dir_path / robot_xml_name
        assets_dir_path = Path(assets_dir)
        # NOTE: currently manually handling duplicate geoms.xml
        # by creating duplicate asset directories. Probably
        # handle that in code through recursive include.

        with open(robot_path, "r", encoding="utf-8") as f:
            robot_content = f.read()

        # Parse robot XML
        robot_tree = ET.ElementTree(ET.fromstring(robot_content))
        robot_root = robot_tree.getroot()
        if robot_root is None:
            raise ValueError("Missing robot element")

        # Update compiler meshdir to absolute path in robot content
        robot_compiler = robot_root.find("compiler")  # type: ignore[union-attr]
        if robot_compiler is not None:
            robot_compiler.set("meshdir", str(assets_dir_path.resolve()))

        # Helper function to recursively make include file paths absolute
        def make_include_paths_absolute(element: ET.Element) -> None:
            """Recursively process an element and its children to make include file
            paths absolute."""
            if element.tag == "include" and element.get("file") is not None:
                file_path = element.get("file")
                if file_path and not Path(file_path).is_absolute():
                    # Make the file path absolute relative to the models directory
                    absolute_path = models_dir_path / file_path
                    element.set("file", str(absolute_path.resolve()))

            # Recursively process all children
            for child_elem in element:
                make_include_paths_absolute(child_elem)

        # Merge the robot content into the input XML
        # Copy all children from robot root to input root (except mujoco tag itself)
        for child in list(robot_root):
            if child.tag == "worldbody":
                # Merge worldbody content
                input_worldbody = input_root.find(  # type:ignore[union-attr]
                    "worldbody"
                )
                if input_worldbody is not None:
                    for robot_body in list(child):
                        # Process any include tags within robot_body and its children
                        make_include_paths_absolute(robot_body)
                        input_worldbody.append(robot_body)
                else:
                    input_root.append(child)  # type: ignore[union-attr]
            elif child.tag == "default":
                # Merge or append default sections
                input_section = input_root.find(child.tag)  # type: ignore[union-attr]
                if input_section is not None:
                    for sub_child in list(child):
                        input_section.append(sub_child)
                else:
                    input_root.append(child)  # type: ignore[union-attr]
            elif child.tag == "asset":
                # Merge or append asset sections
                input_section = input_root.find(child.tag)  # type: ignore[union-attr]
                if input_section is not None:
                    for sub_child in list(child):
                        # Check if the asset element has a "file" attribute
                        # and make it absolute
                        if sub_child.get("file") is not None:
                            file_path = sub_child.get("file")
                            if file_path and not Path(file_path).is_absolute():
                                # Make the file path absolute relative to the
                                # assets directory
                                absolute_path = assets_dir_path / file_path
                                sub_child.set("file", str(absolute_path.resolve()))
                        input_section.append(sub_child)
                else:
                    input_root.append(child)  # type: ignore[union-attr]
            else:
                # For other sections (compiler, actuator, contact, etc.), just append
                input_root.append(child)  # type: ignore[union-attr]

        if input_root is None:
            raise ValueError("input_root is None, cannot serialize to string")

        # Return the merged XML as string
        return ET.tostring(input_root, encoding="unicode")

    @abc.abstractmethod
    def reward(self, obs: MjObs) -> float:
        """Compute the reward from an observation.

        Args:
            obs: The observation to compute reward from.

        Returns:
            The computed reward value.
        """
