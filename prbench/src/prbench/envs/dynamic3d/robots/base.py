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

    def insert_visual_stage_into_xml(
        self, xml_string: str, visual_stage_path: str, scale: float = 10.0
    ) -> str:
        """Insert a visual stage model into the provided XML string.

        Args:
            xml_string: The base XML string to insert the visual stage into.
            visual_stage_path: Absolute path to the visual stage model.xml file.
            scale: Scale factor for the visual stage (default: 10.0 for apartment size).

        Returns:
            Modified XML string with visual stage included.
        """
        # Parse the provided XML string
        input_tree = ET.ElementTree(ET.fromstring(xml_string))
        input_root = input_tree.getroot()

        # Read the visual stage XML content
        visual_stage_path_obj = Path(visual_stage_path)
        if not visual_stage_path_obj.exists():
            raise FileNotFoundError(f"Visual stage file not found: {visual_stage_path}")

        with open(visual_stage_path_obj, "r", encoding="utf-8") as f:
            stage_content = f.read()

        # Parse visual stage XML
        stage_tree = ET.ElementTree(ET.fromstring(stage_content))
        stage_root = stage_tree.getroot()
        if stage_root is None:
            raise ValueError("Missing visual stage element")

        # Get the directory of the stage model for resolving relative paths
        stage_dir = visual_stage_path_obj.parent

        # Helper function to make file paths absolute and apply scale
        def make_file_paths_absolute_and_scale(
            element: ET.Element, base_dir: Path
        ) -> None:
            """Recursively make file paths absolute and apply scale to meshes."""
            # Check for 'file' attribute in mesh/texture elements
            if element.get("file") is not None:
                file_path = element.get("file")
                if file_path and not Path(file_path).is_absolute():
                    absolute_path = base_dir / file_path
                    element.set("file", str(absolute_path.resolve()))

            # Apply scale to mesh elements
            if element.tag == "mesh" and element.get("scale") is not None:
                # Parse existing scale and multiply by the scale factor
                existing_scale = element.get("scale", "1.0 1.0 1.0")
                scale_values = [float(s) * scale for s in existing_scale.split()]
                element.set("scale", f"{scale_values[0]} {scale_values[1]} {scale_values[2]}")

            # Recursively process all children
            for child_elem in element:
                make_file_paths_absolute_and_scale(child_elem, base_dir)

        # Merge the visual stage content into the input XML
        if input_root is None:
            raise ValueError("input_root is None, cannot merge visual stage")

        for child in list(stage_root):
            if child.tag == "worldbody":
                # Merge worldbody content
                input_worldbody = input_root.find("worldbody")
                if input_worldbody is not None:
                    for stage_body in list(child):
                        make_file_paths_absolute_and_scale(stage_body, stage_dir)
                        # Insert at the beginning to place stage behind other objects
                        input_worldbody.insert(0, stage_body)
                else:
                    input_root.append(child)
            elif child.tag == "asset":
                # Merge or append asset sections
                input_section = input_root.find(child.tag)
                if input_section is not None:
                    for sub_child in list(child):
                        make_file_paths_absolute_and_scale(sub_child, stage_dir)
                        input_section.append(sub_child)
                else:
                    make_file_paths_absolute_and_scale(child, stage_dir)
                    input_root.append(child)
            else:
                # For other sections, just append
                input_root.append(child)

        # Return the merged XML as string
        return ET.tostring(input_root, encoding="unicode")

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
