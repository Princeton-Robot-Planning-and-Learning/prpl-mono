"""Object definitions for TidyBot environments."""

from __future__ import annotations

import abc
import math
import xml.etree.ElementTree as ET
from typing import TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from relational_structs import Object

from prbench.envs.dynamic3d.mujoco_utils import MujocoEnv
from prbench.envs.dynamic3d.object_types import (
    MujocoFixtureObjectType,
    MujocoObjectType,
)

# Type variables for decorator type preservation
FixtureT = TypeVar("FixtureT", bound="MujocoFixture")
ObjectT = TypeVar("ObjectT", bound="MujocoObject")

REGISTERED_FIXTURES: dict[str, type[MujocoFixture]] = {}
REGISTERED_OBJECTS: dict[str, type[MujocoObject]] = {}


def register_fixture(cls: type[FixtureT]) -> type[FixtureT]:
    """Register fixture classes for TidyBot environments."""
    REGISTERED_FIXTURES[cls.__name__.lower()] = cls
    return cls


def register_object(cls: type[ObjectT]) -> type[ObjectT]:
    """Register object classes for TidyBot environments."""
    REGISTERED_OBJECTS[cls.__name__.lower()] = cls
    return cls


def get_fixture_class(name: str) -> type[MujocoFixture]:
    """Get a fixture class by name.

    Args:
        name: Name of the fixture class (case-insensitive)

    Returns:
        The fixture class

    Raises:
        ValueError: If the fixture class is not found
    """
    name_lower = name.lower()
    if name_lower not in REGISTERED_FIXTURES:
        available_fixtures = list(REGISTERED_FIXTURES.keys())
        raise ValueError(
            f"Fixture class '{name}' not found. "
            f"Available fixtures: {available_fixtures}"
        )
    return REGISTERED_FIXTURES[name_lower]


def get_object_class(name: str) -> type[MujocoObject]:
    """Get an object class by name.

    Args:
        name: Name of the object class (case-insensitive)

    Returns:
        The object class

    Raises:
        ValueError: If the object class is not found
    """
    name_lower = name.lower()
    if name_lower not in REGISTERED_OBJECTS:
        available_objects = list(REGISTERED_OBJECTS.keys())
        raise ValueError(
            f"Object class '{name}' not found. "
            f"Available objects: {available_objects}"
        )
    return REGISTERED_OBJECTS[name_lower]


class MujocoObject:
    """Base class for MuJoCo objects with position and orientation control."""

    def __init__(
        self,
        name: str,
        env: MujocoEnv | None = None,
        options: dict | None = None,
    ) -> None:
        """Initialize a MujocoObject.

        Args:
            name: Name of the object body in the XML
            env: Reference to the environment (needed for position get/set operations)
        """
        self.name = name
        self.joint_name = f"{name}_joint"
        self.env = env
        self.options = options if options is not None else {}

        # Create the corresponding Object for state representation key
        self.symbolic_object = Object(self.name, MujocoObjectType)

        self.xml_element: ET.Element  # To be defined in subclasses

    def get_position(self) -> NDArray[np.float32]:
        """Get the object's current position.

        Returns:
            Position as [x, y, z] array

        Raises:
            ValueError: If environment is not set
        """
        if self.env is None:
            raise ValueError("Environment must be set to get position")

        pos, _ = self.env.get_joint_pos_quat(self.joint_name)
        return pos

    def get_orientation(self) -> NDArray[np.float32]:
        """Get the object's current orientation.

        Returns:
            Orientation as quaternion [w, x, y, z] array

        Raises:
            ValueError: If environment is not set
        """
        if self.env is None:
            raise ValueError("Environment must be set to get orientation")

        _, quat = self.env.get_joint_pos_quat(self.joint_name)
        return quat

    def get_pose(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Get the object's current position and orientation.

        Returns:
            Tuple of (position, quaternion)

        Raises:
            ValueError: If environment is not set
        """
        if self.env is None:
            raise ValueError("Environment must be set to get pose")

        return self.env.get_joint_pos_quat(self.joint_name)

    def set_position(self, position: Union[list[float], NDArray[np.float32]]) -> None:
        """Set the object's position.

        Args:
            position: New position as [x, y, z]

        Raises:
            ValueError: If environment is not set
        """
        if self.env is None:
            raise ValueError("Environment must be set to set position")

        # Get current orientation to preserve it
        _, current_quat = self.env.get_joint_pos_quat(self.joint_name)

        # Set new position with current orientation
        self.env.set_joint_pos_quat(self.joint_name, np.array(position), current_quat)

    def set_orientation(
        self, quaternion: Union[list[float], NDArray[np.float32]]
    ) -> None:
        """Set the object's orientation.

        Args:
            quaternion: New orientation as quaternion [w, x, y, z]

        Raises:
            ValueError: If environment is not set
        """
        if self.env is None:
            raise ValueError("Environment must be set to set orientation")

        # Get current position to preserve it
        current_pos, _ = self.env.get_joint_pos_quat(self.joint_name)

        # Set new orientation with current position
        self.env.set_joint_pos_quat(self.joint_name, current_pos, np.array(quaternion))

    def set_pose(
        self,
        position: Union[list[float], NDArray[np.float32]],
        quaternion: Union[list[float], NDArray[np.float32]],
    ) -> None:
        """Set the object's position and orientation.

        Args:
            position: New position as [x, y, z]
            quaternion: New orientation as quaternion [w, x, y, z]

        Raises:
            ValueError: If environment is not set
        """
        if self.env is None:
            raise ValueError("Environment must be set to set pose")

        self.env.set_joint_pos_quat(
            self.joint_name, np.array(position), np.array(quaternion)
        )

    def set_velocity(
        self,
        linear_velocity: Union[list[float], NDArray[np.float32]],
        angular_velocity: Union[list[float], NDArray[np.float32]],
    ) -> None:
        """Set the object's linear and angular velocity.

        Args:
            linear_velocity: New linear velocity as [vx, vy, vz]
            angular_velocity: New angular velocity as [wx, wy, wz]

        Raises:
            ValueError: If environment is not set
        """
        if self.env is None:
            raise ValueError("Environment must be set to set velocity")

        self.env.set_joint_vel(
            self.joint_name, np.array(linear_velocity), np.array(angular_velocity)
        )

    @abc.abstractmethod
    def get_bounding_box_dimensions(self) -> tuple[float, float, float]:
        """Get the bounding box dimensions for this object.

        These bounding box dimensions are independent from the object pose.
        """

    def get_object_centric_data(self) -> dict[str, float]:
        """Get the object's current data.

        Returns:
            dict with current position and orientation

        Raises:
            ValueError: If environment is not set
        """
        if self.env is None:
            raise ValueError("Environment must be set to get state")

        pos, quat = self.env.get_joint_pos_quat(self.joint_name)
        linear_vel, angular_vel = self.env.get_joint_vel(self.joint_name)
        bb_x, bb_y, bb_z = self.get_bounding_box_dimensions()

        # Create and return the data
        obj_data = {
            "x": pos[0],
            "y": pos[1],
            "z": pos[2],
            "qw": quat[0],
            "qx": quat[1],
            "qy": quat[2],
            "qz": quat[3],
            "vx": linear_vel[0],
            "vy": linear_vel[1],
            "vz": linear_vel[2],
            "wx": angular_vel[0],
            "wy": angular_vel[1],
            "wz": angular_vel[2],
            "bb_x": bb_x,
            "bb_y": bb_y,
            "bb_z": bb_z,
        }
        return obj_data


@register_object
class Cube(MujocoObject):
    """A cube object for TidyBot environments."""

    def __init__(
        self,
        name: str,
        env: MujocoEnv | None = None,
        options: dict | None = None,
    ) -> None:
        """Initialize a Cube object.

        Args:
            name: Name of the cube body in the XML
            options: Dictionary of cube options:
                - size: Size of the cube (either scalar or [x, y, z] dimensions)
                - rgba: Color of the cube (either string or [r, g, b, a] values)
                - mass: Mass of the cube
            env: Reference to the environment (needed for position get/set operations)
        """
        # Initialize base class
        super().__init__(name, env, options)

        # Override object type
        self.symbolic_object = Object(self.name, MujocoObjectType)

        # Handle size parameter
        size = self.options.get("size", 0.02)
        if isinstance(size, (int, float)):
            self.size = [size, size, size]
        else:
            self.size = list(size)

        # Handle rgba parameter
        rgba = self.options.get("rgba", ".5 .7 .5 1")
        if isinstance(rgba, str):
            self.rgba = rgba
        else:
            self.rgba = " ".join(str(x) for x in rgba)

        self.mass = self.options.get("mass", 0.1)

        # Create the XML element
        self.xml_element = self._create_xml_element()

    def _create_xml_element(self) -> ET.Element:
        """Create the XML Element for this cube.

        Returns:
            ET.Element representing the cube body
        """
        # Create body element
        body = ET.Element("body", name=self.name)

        # Add freejoint for position/orientation control
        ET.SubElement(body, "freejoint", name=self.joint_name)

        # Add geom element with cube properties
        size_str = " ".join(str(x) for x in self.size)
        ET.SubElement(
            body,
            "geom",
            type="box",
            size=size_str,
            rgba=self.rgba,
            mass=str(self.mass),
        )

        return body

    def __str__(self) -> str:
        """String representation of the cube."""
        return (
            f"Cube(name='{self.name}', size={self.size}, "
            f"rgba='{self.rgba}', mass={self.mass})"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the cube."""
        return (
            f"Cube(name='{self.name}', joint_name='{self.joint_name}', "
            f"size={self.size}, rgba='{self.rgba}', mass={self.mass})"
        )

    def get_bounding_box_dimensions(self) -> tuple[float, float, float]:
        return (2 * self.size[0], 2 * self.size[1], 2 * self.size[2])


class MujocoFixture(abc.ABC):
    """Base class for MuJoCo fixtures (static objects).

    These are non-movable objects, like tables, that cannot be manipulated by the robot,
    and cannot change position/orientation after sim initialization.
    """

    def __init__(
        self,
        name: str,
        fixture_config: dict[str, str | float],
        position: list[float] | NDArray[np.float32],
        regions: dict | None = None,
    ) -> None:
        """Initialize a MujocoFixture.

        Args:
            name: Name of the fixture body in the XML
            fixture_config: Dictionary containing fixture configuration
            position: Position of the fixture as [x, y, z]
        """
        self.name = name
        self.fixture_config = fixture_config
        self.position = position
        self.regions = regions

        # Create the corresponding Object for state representation key
        self.symbolic_object = Object(self.name, MujocoFixtureObjectType)

        self.xml_element: ET.Element  # To be defined in subclasses

    def get_position(self) -> NDArray[np.float32]:
        """Get the fixture's position.

        Returns:
            Position as [x, y, z] array
        """
        return np.array(self.position)

    def get_orientation(self) -> NDArray[np.float32]:
        """Get the fixture's orientation.

        Returns:
            Orientation as quaternion [w, x, y, z] array
        """
        # Fixtures are static and assumed to have no rotation by default
        return np.array([1.0, 0.0, 0.0, 0.0])

    def get_object_centric_data(self) -> dict[str, float]:
        """Get the object's current data.

        Returns:
            dict with current position and orientation

        Raises:
            ValueError: If environment is not set
        """
        pos = self.get_position()
        quat = self.get_orientation()

        # Create and return the data
        obj_data = {
            "x": pos[0],
            "y": pos[1],
            "z": pos[2],
            "qw": quat[0],
            "qx": quat[1],
            "qy": quat[2],
            "qz": quat[3],
        }
        return obj_data

    @abc.abstractmethod
    def _create_xml_element(self) -> ET.Element:
        """Create the XML Element for this fixture.

        Returns:
            ET.Element representing the fixture body
        """

    @abc.abstractmethod
    def sample_pose_in_region(
        self,
        regions: list[list[float]],
        np_random: np.random.Generator,
    ) -> tuple[float, float, float]:
        """Sample a pose (x, y, z) uniformly randomly from one of the provided regions.

        Args:
            regions: List of bounding boxes, where each bounding box is a list of
                    4 floats: [x_start, y_start, x_end, y_end] in table-relative
                    coordinates
            np_random: Random number generator. If None, uses numpy's default random

        Returns:
            Tuple of (x, y, z) coordinates in world coordinates (offset by table
            position)

        Raises:
            ValueError: If regions list is empty or if any region has invalid bounds
        """


@register_fixture
class Table(MujocoFixture):
    """A table fixture."""

    def __init__(
        self,
        name: str,
        fixture_config: dict[str, str | float],
        position: list[float] | NDArray[np.float32],
        regions: dict | None = None,
    ) -> None:
        """Initialize a Table object.

        Args:
            name: Name of the table body in the XML
            fixture_config: Dictionary containing table configuration with keys:
                - "shape": Shape of the table - "rectangle" or "circle"
                - "height": Table height in meters
                - "thickness": Table top thickness in meters
                - "length": Total table length in meters (for rectangle)
                - "width": Total table width in meters (for rectangle)
                - "diameter": Diameter of circular table in meters (for circle)
            position: Position of the table as [x, y, z]
        """
        # Initialize base class
        super().__init__(name, fixture_config, position, regions)

        # Parse table configuration
        self.table_shape = str(self.fixture_config["shape"])
        self.table_height = float(self.fixture_config["height"])
        self.table_thickness = float(self.fixture_config["thickness"])
        self.leg_inset = 0.05

        # Optional parameters
        self.table_length: float | None = None
        self.table_width: float | None = None
        self.table_diameter: float | None = None

        # Shape-specific parameters
        if self.table_shape == "rectangle":
            self.table_length = float(self.fixture_config["length"])
            self.table_width = float(self.fixture_config["width"])
        elif self.table_shape == "circle":
            self.table_diameter = float(self.fixture_config["diameter"])
        else:
            raise ValueError(
                f"Unknown table shape: {self.table_shape}. "
                f"Must be 'rectangle' or 'circle'"
            )

        # Create the XML element
        self.xml_element = self._create_xml_element()

    def _create_xml_element(self) -> ET.Element:
        """Create the XML Element for this table.

        Returns:
            ET.Element representing the table body
        """
        # Create table body element
        table_body = ET.Element("body")
        table_body.set("name", self.name)
        position_str = " ".join(str(x) for x in self.position)
        table_body.set("pos", position_str)

        if self.table_shape == "rectangle":
            assert self.table_length is not None
            assert self.table_width is not None

            # Calculate MuJoCo geom sizes (half the actual dimensions)
            table_half_length = float(self.table_length) / 2
            table_half_width = float(self.table_width) / 2
            table_half_thickness = self.table_thickness / 2
            leg_radius = 0.02
            leg_half_height = (self.table_height - self.table_thickness) / 2

            # Calculate leg positions (inset from edges)
            leg_x_offset = table_half_length - self.leg_inset
            leg_y_offset = table_half_width - self.leg_inset
            leg_z_pos = leg_half_height  # Center of leg cylinder
            table_top_z_pos = self.table_height - table_half_thickness

            # Create rectangular table top geom
            table_top = ET.SubElement(table_body, "geom")
            table_top.set("name", f"{self.name}_top")
            table_top.set("type", "box")
            table_top.set(
                "size",
                f"{table_half_length} {table_half_width} {table_half_thickness}",
            )
            table_top.set("pos", f"0 0 {table_top_z_pos}")
            table_top.set("rgba", "0.8 0.6 0.4 1")

            # Create table legs at four corners
            leg_positions = [
                (
                    f"{leg_x_offset} {leg_y_offset} {leg_z_pos}",
                    f"{self.name}_leg1",
                ),
                (
                    f"{-leg_x_offset} {leg_y_offset} {leg_z_pos}",
                    f"{self.name}_leg2",
                ),
                (
                    f"{leg_x_offset} {-leg_y_offset} {leg_z_pos}",
                    f"{self.name}_leg3",
                ),
                (
                    f"{-leg_x_offset} {-leg_y_offset} {leg_z_pos}",
                    f"{self.name}_leg4",
                ),
            ]

            for pos, name in leg_positions:
                leg = ET.SubElement(table_body, "geom")
                leg.set("name", name)
                leg.set("type", "cylinder")
                leg.set("size", f"{leg_radius} {leg_half_height}")
                leg.set("pos", pos)
                leg.set("rgba", "0.6 0.4 0.2 1")

        elif self.table_shape == "circle":
            assert self.table_diameter is not None

            # Calculate MuJoCo geom sizes for circular table
            table_radius = float(self.table_diameter) / 2
            table_half_thickness = self.table_thickness / 2
            leg_radius = 0.02
            leg_half_height = (self.table_height - self.table_thickness) / 2

            # Calculate leg positions (inset from edge on a circle)
            # Place 4 legs at 45-degree intervals from edge
            leg_distance_from_center = table_radius - self.leg_inset
            leg_z_pos = leg_half_height  # Center of leg cylinder
            table_top_z_pos = self.table_height - table_half_thickness

            # Create circular table top geom (using cylinder)
            table_top = ET.SubElement(table_body, "geom")
            table_top.set("name", f"{self.name}_top")
            table_top.set("type", "cylinder")
            table_top.set("size", f"{table_radius} {table_half_thickness}")
            table_top.set("pos", f"0 0 {table_top_z_pos}")
            table_top.set("rgba", "0.8 0.6 0.4 1")

            # Create table legs at 4 positions around the circle
            # (at 45, 135, 225, 315 degrees)
            leg_angles = [
                math.pi / 4,
                3 * math.pi / 4,
                5 * math.pi / 4,
                7 * math.pi / 4,
            ]  # 45, 135, 225, 315 degrees

            for i, angle in enumerate(leg_angles, 1):
                leg_x = leg_distance_from_center * math.cos(angle)
                leg_y = leg_distance_from_center * math.sin(angle)

                leg = ET.SubElement(table_body, "geom")
                leg.set("name", f"{self.name}_leg{i}")
                leg.set("type", "cylinder")
                leg.set("size", f"{leg_radius} {leg_half_height}")
                leg.set("pos", f"{leg_x} {leg_y} {leg_z_pos}")
                leg.set("rgba", "0.6 0.4 0.2 1")

        else:
            raise ValueError(
                f"Unknown table shape: {self.table_shape}. "
                f"Must be 'rectangle' or 'circle'"
            )

        return table_body

    def sample_pose_in_region(
        self,
        regions: list[list[float]],
        np_random: np.random.Generator,
    ) -> tuple[float, float, float]:
        """Sample a pose (x, y, z) uniformly randomly from one of the provided regions.

        Args:
            regions: List of bounding boxes, where each bounding box is a list of
                    4 floats: [x_start, y_start, x_end, y_end] in table-relative
                    coordinates
            np_random: Random number generator. If None, uses numpy's default random

        Returns:
            Tuple of (x, y, z) coordinates in world coordinates (offset by table
            position)

        Raises:
            ValueError: If regions list is empty or if any region has invalid bounds
        """
        if not regions:
            raise ValueError("Regions list cannot be empty")

        # Randomly select one of the regions
        selected_region = np_random.choice(regions)

        # Validate the selected region
        if len(selected_region) != 4:
            raise ValueError(
                f"Each region must have exactly 4 values "
                f"[x_start, y_start, x_end, y_end], got {len(selected_region)}"
            )

        x_start, y_start, x_end, y_end = selected_region

        # Validate bounds
        if x_start >= x_end:
            raise ValueError(f"x_start ({x_start}) must be less than x_end ({x_end})")
        if y_start >= y_end:
            raise ValueError(f"y_start ({y_start}) must be less than y_end ({y_end})")

        # Sample uniformly within the selected region
        x = np_random.uniform(x_start, x_end)
        y = np_random.uniform(y_start, y_end)

        # Sample z coordinate on top of the table
        z = self.table_height + 0.1  # Slightly above the table surface

        # Offset by the table's position to get world coordinates
        world_x = x + self.position[0]
        world_y = y + self.position[1]
        world_z = z + self.position[2]

        return (world_x, world_y, world_z)

    def __str__(self) -> str:
        """String representation of the table."""
        return (
            f"Table(name='{self.name}', shape='{self.table_shape}', "
            f"height={self.table_height}, thickness={self.table_thickness})"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the table."""
        return (
            f"Table(name='{self.name}', "
            f"shape='{self.table_shape}', length={self.table_length}, "
            f"width={self.table_width}, diameter={self.table_diameter}, "
            f"height={self.table_height}, thickness={self.table_thickness}, "
            f"position={self.position}, leg_inset={self.leg_inset})"
        )
