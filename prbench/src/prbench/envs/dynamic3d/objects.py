"""Object definitions for TidyBot environments."""

from __future__ import annotations

import abc
import math
import xml.etree.ElementTree as ET
from typing import TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from relational_structs import Object

from prbench.envs.dynamic3d import utils
from prbench.envs.dynamic3d.mujoco_utils import MujocoEnv
from prbench.envs.dynamic3d.object_types import (
    MujocoFixtureObjectType,
    MujocoMovableObjectType,
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
        self.symbolic_object = Object(self.name, MujocoMovableObjectType)

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
class Cuboid(MujocoObject):
    """A cuboid (rectangular box) object for TidyBot environments."""

    default_edge_size: float = 0.02  # Default edge size in meters

    def __init__(
        self,
        name: str,
        env: MujocoEnv | None = None,
        options: dict | None = None,
    ) -> None:
        """Initialize a Cuboid object.

        Args:
            name: Name of the cuboid body in the XML
            options: Dictionary of cuboid options:
                - size: [x, y, z] dimensions as a list of three floats
                - rgba: Color of the cuboid (either string or [r, g, b, a] values)
                - mass: Mass of the cuboid
            env: Reference to the environment (needed for position get/set operations)
        """
        # Initialize base class
        super().__init__(name, env, options)

        # Override object type
        self.symbolic_object = Object(self.name, MujocoMovableObjectType)

        # Handle size parameter - must be a list of 3 dimensions
        default_size = Cuboid.default_edge_size
        size = self.options.get(
            "size",
            [default_size, default_size, default_size],
        )
        if isinstance(size, (int, float)):
            # If scalar provided, treat as cube
            self.size = [size, size, size]
        else:
            # Expect a list of [x, y, z]
            self.size = list(size)
            if len(self.size) != 3:
                raise ValueError(
                    f"Cuboid size must be a list of 3 values [x, y, z], "
                    f"got {len(self.size)} values"
                )

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
        """Create the XML Element for this cuboid.

        Returns:
            ET.Element representing the cuboid body
        """
        # Create body element
        body = ET.Element("body", name=self.name)

        # Add freejoint for position/orientation control
        ET.SubElement(body, "freejoint", name=self.joint_name)

        # Add geom element with cuboid properties
        size_str = " ".join(str(x) for x in self.size)
        ET.SubElement(
            body,
            "geom",
            type="box",
            size=size_str,
            friction="0.95 0.3 0.1",
            rgba=self.rgba,
            mass=str(self.mass),
        )

        return body

    def __str__(self) -> str:
        """String representation of the cuboid."""
        return (
            f"Cuboid(name='{self.name}', size={self.size}, "
            f"rgba='{self.rgba}', mass={self.mass})"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the cuboid."""
        return (
            f"Cuboid(name='{self.name}', joint_name='{self.joint_name}', "
            f"size={self.size}, rgba='{self.rgba}', mass={self.mass})"
        )

    def get_bounding_box_dimensions(self) -> tuple[float, float, float]:
        return (2 * self.size[0], 2 * self.size[1], 2 * self.size[2])


@register_object
class Cube(Cuboid):
    """A cube object for TidyBot environments.

    This is a special case of Cuboid where all dimensions are equal.
    """

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
        # Normalize size to scalar if all dimensions are equal
        if options is None:
            options = {}

        size = options.get("size", Cuboid.default_edge_size)
        if isinstance(size, (int, float)):
            # Already scalar, keep as is
            pass
        else:
            # Convert to list to check dimensions
            size_list = list(size)
            if len(size_list) == 3 and size_list[0] == size_list[1] == size_list[2]:
                # All dimensions equal, use scalar
                options = dict(options)  # Create a copy
                options["size"] = size_list[0]

        # Initialize parent Cuboid class
        super().__init__(name, env, options)

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
        yaw: float,
        regions: dict | None = None,
    ) -> None:
        """Initialize a MujocoFixture.

        Args:
            name: Name of the fixture body in the XML
            fixture_config: Dictionary containing fixture configuration
            position: Position of the fixture as [x, y, z]
            yaw: Yaw orientation of the fixture in radians
        """
        self.name = name
        self.fixture_config = fixture_config
        self.position = position
        self.yaw = yaw
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

    def get_orientation(self) -> list[float]:
        """Get the fixture's orientation.

        Returns:
            Orientation as quaternion [w, x, y, z] list
        """
        return utils.convert_yaw_to_quaternion(self.yaw)

    @staticmethod
    @abc.abstractmethod
    def get_bounding_box_from_config(
        pos: NDArray[np.float32], fixture_config: dict[str, str | float]
    ) -> list[float]:
        """Get the fixture's bounding box in world coordinates.

        Args:
            pos: Position of the fixture as [x, y, z] array
            fixture_config: Dictionary containing fixture configuration parameters

        Returns:
            Bounding box as [x_min, y_min, z_min, x_max, y_max, z_max] array
        """

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
        region_name: str,
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

    @abc.abstractmethod
    def check_in_region(
        self,
        position: NDArray[np.float32],
        region_name: str,
    ) -> bool:
        """Check if a given position is within the specified region.

        Args:
            position: Position as [x, y, z] array in world coordinates
            region_name: Name of the region to check
        Returns:
            True if the position is within the specified region, False otherwise
        """

    @abc.abstractmethod
    def visualize_regions(self) -> None:
        """Visualize the fixture's regions in the MuJoCo environment.

        This method adds visual elements to the MuJoCo XML to represent the regions
        defined for this fixture.
        """


@register_fixture
class Table(MujocoFixture):
    """A table fixture."""

    def __init__(
        self,
        name: str,
        fixture_config: dict[str, str | float],
        position: list[float] | NDArray[np.float32],
        yaw: float,
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
            yaw: Yaw orientation of the table in radians
        """
        # Initialize base class
        super().__init__(name, fixture_config, position, yaw, regions)

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
        ori_quat = utils.convert_yaw_to_quaternion(self.yaw)
        orientation_str = " ".join(str(x) for x in ori_quat)
        table_body.set("quat", orientation_str)

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

    @staticmethod
    def get_bounding_box_from_config(
        pos: NDArray[np.float32], fixture_config: dict[str, str | float]
    ) -> list[float]:
        """Get bounding box for a table given its position and config.

        Args:
            pos: Position of the table as [x, y, z] array
            fixture_config: Dictionary containing table configuration with keys:
                - "shape": Shape of the table - "rectangle" or "circle"
                - "length": Total table length in meters (for rectangle)
                - "width": Total table width in meters (for rectangle)
                - "diameter": Diameter of circular table in meters (for circle)
                - "height": Table height in meters

        Returns:
            Bounding box as [x_min, y_min, z_min, x_max, y_max, z_max]

        Raises:
            ValueError: If table shape is not supported
        """
        table_height = float(fixture_config["height"])
        z_min = pos[2]
        z_max = pos[2] + table_height

        if fixture_config["shape"] == "rectangle":
            half_length = float(fixture_config["length"]) / 2
            half_width = float(fixture_config["width"]) / 2
            return [
                pos[0] - half_length,  # x_min
                pos[1] - half_width,  # y_min
                z_min,
                pos[0] + half_length,  # x_max
                pos[1] + half_width,  # y_max
                z_max,
            ]
        if fixture_config["shape"] == "circle":
            radius = float(fixture_config["diameter"]) / 2
            return [
                pos[0] - radius,  # x_min
                pos[1] - radius,  # y_min
                z_min,
                pos[0] + radius,  # x_max
                pos[1] + radius,  # y_max
                z_max,
            ]

        raise ValueError(f"Unknown table shape: {fixture_config['shape']}")

    def sample_pose_in_region(
        self,
        region_name: str,
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
        assert self.regions is not None, "Regions must be defined"
        # Randomly select one of the regions
        selected_range = np_random.choice(self.regions[region_name]["ranges"])

        # Validate the selected region
        if len(selected_range) != 4:  # type: ignore[arg-type]
            raise ValueError(
                f"Each region must have exactly 4 values "
                f"[x_start, y_start, x_end, y_end], got {len(selected_range)}"
            )

        x_start, y_start, x_end, y_end = selected_range  # type: ignore[misc]

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

    def check_in_region(
        self,
        position: NDArray[np.float32],
        region_name: str,
    ) -> bool:
        """Check if a given position is within the specified region.

        Args:
            position: Position as [x, y, z] array in world coordinates
            region_name: Name of the region to check
        Returns:
            True if the position is within the specified region, False otherwise
        """
        # Convert world coordinates to table-relative coordinates
        table_x = position[0] - self.position[0]
        table_y = position[1] - self.position[1]
        table_z = position[2] - self.position[2]

        table_placement_threshold = 0.01  # 1cm tolerance for placement

        # Get the bounding box for the specified region
        assert self.regions is not None, "Regions must be defined"
        if region_name not in self.regions:
            raise ValueError(f"Region '{region_name}' not found")

        region_ranges = self.regions[region_name]["ranges"]

        for region_range in region_ranges:
            x_start, y_start, x_end, y_end = region_range

            if (
                x_start <= table_x <= x_end
                and y_start <= table_y <= y_end
                and self.table_height
                <= table_z
                <= (self.table_height + table_placement_threshold)
            ):
                return True

        return False

    def visualize_regions(self) -> None:
        """Visualize the table's regions in the MuJoCo environment.

        This method adds visual elements to the MuJoCo XML to represent the regions
        defined for this table.
        """
        if self.regions is None:
            return

        for region_name, region_config in self.regions.items():
            if "rgba" in region_config:
                region_bounds_list = region_config["ranges"]
                for i_region, region_bounds in enumerate(region_bounds_list):
                    x_start, y_start, x_end, y_end = region_bounds
                    region_center_x = (x_start + x_end) / 2
                    region_center_y = (y_start + y_end) / 2
                    region_center_z = (
                        self.table_height + self.position[2] + 0.01
                    )  # Slightly above

                    region_size_x = (x_end - x_start) / 2
                    region_size_y = (y_end - y_start) / 2
                    region_size_z = 0.005  # Thin box for visualization

                    # Create geom element for the region visualization
                    region_geom = ET.SubElement(self.xml_element, "geom")
                    region_geom.set(
                        "name", f"{self.name}_{region_name}_region_{i_region}"
                    )
                    region_geom.set("type", "box")
                    region_geom.set(
                        "size",
                        f"{region_size_x} {region_size_y} {region_size_z}",
                    )
                    region_geom.set(
                        "pos",
                        f"{region_center_x} {region_center_y} {region_center_z}",
                    )
                    region_geom.set("rgba", " ".join(map(str, region_config["rgba"])))
                    # Disable collision for visual-only representation
                    region_geom.set("contype", "0")
                    region_geom.set("conaffinity", "0")

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


@register_fixture
class Cupboard(MujocoFixture):
    """A cupboard fixture with multiple shelves."""

    default_shelf_thickness: float = 0.02
    default_partition_thickness: float = 0.01  # 1cm thick partitions

    def __init__(
        self,
        name: str,
        fixture_config: dict[str, str | float],
        position: list[float] | NDArray[np.float32],
        yaw: float,
        regions: dict | None = None,
    ) -> None:
        """Initialize a Cupboard object.

        Args:
            name: Name of the cupboard body in the XML
            fixture_config: Dictionary containing cupboard configuration with keys:
                - "length": Total cupboard length in meters
                - "depth": Total cupboard depth in meters
                - "shelf_heights": List of distances between consecutive shelves
                  in meters
                - "shelf_partitions": List of lists, each containing partition
                  distances from left edge
                - "side_and_back_open": Boolean indicating if sides and back are
                  open
                - "shelf_thickness": Thickness of each shelf in meters
                  (optional, default 0.02)
            position: Position of the cupboard as [x, y, z]
            yaw: Yaw orientation of the cupboard in radians
        """
        # Initialize base class
        super().__init__(name, fixture_config, position, yaw, regions)

        # Parse cupboard configuration
        self.cupboard_length = float(self.fixture_config["length"])
        self.cupboard_depth = float(self.fixture_config["depth"])

        # Handle shelf_heights - convert to list of floats
        shelf_heights_raw = self.fixture_config["shelf_heights"]
        self.shelf_heights: list[float] = (
            [float(h) for h in shelf_heights_raw]  # type: ignore
            if hasattr(shelf_heights_raw, "__iter__")
            and not isinstance(shelf_heights_raw, str)
            else [float(shelf_heights_raw)]
        )

        # Handle shelf_partitions - convert to list of lists of floats
        shelf_partitions_raw = self.fixture_config["shelf_partitions"]
        self.shelf_partitions: list[list[float]] = [
            [float(p) for p in partition_list]  # type: ignore
            for partition_list in shelf_partitions_raw  # type: ignore
        ]
        self.side_and_back_open: bool = bool(self.fixture_config["side_and_back_open"])
        self.shelf_thickness: float = float(
            self.fixture_config.get("shelf_thickness", Cupboard.default_shelf_thickness)
        )
        self.panel_thickness: float = 0.01  # Thickness of side and back panels
        # Set leg thickness: thin when panels present, thicker when open
        self.leg_thickness: float = (
            self.panel_thickness if not self.side_and_back_open else 0.03
        )

        # Calculate derived properties
        self.num_shelves: int = len(self.shelf_heights) + 1  # +1 for the top shelf
        self.cupboard_height: float = (
            sum(self.shelf_heights) + self.num_shelves * self.shelf_thickness
        )

        # Validate configuration
        if len(self.shelf_heights) < 1:
            raise ValueError("Number of shelf heights must be at least 1")

        if len(self.shelf_partitions) != len(self.shelf_heights):
            raise ValueError(
                f"shelf_partitions must have {len(self.shelf_heights)} lists, "
                f"got {len(self.shelf_partitions)} (one list per shelf gap, "
                f"not including top shelf)"
            )

        # Validate partition positions
        for i, partitions in enumerate(self.shelf_partitions):
            for partition_pos in partitions:
                if (
                    partition_pos <= -self.cupboard_length / 2
                    or partition_pos >= self.cupboard_length / 2
                ):
                    raise ValueError(
                        f"Partition position {partition_pos} on shelf {i} must be "
                        f"between -{self.cupboard_length/2} and "
                        f"{self.cupboard_length/2} "
                        f"(cupboard length is {self.cupboard_length})"
                    )

        # Precompute shelf z positions for efficiency
        self._shelf_z_positions = self._compute_shelf_z_positions()

        # Create the XML element
        self.xml_element = self._create_xml_element()

    def _compute_shelf_z_positions(self) -> list[float]:
        """Compute the z position of each shelf surface.

        Returns:
            List of z positions for each shelf surface (relative to cupboard base)
        """
        shelf_z_positions = []
        current_z = self.shelf_thickness / 2

        for i in range(self.num_shelves):
            # Z position of shelf surface (top of shelf)
            shelf_surface_z = current_z + self.shelf_thickness / 2
            shelf_z_positions.append(shelf_surface_z)

            # Move to next shelf if not the last one
            if i < len(self.shelf_heights):
                current_z += (
                    self.shelf_thickness / 2
                    + self.shelf_heights[i]
                    + self.shelf_thickness / 2
                )

        return shelf_z_positions

    def _create_xml_element(self) -> ET.Element:
        """Create the XML Element for this cupboard.

        Returns:
            ET.Element representing the cupboard body
        """
        # Create cupboard body element
        cupboard_body = ET.Element("body")
        cupboard_body.set("name", self.name)
        position_str = " ".join(str(x) for x in self.position)
        cupboard_body.set("pos", position_str)
        ori_quat = utils.convert_yaw_to_quaternion(self.yaw)
        orientation_str = " ".join(str(x) for x in ori_quat)
        cupboard_body.set("quat", orientation_str)

        # Calculate dimensions
        cupboard_half_length = self.cupboard_length / 2
        cupboard_half_depth = self.cupboard_depth / 2
        shelf_half_thickness = self.shelf_thickness / 2

        # Calculate the height of the topmost shelf
        top_shelf_z = shelf_half_thickness
        for height in self.shelf_heights:
            top_shelf_z += shelf_half_thickness + height + shelf_half_thickness

        # Make legs flush with the top shelf (leg height = top shelf height)
        leg_half_height = top_shelf_z / 2

        # Calculate leg positions (at the edges)
        leg_x_offset = cupboard_half_length - self.leg_thickness / 2
        leg_y_offset = cupboard_half_depth - self.leg_thickness / 2

        # Create vertical legs at four corners
        leg_positions = [
            (f"{leg_x_offset} {leg_y_offset}", f"{self.name}_leg1"),
            (f"{-leg_x_offset} {leg_y_offset}", f"{self.name}_leg2"),
            (f"{leg_x_offset} {-leg_y_offset}", f"{self.name}_leg3"),
            (f"{-leg_x_offset} {-leg_y_offset}", f"{self.name}_leg4"),
        ]

        for pos, name in leg_positions:
            leg = ET.SubElement(cupboard_body, "geom")
            leg.set("name", name)
            leg.set("type", "box")
            leg.set(
                "size",
                f"{self.leg_thickness/2} {self.leg_thickness/2} {leg_half_height}",
            )
            leg.set(
                "pos", f"{pos.split()[0]} {pos.split()[1]} {leg_half_height}"
            )  # Position leg center at half its height
            leg.set("rgba", "0.6 0.4 0.2 1")  # Brown color for legs

        # Calculate cumulative shelf positions
        current_z = shelf_half_thickness
        shelf_positions = [current_z]

        for height in self.shelf_heights:  # Include all heights to get the top shelf
            current_z += shelf_half_thickness + height + shelf_half_thickness
            shelf_positions.append(current_z)

        # Create horizontal shelves (including the top shelf)
        for i, shelf_z in enumerate(shelf_positions):
            shelf = ET.SubElement(cupboard_body, "geom")
            shelf.set("name", f"{self.name}_shelf{i+1}")
            shelf.set("type", "box")
            shelf.set(
                "size",
                f"{cupboard_half_length} {cupboard_half_depth} {shelf_half_thickness}",
            )
            shelf.set("pos", f"0 0 {shelf_z}")
            shelf.set("rgba", "0.8 0.6 0.4 1")  # Light brown color for shelves

            # Create vertical partitions for this shelf
            # (if we have partition data for it)
            if i < len(self.shelf_partitions):
                partitions = self.shelf_partitions[i]
                shelf_height = (
                    self.shelf_heights[i]
                    if i < len(self.shelf_heights)
                    else self.shelf_heights[-1]
                )

                for j, partition_x in enumerate(partitions):
                    # partition_x is already in center-relative coordinates
                    # Calculate partition dimensions
                    partition_half_thickness = Cupboard.default_partition_thickness / 2
                    partition_half_height = shelf_height / 2
                    partition_z = shelf_z + shelf_half_thickness + partition_half_height

                    partition = ET.SubElement(cupboard_body, "geom")
                    partition.set("name", f"{self.name}_shelf{i+1}_partition{j+1}")
                    partition.set("type", "box")
                    partition.set(
                        "size",
                        f"{partition_half_thickness} {cupboard_half_depth} "
                        f"{partition_half_height}",
                    )
                    partition.set("pos", f"{partition_x} 0 {partition_z}")
                    partition.set(
                        "rgba", "0.7 0.5 0.3 1"
                    )  # Slightly different color for partitions

        # Create side and back panels if not open
        if not self.side_and_back_open:
            panel_half_thickness = self.panel_thickness / 2
            # Make panels flush with the top shelf (same height as legs)
            panel_half_height = top_shelf_z / 2

            # Back panel (at -Y edge)
            back_panel = ET.SubElement(cupboard_body, "geom")
            back_panel.set("name", f"{self.name}_back_panel")
            back_panel.set("type", "box")
            back_panel.set(
                "size",
                f"{cupboard_half_length} {panel_half_thickness} {panel_half_height}",
            )
            back_panel.set(
                "pos",
                f"0 {-cupboard_half_depth + panel_half_thickness} {panel_half_height}",
            )
            back_panel.set("rgba", "0.7 0.5 0.3 1")

            # Left side panel (at -X edge)
            left_panel = ET.SubElement(cupboard_body, "geom")
            left_panel.set("name", f"{self.name}_left_panel")
            left_panel.set("type", "box")
            left_panel.set(
                "size",
                f"{panel_half_thickness} {cupboard_half_depth} {panel_half_height}",
            )
            left_panel.set(
                "pos",
                f"{-cupboard_half_length + panel_half_thickness} 0 {panel_half_height}",
            )
            left_panel.set("rgba", "0.7 0.5 0.3 1")

            # Right side panel (at +X edge)
            right_panel = ET.SubElement(cupboard_body, "geom")
            right_panel.set("name", f"{self.name}_right_panel")
            right_panel.set("type", "box")
            right_panel.set(
                "size",
                f"{panel_half_thickness} {cupboard_half_depth} {panel_half_height}",
            )
            right_panel.set(
                "pos",
                f"{cupboard_half_length - panel_half_thickness} 0 {panel_half_height}",
            )
            right_panel.set("rgba", "0.7 0.5 0.3 1")

        return cupboard_body

    @staticmethod
    def get_bounding_box_from_config(
        pos: NDArray[np.float32], fixture_config: dict[str, str | float]
    ) -> list[float]:
        """Get bounding box for a cupboard given its position and config.

        Args:
            pos: Position of the cupboard as [x, y, z] array
            fixture_config: Dictionary containing cupboard configuration with keys:
                - "length": Total cupboard length in meters
                - "depth": Total cupboard depth in meters
                - "shelf_heights": List of distances between consecutive shelves
                - "shelf_thickness": Thickness of each shelf in meters
                  (optional, default 0.02)

        Returns:
            Bounding box as [x_min, y_min, z_min, x_max, y_max, z_max]

        Raises:
            ValueError: If required keys are missing in fixture_config
        """
        if "length" not in fixture_config or "depth" not in fixture_config:
            raise ValueError("fixture_config must contain 'length' and 'depth' keys")

        half_length = float(fixture_config["length"]) / 2
        half_depth = float(fixture_config["depth"]) / 2

        # Calculate cupboard height from shelf configuration
        shelf_heights_config: list[float] = fixture_config.get(
            "shelf_heights", []
        )  # type: ignore
        shelf_heights_float: list[float] = [float(h) for h in shelf_heights_config]
        shelf_thickness = float(
            fixture_config.get("shelf_thickness", Cupboard.default_shelf_thickness)
        )
        num_shelves = len(shelf_heights_float) + 1  # +1 for the top shelf
        cupboard_height = sum(shelf_heights_float) + num_shelves * shelf_thickness

        return [
            pos[0] - half_length,  # x_min
            pos[1] - half_depth,  # y_min
            pos[2],  # z_min
            pos[0] + half_length,  # x_max
            pos[1] + half_depth,  # y_max
            pos[2] + cupboard_height,  # z_max
        ]

    def sample_pose_in_region(
        self,
        region_name: str,
        np_random: np.random.Generator,
    ) -> tuple[float, float, float]:
        """Sample a pose (x, y, z) uniformly randomly from one of the provided regions.

        For cupboards, this samples on the top shelf surface.

        Args:
            regions: List of bounding boxes, where each bounding box is a list of
                    4 floats: [x_start, y_start, x_end, y_end] in cupboard-relative
                    coordinates
            np_random: Random number generator

        Returns:
            Tuple of (x, y, z) coordinates in world coordinates (offset by cupboard
            position)

        Raises:
            ValueError: If regions list is empty or if any region has invalid bounds
        """
        assert self.regions is not None, "Regions must be defined"

        # Ensure region exists
        region = self.regions.get(region_name)
        if region is None:
            raise ValueError(f"Region '{region_name}' not found")

        # Ensure shelf index is specified
        if "shelf" not in region:
            raise ValueError(
                f"Cupboard region '{region_name}' must specify 'shelf' to sample on"
            )
        shelf = region["shelf"]
        assert 0 <= shelf < self.num_shelves, (
            f"Shelf index {shelf} out of range for cupboard with "
            f"{self.num_shelves} shelves"
        )

        # Randomly select one of the regions
        selected_range = np_random.choice(region["ranges"])

        # Validate the selected region
        if len(selected_range) != 4:  # type: ignore[arg-type]
            raise ValueError(
                f"Each region must have exactly 4 values "
                f"[x_start, y_start, x_end, y_end], got {len(selected_range)}"
            )

        x_start, y_start, x_end, y_end = selected_range  # type: ignore[misc]

        # Validate bounds
        if x_start >= x_end:
            raise ValueError(f"x_start ({x_start}) must be less than x_end ({x_end})")
        if y_start >= y_end:
            raise ValueError(f"y_start ({y_start}) must be less than y_end ({y_end})")

        # Sample uniformly within the selected region
        x = np_random.uniform(x_start, x_end)
        y = np_random.uniform(y_start, y_end)

        # Get z position from precomputed shelf positions
        shelf_z = self._shelf_z_positions[shelf]
        z = shelf_z + 0.01  # Slightly above shelf surface

        # Offset by the cupboard's position to get world coordinates
        world_x = x + self.position[0]
        world_y = y + self.position[1]
        world_z = z + self.position[2]

        return (world_x, world_y, world_z)

    def check_in_region(
        self,
        position: NDArray[np.float32],
        region_name: str,
    ) -> bool:
        """Check if a given position is within the specified region.

        This checks if the
        position is on the top shelf surface.
        Args:
            position: Position as [x, y, z] array in world coordinates
            region_name: Name of the region to check
        Returns:
            True if the position is within the specified region, False otherwise
        """
        # Convert world coordinates to cupboard-relative coordinates
        cupboard_x = position[0] - self.position[0]
        cupboard_y = position[1] - self.position[1]
        cupboard_z = position[2] - self.position[2]

        cupboard_placement_threshold = 0.02  # 2cm tolerance for placement

        # Get the bounding box for the specified region
        assert self.regions is not None, "Regions must be defined"
        if region_name not in self.regions:
            raise ValueError(f"Region '{region_name}' not found")

        region = self.regions[region_name]
        region_ranges = region["ranges"]

        # Get shelf index if specified
        if "shelf" not in region:
            raise ValueError(
                f"Cupboard region '{region_name}' must specify 'shelf' for checking"
            )
        shelf = region["shelf"]

        # Get z position from precomputed shelf positions
        shelf_z = self._shelf_z_positions[shelf]

        for region_range in region_ranges:
            x_start, y_start, x_end, y_end = region_range

            if (
                x_start <= cupboard_x <= x_end
                and y_start <= cupboard_y <= y_end
                and shelf_z <= cupboard_z <= (shelf_z + cupboard_placement_threshold)
            ):
                return True

        return False

    def visualize_regions(self) -> None:
        """Visualize the cupboard's regions in the MuJoCo environment.

        This method adds visual elements to the MuJoCo XML to represent the regions
        defined for this cupboard.
        """
        if self.regions is None:
            return

        for region_name, region_config in self.regions.items():
            if "rgba" in region_config and "shelf" in region_config:
                shelf = region_config["shelf"]
                region_bounds_list = region_config["ranges"]

                # Get z position from precomputed shelf positions
                shelf_z = self._shelf_z_positions[shelf]

                for i_region, region_bounds in enumerate(region_bounds_list):
                    x_start, y_start, x_end, y_end = region_bounds
                    region_center_x = (x_start + x_end) / 2
                    region_center_y = (y_start + y_end) / 2
                    region_center_z = shelf_z + 0.01  # Slightly above shelf surface

                    region_size_x = (x_end - x_start) / 2
                    region_size_y = (y_end - y_start) / 2
                    region_size_z = 0.005  # Thin box for visualization

                    # Create geom element for the region visualization
                    region_geom = ET.SubElement(self.xml_element, "geom")
                    region_geom.set(
                        "name", f"{self.name}_{region_name}_region_{i_region}"
                    )
                    region_geom.set("type", "box")
                    region_geom.set(
                        "size",
                        f"{region_size_x} {region_size_y} {region_size_z}",
                    )
                    region_geom.set(
                        "pos",
                        f"{region_center_x} {region_center_y} {region_center_z}",
                    )
                    region_geom.set("rgba", " ".join(map(str, region_config["rgba"])))
                    # Disable collision for visual-only representation
                    region_geom.set("contype", "0")
                    region_geom.set("conaffinity", "0")

    def __str__(self) -> str:
        """String representation of the cupboard."""
        return (
            f"Cupboard(name='{self.name}', length={self.cupboard_length}, "
            f"depth={self.cupboard_depth}, height={self.cupboard_height}, "
            f"num_shelves={self.num_shelves}, shelf_heights={self.shelf_heights})"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the cupboard."""
        return (
            f"Cupboard(name='{self.name}', "
            f"length={self.cupboard_length}, depth={self.cupboard_depth}, "
            f"height={self.cupboard_height}, num_shelves={self.num_shelves}, "
            f"shelf_heights={self.shelf_heights}, "
            f"shelf_partitions={self.shelf_partitions}, "
            f"shelf_thickness={self.shelf_thickness}, "
            f"side_and_back_open={self.side_and_back_open}, "
            f"position={self.position}, leg_thickness={self.leg_thickness})"
        )
