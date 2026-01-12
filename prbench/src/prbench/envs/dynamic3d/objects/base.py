"""Base classes for MuJoCo objects and fixtures."""

from __future__ import annotations

import abc
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Callable, TypeVar, Union, overload

import numpy as np
from numpy.typing import NDArray
from relational_structs import Object

from prbench.envs.dynamic3d.mujoco_utils import MujocoEnv
from prbench.envs.dynamic3d.object_types import (
    MujocoFixtureObjectType,
    MujocoMovableObjectType,
)

if TYPE_CHECKING:
    from prbench.envs.dynamic3d import utils

# Type variables for decorator type preservation
FixtureT = TypeVar("FixtureT", bound="MujocoFixture")
ObjectT = TypeVar("ObjectT", bound="MujocoObject")

REGISTERED_FIXTURES: dict[str, type[MujocoFixture]] = {}
REGISTERED_OBJECTS: dict[str, type[MujocoObject]] = {}


def register_fixture(cls: type[FixtureT]) -> type[FixtureT]:
    """Register fixture classes for TidyBot environments."""
    REGISTERED_FIXTURES[cls.__name__.lower()] = cls
    return cls


@overload
def register_object(cls: type[ObjectT]) -> type[ObjectT]: ...


@overload
def register_object(
    cls: None = None, name: str | None = None
) -> Callable[[type[ObjectT]], type[ObjectT]]: ...


def register_object(
    cls: type[ObjectT] | None = None, name: str | None = None
) -> type[ObjectT] | Callable[[type[ObjectT]], type[ObjectT]]:
    """Register object classes for TidyBot environments.

    Can be used as:
    - @register_object (uses class name in lowercase)
    - @register_object(name='custom_name') (uses provided name)

    Args:
        cls: The class to register (when used without parentheses)
        name: Optional name to register with (defaults to lowercase class name)

    Returns:
        The registered class or a decorator function
    """

    def decorator(c: type[ObjectT]) -> type[ObjectT]:
        registry_name = name if name is not None else c.__name__.lower()
        REGISTERED_OBJECTS[registry_name] = c
        c.REGISTERED_NAME = registry_name  # type: ignore[attr-defined]
        return c

    # If used without parentheses: @register_object
    if cls is not None:
        return decorator(cls)

    # If used with parentheses: @register_object() or @register_object(name='...')
    return decorator


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
            options: Optional dictionary of configuration options. Can include:
                - "regions": Optional dictionary of regions for this object
        """
        self.name = name
        self.joint_name = f"{name}_joint"
        self.env = env
        self.options = options if options is not None else {}
        self.regions = self.options.get("regions")

        # Create the corresponding Object for state representation key
        self.symbolic_object = Object(self.name, MujocoMovableObjectType)

        # Create region bounding boxes if regions are defined
        self.region_bboxes: dict[str, list[list[float]]] = {}
        if self.regions is not None:
            self._create_region_bboxes()

        self.xml_element: ET.Element  # To be defined in subclasses

    def _create_region_bboxes(self) -> None:
        """Create 3D bounding boxes for each region.

        Each region's 2D ranges [x_start, y_start, x_end, y_end] are converted
        to 3D bounding boxes [x_min, y_min, z_min, x_max, y_max, z_max] where
        the z dimension spans a small height above the object surface.
        """
        assert self.regions is not None, "Regions must be defined"
        placement_threshold = 0.01  # 1cm tolerance for placement
        # Note: we are currently hard-coding the z range for the bounding boxes
        # This could potentially be made configurable in the future.

        for region_name, region_config in self.regions.items():
            region_bboxes = []

            for region_range in region_config["ranges"]:
                x_start, y_start, x_end, y_end = region_range

                # Create 3D bounding box with z range and tolerance on x/y bounds
                # Apply tolerance to x and y boundaries
                bbox = [
                    x_start - placement_threshold,
                    y_start - placement_threshold,
                    -placement_threshold,
                    x_end + placement_threshold,
                    y_end + placement_threshold,
                    placement_threshold,
                ]
                region_bboxes.append(bbox)

            self.region_bboxes[region_name] = region_bboxes

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

    @staticmethod
    @abc.abstractmethod
    def get_bounding_box_from_config(
        pos: NDArray[np.float32], object_config: dict[str, str | float]
    ) -> list[float]:
        """Get the object's bounding box in world coordinates.

        Args:
            pos: Position of the object as [x, y, z] array
            object_config: Dictionary containing object configuration parameters

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

        Raises:
            ValueError: If regions are not defined or region not found
            ValueError: If environment is not set (needed to get object position)
        """
        if self.regions is None:
            raise ValueError("Regions must be defined for this object")
        if region_name not in self.region_bboxes:
            raise ValueError(f"Region {region_name} not found in object regions")

        # Get the object's current position
        obj_pos = self.get_position()

        # Convert world coordinates to object-relative coordinates
        object_relative_pos = np.array(
            [
                position[0] - obj_pos[0],
                position[1] - obj_pos[1],
                position[2] - obj_pos[2],
            ]
        )

        # Import here to avoid circular dependency
        # pylint: disable=import-outside-toplevel
        from prbench.envs.dynamic3d import utils

        # Check if position is in any of the region's bounding boxes
        region_bboxes = self.region_bboxes[region_name]
        for bbox in region_bboxes:
            if utils.point_in_bbox_3d(object_relative_pos, bbox):
                return True

        return False

    def visualize_regions(self) -> None:
        """Visualize the object's regions in the MuJoCo environment.

        This method adds visual elements to the MuJoCo XML to represent the regions
        defined for this object in object-relative coordinates.
        """
        if self.regions is None:
            return

        for region_name, region_config in self.regions.items():
            # Get the bounding boxes for this region
            region_bboxes = self.region_bboxes.get(region_name, [])

            for i_region, bbox in enumerate(region_bboxes):
                # bbox is [x_min, y_min, z_min, x_max, y_max, z_max]
                x_min, y_min, z_min, x_max, y_max, z_max = bbox

                # Calculate center and half-sizes for MuJoCo box geom
                region_center_x = (x_min + x_max) / 2
                region_center_y = (y_min + y_max) / 2
                region_center_z = (z_min + z_max) / 2

                region_size_x = (x_max - x_min) / 2
                region_size_y = (y_max - y_min) / 2
                region_size_z = (z_max - z_min) / 2

                # Create site element for the region visualization
                region_site = ET.SubElement(self.xml_element, "site")
                region_site.set("name", f"{self.name}_{region_name}_region_{i_region}")
                region_site.set("type", "box")
                region_site.set(
                    "size",
                    f"{region_size_x} {region_size_y} {region_size_z}",
                )
                region_site.set(
                    "pos",
                    f"{region_center_x} {region_center_y} {region_center_z}",
                )
                rgba_values = region_config.get("rgba", [1.0, 0.0, 0.0, 0.0])
                region_site.set("rgba", " ".join(map(str, rgba_values)))
                region_site.set("group", "0")


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
        # Import here to avoid circular dependency
        # pylint: disable=import-outside-toplevel
        from prbench.envs.dynamic3d import utils

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
    ) -> tuple[float, float, float, float]:
        """Sample a pose (x, y, z, yaw) uniformly randomly from one of the provided
        regions.

        Args:
            region_name: Name of the region to sample from
            np_random: Random number generator

        Returns:
            Tuple of (x, y, z, yaw) coordinates in world coordinates (offset by
            fixture position), where yaw is in radians. The yaw range is read from
            self.regions[region_name]["yaw_ranges"] if it exists, otherwise
            defaults to (0.0, 360.0) degrees.

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


class MujocoGround:
    """A ground fixture for simple region sampling on the ground plane."""

    def __init__(
        self,
        regions: dict | None = None,
    ) -> None:
        """Initialize a Ground object.

        Args:
            name: Name of the ground body in the XML
            fixture_config: Dictionary containing ground configuration with keys:
                - "length": Total ground length in meters
                - "width": Total ground width in meters
            position: Position of the ground as [x, y, z]
            yaw: Yaw orientation of the ground in radians
            regions: Dictionary of regions defined on the ground plane
        """
        self.name = "ground"
        self.regions = regions
        self.position = np.array([0.0, 0.0, 0.0])  # Ground at origin
        self.ground_thickness = 0.01  # Default ground thickness
        self.xml_element = ET.Element("body", name=self.name)  # Empty body element

    def sample_pose_in_region(
        self,
        region_name: str,
        np_random: np.random.Generator,
    ) -> tuple[float, float, float, float]:
        """Sample a pose (x, y, z, yaw) uniformly randomly from one of the provided
        regions.

        For ground, this samples on the ground plane surface.

        Args:
            region_name: Name of the region to sample from
            np_random: Random number generator

        Returns:
            Tuple of (x, y, z, yaw) coordinates in world coordinates (offset by ground
            position), where yaw is in radians. The yaw range is read from
            self.regions[region_name]["yaw_ranges"] if it exists, otherwise
            defaults to (0.0, 360.0) degrees.

        Raises:
            ValueError: If regions list is empty or if any region has invalid bounds
        """
        assert self.regions is not None, "Regions must be defined"
        region_config = self.regions[region_name]

        # Randomly select one of the regions
        selected_range_index = np_random.choice(len(region_config["ranges"]))

        # Validate the selected region
        selected_range = region_config["ranges"][selected_range_index]
        if len(selected_range) != 4:
            raise ValueError(
                f"Each region must have exactly 4 values "
                f"[x_start, y_start, x_end, y_end], got {len(selected_range)}"
            )

        (x_start, y_start, x_end, y_end) = selected_range  # type: ignore[misc]

        # Validate bounds
        if x_start > x_end:
            raise ValueError(f"x_start ({x_start}) must be less than x_end ({x_end})")
        if y_start > y_end:
            raise ValueError(f"y_start ({y_start}) must be less than y_end ({y_end})")

        # Sample uniformly within the selected region
        x = np_random.uniform(x_start, x_end)
        y = np_random.uniform(y_start, y_end)

        # Sample z coordinate on the ground surface
        z = self.ground_thickness / 2  # Slightly above ground

        # Sample yaw from the specified range (convert from degrees to radians)
        yaw_range = (0.0, 360.0)  # Default range
        if "yaw_ranges" in region_config:
            selected_yaw_range = region_config["yaw_ranges"][selected_range_index]
            yaw_range = selected_yaw_range
        yaw_deg = np_random.uniform(yaw_range[0], yaw_range[1])
        yaw = np.radians(yaw_deg)

        # Offset by the ground's position to get world coordinates
        world_x = x + self.position[0]
        world_y = y + self.position[1]
        world_z = z + self.position[2]

        return (world_x, world_y, world_z, yaw)

    def check_in_region(
        self,
        position: NDArray[np.float32],
        region_name: str,
    ) -> bool:
        """Check if a given position is within the specified region on the ground.

        Args:
            position: Position as [x, y, z] array in world coordinates
            region_name: Name of the region to check
        Returns:
            True if the position is within the specified region, False otherwise
        """
        # Convert world coordinates to ground-relative coordinates
        ground_x = position[0] - self.position[0]
        ground_y = position[1] - self.position[1]

        ground_placement_threshold = 0.01  # 1cm tolerance for placement

        # Get the bounding box for the specified region
        assert self.regions is not None, "Regions must be defined"
        if region_name not in self.regions:
            raise ValueError(f"Region {region_name} not found in ground regions")

        region_ranges = self.regions[region_name]["ranges"]

        for region_range in region_ranges:
            x_start, y_start, x_end, y_end = region_range
            # Check if position is within this region (with tolerance)
            if (
                x_start - ground_placement_threshold
                <= ground_x
                <= x_end + ground_placement_threshold
            ) and (
                y_start - ground_placement_threshold
                <= ground_y
                <= y_end + ground_placement_threshold
            ):
                return True

        return False

    def visualize_regions(self) -> None:
        """Visualize the ground's regions in the MuJoCo environment.

        This method adds visual elements to the MuJoCo XML to represent the regions
        defined for this ground.
        """
        if self.regions is None:
            return

        for region_name, region_config in self.regions.items():
            for i_region, region_range in enumerate(region_config["ranges"]):
                x_start, y_start, x_end, y_end = region_range

                # Calculate region center and size
                region_center_x = (x_start + x_end) / 2
                region_center_y = (y_start + y_end) / 2
                region_center_z = 0.001  # Just above ground
                region_size_x = (x_end - x_start) / 2
                region_size_y = (y_end - y_start) / 2
                region_size_z = 0.001  # Very thin box for visualization

                # Create site element for the region visualization
                region_site = ET.SubElement(self.xml_element, "site")
                region_site.set("name", f"{self.name}_{region_name}_region_{i_region}")
                region_site.set("type", "box")
                region_site.set(
                    "size",
                    f"{region_size_x} {region_size_y} {region_size_z}",
                )
                region_site.set(
                    "pos",
                    f"{region_center_x} {region_center_y} {region_center_z}",
                )
                rgba_values = region_config.get("rgba", [1.0, 0.0, 0.0, 0.0])
                region_site.set("rgba", " ".join(map(str, rgba_values)))
                region_site.set("group", "1")

    def __str__(self) -> str:
        """String representation of the ground."""
        return f"Ground(name='{self.name}', " f"position={self.position})"

    def __repr__(self) -> str:
        """Detailed string representation of the ground."""
        return f"Ground(name='{self.name}', " f"position={self.position})"
