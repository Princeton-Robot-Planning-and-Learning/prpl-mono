"""Primitive MuJoCo object classes such as Cube and Cuboid."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
from numpy.typing import NDArray
from relational_structs import Object

from prbench.envs.dynamic3d.mujoco_utils import MujocoEnv
from prbench.envs.dynamic3d.object_types import MujocoMovableObjectType
from prbench.envs.dynamic3d.objects.base import MujocoObject, register_object


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

        # Handle rgba parameter - can be string or list of values
        rgba = self.options.get("rgba", [0.5, 0.7, 0.5, 1])
        if isinstance(rgba, str):
            self.rgba = rgba
        else:
            self.rgba = " ".join(str(x) for x in rgba)

        # Handle mass parameter with default
        self.mass = self.options.get("mass", 0.1)

        # Create the XML element
        self.xml_element = self._create_xml_element()

    def _create_xml_element(self) -> ET.Element:
        """Create the XML Element for this cuboid.
        
        The cuboid is created as a single box geom centered at the body's origin.
        The origin (0, 0, 0) is located at the center of the cuboid.
        The cuboid extends by size[i]/2 in each direction (+/- x, +/- y, +/- z) from the origin.

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
            # friction="2.0 0.2 0.02",
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

    @staticmethod
    def get_bounding_box_from_config(
        pos: NDArray[np.float32], object_config: dict[str, str | float]
    ) -> list[float]:
        """Get bounding box for a cuboid given its position and config.

        Args:
            pos: Position of the cuboid as [x, y, z] array
            object_config: Dictionary containing cuboid configuration with keys:
                - "size": Either a scalar (for cube) or [x, y, z] half-extents

        Returns:
            Bounding box as [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        # Handle size parameter - can be scalar or list of 3 dimensions
        default_size = Cuboid.default_edge_size
        size = object_config.get("size", default_size)

        if isinstance(size, (int, float)):
            # Scalar size - cube
            half_extents = [float(size), float(size), float(size)]
        else:
            # List of [x, y, z] half-extents
            half_extents = [float(s) for s in size]  # type: ignore[union-attr]
            if len(half_extents) != 3:
                raise ValueError(
                    f"Cuboid size must be a scalar or list of 3 values [x, y, z], "
                    f"got {len(half_extents)} values"
                )

        return [
            pos[0] - half_extents[0],  # x_min
            pos[1] - half_extents[1],  # y_min
            pos[2] - half_extents[2],  # z_min
            pos[0] + half_extents[0],  # x_max
            pos[1] + half_extents[1],  # y_max
            pos[2] + half_extents[2],  # z_max
        ]


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


@register_object
class Bin(MujocoObject):
    """A bin (rectangular container with open top) object for TidyBot environments.
    
    The bin is constructed using multiple MuJoCo box primitives:
    - 1 bottom panel
    - 4 wall panels (front, back, left, right)
    """

    default_wall_thickness: float = 0.005  # Default wall thickness in meters

    def __init__(
        self,
        name: str,
        env: MujocoEnv | None = None,
        options: dict | None = None,
    ) -> None:
        """Initialize a Bin object.

        Args:
            name: Name of the bin body in the XML
            options: Dictionary of bin options:
                - length: Length of bin (x dimension, outer)
                - width: Width of bin (y dimension, outer)
                - height: Height of bin (z dimension)
                - wall_thickness: Thickness of walls (default: 0.005)
                - rgba: Color of the bin (either string or [r, g, b, a] values)
                - mass: Mass of the bin
            env: Reference to the environment (needed for position get/set operations)
        """
        # Initialize base class
        super().__init__(name, env, options)

        # Override object type
        self.symbolic_object = Object(self.name, MujocoMovableObjectType)

        # Bin dimensions
        self.length = float(self.options.get("length", 0.1))  # x dimension
        self.width = float(self.options.get("width", 0.1))    # y dimension
        self.height = float(self.options.get("height", 0.05)) # z dimension
        self.wall_thickness = float(
            self.options.get("wall_thickness", Bin.default_wall_thickness)
        )

        # Handle rgba parameter - can be string or list of values
        rgba = self.options.get("rgba", [0.5, 0.5, 0.5, 1])
        if isinstance(rgba, str):
            self.rgba = rgba
        else:
            self.rgba = " ".join(str(x) for x in rgba)

        # Handle mass parameter with default
        self.mass = self.options.get("mass", 0.1)

        # Create the XML element
        self.xml_element = self._create_xml_element()

    def _create_xml_element(self) -> ET.Element:
        """Create the XML Element for this bin using multiple box geoms.
        
        The bin is constructed from 5 box geoms:
        - 1 bottom panel: full outer dimensions (length x width), at z in [0, wall_thickness]
        - 4 wall panels: back, front, left, right walls with thickness wall_thickness,
          extending from z = wall_thickness to z = height
        
        The origin (0, 0, 0) is located at the base center of the bin (center of bottom surface).
        The bin extends in the positive z direction.

        Returns:
            ET.Element representing the bin body with all geoms
        """
        # Create body element
        body = ET.Element("body", name=self.name)

        # Add freejoint for position/orientation control
        ET.SubElement(body, "freejoint", name=self.joint_name)

        # Calculate half dimensions
        half_length = self.length / 2
        half_width = self.width / 2
        half_wall = self.wall_thickness / 2

        # Calculate inner dimensions
        inner_half_length = half_length - self.wall_thickness
        inner_half_width = half_width - self.wall_thickness

        # Wall height (excluding bottom thickness)
        wall_height = self.height - self.wall_thickness
        half_wall_height = wall_height / 2

        # Mass distribution (divide among 5 components)
        component_mass = self.mass / 5.0

        # Bottom panel (full outer dimensions, at z = 0 to z = wall_thickness)
        bottom_size = [half_length, half_width, half_wall]
        bottom_pos = [0.0, 0.0, half_wall]
        ET.SubElement(
            body,
            "geom",
            type="box",
            size=" ".join(str(x) for x in bottom_size),
            pos=" ".join(str(x) for x in bottom_pos),
            rgba=self.rgba,
            mass=str(component_mass),
        )

        # Back wall (along x-axis, at -y edge)
        back_wall_size = [half_length, half_wall, half_wall_height]
        back_wall_pos = [0.0, -half_width + half_wall, self.wall_thickness + half_wall_height]
        ET.SubElement(
            body,
            "geom",
            type="box",
            size=" ".join(str(x) for x in back_wall_size),
            pos=" ".join(str(x) for x in back_wall_pos),
            rgba=self.rgba,
            mass=str(component_mass),
        )

        # Front wall (along x-axis, at +y edge)
        front_wall_size = [half_length, half_wall, half_wall_height]
        front_wall_pos = [0.0, half_width - half_wall, self.wall_thickness + half_wall_height]
        ET.SubElement(
            body,
            "geom",
            type="box",
            size=" ".join(str(x) for x in front_wall_size),
            pos=" ".join(str(x) for x in front_wall_pos),
            rgba=self.rgba,
            mass=str(component_mass),
        )

        # Left wall (along y-axis, at -x edge)
        left_wall_size = [half_wall, inner_half_width, half_wall_height]
        left_wall_pos = [-half_length + half_wall, 0.0, self.wall_thickness + half_wall_height]
        ET.SubElement(
            body,
            "geom",
            type="box",
            size=" ".join(str(x) for x in left_wall_size),
            pos=" ".join(str(x) for x in left_wall_pos),
            rgba=self.rgba,
            mass=str(component_mass),
        )

        # Right wall (along y-axis, at +x edge)
        right_wall_size = [half_wall, inner_half_width, half_wall_height]
        right_wall_pos = [half_length - half_wall, 0.0, self.wall_thickness + half_wall_height]
        ET.SubElement(
            body,
            "geom",
            type="box",
            size=" ".join(str(x) for x in right_wall_size),
            pos=" ".join(str(x) for x in right_wall_pos),
            rgba=self.rgba,
            mass=str(component_mass),
        )

        return body

    def get_bounding_box_dimensions(self) -> tuple[float, float, float]:
        """Get the bounding box dimensions for this bin.

        Returns:
            Tuple of (length, width, height) for the bounding box
        """
        return (self.length, self.width, self.height)

    @staticmethod
    def get_bounding_box_from_config(
        pos: NDArray[np.float32], object_config: dict[str, str | float]
    ) -> list[float]:
        """Get bounding box for a bin given its position and config.

        Args:
            pos: Position of the bin base as [x, y, z] array
            object_config: Dictionary containing bin configuration with keys:
                - "length": Length of bin (x dimension)
                - "width": Width of bin (y dimension)
                - "height": Height of bin (z dimension)

        Returns:
            Bounding box as [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        # Extract bin parameters
        length = float(object_config.get("length", 0.1))
        width = float(object_config.get("width", 0.1))
        height = float(object_config.get("height", 0.05))

        # Half-extents
        half_length = length / 2
        half_width = width / 2

        return [
            pos[0] - half_length,  # x_min
            pos[1] - half_width,   # y_min
            pos[2],                # z_min (at base)
            pos[0] + half_length,  # x_max
            pos[1] + half_width,   # y_max
            pos[2] + height,       # z_max
        ]

    def __str__(self) -> str:
        """String representation of the bin."""
        return (
            f"Bin(name='{self.name}', length={self.length}, "
            f"width={self.width}, height={self.height}, "
            f"wall_thickness={self.wall_thickness})"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the bin."""
        return (
            f"Bin(name='{self.name}', joint_name='{self.joint_name}', "
            f"length={self.length}, width={self.width}, height={self.height}, "
            f"wall_thickness={self.wall_thickness}, rgba='{self.rgba}', "
            f"mass={self.mass})"
        )
