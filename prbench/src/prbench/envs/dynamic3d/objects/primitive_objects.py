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

        if self.regions is not None:
            self._create_regions()

    def _create_xml_element(self) -> ET.Element:
        """Create the XML Element for this cuboid.

        The cuboid is created as a single box geom centered at the body's origin.
        The origin (0, 0, 0) is located at the center of the cuboid.
        The cuboid extends by size[i]/2 in each direction
        (+/- x, +/- y, +/- z) from the origin.

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
        self.width = float(self.options.get("width", 0.1))  # y dimension
        self.height = float(self.options.get("height", 0.05))  # z dimension
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

        if self.regions is not None:
            self._create_regions()

    def _create_xml_element(self) -> ET.Element:
        """Create the XML Element for this bin using multiple box geoms.

        The bin is constructed from 5 box geoms:
        - 1 bottom panel: full outer dimensions (length x width),
          at z in [0, wall_thickness]
        - 4 wall panels: back, front, left, right walls with thickness
          wall_thickness, extending from z = wall_thickness to z = height

        The origin (0, 0, 0) is located at the base center of the bin
        (center of bottom surface). The bin extends in the positive z direction.

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
        back_wall_z = self.wall_thickness + half_wall_height
        back_wall_pos = [0.0, -half_width + half_wall, back_wall_z]
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
        front_wall_z = self.wall_thickness + half_wall_height
        front_wall_pos = [0.0, half_width - half_wall, front_wall_z]
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
        left_wall_z = self.wall_thickness + half_wall_height
        left_wall_pos = [-half_length + half_wall, 0.0, left_wall_z]
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
        right_wall_z = self.wall_thickness + half_wall_height
        right_wall_pos = [half_length - half_wall, 0.0, right_wall_z]
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
            pos[1] - half_width,  # y_min
            pos[2],  # z_min (at base)
            pos[0] + half_length,  # x_max
            pos[1] + half_width,  # y_max
            pos[2] + height,  # z_max
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


@register_object
class Wiper(MujocoObject):
    """A wiper object composed of a long handle and a perpendicular blade head."""

    default_handle_width: float = 0.01  # Default handle width in meters
    default_handle_height: float = 0.01  # Default handle height in meters
    default_head_length: float = 0.15  # Default head length in meters
    default_head_height: float = 0.01  # Default head height in meters

    def __init__(
        self,
        name: str,
        env: MujocoEnv | None = None,
        options: dict | None = None,
    ) -> None:
        """Initialize a Wiper object.

        Args:
            name: Name of the wiper body in the XML
            options: Dictionary of wiper options:
                - handle_width: Width of the handle in both x and y dimensions
                  (default: 0.01)
                - handle_height: Height of the handle in z dimension (default: 0.01)
                - head_length: Length of the blade head in x dimension (default: 0.15)
                - head_height: Height of the blade head in z dimension (default: 0.01)
                - handle_rgba: Color of the handle (default: [0.5, 0.5, 0.5, 1])
                - head_rgba: Color of the head (default: [0.5, 0.5, 0.5, 1])
                - mass: Mass of the wiper (default: 0.1)
            env: Reference to the environment
        """
        super().__init__(name, env, options)

        # Override object type
        self.symbolic_object = Object(self.name, MujocoMovableObjectType)

        # Handle parameters
        self.handle_width = float(self.options.get("handle_width", 0.01))
        self.handle_height = float(self.options.get("handle_height", 0.01))

        # Blade head parameters
        self.head_length = float(self.options.get("head_length", 0.15))
        self.head_height = float(self.options.get("head_height", 0.01))

        # Handle rgba parameter - can be string or list of values
        handle_rgba = self.options.get("handle_rgba", [0.5, 0.5, 0.5, 1])
        if isinstance(handle_rgba, str):
            self.handle_rgba = handle_rgba
        else:
            self.handle_rgba = " ".join(str(x) for x in handle_rgba)

        # Head rgba parameter - can be string or list of values
        head_rgba = self.options.get("head_rgba", [0.5, 0.5, 0.5, 1])
        if isinstance(head_rgba, str):
            self.head_rgba = head_rgba
        else:
            self.head_rgba = " ".join(str(x) for x in head_rgba)

        # Handle mass parameter with default
        self.mass = self.options.get("mass", 0.1)

        # Create the XML element
        self.xml_element = self._create_xml_element()

        if self.regions is not None:
            self._create_regions()

    def _create_xml_element(self) -> ET.Element:
        """Create the XML Element for this wiper.

        The wiper consists of:
        - A handle: a box with width x width x height (in x, y, z)
        - A blade head: a box with head_length x width x head_height (in x, y, z)
          positioned at the end of the handle

        Returns:
            ET.Element representing the wiper body with both geoms
        """
        # Create body element
        body = ET.Element("body", name=self.name)

        # Add freejoint for position/orientation control
        ET.SubElement(body, "freejoint", name=self.joint_name)

        # Mass distribution (divide between handle and head)
        component_mass = self.mass / 2.0

        # Handle: a box with square cross-section in x-y plane
        # MuJoCo box size is half-extent in each direction
        handle_size = (
            f"{self.handle_width / 2} {self.handle_width / 2} {self.handle_height / 2}"
        )
        handle_pos_z = self.handle_height / 2 + self.head_height
        ET.SubElement(
            body,
            "geom",
            type="box",
            size=handle_size,
            pos=f"0 0 {handle_pos_z}",
            rgba=self.handle_rgba,
            mass=str(component_mass),
        )

        # Blade head: box at the end of the handle
        # Position: at the end of the handle along x-axis
        head_pos = f"0 0 {self.head_height / 2}"
        # Size: head_length in x, width in y, and head_height in z
        head_size = (
            f"{self.head_length / 2} {self.handle_width / 2} {self.head_height / 2}"
        )
        ET.SubElement(
            body,
            "geom",
            type="box",
            size=head_size,
            pos=head_pos,
            rgba=self.head_rgba,
            mass=str(component_mass),
        )

        return body

    def get_bounding_box_dimensions(self) -> tuple[float, float, float]:
        """Get the bounding box dimensions for this wiper.

        Returns:
            Tuple of (length, width, height) encompassing both handle and blade
        """
        # Total length is handle_width + head_length
        total_length = self.handle_width + self.head_length
        total_width = self.handle_width
        total_height = self.handle_width
        return (total_length, total_width, total_height)

    @staticmethod
    def get_bounding_box_from_config(
        pos: NDArray[np.float32], object_config: dict[str, str | float]
    ) -> list[float]:
        """Get bounding box for a wiper given its position and config.

        Args:
            pos: Position of the wiper as [x, y, z] array
            object_config: Dictionary containing wiper configuration with keys:
                - "handle_width": Width of the handle in x and y dimensions
                - "handle_height": Height of the handle in z dimension
                - "head_length": Length of the blade head in x dimension
                - "head_height": Height of the blade head in z dimension

        Returns:
            Bounding box as [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        # Extract wiper parameters
        handle_width = float(object_config.get("handle_width", 0.01))
        handle_height = float(object_config.get("handle_height", 0.01))
        head_length = float(object_config.get("head_length", 0.15))
        head_height = float(object_config.get("head_height", 0.01))

        # Head geom: size=[head_length/2, handle_width/2, head_height/2], pos=[0, 0, head_height/2]
        #   Extends x: ±head_length/2, y: ±handle_width/2, z: [0, head_height]
        # Handle geom: size=[handle_width/2, handle_width/2, handle_height/2], pos=[0, 0, head_height + handle_height/2]
        #   Extends x: ±handle_width/2, y: ±handle_width/2, z: [head_height, head_height + handle_height]

        # Overall bounds relative to body origin:
        # x: [-head_length/2, head_length/2] (head is longer)
        # y: [-handle_width/2, handle_width/2]
        # z: [0, head_height + handle_height]

        x_min = pos[0] - head_length / 2
        x_max = pos[0] + head_length / 2

        y_min = pos[1] - handle_width / 2
        y_max = pos[1] + handle_width / 2

        z_min = pos[2]
        z_max = pos[2] + head_height + handle_height

        return [x_min, y_min, z_min, x_max, y_max, z_max]

    def __str__(self) -> str:
        """String representation of the wiper."""
        return (
            f"Wiper(name='{self.name}', handle_width={self.handle_width}, "
            f"handle_height={self.handle_height}, head_length={self.head_length}, "
            f"head_height={self.head_height}, handle_rgba='{self.handle_rgba}', "
            f"head_rgba='{self.head_rgba}', mass={self.mass})"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the wiper."""
        return (
            f"Wiper(name='{self.name}', joint_name='{self.joint_name}', "
            f"handle_width={self.handle_width}, handle_height={self.handle_height}, "
            f"head_length={self.head_length}, head_height={self.head_height}, "
            f"handle_rgba='{self.handle_rgba}', head_rgba='{self.head_rgba}', "
            f"mass={self.mass})"
        )
