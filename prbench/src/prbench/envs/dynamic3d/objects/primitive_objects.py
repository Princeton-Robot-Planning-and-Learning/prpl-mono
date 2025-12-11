"""Primitive MuJoCo object classes such as Cube and Cuboid."""

from __future__ import annotations

import xml.etree.ElementTree as ET
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
