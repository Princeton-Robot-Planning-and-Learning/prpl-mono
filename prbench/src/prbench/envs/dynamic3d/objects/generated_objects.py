"""Generated mesh objects for dynamic3d environments."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from relational_structs import Object

from prbench.envs.dynamic3d.mujoco_utils import MujocoEnv
from prbench.envs.dynamic3d.object_types import MujocoMovableObjectType
from prbench.envs.dynamic3d.objects.base import MujocoObject, register_object
from prbench.envs.dynamic3d.objects.utils import save_mesh


@register_object(name="generated_bowl")
class GeneratedBowl(MujocoObject):
    """A procedurally generated bowl object for TidyBot environments."""

    def __init__(
        self,
        name: str,
        env: MujocoEnv | None = None,
        options: dict | None = None,
    ) -> None:
        """Initialize a GeneratedBowl object.

        Args:
            name: Name of the bowl body in the XML
            env: Reference to the environment
            options: Dictionary of bowl options:
                - outer_radius: Outer radius at rim (default: 0.05m)
                - inner_radius: Inner radius at rim (default: 0.045m)
                - height: Height/depth of the bowl (default: 0.025m)
                - wall_thickness: Bowl wall thickness at bottom (default: 0.003m)
                - radial_segments: Segments around circumference (default: 32)
                - vertical_segments: Segments from rim to bottom (default: 16)
                - rgba: Color as string or [r, g, b, a] (default: ".5 .5 .5 1")
                - mass: Mass of the bowl (default: 0.05)
        """
        # Initialize base class
        super().__init__(name, env, options)

        # Override object type
        self.symbolic_object = Object(self.name, MujocoMovableObjectType)

        # Bowl parameters
        self.outer_radius: float = float(self.options.get("outer_radius", 0.05))
        self.inner_radius: float = float(self.options.get("inner_radius", 0.045))
        self.height: float = float(self.options.get("height", 0.025))
        self.wall_thickness: float = float(self.options.get("wall_thickness", 0.003))
        self.radial_segments: int = int(self.options.get("radial_segments", 32))
        self.vertical_segments: int = int(self.options.get("vertical_segments", 16))

        # Handle rgba parameter
        rgba = self.options.get("rgba", ".5 .5 .5 1")
        if isinstance(rgba, str):
            self.rgba = rgba
        else:
            self.rgba = " ".join(str(x) for x in rgba)

        self.mass: float = float(self.options.get("mass", 0.05))

        # Generate mesh and create temporary OBJ file
        self.mesh_file = self._generate_and_save_mesh()
        self.mesh_name = f"{self.name}_bowl_mesh"

        # Create the XML element
        self.xml_element = self._create_xml_element()

    def _generate_and_save_mesh(self) -> str:
        """Generate bowl mesh and save to a temporary OBJ file.

        Returns:
            Path to the generated OBJ file
        """
        # Generate mesh vertices and faces
        vertices, faces = self._generate_bowl_mesh()

        # Define directory for temporary meshes
        mesh_dir = Path(__file__).parents[1] / "models" / "assets" / ".tmp"

        # Save mesh using utility function
        return save_mesh(vertices, faces, mesh_dir)

    def _generate_bowl_mesh(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a bowl mesh with specified dimensions.

        Returns:
            Tuple of (vertices, faces) arrays
        """
        vertices_list = []
        faces_list = []

        # Generate outer surface (true hemisphere shape)
        for i in range(self.vertical_segments + 1):
            # Angle from vertical: pi/2 at rim (i=0), 0 at bottom
            theta = (np.pi / 2) * (1 - i / self.vertical_segments)

            # For a true hemisphere bowl:
            # At i=0 (rim/top): theta=pi/2, r=outer_radius, z=0
            # At i=vertical_segments (bottom): theta=0, r=0, z=-height
            r = self.outer_radius * np.sin(theta)
            z = -self.height * np.cos(theta)

            for j in range(self.radial_segments):
                phi = (j / self.radial_segments) * 2 * np.pi
                x = r * np.cos(phi)
                y = r * np.sin(phi)
                vertices_list.append([x, y, z])

        # Generate inner surface (slightly smaller hemisphere)
        for i in range(self.vertical_segments + 1):
            theta = (np.pi / 2) * (1 - i / self.vertical_segments)
            r = self.inner_radius * np.sin(theta)
            z = -self.height * np.cos(theta)

            for j in range(self.radial_segments):
                phi = (j / self.radial_segments) * 2 * np.pi
                x = r * np.cos(phi)
                y = r * np.sin(phi)
                vertices_list.append([x, y, z])

        vertices = np.array(vertices_list)

        # Create faces for outer surface
        num_outer_vertices = (self.vertical_segments + 1) * self.radial_segments
        for i in range(self.vertical_segments):
            for j in range(self.radial_segments):
                # Current quad vertices
                v0 = i * self.radial_segments + j
                v1 = i * self.radial_segments + (j + 1) % self.radial_segments
                v2 = (i + 1) * self.radial_segments + (
                    j + 1
                ) % self.radial_segments
                v3 = (i + 1) * self.radial_segments + j

                # Two triangles per quad (outer surface faces outward)
                faces_list.append([v0, v2, v1])
                faces_list.append([v0, v3, v2])

        # Create faces for inner surface
        for i in range(self.vertical_segments):
            for j in range(self.radial_segments):
                # Current quad vertices (offset by outer surface vertices)
                v0 = num_outer_vertices + i * self.radial_segments + j
                v1 = num_outer_vertices + i * self.radial_segments + (
                    j + 1
                ) % self.radial_segments
                v2 = (
                    num_outer_vertices
                    + (i + 1) * self.radial_segments
                    + (j + 1) % self.radial_segments
                )
                v3 = num_outer_vertices + (i + 1) * self.radial_segments + j

                # Two triangles per quad (inner surface faces inward, so reverse winding)
                faces_list.append([v0, v1, v2])
                faces_list.append([v0, v2, v3])

        # Create rim (flat surface connecting outer rim to inner rim at the top)
        for j in range(self.radial_segments):
            # Outer rim vertices (i=0, at the opening)
            v0_outer = j
            v1_outer = (j + 1) % self.radial_segments

            # Inner rim vertices (i=0, at the opening)
            v0_inner = num_outer_vertices + j
            v1_inner = num_outer_vertices + (j + 1) % self.radial_segments

            # Two triangles to create flat rim surface (facing upward)
            faces_list.append([v0_outer, v1_outer, v1_inner])
            faces_list.append([v0_outer, v1_inner, v0_inner])

        faces = np.array(faces_list)

        return vertices, faces

    def get_assets(self) -> list[ET.Element]:
        """Get the asset elements (mesh) for this bowl.

        Returns:
            List of ET.Element containing mesh asset
        """
        # Create mesh asset element
        mesh_elem = ET.Element("mesh")
        mesh_elem.set("file", self.mesh_file)
        mesh_elem.set("name", self.mesh_name)

        return [mesh_elem]

    def _create_xml_element(self) -> ET.Element:
        """Create the XML Element for this bowl.

        Returns:
            ET.Element representing the bowl body
        """
        # Create body element
        body = ET.Element("body", name=self.name)

        # Add freejoint for position/orientation control
        ET.SubElement(body, "freejoint", name=self.joint_name)

        # Add geom element with mesh reference (mesh will be added to assets)
        ET.SubElement(
            body,
            "geom",
            type="mesh",
            mesh=self.mesh_name,
            rgba=self.rgba,
            mass=str(self.mass),
        )

        return body

    def get_bounding_box_dimensions(self) -> tuple[float, float, float]:
        """Get the bounding box dimensions for this bowl.

        Returns:
            Tuple of (width, depth, height) for the bounding box
        """
        # Bowl dimensions: diameter x diameter x height
        diameter = 2 * self.outer_radius
        return (diameter, diameter, self.height)

    @staticmethod
    def get_bounding_box_from_config(
        pos: NDArray[np.float32], object_config: dict[str, str | float]
    ) -> list[float]:
        """Get bounding box for a bowl given its position and config.

        Args:
            pos: Position of the bowl as [x, y, z] array
            object_config: Dictionary containing bowl configuration

        Returns:
            Bounding box as [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        # Extract bowl parameters
        outer_radius = float(object_config.get("outer_radius", 0.05))
        height = float(object_config.get("height", 0.025))

        # Half-extents
        half_diameter = outer_radius
        half_height = height / 2

        return [
            float(pos[0]) - half_diameter,  # x_min
            float(pos[1]) - half_diameter,  # y_min
            float(pos[2]) - half_height,  # z_min
            float(pos[0]) + half_diameter,  # x_max
            float(pos[1]) + half_diameter,  # y_max
            float(pos[2]) + half_height,  # z_max
        ]

    def __str__(self) -> str:
        """String representation of the bowl."""
        return (
            f"GeneratedBowl(name='{self.name}', "
            f"outer_radius={self.outer_radius}, inner_radius={self.inner_radius}, "
            f"height={self.height})"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the bowl."""
        return (
            f"GeneratedBowl(name='{self.name}', joint_name='{self.joint_name}', "
            f"outer_radius={self.outer_radius}, inner_radius={self.inner_radius}, "
            f"height={self.height}, mass={self.mass})"
        )
