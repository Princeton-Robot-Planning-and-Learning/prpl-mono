"""Utility functions for dynamic3d objects."""

import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def euler_to_quat(euler: list[float]) -> str:
    """Convert euler angles (roll, pitch, yaw) in degrees to MuJoCo quaternion string.

    Args:
        euler: [roll, pitch, yaw] in degrees

    Returns:
        Quaternion string "w x y z" for MuJoCo
    """
    # Convert degrees to radians
    roll = np.radians(euler[0])
    pitch = np.radians(euler[1])
    yaw = np.radians(euler[2])

    # Calculate quaternion components
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return f"{w} {x} {y} {z}"


def save_mesh(
    vertices: NDArray[np.float32],
    faces: NDArray[np.int32],
    mesh_dir: Path | str | None = None,
) -> str:
    """Save mesh vertices and faces to a temporary OBJ file.

    Args:
        vertices: Array of vertices with shape (N, 3) containing [x, y, z] coordinates
        faces: Array of faces with shape (M, 3) containing vertex indices
        mesh_dir: Directory to store the mesh file. If None, uses a temporary directory

    Returns:
        Path to the generated OBJ file
    """
    # Convert to Path object if string provided
    if isinstance(mesh_dir, str):
        mesh_dir = Path(mesh_dir)

    # Create target directory for temporary meshes if not provided
    if mesh_dir is None:
        mesh_dir = Path(__file__).parents[1] / "models" / "assets" / ".tmp"

    mesh_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary file for the mesh
    temp_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".obj", delete=False, dir=str(mesh_dir), encoding="utf-8"
    )
    mesh_file = temp_file.name

    # Write OBJ file
    with open(mesh_file, "w", encoding="utf-8") as f:
        f.write("# Generated mesh\n")

        # Write vertices
        for vertex in vertices:
            f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")

        # Write faces (OBJ uses 1-based indexing)
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    temp_file.close()
    return mesh_file
