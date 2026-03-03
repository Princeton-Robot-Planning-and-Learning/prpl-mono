"""Tests for dynamic3d objects utility functions."""

import tempfile
from pathlib import Path

import numpy as np

from kinder.envs.dynamic3d.objects.utils import save_mesh


def test_save_mesh_basic():
    """Test basic mesh saving functionality."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    with tempfile.TemporaryDirectory() as temp_dir:
        mesh_file = save_mesh(vertices, faces, temp_dir)

        assert Path(mesh_file).exists()
        assert mesh_file.endswith(".obj")


def test_save_mesh_with_string_path():
    """Test save_mesh with directory as string."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    with tempfile.TemporaryDirectory() as temp_dir:
        mesh_file = save_mesh(vertices, faces, str(temp_dir))

        assert Path(mesh_file).exists()
        assert mesh_file.endswith(".obj")
        assert temp_dir in mesh_file


def test_save_mesh_with_path_object():
    """Test save_mesh with directory as Path object."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    with tempfile.TemporaryDirectory() as temp_dir:
        mesh_file = save_mesh(vertices, faces, Path(temp_dir))

        assert Path(mesh_file).exists()
        assert mesh_file.endswith(".obj")


def test_save_mesh_creates_directory():
    """Test that save_mesh creates the directory if it doesn't exist."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    with tempfile.TemporaryDirectory() as temp_dir:
        nested_dir = Path(temp_dir) / "nested" / "directory"
        mesh_file = save_mesh(vertices, faces, nested_dir)

        assert nested_dir.exists()
        assert Path(mesh_file).exists()


def test_save_mesh_content_vertices():
    """Test that vertices are written correctly to the OBJ file."""
    vertices = np.array(
        [[0.5, 1.5, 2.5], [3.7, 4.2, 5.1], [-1.0, 0.0, 1.0]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    with tempfile.TemporaryDirectory() as temp_dir:
        mesh_file = save_mesh(vertices, faces, temp_dir)

        with open(mesh_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that vertices are in the file with correct formatting
        assert "v 0.500000 1.500000 2.500000" in content
        assert "v 3.700000 4.200000 5.100000" in content
        assert "v -1.000000 0.000000 1.000000" in content


def test_save_mesh_content_faces():
    """Test that faces are written correctly with 1-based indexing."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)

    with tempfile.TemporaryDirectory() as temp_dir:
        mesh_file = save_mesh(vertices, faces, temp_dir)

        with open(mesh_file, "r", encoding="utf-8") as f:
            content = f.read()

        # OBJ uses 1-based indexing, so face [0, 1, 2] becomes [1, 2, 3]
        assert "f 1 2 3" in content
        assert "f 2 4 3" in content


def test_save_mesh_content_header():
    """Test that the OBJ file has the correct header."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    with tempfile.TemporaryDirectory() as temp_dir:
        mesh_file = save_mesh(vertices, faces, temp_dir)

        with open(mesh_file, "r", encoding="utf-8") as f:
            first_line = f.readline()

        assert first_line.startswith("# Generated mesh")


def test_save_mesh_multiple_faces():
    """Test save_mesh with multiple faces."""
    # Create a simple pyramid with 4 triangular faces
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # base center
            [1.0, 0.0, 0.0],  # base corner 1
            [0.0, 1.0, 0.0],  # base corner 2
            [-1.0, 0.0, 0.0],  # base corner 3
            [0.0, 0.0, 1.0],  # apex
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 4], [0, 2, 4], [0, 3, 4], [0, 4, 1]], dtype=np.int32)

    with tempfile.TemporaryDirectory() as temp_dir:
        mesh_file = save_mesh(vertices, faces, temp_dir)

        with open(mesh_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Verify all faces are present (with 1-based indexing)
        lines = content.strip().split("\n")
        face_lines = [line for line in lines if line.startswith("f ")]
        assert len(face_lines) == 4


def test_save_mesh_large_mesh():
    """Test save_mesh with a larger mesh."""
    # Create a simple icosphere-like mesh with many vertices
    rng = np.random.default_rng(0)
    num_vertices = 100
    vertices = rng.standard_normal((num_vertices, 3)).astype(np.float32)
    num_faces = 50
    faces = rng.integers(0, num_vertices, (num_faces, 3)).astype(np.int32)

    with tempfile.TemporaryDirectory() as temp_dir:
        mesh_file = save_mesh(vertices, faces, temp_dir)

        with open(mesh_file, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.strip().split("\n")
        vertex_lines = [line for line in lines if line.startswith("v ")]
        face_lines = [line for line in lines if line.startswith("f ")]

        assert len(vertex_lines) == num_vertices
        assert len(face_lines) == num_faces


def test_save_mesh_precision():
    """Test that vertex coordinates maintain proper precision."""
    vertices = np.array([[0.123456, 0.654321, 0.111111]], dtype=np.float32)
    faces = np.array([[0, 0, 0]], dtype=np.int32)

    with tempfile.TemporaryDirectory() as temp_dir:
        mesh_file = save_mesh(vertices, faces, temp_dir)

        with open(mesh_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that coordinates are formatted to 6 decimal places
        assert "v 0.123456 0.654321 0.111111" in content


def test_save_mesh_default_directory():
    """Test that save_mesh uses a default directory when None is provided."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    # When mesh_dir is None, it should use a default location
    mesh_file = save_mesh(vertices, faces, None)

    assert Path(mesh_file).exists()
    assert mesh_file.endswith(".obj")

    # Clean up the created file
    Path(mesh_file).unlink()


def test_save_mesh_file_format():
    """Test the overall format of the generated OBJ file."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    with tempfile.TemporaryDirectory() as temp_dir:
        mesh_file = save_mesh(vertices, faces, temp_dir)

        with open(mesh_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]

        # First line should be comment
        assert lines[0].startswith("#")

        # Next 3 lines should be vertices
        vertex_lines = [line for line in lines if line.startswith("v ")]
        assert len(vertex_lines) == 3

        # Last line should be face
        face_lines = [line for line in lines if line.startswith("f ")]
        assert len(face_lines) == 1
        assert face_lines[0] == "f 1 2 3"


def test_save_mesh_returns_string_path():
    """Test that save_mesh returns a string path."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    with tempfile.TemporaryDirectory() as temp_dir:
        mesh_file = save_mesh(vertices, faces, temp_dir)

        assert isinstance(mesh_file, str)
        assert Path(mesh_file).exists()
