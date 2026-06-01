"""Unit tests for mesh conversion helpers."""

import os

import pybullet as p
import trimesh

from prpl_kinematics.meshes import to_pybullet_mesh


def test_native_mesh_passthrough():
    """Native mesh formats are returned unchanged, without conversion."""
    assert to_pybullet_mesh("/some/dir/link.obj") == "/some/dir/link.obj"
    assert to_pybullet_mesh("/some/dir/link.STL") == "/some/dir/link.STL"


def test_glb_converted_to_loadable_obj(physics_client_id, tmp_path):
    """A non-native .glb mesh is converted to a .obj that PyBullet can load."""
    glb = tmp_path / "box.glb"
    trimesh.creation.box((0.2, 0.2, 0.2)).export(str(glb))
    obj = to_pybullet_mesh(str(glb))
    assert obj.endswith(".obj")
    assert os.path.exists(obj)
    shape = p.createVisualShape(
        p.GEOM_MESH, fileName=obj, physicsClientId=physics_client_id
    )
    assert shape >= 0
