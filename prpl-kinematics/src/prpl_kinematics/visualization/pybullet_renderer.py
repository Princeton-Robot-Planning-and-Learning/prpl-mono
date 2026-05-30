"""PyBullet shape-soup renderer: one visual body per node shape, FK-positioned.

``PyBulletRenderer`` creates one PyBullet visual body per node shape and, each
frame, positions every body from the tree's forward kinematics. Nothing uses
PyBullet's articulation, so a grasped object (re-parented in the tree) renders
correctly with no special handling.
"""

from __future__ import annotations

import numpy as np
import pybullet as p

from prpl_kinematics.geometry.shapes import BoxShape, CylinderShape, MeshShape, Shape
from prpl_kinematics.geometry.transforms import pose_to_pybullet
from prpl_kinematics.meshes import to_pybullet_mesh
from prpl_kinematics.tree.kinematic_tree import Configuration, KinematicTree
from prpl_kinematics.visualization.interface import CameraParams, Renderer


def _create_visual_shape(physics_client_id: int, shape: Shape) -> int:
    position, orientation = pose_to_pybullet(shape.origin)
    common = {
        "visualFramePosition": position,
        "visualFrameOrientation": orientation,
        "physicsClientId": physics_client_id,
    }
    if shape.color is not None:
        common["rgbaColor"] = list(shape.color)
    if isinstance(shape, MeshShape):
        return int(
            p.createVisualShape(
                p.GEOM_MESH,
                fileName=to_pybullet_mesh(shape.filename),
                meshScale=list(shape.scale),
                **common,
            )
        )
    if isinstance(shape, BoxShape):
        half_extents = [shape.size[0] / 2, shape.size[1] / 2, shape.size[2] / 2]
        return int(p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, **common))
    if isinstance(shape, CylinderShape):
        return int(
            p.createVisualShape(
                p.GEOM_CYLINDER, radius=shape.radius, length=shape.length, **common
            )
        )
    return int(p.createVisualShape(p.GEOM_SPHERE, radius=shape.radius, **common))


class PyBulletRenderer(Renderer):
    """Renders a KinematicTree by positioning per-shape visual bodies via FK."""

    def __init__(self, physics_client_id: int) -> None:
        self._physics_client_id = physics_client_id
        self._tree: KinematicTree | None = None
        self._bodies: list[tuple[int, str]] = []

    @property
    def physics_client_id(self) -> int:
        """The PyBullet client this renderer draws into."""
        return self._physics_client_id

    def load(self, tree: KinematicTree) -> None:
        """Create one visual body per node shape (positioned later by FK)."""
        self._tree = tree
        for name, node in tree.nodes.items():
            for shape in node.visuals:
                visual = _create_visual_shape(self._physics_client_id, shape)
                body = int(
                    p.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=visual,
                        physicsClientId=self._physics_client_id,
                    )
                )
                self._bodies.append((body, name))

    def render(self, config: Configuration) -> None:
        """Move every visual body to its node's world pose under ``config``."""
        assert self._tree is not None, "call load() before render()"
        for body, name in self._bodies:
            position, orientation = pose_to_pybullet(
                self._tree.forward_kinematics(name, config)
            )
            p.resetBasePositionAndOrientation(
                body, position, orientation, physicsClientId=self._physics_client_id
            )

    def capture_image(self, camera: CameraParams = CameraParams()) -> np.ndarray:
        """Capture an RGB image of the current scene from a synthetic camera."""
        return capture_image(self._physics_client_id, camera)


def capture_image(
    physics_client_id: int, camera: CameraParams = CameraParams()
) -> np.ndarray:
    """Capture an RGB image (H, W, 3) ``uint8`` from a synthetic camera."""
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera.target,
        distance=camera.distance,
        yaw=camera.yaw,
        pitch=camera.pitch,
        roll=0.0,
        upAxisIndex=2,
    )
    projection = p.computeProjectionMatrixFOV(
        fov=camera.fov,
        aspect=camera.width / camera.height,
        nearVal=camera.near,
        farVal=camera.far,
    )
    _, _, rgba, _, _ = p.getCameraImage(
        camera.width,
        camera.height,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=physics_client_id,
    )
    image = np.reshape(
        np.asarray(rgba, dtype=np.uint8), (camera.height, camera.width, 4)
    )
    return image[:, :, :3]
