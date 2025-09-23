"""PyBullet helpers for cameras and rendering."""

import numpy as np
import pybullet as p
from tomsutils.structs import Image

from pybullet_helpers.geometry import Pose3D


def capture_image(
    physics_client_id: int,
    camera_distance: float = 1.5,
    camera_yaw: float = 0,
    camera_pitch: float = -15,
    camera_target: Pose3D = (0, 0, 0.5),
    image_width: int = 1674,
    image_height: int = 900,
) -> Image:
    """Capture an image."""
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera_target,
        distance=camera_distance,
        yaw=camera_yaw,
        pitch=camera_pitch,
        roll=0,
        upAxisIndex=2,
        physicsClientId=physics_client_id,
    )

    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=float(image_width / image_height),
        nearVal=0.1,
        farVal=100.0,
        physicsClientId=physics_client_id,
    )

    (_, _, px, _, _) = p.getCameraImage(
        width=image_width,
        height=image_height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        physicsClientId=physics_client_id,
    )

    rgb_array = np.array(px).reshape((image_height, image_width, 4))
    rgb_array = rgb_array[:, :, :3]  # type: ignore
    return rgb_array.astype(np.uint8)


def capture_superimposed_image(
    physics_client_id: int,
    outer_camera_distance: float = 1.5,
    outer_camera_yaw: float = 0,
    outer_camera_pitch: float = -15,
    outer_camera_target: Pose3D = (0, 0, 0.5),
    outer_image_width: int = 900,
    outer_image_height: int = 900,
    inner_camera_distance: float = 1.0,
    inner_camera_yaw: float = 0,
    inner_camera_pitch: float = -15,
    inner_camera_target: Pose3D = (0, 0, 0.5),
    inner_image_width: int = 300,
    inner_image_height: int = 300,
    inner_row_offset: int = 5,
    inner_col_offset: int = 5,
    inner_border_size: int = 2,
    inner_border_color: tuple[float, float, float] = (200.0, 200.0, 200.0),
) -> Image:
    """Get two images and superimpose them."""

    outer_image = capture_image(
        physics_client_id,
        camera_target=outer_camera_target,
        camera_yaw=outer_camera_yaw,
        camera_distance=outer_camera_distance,
        camera_pitch=outer_camera_pitch,
        image_width=outer_image_width,
        image_height=outer_image_height,
    )

    inner_image = capture_image(
        physics_client_id,
        camera_target=inner_camera_target,
        camera_yaw=inner_camera_yaw,
        camera_distance=inner_camera_distance,
        camera_pitch=inner_camera_pitch,
        image_width=inner_image_width,
        image_height=inner_image_height,
    )

    combined_image = outer_image.copy()

    r_start = inner_row_offset
    c_start = inner_col_offset
    r_end = inner_image_height + 2 * inner_border_size + inner_row_offset
    c_end = inner_image_width + 2 * inner_border_size + inner_col_offset
    for i, v in enumerate(inner_border_color):
        combined_image[r_start:r_end, c_start:c_end, i] = v

    r_start = inner_row_offset + inner_border_size
    c_start = inner_col_offset + inner_border_size
    r_end = inner_image_height + inner_row_offset + inner_border_size
    c_end = inner_image_width + inner_col_offset + inner_border_size
    combined_image[r_start:r_end, c_start:c_end] = inner_image

    return combined_image.astype(np.uint8)
