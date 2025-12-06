"""Tests for utils.py."""

import pybullet as p

from pybullet_helpers.geometry import Pose, set_pose
from pybullet_helpers.utils import create_pybullet_shelf


def test_create_pybullet_shelf():
    """Tests for create_pybullet_shelf()."""

    # Set up a scene to test manipuation.
    physics_client_id = p.connect(p.DIRECT)

    # Uncomment to debug.
    # from pybullet_helpers.gui import create_gui_connection
    # physics_client_id = create_gui_connection(camera_yaw=0)

    shelf_id, surface_ids = create_pybullet_shelf(
        color=(0.5, 0.5, 0.5, 1.0),
        shelf_width=0.8,
        shelf_depth=0.3,
        shelf_height=0.1,
        spacing=0.3,
        support_width=0.05,
        num_layers=3,
        physics_client_id=physics_client_id,
    )
    assert isinstance(shelf_id, int)
    assert len(surface_ids) == 3
    set_pose(shelf_id, Pose((0.0, 0.0, 0.0)), physics_client_id)

    taller_shelf_id, taller_surface_ids = create_pybullet_shelf(
        color=(0.5, 0.5, 0.5, 1.0),
        shelf_width=0.8,
        shelf_depth=0.3,
        shelf_height=0.1,
        spacing=0.3,
        support_width=0.05,
        num_layers=5,
        physics_client_id=physics_client_id,
    )
    assert isinstance(taller_shelf_id, int)
    assert len(taller_surface_ids) == 5
    set_pose(taller_shelf_id, Pose((0.0, 1.0, 0.0)), physics_client_id)

    # Uncomment to debug.
    # while True:
    #     p.getMouseEvents(physics_client_id)
