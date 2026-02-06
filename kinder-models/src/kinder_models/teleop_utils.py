"""Teleoperation interface for kinder environments using WebXR phone app."""

import logging
import math
import socket
import threading
import time
from queue import Queue

import cv2 as cv
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R  # type: ignore


def _visualize_image_in_window(image: NDArray[np.uint8], window_name: str) -> None:
    """Visualize an image in an OpenCV window."""
    if image.dtype == np.uint8 and len(image.shape) == 3:
        # Convert RGB to BGR for proper color display in OpenCV
        display_image = cv.cvtColor(  # pylint: disable=no-member
            image, cv.COLOR_RGB2BGR  # pylint: disable=no-member
        )
        cv.imshow(window_name, display_image)  # pylint: disable=no-member
        cv.waitKey(1)  # pylint: disable=no-member


# ============================================================================
# WebXR Web Server for Phone Teleoperation
# ============================================================================


class WebServer:
    """Flask web server for serving the WebXR phone web app."""

    def __init__(self, queue: Queue, port: int = 5000):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.queue = queue
        self.port = port

        @self.app.route("/")
        def index():
            return render_template("index.html")

        @self.socketio.on("message")
        def handle_message(data):
            # Send the timestamp back for RTT calculation
            # (expected RTT on 5 GHz Wi-Fi is 7 ms)
            emit("echo", data["timestamp"])

            # Add data to queue for processing
            self.queue.put(data)

        # Reduce verbose Flask log output
        logging.getLogger("werkzeug").setLevel(logging.WARNING)

    def run(self):
        """Start the web server."""
        # Get IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            s.connect(("8.8.8.8", 1))
            address = s.getsockname()[0]
        except Exception:
            address = "127.0.0.1"
        finally:
            s.close()
        print(f"Starting server at {address}:{self.port}")
        self.socketio.run(self.app, host="0.0.0.0", port=self.port)


# ============================================================================
# WebXR Coordinate Conversion
# ============================================================================

DEVICE_CAMERA_OFFSET = np.array([0.0, 0.02, -0.04])  # iPhone 14 Pro


def convert_webxr_pose(pos: dict, quat: dict) -> tuple[np.ndarray, R]:
    """Convert coordinate system from WebXR to robot.

    WebXR: +x right, +y up, +z back
    Robot: +x forward, +y left, +z up
    """
    pos_array = np.array([-pos["z"], -pos["x"], pos["y"]], dtype=np.float64)
    rot = R.from_quat([-quat["z"], -quat["x"], quat["y"], quat["w"]])

    # Apply offset so that rotations are around device center instead of device camera
    pos_array = pos_array + rot.apply(DEVICE_CAMERA_OFFSET)

    return pos_array, rot


TWO_PI = 2 * math.pi


# ============================================================================
# Teleop Controller
# ============================================================================


class TeleopController:
    """Controller that processes WebXR input to generate robot commands."""

    def __init__(self) -> None:
        # Teleop device IDs
        self.primary_device_id = None  # Primary device controls either arm or base
        self.secondary_device_id = None  # Optional secondary device controls base
        self.enabled_counts: dict = {}

        # Mobile base pose
        self.base_pose = None

        # Teleop targets
        self.targets_initialized = False
        self.base_target_pose = None
        self.arm_target_pos = None
        self.arm_target_rot = None
        self.gripper_target_pos = None

        # WebXR reference poses
        self.base_xr_ref_pos = None
        self.base_xr_ref_rot_inv = None
        self.arm_xr_ref_pos = None
        self.arm_xr_ref_rot_inv = None

        # Robot reference poses
        self.base_ref_pose = None
        self.arm_ref_pos = None
        self.arm_ref_rot = None
        self.arm_ref_base_pose = None  # For optional secondary control of base
        self.gripper_ref_pos = None

    def process_message(self, data: dict) -> None:
        """Process incoming WebXR message."""
        if not self.targets_initialized:
            return

        # Use device ID to disambiguate between primary and secondary devices
        device_id = data["device_id"]

        # Update enabled count for the device that sent this message
        self.enabled_counts[device_id] = (
            self.enabled_counts.get(device_id, 0) + 1 if "teleop_mode" in data else 0
        )

        # Assign primary and secondary devices
        if self.enabled_counts[device_id] > 2:
            if self.primary_device_id is None and device_id != self.secondary_device_id:
                # Skip first 2 steps: WebXR pose updates have higher latency than touch
                self.primary_device_id = device_id
            elif (
                self.secondary_device_id is None and device_id != self.primary_device_id
            ):
                self.secondary_device_id = device_id
        elif self.enabled_counts[device_id] == 0:
            if device_id == self.primary_device_id:
                self.primary_device_id = None  # Primary device no longer enabled
                self.base_xr_ref_pos = None
                self.arm_xr_ref_pos = None
            elif device_id == self.secondary_device_id:
                self.secondary_device_id = None
                self.base_xr_ref_pos = None

        # Teleop is enabled
        if self.primary_device_id is not None and "teleop_mode" in data:  # type: ignore
            pos, rot = convert_webxr_pose(data["position"], data["orientation"])  # type: ignore # pylint: disable=line-too-long

            # Base movement
            if data["teleop_mode"] == "base" or device_id == self.secondary_device_id:
                # Note: Secondary device can only control base
                # Store reference poses
                if self.base_xr_ref_pos is None:
                    self.base_ref_pose = self.base_pose.copy()
                    self.base_xr_ref_pos = pos[:2]
                    self.base_xr_ref_rot_inv = rot.inv()

                # Position
                self.base_target_pose[:2] = self.base_ref_pose[:2] + (
                    pos[:2] - self.base_xr_ref_pos
                )

                # Orientation
                base_fwd_vec_rotated = (rot * self.base_xr_ref_rot_inv).apply(
                    [1.0, 0.0, 0.0]
                )
                base_target_theta = self.base_ref_pose[2] + math.atan2(
                    base_fwd_vec_rotated[1], base_fwd_vec_rotated[0]
                )
                self.base_target_pose[2] += (
                    base_target_theta - self.base_target_pose[2] + math.pi
                ) % TWO_PI - math.pi  # Unwrapped

            # Arm movement
            elif data["teleop_mode"] == "arm":
                # Store reference poses
                if self.arm_xr_ref_pos is None:
                    self.arm_xr_ref_pos = pos
                    self.arm_xr_ref_rot_inv = rot.inv()
                    self.arm_ref_pos = self.arm_target_pos.copy()
                    self.arm_ref_rot = self.arm_target_rot
                    self.arm_ref_base_pose = self.base_pose.copy()
                    self.gripper_ref_pos = self.gripper_target_pos

                # Rotations around z-axis: global frame (base) <-> local frame (arm)
                z_rot = R.from_rotvec(np.array([0.0, 0.0, 1.0]) * self.base_pose[2])
                z_rot_inv = z_rot.inv()
                ref_z_rot = R.from_rotvec(
                    np.array([0.0, 0.0, 1.0]) * self.arm_ref_base_pose[2]
                )

                # Position
                pos_diff = pos - self.arm_xr_ref_pos  # WebXR
                pos_diff += ref_z_rot.apply(self.arm_ref_pos) - z_rot.apply(
                    self.arm_ref_pos
                )  # Compensate for base rotation
                pos_diff[:2] += (
                    self.arm_ref_base_pose[:2] - self.base_pose[:2]
                )  # Compensate for base translation
                self.arm_target_pos = self.arm_ref_pos + z_rot_inv.apply(pos_diff)

                # Orientation
                self.arm_target_rot = (
                    z_rot_inv * (rot * self.arm_xr_ref_rot_inv) * ref_z_rot
                ) * self.arm_ref_rot

                # Gripper position
                self.gripper_target_pos = np.clip(
                    self.gripper_ref_pos + data["gripper_delta"], 0.0, 1.0
                )

        # Teleop is disabled
        elif self.primary_device_id is None:
            # Update target pose in case base is pushed while teleop is disabled
            self.base_target_pose = self.base_pose

    def step(self, obs: dict) -> dict | None:
        """Generate action from current teleop state.

        Args:
            obs: Observation dictionary with robot state.

        Returns:
            Action dictionary or None if teleop is not enabled.
        """
        # Update robot state
        self.base_pose = obs["base_pose"]

        # Initialize targets
        if not self.targets_initialized:
            self.base_target_pose = obs["base_pose"].copy()
            self.arm_target_pos = obs["arm_pos"].copy()
            self.arm_target_rot = R.from_quat(obs["arm_quat"])  # type: ignore
            self.gripper_target_pos = obs["gripper_pos"].copy()
            self.targets_initialized = True

        # Return no action if teleop is not enabled
        if self.primary_device_id is None:
            return None

        # Get most recent teleop command
        arm_quat = self.arm_target_rot.as_quat()  # type: ignore
        if arm_quat[3] < 0.0:  # Enforce quaternion uniqueness
            np.negative(arm_quat, out=arm_quat)
        action = {
            "base_pose": self.base_target_pose.copy(),
            "arm_pos": self.arm_target_pos.copy(),
            "arm_quat": arm_quat,
            "gripper_pos": self.gripper_target_pos.copy(),
        }

        return action


# ============================================================================
# Teleop Policy
# ============================================================================


class Policy:
    """Base policy interface."""

    def reset(self) -> None:
        """Reset the policy."""
        raise NotImplementedError

    def step(self, obs: dict) -> dict | str | None:
        """Get action from observation."""
        raise NotImplementedError


class TeleopPolicy(Policy):
    """Teleop using WebXR phone web app."""

    def __init__(self, enable_web_server: bool = True, port: int = 5000):
        self.web_server_queue: Queue = Queue()
        self.teleop_controller: TeleopController | None = None
        self.teleop_state: str | None = (
            None  # episode_started -> episode_ended -> reset_env
        )
        self.episode_ended = False
        self.enable_web_server = enable_web_server

        if self.enable_web_server:
            # Web server for serving the WebXR phone web app
            server = WebServer(self.web_server_queue, port=port)
            threading.Thread(target=server.run, daemon=True).start()
            print("Web server started for teleop interface")
        else:
            print("Web server disabled (simulation mode)")

        # Listener thread to process messages from WebXR client
        threading.Thread(target=self._listener_loop, daemon=True).start()

    def reset(self) -> None:
        """Reset the policy for a new episode."""
        self.teleop_controller = TeleopController()
        self.episode_ended = False

        if self.enable_web_server:
            # Wait for user to signal that episode has started
            self.teleop_state = None
            while self.teleop_state != "episode_started":
                time.sleep(0.01)
        else:
            # In sim mode without web server, start episode immediately
            self.teleop_state = "episode_started"

    def step(self, obs: dict) -> dict | str | None:
        """Get action from teleop input.

        Args:
            obs: Observation dictionary with robot state.

        Returns:
            Action dictionary, control signal string, or None.
        """
        # Signal that user has ended episode
        if not self.episode_ended and self.teleop_state == "episode_ended":
            self.episode_ended = True
            return "end_episode"

        # Signal that user is ready for env reset (after ending the episode)
        if self.teleop_state == "reset_env":
            return "reset_env"

        return self._step(obs)

    def _step(self, obs: dict) -> dict | None:
        """Internal step function."""
        if self.teleop_controller is None:
            return None
        return self.teleop_controller.step(obs)

    def _listener_loop(self) -> None:
        """Background thread to process WebXR messages."""
        while True:
            if not self.web_server_queue.empty():
                data = self.web_server_queue.get()

                # Update state
                if "state_update" in data:
                    self.teleop_state = data["state_update"]

                # Process message if not stale
                elif 1000 * time.time() - data["timestamp"] < 250:  # 250 ms
                    self._process_message(data)

            time.sleep(0.001)

    def _process_message(self, data: dict) -> None:
        """Process a WebXR message."""
        if self.teleop_controller is not None:
            self.teleop_controller.process_message(data)

    def close(self) -> None:
        """Clean up resources."""
        # No explicit cleanup needed for teleop policy
