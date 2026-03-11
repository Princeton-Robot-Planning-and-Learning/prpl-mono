"""Teleoperation interface for kinder environments using WebXR phone app."""

import logging
import math
import socket
import threading
import time
from queue import Queue

import cv2 as cv  # type: ignore[import-untyped]
import numpy as np
from flask import Flask, render_template  # type: ignore[import-untyped]
from flask_socketio import SocketIO, emit  # type: ignore[import-untyped]
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


# ============================================================================
# Quest Teleop Controller
# ============================================================================


class QuestImageViewer:
    """Utility class to display images in OpenCV windows.

    Image can be updated dynamically
    """

    def __init__(self, window_name: str) -> None:
        self.window_name = window_name
        cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)  # pylint: disable=no-member

    def show_image(self, image: np.ndarray) -> None:
        """Display image in OpenCV window."""
        cv.imshow(self.window_name, image)  # pylint: disable=no-member
        cv.waitKey(1)  # pylint: disable=no-member


class QuestController:
    """Controller that processes Quest VR input to generate robot commands."""

    def __init__(self, debug: bool = False) -> None:
        try:
            from oculus_reader import (  # type: ignore  # pylint: disable=import-outside-toplevel
                OculusReader,
            )
        except ImportError:
            raise ImportError(
                "oculus_reader not found. Install it to use Quest controllers."
            )

        self.device = OculusReader()
        self.verbose = debug

        # Controller state tracking
        self.left_trigger_pressed = False
        self.right_trigger_pressed = False
        self.left_grip_pressed = False
        self.right_grip_pressed = False
        self.initialize_left_pose = True
        self.initialize_right_pose = True

        # Mobile base pose
        self.base_pose = None

        # Teleop targets
        self.targets_initialized = False
        self.base_target_pose = None
        self.arm_target_pos = None
        self.arm_target_rot = None
        self.gripper_target_pos = None

        # Quest controller reference poses
        self.left_controller_init_pos = np.zeros(3)
        self.left_controller_init_rot = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz
        self.right_controller_init_pos = np.zeros(3)
        self.right_controller_init_rot = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz

        # Robot reference poses
        self.base_ref_pose: np.ndarray | None = None
        self.arm_ref_pos: np.ndarray | None = None
        self.arm_ref_rot: R | None = None
        self.arm_ref_base_pose: np.ndarray | None = None
        self.gripper_ref_pos: np.ndarray | None = None

        # Controller offset from robot base frame
        # Quest coordinate system: adjust as needed for your setup
        self.controller_offset = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        self.set_constant_controller_offset()

        if self.verbose:
            self._display_controls()

    def _display_controls(self) -> None:
        """Display Quest controller mappings."""
        print("\n=== Quest Controller Mappings ===")
        print("Left Controller:")
        print("  - Trigger: Enable base XY control")
        print("  - Position: Controls base XY position")
        print("  - Orientation: Controls base rotation")
        print("\nRight Controller:")
        print("  - Trigger: Enable end-effector control")
        print("  - Position: Controls end-effector position")
        print("  - Orientation: Controls end-effector orientation")
        print("  - Grip: Close gripper")
        print("\nButtons:")
        print("  - A: Mark episode ended")
        print("  - B: Request environment reset")
        print("  - Press Right Joystick: Recalibrate controller to robot")
        print("==================================\n")

    def set_constant_controller_offset(self, inverted: bool = False) -> None:
        """Set the controller rotation offset from the robot base frame.

        Args:
            inverted: If True, use inverted (upside down) configuration.
        """
        # Headset to robot offset when headset faces same direction as the robot.
        self.controller_offset = np.array(
            [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]  # robot_T_headset
        )
        if inverted:
            # Headset to robot offset when headset is upside down.
            self.controller_offset = np.array(
                [[0, 0, -1], [1, 0, 0], [0, -1, 0]]  # robot_T_invertedHeadset
            )

        if self.verbose:
            print(f"Controller offset set to:\n{self.controller_offset}")

    def calibrate_controller(self) -> bool:
        """Calibrate the controller axes to the robot frame.

        Returns:
            bool: True if calibration was successful, False otherwise.
        """
        print("\n=== Controller Calibration ===")
        print(
            "Press and hold the A button while moving the RIGHT controller "
            "along the robot's X-axis"
        )
        print("(robot's forward direction)...")

        # Wait until the button is pressed for the first time
        controller_data = self.device.get_transformations_and_buttons()
        while not controller_data or not controller_data[1].get("A", False):
            time.sleep(0.01)
            controller_data = self.device.get_transformations_and_buttons()

        ori_tf = controller_data[0].get("r")
        if ori_tf is None:
            print("ERROR: Right controller not detected")
            return False

        print("Button pressed! Hold and move along X-axis...")

        # Wait until the button is released
        buttons_data = controller_data[1]
        while buttons_data.get("A", False):
            controller_data = self.device.get_transformations_and_buttons()
            if controller_data:
                buttons_data = controller_data[1]
            time.sleep(0.01)

        end_tf = controller_data[0].get("r")
        if end_tf is None:
            print("ERROR: Right controller not detected")
            return False

        print("Button released! Calibrating X-axis...")

        # Calculate X-axis
        delta = end_tf[:3, 3] - ori_tf[:3, 3]
        kx = np.argmax(np.abs(delta))
        x_axis = np.zeros(3)
        x_axis[kx] = np.sign(delta[kx])

        print(f"X-axis mapped to controller axis {kx} with sign {np.sign(delta[kx])}")
        print(
            "\nNow press and hold the A button while moving the RIGHT controller "
            "along the robot's Y-axis"
        )
        print("(robot's left direction)...")

        # Wait until the button is pressed for the first time
        controller_data = self.device.get_transformations_and_buttons()
        while not controller_data or not controller_data[1].get("A", False):
            time.sleep(0.01)
            controller_data = self.device.get_transformations_and_buttons()

        ori_tf = controller_data[0].get("r")
        if ori_tf is None:
            print("ERROR: Right controller not detected")
            return False

        print("Button pressed! Hold and move along Y-axis...")

        # Wait until the button is released
        buttons_data = controller_data[1]
        while buttons_data.get("A", False):
            controller_data = self.device.get_transformations_and_buttons()
            if controller_data:
                buttons_data = controller_data[1]
            time.sleep(0.01)

        end_tf = controller_data[0].get("r")
        if end_tf is None:
            print("ERROR: Right controller not detected")
            return False

        print("Button released! Calibrating Y-axis...")

        # Calculate Y-axis
        delta = end_tf[:3, 3] - ori_tf[:3, 3]
        ky = np.argmax(np.abs(delta))
        y_axis = np.zeros(3)
        y_axis[ky] = np.sign(delta[ky])

        if kx == ky:
            print("ERROR: Calibration failed - same axis provided twice")
            print("Please try again with more distinct movements.")
            return False

        print(f"Y-axis mapped to controller axis {ky} with sign {np.sign(delta[ky])}")

        # Calculate Z-axis via cross product
        z_axis = np.cross(x_axis, y_axis)
        self.controller_offset = np.array([x_axis, y_axis, z_axis])

        print("\n=== Calibration Successful! ===")
        print(f"Controller to Robot transform:\n{self.controller_offset}")
        print("================================\n")

        # Reset initialization flags to use new calibration
        self.initialize_left_pose = True
        self.initialize_right_pose = True

        return True

    def get_controller_data(self) -> tuple[dict, dict] | None:
        """Get data from Quest controllers."""
        controller_data = self.device.get_transformations_and_buttons()
        if not controller_data or controller_data[0] == {}:
            return None
        return controller_data

    def process_controllers(self, obs: dict) -> dict | str | None:  # pylint: disable=unused-argument
        """Process Quest controller input and generate action."""
        if not self.targets_initialized:
            return None

        # Get controller data
        controller_data = self.get_controller_data()
        if controller_data is None:
            return None

        transforms_data, buttons_data = controller_data

        # Check if user wants to recalibrate controller to robot transform
        if buttons_data.get("RJ", False):  # Right joystick pressed
            print("\nCalibration requested...")
            # Keep trying until calibration succeeds
            while not self.calibrate_controller():
                print("Retrying calibration...")
                time.sleep(0.5)
            return None  # Return None to continue normal operation

        # Check for episode control buttons
        if buttons_data.get("A", False):
            return "end_episode"
        if buttons_data.get("B", False):
            return "reset_env"

        # Update trigger states
        left_trigger_now = buttons_data.get("LTr", False)
        right_trigger_now = buttons_data.get("RTr", False)

        # Left controller: base control
        if "l" in transforms_data and left_trigger_now:
            if not self.left_trigger_pressed:
                # Trigger just pressed, initialize reference pose
                self.initialize_left_pose = True
                self.left_trigger_pressed = True

            left_transform = transforms_data["l"]
            self._process_left_controller(left_transform)
        else:
            self.left_trigger_pressed = False
            if not left_trigger_now:
                self.initialize_left_pose = True

        # Right controller: end-effector control
        if "r" in transforms_data and right_trigger_now:
            if not self.right_trigger_pressed:
                # Trigger just pressed, initialize reference pose
                self.initialize_right_pose = True
                self.right_trigger_pressed = True

            right_transform = transforms_data["r"]
            self.right_grip_pressed = buttons_data.get("RG", False)
            if self.verbose:
                print(f"Right grip pressed: {self.right_grip_pressed}")
            self._process_right_controller(right_transform)
        else:
            self.right_trigger_pressed = False
            if not right_trigger_now:
                self.initialize_right_pose = True

        return None

    def _process_left_controller(self, transform: np.ndarray) -> None:
        """Process left controller for base control."""
        controller_curr_pos = transform[:3, 3]
        controller_curr_ori = transform[:3, :3]

        if self.initialize_left_pose:
            # Initialize reference pose
            self.left_controller_init_pos = controller_curr_pos.copy()
            self.left_controller_init_rot = self.controller_offset @ controller_curr_ori
            # Type ignore: base_pose is set in step() before this is called
            self.base_ref_pose = self.base_pose.copy()  # type: ignore[attr-defined]
            self.initialize_left_pose = False
            if self.verbose:
                print("Initialized left controller (base) reference pose")

        # Compute delta position in robot frame
        dpos = self.controller_offset @ (
            controller_curr_pos - self.left_controller_init_pos
        )

        # Update base XY position
        # Type ignore: checked as non-None in __init__ or step()
        self.base_target_pose[:2] = (  # type: ignore[index]
            self.base_ref_pose[:2] + dpos[:2]  # type: ignore[index]
        )

        # Update base orientation from controller rotation
        controller_curr_rot = self.controller_offset @ controller_curr_ori
        rot_delta = controller_curr_rot @ np.linalg.inv(self.left_controller_init_rot)

        # Extract yaw angle from rotation matrix
        base_fwd_vec = rot_delta @ np.array([1.0, 0.0, 0.0])
        # Type ignore: base_ref_pose is set in initialize block or step()
        base_target_theta = (
            self.base_ref_pose[2]  # type: ignore[index]
            + math.atan2(base_fwd_vec[1], base_fwd_vec[0])
        )

        # Unwrap angle
        # Type ignore: base_target_pose is set in step()
        angle_diff = (
            base_target_theta - self.base_target_pose[2] + math.pi  # type: ignore[index]
        )
        self.base_target_pose[2] += (  # type: ignore[index]
            angle_diff % TWO_PI - math.pi
        )

    def _process_right_controller(self, transform: np.ndarray) -> None:
        """Process right controller for end-effector control."""
        controller_curr_pos = transform[:3, 3]
        controller_curr_ori = transform[:3, :3]

        if self.initialize_right_pose:
            # Initialize reference pose
            self.right_controller_init_pos = controller_curr_pos.copy()
            self.right_controller_init_rot = (
                self.controller_offset @ controller_curr_ori
            )
            # Type ignore: these are set in step() before this is called
            self.arm_ref_pos = (
                self.arm_target_pos.copy()  # type: ignore[attr-defined]
            )
            self.arm_ref_rot = self.arm_target_rot
            self.arm_ref_base_pose = (
                self.base_pose.copy()  # type: ignore[attr-defined]
            )
            self.gripper_ref_pos = (
                self.gripper_target_pos.copy()  # type: ignore[attr-defined]
            )
            self.initialize_right_pose = False
            if self.verbose:
                print("Initialized right controller (arm) reference pose")

        # Rotations around z-axis: global frame (base) <-> local frame (arm)
        # Type ignore: base_pose is set in step()
        z_rot = R.from_rotvec(
            np.array([0.0, 0.0, 1.0]) * self.base_pose[2]  # type: ignore[index]
        )
        z_rot_inv = z_rot.inv()
        # Type ignore: arm_ref_base_pose is set in initialize block
        ref_z_rot = R.from_rotvec(
            np.array([0.0, 0.0, 1.0]) * self.arm_ref_base_pose[2]  # type: ignore[index]
        )

        # Compute position delta in robot frame
        dpos_controller = self.controller_offset @ (
            controller_curr_pos - self.right_controller_init_pos
        )

        # Compensate for base rotation and translation
        # Type ignore: arm_ref_pos is set in initialize block
        pos_diff = dpos_controller
        pos_diff += ref_z_rot.apply(
            self.arm_ref_pos  # type: ignore[arg-type]
        ) - z_rot.apply(self.arm_ref_pos)  # type: ignore[arg-type]
        # Type ignore: arm_ref_base_pose, base_pose set earlier
        base_delta = (
            self.arm_ref_base_pose[:2]  # type: ignore[index]
            - self.base_pose[:2]  # type: ignore[index,operator]
        )
        pos_diff[:2] += base_delta  # type: ignore

        # Update arm target position
        # Type ignore: arm_ref_pos set in initialize block
        arm_delta = z_rot_inv.apply(pos_diff)
        self.arm_target_pos = (
            self.arm_ref_pos + arm_delta  # type: ignore[operator,assignment]
        )

        # Compute orientation delta
        controller_curr_rot = self.controller_offset @ controller_curr_ori
        rot_delta = controller_curr_rot @ np.linalg.inv(self.right_controller_init_rot)

        # Convert to scipy Rotation and apply
        rot_delta_R = R.from_matrix(rot_delta)
        # Type ignore: arm_ref_rot is set in initialize block
        rot_combined = z_rot_inv * rot_delta_R * ref_z_rot
        self.arm_target_rot = (
            rot_combined * self.arm_ref_rot  # type: ignore[operator,assignment]
        )

        # Type ignore: gripper_ref_pos is set in initialize block
        if self.right_grip_pressed:
            # Close gripper when grip button is pressed
            self.gripper_target_pos = np.array([1.0])  # type: ignore[assignment]
        else:
            # Otherwise, keep gripper at reference position
            self.gripper_target_pos = self.gripper_ref_pos  # type: ignore[assignment]

    def step(self, obs: dict) -> dict | str | None:  # type: ignore[no-untyped-def]
        """Generate action from current Quest controller state.

        Args:
            obs: Observation dictionary with robot state.

        Returns:
            Action dictionary, control signal string, or None.
        """
        # Update robot state
        self.base_pose = obs["base_pose"]

        # Initialize targets
        if not self.targets_initialized:
            # Type ignore: obs values are expected to be numpy arrays
            self.base_target_pose = (
                obs["base_pose"].copy()  # type: ignore
            )
            self.arm_target_pos = (
                obs["arm_pos"].copy()  # type: ignore
            )
            self.arm_target_rot = R.from_quat(obs["arm_quat"])  # type: ignore
            self.gripper_target_pos = (
                obs["gripper_pos"].copy()  # type: ignore
            )
            self.targets_initialized = True

        # Process controllers and check for control signals
        control_signal = self.process_controllers(obs)
        if isinstance(control_signal, str):
            return control_signal

        # Return no action if neither controller is engaged
        if not self.left_trigger_pressed and not self.right_trigger_pressed:
            return None

        # Generate action
        arm_quat = self.arm_target_rot.as_quat()  # type: ignore
        if arm_quat[3] < 0.0:  # Enforce quaternion uniqueness
            np.negative(arm_quat, out=arm_quat)

        action = {
            "base_pose": self.base_target_pose.copy(),  # type: ignore[attr-defined]
            "arm_pos": self.arm_target_pos.copy(),  # type: ignore[attr-defined]
            "arm_quat": arm_quat,
            "gripper_pos": self.gripper_target_pos.copy(),  # type: ignore[attr-defined]
        }

        return action


class QuestTeleopPolicy(Policy):
    """Teleop policy using Meta Quest VR controllers."""

    def __init__(self, debug: bool = False):
        self.quest_controller: QuestController | None = None
        self.debug = debug
        print("Quest VR controller policy initialized")

    def reset(self) -> None:
        """Reset the policy for a new episode."""
        self.quest_controller = QuestController(debug=self.debug)
        print("Quest controller reset for new episode")

    def step(self, obs: dict) -> dict | str | None:
        """Get action from Quest controller input.

        Args:
            obs: Observation dictionary with robot state.

        Returns:
            Action dictionary, control signal string, or None.
        """
        if self.quest_controller is None:
            return None
        return self.quest_controller.step(obs)

    def close(self) -> None:
        """Clean up resources."""
        self.quest_controller = None
