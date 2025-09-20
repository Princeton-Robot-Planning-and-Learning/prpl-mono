"""Utilities for Dynamic2D PyMunk-based environments."""

import math
from typing import Any

import numpy as np
import pymunk
from numpy.typing import NDArray
from pymunk.vec2d import Vec2d
from relational_structs import Object

from prbench.envs.dynamic2d.object_types import (
    KinRectangleType,
)
from prbench.envs.geom2d.structs import (
    SE2Pose,
    ZOrder,
)
from prbench.envs.utils import BLACK, RobotActionSpace

# Collision types from the basic_pymunk.py script
STATIC_COLLISION_TYPE = 0
DYNAMIC_COLLISION_TYPE = 1
ROBOT_COLLISION_TYPE = 2


class KinRobotActionSpace(RobotActionSpace):
    """An action space for a fingered robot with gripper control.

    Actions are bounded relative movements of the base, arm extension, and gripper
    opening/closing.
    """

    def __init__(
        self,
        min_dx: float = -5e-1,
        max_dx: float = 5e-1,
        min_dy: float = -5e-1,
        max_dy: float = 5e-1,
        min_dtheta: float = -np.pi / 16,
        max_dtheta: float = np.pi / 16,
        min_darm: float = -1e-1,
        max_darm: float = 1e-1,
        min_dgripper: float = -0.02,
        max_dgripper: float = 0.02,
    ) -> None:
        low = np.array([min_dx, min_dy, min_dtheta, min_darm, min_dgripper])
        high = np.array([max_dx, max_dy, max_dtheta, max_darm, max_dgripper])
        super().__init__(low, high)

    def create_markdown_description(self) -> str:
        """Create a human-readable markdown description of this space."""
        features = [
            ("dx", "Change in robot x position (positive is right)"),
            ("dy", "Change in robot y position (positive is up)"),
            ("dtheta", "Change in robot angle in radians (positive is ccw)"),
            ("darm", "Change in robot arm length (positive is out)"),
            ("dgripper", "Change in gripper gap (positive is open)"),
        ]
        md_table_str = (
            "| **Index** | **Feature** | **Description** | **Min** | **Max** |"
        )
        md_table_str += "\n| --- | --- | --- | --- | --- |"
        for idx, (feature, description) in enumerate(features):
            lb = self.low[idx]
            ub = self.high[idx]
            md_table_str += (
                f"\n| {idx} | {feature} | {description} | {lb:.3f} | {ub:.3f} |"
            )
        return (
            f"The entries of an array in this Box space correspond to the "
            f"following action features:\n{md_table_str}\n"
        )


class KinRobot:
    """Robot implementation using PyMunk physics engine with four bodies.

    The robot has a circular base, a rectangular gripper base (attached one arm and one
    right finger), and a left fingers. The gripper base is attached to the robot base
    via a kinematic arm that can extend and retract. The fingers can open and close to
    grasp objects.

    The robot can held objects by closing the fingers around them.

    The robot will be revert to the last valid state when colliding with static objects.

    The robot is controlled via setting the velocities of the bodies, which can be
    computed using a PD controller.
    """

    def __init__(
        self,
        init_pos: Vec2d = Vec2d(5.0, 5.0),
        base_radius: float = 0.4,
        arm_length_max: float = 0.8,
        gripper_base_width: float = 0.01,
        gripper_base_height: float = 0.1,
        gripper_finger_width: float = 0.1,
        gripper_finger_height: float = 0.01,
        finger_move_thresh: float = 0.001,
        grasping_theta_thresh: float = 0.1,
    ) -> None:
        # Robot parameters
        self.base_radius = base_radius
        self.gripper_base_width = gripper_base_width
        self.gripper_base_height = gripper_base_height
        self.gripper_finger_width = gripper_finger_width
        self.gripper_finger_height = gripper_finger_height
        self.arm_length_max = arm_length_max
        self.gripper_gap_max = gripper_base_height
        self.finger_move_thresh = finger_move_thresh
        self.grasping_theta_thresh = grasping_theta_thresh

        # Track last robot state
        self._base_position = init_pos
        self._base_angle = 0.0
        self._arm_length = base_radius
        self._gripper_gap = gripper_base_height
        self.held_objects: list[
            tuple[tuple[pymunk.Body, pymunk.Shape], float, SE2Pose]
        ] = []

        # Updated by env.step()
        self.is_opening_finger = False
        self.is_closing_finger = False

        # Body and shape references
        self.create_base()
        self.create_gripper_base()
        self._left_finger_body, self._left_finger_shape = self.create_finger()

    def add_to_space(self, space: pymunk.Space) -> None:
        """Add robot components to the PyMunk space."""
        space.add(self._base_body, self._base_shape)
        space.add(
            self._gripper_base_body,
            self._gripper_base_shape,
            self._arm_shape,
            self._right_finger_shape,
        )
        space.add(self._left_finger_body, self._left_finger_shape)

    def create_base(self) -> None:
        """Create the robot base."""
        self._base_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self._base_shape = pymunk.Circle(self._base_body, self.base_radius)
        self._base_shape.color = (255, 50, 50, 255)
        self._base_shape.friction = 1
        self._base_shape.collision_type = ROBOT_COLLISION_TYPE
        self._base_shape.density = 1.0
        self._base_body.position = self._base_position
        self._base_body.angle = self._base_angle

    @property
    def base_pose(self) -> SE2Pose:
        """Get the base pose as SE2Pose."""
        return SE2Pose(
            x=self._base_body.position.x,
            y=self._base_body.position.y,
            theta=self._base_body.angle,
        )

    @property
    def base_vel(self) -> tuple[Vec2d, float]:
        """Get the base linear and angular velocity."""
        return self._base_body.velocity, self._base_body.angular_velocity

    def create_gripper_base(self) -> None:
        """Create the gripper base."""
        half_w = self.gripper_base_width / 2
        half_h = self.gripper_base_height / 2
        vs = [
            (-half_w, half_h),
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
        ]
        self._gripper_base_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self._gripper_base_shape = pymunk.Poly(self._gripper_base_body, vs)
        self._gripper_base_shape.friction = 1
        self._gripper_base_shape.collision_type = ROBOT_COLLISION_TYPE
        self._gripper_base_shape.density = 1.0

        vs_arm = [
            (-self.arm_length_max / 2, half_w),
            (-self.arm_length_max / 2, -half_w),
            (self.arm_length_max / 2, -half_w),
            (self.arm_length_max / 2, half_w),
        ]
        ts_arm = pymunk.Transform(tx=-self.arm_length_max / 2 - half_w, ty=0)
        self._arm_shape = pymunk.Poly(self._gripper_base_body, vs_arm, transform=ts_arm)
        self._arm_shape.friction = 1
        self._arm_shape.collision_type = ROBOT_COLLISION_TYPE
        self._arm_shape.density = 1.0

        vs_right_finger = [
            (-self.gripper_finger_width / 2, -self.gripper_finger_height / 2),
            (-self.gripper_finger_width / 2, self.gripper_finger_height / 2),
            (self.gripper_finger_width / 2, -self.gripper_finger_height / 2),
            (self.gripper_finger_width / 2, self.gripper_finger_height / 2),
        ]
        ts_finger = pymunk.Transform(
            tx=self.gripper_finger_width / 2, ty=-self.gripper_base_height / 2
        )
        self._right_finger_shape = pymunk.Poly(
            self._gripper_base_body, vs_right_finger, transform=ts_finger
        )
        self._right_finger_shape.friction = 1
        self._right_finger_shape.collision_type = ROBOT_COLLISION_TYPE
        self._right_finger_shape.density = 1.0

        init_rel_pos = SE2Pose(x=self._arm_length, y=0.0, theta=0.0)
        init_pose = self.base_pose * init_rel_pos
        self._gripper_base_body.position = (init_pose.x, init_pose.y)
        self._gripper_base_body.angle = init_pose.theta

    @property
    def gripper_base_pose(self) -> SE2Pose:
        """Get the gripper base pose as SE2Pose."""
        return SE2Pose(
            x=self._gripper_base_body.position.x,
            y=self._gripper_base_body.position.y,
            theta=self._gripper_base_body.angle,
        )

    @property
    def gripper_base_vel(self) -> tuple[Vec2d, float]:
        """Get the gripper base linear and angular velocity."""
        return (
            self._gripper_base_body.velocity,
            self._gripper_base_body.angular_velocity,
        )

    def create_finger(self) -> tuple[pymunk.Body, pymunk.Shape]:
        """Create a gripper finger."""
        half_w = self.gripper_finger_width / 2
        half_h = self.gripper_finger_height / 2
        vs = [
            (-half_w, half_h),
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
        ]
        finger_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        finger_shape = pymunk.Poly(finger_body, vs)
        finger_shape.friction = 1
        finger_shape.density = 1.0
        finger_shape.collision_type = ROBOT_COLLISION_TYPE

        init_rel_pos = SE2Pose(
            x=half_w, y=self._gripper_gap - self.gripper_base_height / 2, theta=0.0
        )
        init_pose = self.gripper_base_pose * init_rel_pos
        finger_body.position = (init_pose.x, init_pose.y)
        finger_body.angle = init_pose.theta
        return finger_body, finger_shape

    @property
    def finger_poses(self) -> SE2Pose:
        """Get the left finger pose as SE2Pose."""
        return SE2Pose(
            x=self._left_finger_body.position.x,
            y=self._left_finger_body.position.y,
            theta=self._left_finger_body.angle,
        )

    @property
    def finger_vels(self) -> tuple[Vec2d, float]:
        """Get the left finger linear and angular velocity."""
        return (
            self._left_finger_body.velocity,
            self._left_finger_body.angular_velocity,
        )

    @property
    def held_object_vels(self) -> list[tuple[Vec2d, float]]:
        """Get the held object linear and angular velocity."""
        vel_list = []
        for obj, _, _ in self.held_objects:
            obj_body, _ = obj
            vel_list.append((obj_body.velocity, obj_body.angular_velocity))
        return vel_list

    @property
    def curr_gripper(self) -> float:
        """Get the current gripper opening."""
        relative_finger_pose = self.gripper_base_pose.inverse * self.finger_poses
        return relative_finger_pose.y + self.gripper_base_height / 2

    @property
    def curr_arm_length(self) -> float:
        """Get the current arm length."""
        relative_pose = self.base_pose.inverse * self.gripper_base_pose
        return relative_pose.x

    @property
    def body_id(self) -> int:
        """Get the base id in pymunk space."""
        return self._base_body.id

    def reset_positions(
        self,
        base_x: float,
        base_y: float,
        base_theta: float,
        arm_length: float,
        gripper_gap: float,
    ) -> None:
        """Reset robot to specified positions with zero velocity."""
        self._base_body.position = (base_x, base_y)
        self._base_body.velocity = (0.0, 0.0)
        self._base_body.angle = base_theta
        self._base_body.angular_velocity = 0.0

        base_to_gripper = SE2Pose(x=arm_length, y=0.0, theta=0.0)
        gripper_pose = self.base_pose * base_to_gripper
        self._gripper_base_body.position = (gripper_pose.x, gripper_pose.y)
        self._gripper_base_body.velocity = (0.0, 0.0)
        self._gripper_base_body.angle = gripper_pose.theta
        self._gripper_base_body.angular_velocity = 0.0

        gripper_to_left_finger = SE2Pose(
            x=self.gripper_finger_width / 2,
            y=gripper_gap - self.gripper_base_height / 2,
            theta=0.0,
        )
        left_finger_pose = gripper_pose * gripper_to_left_finger
        self._left_finger_body.position = (left_finger_pose.x, left_finger_pose.y)
        self._left_finger_body.velocity = (0.0, 0.0)
        self._left_finger_body.angle = left_finger_pose.theta
        self._left_finger_body.angular_velocity = 0.0

        # Update last state
        self.update_last_state()

    def revert_to_last_state(self) -> None:
        """Reset to last state when collide with static objects."""
        self._base_body.position = self._base_position
        self._base_body.angle = self._base_angle

        gripper_base_rel_pos = SE2Pose(x=self._arm_length, y=0.0, theta=0.0)
        gripper_base_pose = self.base_pose * gripper_base_rel_pos
        self._gripper_base_body.position = (gripper_base_pose.x, gripper_base_pose.y)
        self._gripper_base_body.angle = gripper_base_pose.theta

        left_finger_rel_pos = SE2Pose(
            x=self.gripper_finger_width / 2,
            y=self._gripper_gap - self.gripper_base_height / 2,
            theta=0.0,
        )
        left_finger_pose = gripper_base_pose * left_finger_rel_pos
        self._left_finger_body.position = (left_finger_pose.x, left_finger_pose.y)
        self._left_finger_body.angle = left_finger_pose.theta

        # Update held objects
        for obj, _, relative_pose in self.held_objects:
            obj_body, _ = obj
            new_obj_pose = gripper_base_pose * relative_pose
            obj_body.position = (new_obj_pose.x, new_obj_pose.y)
            obj_body.angle = new_obj_pose.theta

    def update_last_state(self) -> None:
        """Update the last state tracking variables."""
        self._base_position = Vec2d(
            self._base_body.position.x, self._base_body.position.y
        )
        self._base_angle = self._base_body.angle

        relative_pose = self.base_pose.inverse * self.gripper_base_pose
        self._arm_length = relative_pose.x
        relative_finger_pose = self.gripper_base_pose.inverse * self.finger_poses
        self._gripper_gap = relative_finger_pose.y + self.gripper_base_height / 2

    def update(
        self,
        base_vel: Vec2d,
        base_ang_vel: float,
        gripper_base_vel: Vec2d,
        finger_vel_l: Vec2d,
        helder_object_vels: list[Vec2d],
    ) -> None:
        """Update the body velocities."""
        # Update robot last state
        self.update_last_state()
        # Update velocities
        self._base_body.velocity = base_vel
        self._base_body.angular_velocity = base_ang_vel
        # Calculate target gripper base
        # It only has relative translational velocity
        self._gripper_base_body.velocity = gripper_base_vel
        self._gripper_base_body.angular_velocity = base_ang_vel
        # Left Finger
        self._left_finger_body.velocity = finger_vel_l
        self._left_finger_body.angular_velocity = base_ang_vel

        # Update held objects - they have the same velocity as gripper base
        for i, (obj, _, _) in enumerate(self.held_objects):
            obj_body, _ = obj
            obj_body.velocity = helder_object_vels[i]
            obj_body.angular_velocity = self._gripper_base_body.angular_velocity

    def is_grasping(
        self, contact_point_set: pymunk.ContactPointSet, tgt_body: pymunk.Body
    ) -> bool:
        """Check if robot is grasping a target body."""
        # Checker 0: If robot is closing gripper
        if not self.is_closing_finger:
            return False
        # Checker 1: If contact normal is roughly perpendicular
        # to gripper_base_body base
        normal = contact_point_set.normal
        if not self._gripper_base_body:
            return False
        dtheta = abs(self._gripper_base_body.angle - normal.angle)
        dtheta = min(dtheta, 2 * np.pi - dtheta)
        theta_ok = abs(dtheta - np.pi / 2) < self.grasping_theta_thresh
        if not theta_ok:
            return False
        # Checker 2: If exist contact points in hand and target body is within
        # the gripper height
        rel_body = self.gripper_base_pose.inverse * SE2Pose(
            x=tgt_body.position.x, y=tgt_body.position.y, theta=0.0
        )
        if abs(rel_body.y) > self.gripper_base_height / 2:
            return False
        for pt in contact_point_set.points:
            pt_a = pt.point_a
            p_a = SE2Pose(x=pt_a.x, y=pt_a.y, theta=0.0)
            rel_a = self.gripper_base_pose.inverse * p_a
            if (abs(rel_a.y) < self.gripper_base_height / 2) and (
                (rel_a.x < self.gripper_finger_width / 2) and rel_a.x > 0
            ):
                return True
        return False

    def add_to_hand(self, obj: tuple[pymunk.Body, pymunk.Shape], mass: float) -> None:
        """Add an object to the robot's hand."""
        obj_body, _ = obj
        obj_pose = SE2Pose(
            x=obj_body.position.x, y=obj_body.position.y, theta=obj_body.angle
        )
        gripper_base_pose = SE2Pose(
            x=self._gripper_base_body.position.x,
            y=self._gripper_base_body.position.y,
            theta=self._gripper_base_body.angle,
        )
        relative_obj_pose = gripper_base_pose.inverse * obj_pose
        self.held_objects.append((obj, mass, relative_obj_pose))


class PDController:
    """A simple PD controller for the robot."""

    def __init__(
        self,
        kp_pos: float = 100.0,
        kv_pos: float = 20.0,
        kp_rot: float = 500.0,
        kv_rot: float = 50.0,
    ) -> None:
        self.kp_pos = kp_pos
        self.kv_pos = kv_pos
        self.kp_rot = kp_rot
        self.kv_rot = kv_rot

    def compute_control(
        self,
        robot: KinRobot,
        tgt_x: float,
        tgt_y: float,
        tgt_theta: float,
        tgt_arm: float,  # target arm length L*
        tgt_gripper: float,  # target finger opening g*
        dt: float,
    ) -> tuple[Vec2d, float, Vec2d, Vec2d, list[Vec2d]]:
        """Compute base vel, base ang vel, gripper-base vel (world), finger vel (world),
        and held object vels (world) using PD control."""
        # === 0) Read current state ===
        base_pos_curr = Vec2d(robot.base_pose.x, robot.base_pose.y)
        base_vel_curr = robot.base_vel[0]  # Vec2d(vx, vy) in world
        base_ang_curr = robot.base_pose.theta
        base_ang_vel_curr = robot.base_vel[1]  # scalar omega
        base_rot_omega_vec = Vec2d(
            math.cos(base_ang_curr + math.pi / 2), math.sin(base_ang_curr + math.pi / 2)
        )

        L_curr = robot.curr_arm_length  # current arm length (scalar)
        Ldot_curr = robot.gripper_base_vel[0]

        # If available (recommended), provide current gripper opening and its rate:
        g_curr = robot.curr_gripper  # opening distance
        finger_vel_abs_w = robot.finger_vels[0]  # Vec2d

        # === 1) Base PD ===
        base_pos_tgt = Vec2d(tgt_x, tgt_y)
        a_base_lin = self.kp_pos * (base_pos_tgt - base_pos_curr) + self.kv_pos * (
            Vec2d(0, 0) - base_vel_curr
        )
        base_vel = base_vel_curr + a_base_lin * dt

        a_base_ang = self.kp_rot * (tgt_theta - base_ang_curr) + self.kv_rot * (
            0.0 - base_ang_vel_curr
        )
        base_ang_vel = base_ang_vel_curr + a_base_ang * dt

        # === 2) Arm prismatic rate via PD on length in the base frame ===
        # PD on arm length
        kp_arm = getattr(self, "kp_arm", self.kp_pos)
        kv_arm = getattr(self, "kv_arm", self.kv_pos)
        arm_center_omega_vec = (
            base_rot_omega_vec * L_curr * base_ang_vel_curr
        )  # omega x r
        # Extract prismatic vel from a moving base
        rel_Ldot_curr = (
            (Ldot_curr - base_vel_curr - arm_center_omega_vec).rotated(-base_ang_curr).x
        )  # R^T * v_gripper_base
        a_L = kp_arm * (tgt_arm - L_curr) + kv_arm * (0.0 - rel_Ldot_curr)

        # Integrate prismatic rate
        rel_Ldot_next = rel_Ldot_curr + a_L * dt
        # Note: We need to use the *next base_ang_vel* to compute the
        # world-frame gripper-base velocity
        v_gripper_base = (
            base_vel
            + base_rot_omega_vec * L_curr * base_ang_vel
            + Vec2d(rel_Ldot_next, 0.0).rotated(base_ang_curr)
        )

        # Held object vel (world), calculated the same way as gripper-base vel
        helde_object_vels = []
        for kin_obj, _, _ in robot.held_objects:
            obj_body, _ = kin_obj
            obj_x_world = obj_body.position.x
            obj_y_world = obj_body.position.y
            obj_pos = Vec2d(obj_x_world, obj_y_world)
            relative_pos = obj_pos - base_pos_curr
            obj_rot_omega_vec_base = relative_pos.normalized().rotated(math.pi / 2)
            # We assume held object does not have relative velocity in the gripper frame
            # So we can just use rel_Ldot_next in the x-dir.
            v_held_obj = (
                base_vel
                + obj_rot_omega_vec_base * relative_pos.length * base_ang_vel
                + Vec2d(rel_Ldot_next, 0.0).rotated(base_ang_curr)
            )
            helde_object_vels.append(v_held_obj)

        # === 3) Gripper-base world velocity = rigid motion + prismatic contribution ===
        # Use *updated* base_vel & base_ang_vel for consistency in this control step
        kp_finger = getattr(self, "kp_finger", self.kp_pos)
        kv_finger = getattr(self, "kv_finger", self.kv_pos)
        gripper_centr = Vec2d(robot.finger_poses.x, robot.finger_poses.y)
        # Extract the rotate omega x r contribution
        relative_pos = gripper_centr - base_pos_curr
        finger_rot_omega_vec_base = relative_pos.normalized().rotated(math.pi / 2)
        finger_rot_omega_vec = (
            finger_rot_omega_vec_base * relative_pos.length * base_ang_vel_curr
        )
        # We only care about y-dir as finger only moves in y in the base frame
        rel_gdot_curr = (
            (finger_vel_abs_w - base_vel_curr - finger_rot_omega_vec)
            .rotated(-base_ang_curr)
            .y
        )
        a_g = kp_finger * (tgt_gripper - g_curr) + kv_finger * (0.0 - rel_gdot_curr)
        rel_gdot_next = rel_gdot_curr + a_g * dt
        # Finger world velocity, similar to gripper-base vel but with the additional
        # primistamic part in y-dir
        finger_vel_l = (
            base_vel
            + finger_rot_omega_vec_base * relative_pos.length * base_ang_vel
            + Vec2d(rel_Ldot_next, rel_gdot_next).rotated(base_ang_curr)
        )

        return (
            base_vel,
            base_ang_vel,
            v_gripper_base,
            finger_vel_l,
            helde_object_vels,
        )


def on_gripper_grasp(
    arbiter: pymunk.Arbiter, space: pymunk.Space, robot: KinRobot
) -> None:
    """Collision callback for gripper grasping objects."""
    dynamic_body = arbiter.bodies[0]
    if robot.is_grasping(arbiter.contact_point_set, dynamic_body):
        # Create a new kinematic object
        kinematic_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        kinematic_body.position = dynamic_body.position
        kinematic_body.angle = dynamic_body.angle
        points = arbiter.shapes[0].get_vertices()
        shape = pymunk.Poly(kinematic_body, points)
        shape.friction = 1
        shape.density = 1.0
        # Held object becomes part of the robot.
        shape.collision_type = ROBOT_COLLISION_TYPE
        space.add(kinematic_body, shape)
        robot.add_to_hand((kinematic_body, shape), dynamic_body.mass)
        # Remove the dynamic body from the space
        space.remove(dynamic_body, arbiter.shapes[0])


def on_collision_w_static(
    arbiter: pymunk.Arbiter, space: pymunk.Space, robot: KinRobot
) -> None:
    """Collision callback for robot colliding with static objects."""
    del arbiter
    del space
    robot.revert_to_last_state()


def create_walls_from_world_boundaries(
    world_min_x: float,
    world_max_x: float,
    world_min_y: float,
    world_max_y: float,
    min_dx: float,
    max_dx: float,
    min_dy: float,
    max_dy: float,
) -> dict[Object, dict[str, float]]:
    """Create wall objects and feature dicts based on world boundaries.

    Velocities are used to determine how large the walls need to be to avoid the
    possibility that the robot will transport over the wall.

    Left and right walls are considered "surfaces" (z_order=1) while top and bottom
    walls are considered "floors" (z_order=0). Otherwise there might be weird collision
    betweent left/right walls and top/bottom walls.
    """
    state_dict: dict[Object, dict[str, float]] = {}
    # Right wall.
    right_wall = Object("right_wall", KinRectangleType)
    side_wall_height = world_max_y - world_min_y
    state_dict[right_wall] = {
        "x": world_max_x + max_dx,
        "vx": 0.0,
        "y": (world_min_y + world_max_y) / 2,
        "vy": 0.0,
        "width": 2 * max_dx,  # 2x just for safety
        "height": side_wall_height,
        "theta": 0.0,
        "omega": 0.0,
        "static": True,
        "color_r": BLACK[0],
        "color_g": BLACK[1],
        "color_b": BLACK[2],
        "z_order": ZOrder.ALL.value,
    }
    # Left wall.
    left_wall = Object("left_wall", KinRectangleType)
    state_dict[left_wall] = {
        "x": world_min_x + min_dx,
        "vx": 0.0,
        "y": (world_min_y + world_max_y) / 2,
        "vy": 0.0,
        "width": 2 * abs(min_dx),  # 2x just for safety
        "height": side_wall_height,
        "theta": 0.0,
        "omega": 0.0,
        "static": True,
        "color_r": BLACK[0],
        "color_g": BLACK[1],
        "color_b": BLACK[2],
        "z_order": ZOrder.ALL.value,
    }
    # Top wall.
    top_wall = Object("top_wall", KinRectangleType)
    horiz_wall_width = 2 * 2 * abs(min_dx) + world_max_x - world_min_x
    state_dict[top_wall] = {
        "x": (world_min_x + world_max_x) / 2,
        "vx": 0.0,
        "y": world_max_y + max_dy,
        "vy": 0.0,
        "width": horiz_wall_width,
        "height": 2 * max_dy,
        "theta": 0.0,
        "omega": 0.0,
        "static": True,
        "color_r": BLACK[0],
        "color_g": BLACK[1],
        "color_b": BLACK[2],
        "z_order": ZOrder.ALL.value,
    }
    # Bottom wall.
    bottom_wall = Object("bottom_wall", KinRectangleType)
    state_dict[bottom_wall] = {
        "x": (world_min_x + world_max_x) / 2,
        "vx": 0.0,
        "y": world_min_y + min_dy,
        "vy": 0.0,
        "width": horiz_wall_width,
        "height": 2 * abs(min_dy),
        "theta": 0.0,
        "omega": 0.0,
        "static": True,
        "color_r": BLACK[0],
        "color_g": BLACK[1],
        "color_b": BLACK[2],
        "z_order": ZOrder.ALL.value,
    }
    return state_dict


def get_fingered_robot_action_from_gui_input(
    action_space: KinRobotActionSpace, gui_input: dict[str, Any]
) -> NDArray[np.float32]:
    """Get the mapping from human inputs to actions, derived from action space."""
    # This will be implemented later - placeholder for now
    keys_pressed = gui_input["keys"]
    right_x, right_y = gui_input["right_stick"]
    left_x, _ = gui_input["left_stick"]

    # Initialize the action.
    low = action_space.low
    high = action_space.high
    action = np.zeros(action_space.shape, action_space.dtype)

    def _rescale(x: float, lb: float, ub: float) -> float:
        """Rescale from [-1, 1] to [lb, ub]."""
        return lb + (x + 1) * (ub - lb) / 2

    # The right stick controls the x, y movement of the base.
    action[0] = _rescale(right_x, low[0], high[0])
    action[1] = _rescale(right_y, low[1], high[1])

    # The left stick controls the rotation of the base. Only the x axis
    # is used right now.
    action[2] = _rescale(left_x, low[2], high[2])

    # The w/s mouse keys are used to adjust the robot arm.
    if "a" in keys_pressed:
        action[3] = low[3]
    if "s" in keys_pressed:
        action[3] = high[3]

    # The space bar is used to close the gripper.
    # Open the gripper by default.
    if "d" in keys_pressed:
        action[4] = low[4]
    if "f" in keys_pressed:
        action[4] = high[4]

    return action
