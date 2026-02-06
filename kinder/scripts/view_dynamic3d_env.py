#!/usr/bin/env python3
"""View dynamic3D environments with an interactive MuJoCo viewer.

This script uses MuJoCo's passive viewer which allows interactive
camera control while the simulation runs (similar to PyBullet's GUI mode).

Usage:
    python kinder/scripts/view_dynamic3d_env.py \
kinder/TidyBot3D-table-o1-v0
    python kinder/scripts/view_dynamic3d_env.py \
kinder/TidyBot3D-table-o1-v0 --seed 42
    python kinder/scripts/view_dynamic3d_env.py \
kinder/TidyBot3D-table-o3-v0 --episodes 5
    python kinder/scripts/view_dynamic3d_env.py \
kinder/TidyBot3D-table-o3-v0 --use-opencv

Interactive MuJoCo Viewer Controls (default mode):
    - Left mouse drag: Rotate camera
    - Right mouse drag: Move camera
    - Scroll wheel: Zoom in/out
    - Double-click: Select/track object
    - Close window to quit

OpenCV Viewer Controls (--use-opencv mode):
    - Press 'q' to quit
    - Press 'r' to reset episode
    - Press SPACE to pause/unpause
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from typing import Any

import cv2  # type: ignore
import mujoco
import mujoco.viewer
import numpy as np

import kinder


def update_xml_with_state(
    xml_path: str,
    model: "mujoco.MjModel",  # type: ignore  # pylint: disable=no-member
    data: "mujoco.MjData",  # type: ignore  # pylint: disable=no-member
) -> None:
    """Update XML file with current simulation state (object positions/orientations).

    This ensures objects appear at their current positions in the viewer, not at origin.
    """
    # Parse the XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get all bodies in worldbody
    worldbody = root.find("worldbody")
    if worldbody is None:
        return

    updated_count = 0

    # Iterate through all bodies and update positions for freejoint bodies
    for body in worldbody.iter("body"):
        body_name = body.get("name")
        if body_name is None:
            continue

        # Check if this body has a freejoint
        freejoint = None
        for child in body:
            if child.tag == "freejoint":
                freejoint = child
                break
            if child.tag == "joint" and child.get("type") == "free":
                freejoint = child
                break

        if freejoint is None:
            continue

        # Get body ID
        try:
            # pylint: disable=no-member
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        except Exception:  # pylint: disable=broad-except
            continue

        # Get the joint ID for this body's freejoint
        joint_id = None
        for jnt_id in range(model.njnt):
            jnt_body_id = model.jnt_bodyid[jnt_id]
            # pylint: disable=no-member
            if (
                jnt_body_id == body_id
                and model.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_FREE
            ):
                joint_id = jnt_id
                break

        if joint_id is None:
            continue

        # Get qpos address for this joint
        qpos_addr = model.jnt_qposadr[joint_id]

        # Extract position (first 3 values) and quaternion (next 4 values)
        pos = data.qpos[qpos_addr : qpos_addr + 3]
        quat = data.qpos[qpos_addr + 3 : qpos_addr + 7]

        # Update body position in XML
        body.set("pos", f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
        body.set("quat", f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}")

        updated_count += 1

    if updated_count > 0:
        print(f"Updated {updated_count} objects with current positions")

    # Save updated XML
    tree.write(xml_path)


def run_with_standalone_viewer(env: Any, args: Any) -> None:
    """Run using standalone mujoco.viewer app (works great on macOS!).

    This launches the MuJoCo viewer as a separate process, avoiding threading issues.
    Shows the initial environment state with full interactive camera controls.
    """
    # Get the underlying MuJoCo sim from the environment
    unwrapped_env = env.unwrapped

    # Navigate to robot env
    # pylint: disable=protected-access
    if hasattr(unwrapped_env, "_object_centric_env"):
        object_centric_env = unwrapped_env._object_centric_env
        if hasattr(object_centric_env, "_robot_env"):
            robot_env = object_centric_env._robot_env
        else:
            print("Error: Could not find _robot_env")
            return
    else:
        print("Error: Could not find object_centric_env")
        return

    current_seed = args.seed

    for episode in range(args.episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{args.episodes}")
        print("=" * 60)

        # Reset environment to get a new configuration
        _, _ = env.reset(seed=current_seed)
        print(f"Environment reset with seed: {current_seed}")

        # Get the sim after reset
        sim = robot_env.sim
        if sim is None:
            print("Error: sim is None after reset")
            return

        # Run a few simulation steps to let objects settle to their proper positions
        for _ in range(10):
            sim.forward()  # Update derived quantities
            sim.step()  # Step physics

        # Get the XML model and export to temporary file
        model = sim.model.mj_model
        temp_fd, temp_xml_path = tempfile.mkstemp(suffix=".xml")
        os.close(temp_fd)
        mujoco.mj_saveLastXML(temp_xml_path, model)  # pylint: disable=no-member

        # Update XML with current state (positions/orientations of objects)
        update_xml_with_state(temp_xml_path, model, sim.data.mj_data)

        # If user wants to save XML, copy it to their specified path
        if args.save_xml:
            save_path = args.save_xml
            if args.episodes > 1:
                # Add episode number to filename
                base, ext = os.path.splitext(save_path)
                save_path = f"{base}_episode{episode}{ext}"
            shutil.copy(temp_xml_path, save_path)
            print(f"XML saved to: {save_path}")

        print("\nLaunching standalone MuJoCo viewer...")
        print(f"Model file: {temp_xml_path}")
        print("\nInteractive Viewer Controls:")
        print("  - Left mouse drag: Rotate camera")
        print("  - Right mouse drag: Move camera")
        print("  - Scroll wheel: Zoom in/out")
        print("  - Double-click object: Select it")
        print("  - Ctrl+Left drag: Apply forces to selected object")
        print("  - Ctrl+Right drag: Apply torques to selected object")
        print("  - Space: Play/pause physics")
        print("  - Ctrl+Q or close window: Quit viewer")
        print("\nClose the viewer window to continue to next episode...\n")

        # Launch standalone viewer
        subprocess.run(
            [sys.executable, "-m", "mujoco.viewer", f"--mjcf={temp_xml_path}"],
            check=False,
        )

        # Clean up temp file
        os.unlink(temp_xml_path)

        if current_seed is not None:
            current_seed += 1


def run_with_mujoco_viewer(env: Any, args: Any) -> None:
    """Run environment with interactive MuJoCo viewer using blocking approach.

    Note: On macOS, passive viewer has threading issues, so this will automatically
    fall back to standalone viewer or OpenCV viewer.
    """
    # On macOS, use standalone viewer for best experience
    if platform.system() == "Darwin":
        print("\n" + "!" * 60)
        print("INFO: On macOS, using standalone MuJoCo viewer.")
        print("This provides full interactive 3D viewing without threading issues!")
        print("!" * 60 + "\n")
        run_with_standalone_viewer(env, args)
        return

    # Linux/Windows: use passive viewer
    # Get the underlying MuJoCo sim from the environment
    unwrapped_env = env.unwrapped

    # Try to find the object_centric_env first
    # pylint: disable=protected-access
    if hasattr(unwrapped_env, "_object_centric_env"):
        object_centric_env = unwrapped_env._object_centric_env
        if hasattr(object_centric_env, "_robot_env"):
            robot_env = object_centric_env._robot_env
        else:
            print("Error: Could not find _robot_env in object_centric_env")
            return
    elif hasattr(unwrapped_env, "_robot_env"):
        robot_env = unwrapped_env._robot_env
    else:
        print("Error: Could not find robot environment with MuJoCo simulation")
        return

    if not hasattr(robot_env, "sim"):
        print("Error: Robot environment does not have a MuJoCo simulation")
        return

    print("\n" + "=" * 60)
    print("Interactive MuJoCo Viewer Controls:")
    print("  - Left mouse drag: Rotate camera")
    print("  - Right mouse drag: Move camera")
    print("  - Scroll wheel: Zoom in/out")
    print("  - Close window to quit")
    print("=" * 60 + "\n")

    current_seed = args.seed

    # Run episodes
    for episode in range(args.episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{args.episodes}")
        print("=" * 60)

        # Reset environment (this creates the sim)
        _, _ = env.reset(seed=current_seed)
        print(f"Environment reset with seed: {current_seed}")

        # Get the sim after reset
        sim = robot_env.sim
        if sim is None:
            print("Error: sim is None after reset")
            return

        # Launch passive viewer
        with mujoco.viewer.launch_passive(
            sim.model.mj_model,
            sim.data.mj_data,
        ) as viewer:
            viewer.sync()

            total_reward = 0.0
            step = 0

            while viewer.is_running() and step < args.max_steps:
                if not args.no_random:
                    action = env.action_space.sample()
                    _, reward, terminated, truncated, _ = env.step(action)
                    total_reward += float(reward)
                    step += 1

                    viewer.sync()

                    if step % 100 == 0:
                        msg = (
                            f"  Step {step}/{args.max_steps} | "
                            f"Total reward: {total_reward:.2f}"
                        )
                        print(msg)

                    if terminated or truncated:
                        status = "✓" if terminated else "⚠"
                        print(f"\n{status} Episode ended at step {step}")
                        print(f"  Total reward: {total_reward:.2f}")
                        time.sleep(2)
                        break
                else:
                    viewer.sync()
                    time.sleep(0.01)

            if not viewer.is_running():
                print("\nViewer closed by user")
                break

        if current_seed is not None:
            current_seed += 1


def run_with_opencv_viewer(env: Any, args: Any) -> None:
    """Run environment with OpenCV window (fallback mode)."""

    window_name = f"KinDER Viewer - {args.env_id}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.window_width, args.window_height)

    paused = False
    current_seed = args.seed
    quit_requested = False
    frame_delay = int(1000 / args.fps)

    print("\n" + "=" * 60)
    print("OpenCV Viewer Controls:")
    print("  - Press 'q': Quit")
    print("  - Press 'r': Reset episode")
    print("  - Press SPACE: Pause/unpause")
    print("=" * 60 + "\n")

    for episode in range(args.episodes):
        if quit_requested:
            break

        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{args.episodes}")
        print("=" * 60)

        _, _ = env.reset(seed=current_seed)
        print(f"Environment reset with seed: {current_seed}")

        frame = env.render()
        if frame is None or not isinstance(frame, np.ndarray):
            print(f"Error: render() returned invalid type: {type(frame)}")
            break

        total_reward = 0.0
        step = 0

        while step < args.max_steps:
            display_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

            status_text = (
                f"Episode {episode + 1}/{args.episodes} | "
                f"Step {step}/{args.max_steps} | Reward: {total_reward:.2f}"
            )
            if paused:
                status_text += " | PAUSED"

            cv2.putText(
                display_frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(frame_delay) & 0xFF

            if key == ord("q"):
                quit_requested = True
                break
            if key == ord("r"):
                break
            if key == ord(" "):
                paused = not paused

            if paused or args.no_random:
                continue

            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            step += 1
            frame = env.render()

            if step % 100 == 0:
                msg = (
                    f"  Step {step}/{args.max_steps} | "
                    f"Total reward: {total_reward:.2f}"
                )
                print(msg)

            if terminated or truncated:
                status = "✓" if terminated else "⚠"
                print(f"\n{status} Episode ended at step {step}")
                print(f"  Total reward: {total_reward:.2f}")
                time.sleep(1)
                break

        if current_seed is not None:
            current_seed += 1

    cv2.destroyAllWindows()


def main() -> None:
    """Parse command line arguments and view the environment."""
    parser = argparse.ArgumentParser(
        description="View dynamic3D environments with interactive MuJoCo viewer"
    )
    parser.add_argument(
        "env_id",
        type=str,
        help="Environment ID (e.g., kinder/TidyBot3D-ground-o1-v0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for environment reset (default: None)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run (default: 1)",
    )
    parser.add_argument(
        "--no-random",
        action="store_true",
        help="Don't take random actions, just show initial state",
    )
    parser.add_argument(
        "--realistic-bg",
        action="store_true",
        help="Use realistic background images (if supported by env)",
    )
    parser.add_argument(
        "--use-opencv",
        action="store_true",
        help="Use OpenCV viewer instead of interactive MuJoCo viewer",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target FPS for OpenCV viewer (default: 30)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode (default: 1000)",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1280,
        help="OpenCV window width in pixels (default: 1280)",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=720,
        help="OpenCV window height in pixels (default: 720)",
    )
    parser.add_argument(
        "--save-xml",
        type=str,
        default=None,
        help="Save the MuJoCo XML file to this path (useful for inspection)",
    )

    args = parser.parse_args()

    if not args.env_id.startswith("kinder/"):
        print("Error: Environment ID must start with 'kinder/'")
        print("Example: kinder/TidyBot3D-ground-o1-v0")
        return

    print(f"Loading environment: {args.env_id}")
    print(f"Viewer mode: {'OpenCV' if args.use_opencv else 'Interactive MuJoCo'}")
    print(f"Seed: {args.seed}")
    print(f"Episodes: {args.episodes}")
    print(f"Random actions: {not args.no_random}")
    print("-" * 60)

    # Register all environments
    kinder.register_all_environments()

    try:
        if args.use_opencv:
            # OpenCV mode needs render_mode
            env = kinder.make(
                args.env_id, render_mode="rgb_array", scene_bg=args.realistic_bg
            )
        else:
            # MuJoCo viewer mode - no render_mode needed
            env = kinder.make(args.env_id, scene_bg=args.realistic_bg)
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("\nAvailable dynamic3D environments:")
        all_envs = kinder.get_all_env_ids()
        dynamic3d_envs = sorted(
            [e for e in all_envs if "TidyBot3D" in e or "RBY1A3D" in e]
        )
        for env_id in dynamic3d_envs:
            print(f"  {env_id}")
        sys.exit(1)

    # Run with appropriate viewer
    try:
        if args.use_opencv:
            run_with_opencv_viewer(env, args)
        else:
            run_with_mujoco_viewer(env, args)
    finally:
        print("\n" + "=" * 60)
        print("Closing environment...")
        env.close()  # type: ignore
        print("Done!")


if __name__ == "__main__":
    main()
