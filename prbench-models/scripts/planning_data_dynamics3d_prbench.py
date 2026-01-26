"""Dataset collection using bilevel planning parameterized skills for dynamics3d
environments."""

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import prbench
from relational_structs.spaces import ObjectCentricBoxSpace

from prbench_models.dynamic3d.ground.parameterized_skills import (
    PyBulletSim,
    create_lifted_controllers,
)
from prbench_models.teleop_utils import _visualize_image_in_window

prbench.register_all_environments()

# Default demos directory: ../prbench/demos relative to this script
# Script: prpl-mono/prbench-models/scripts/teleop_dynamics3d_prbench.py
# Demos:  prpl-mono/prbench/demos
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DEMOS_DIR = _SCRIPT_DIR.parent.parent / "prbench" / "demos"


def sanitize_env_id(env_id: str) -> str:
    """Remove unnecessary stuff from the env ID.

    Mirrors the function in prbench/scripts/generate_env_docs.py and
    collect_demos_ds.py for consistent directory naming.
    """
    if env_id.startswith("prbench/"):
        env_id = env_id[len("prbench/") :]
    env_id = env_id.replace("/", "_")
    if len(env_id) >= 3 and env_id[-3:-1] == "-v":
        return env_id[:-3]
    return env_id


def save_demo(
    demo_dir: Path,
    env_id: str,
    seed: int,
    observations: list[Any],
    actions: list[Any],
    rewards: list[float],
    terminated: bool,
    truncated: bool,
) -> Path:
    """Save a demo to disk in the same format as collect_demos_ds.py.

    Directory structure: {demo_dir}/{sanitized_env_id}/{seed}/{timestamp}.p
    """
    timestamp = int(time.time())
    demo_subdir = demo_dir / sanitize_env_id(env_id) / str(seed)
    demo_subdir.mkdir(parents=True, exist_ok=True)
    demo_path = demo_subdir / f"{timestamp}.p"
    demo_data = {
        "env_id": env_id,
        "timestamp": timestamp,
        "seed": seed,
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminated": terminated,
        "truncated": truncated,
    }
    with open(demo_path, "wb") as f:
        pkl.dump(demo_data, f)
    return demo_path


def collect_data(
    output_dir: str = "data/demos",
    seed: int = 123,
    save: bool = True,
    grasping_only: bool = False,
    show_images: bool = False,
):
    """Collect pick and place demonstration data in ground environment.

    Args:
        output_dir: Directory to save episode data.
        seed: Random seed for reproducibility.
        save: Whether to save the episode data to disk.
    """

    # Create the environment.
    num_cubes = 1
    env = prbench.make(
        f"prbench/TidyBot3D-cupboard_real-o{num_cubes}-v0", render_mode="rgb_array", scene_bg=True
    )

    

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=seed)  # type: ignore
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    # Initialize demo collection lists (same format as collect_demos_ds.py)
    observations: list[Any] = [obs]  # Start with initial observation
    actions: list[Any] = []
    rewards: list[float] = []

    assert state is not None
    pybullet_sim = PyBulletSim(state, rendering=False)

    controllers = create_lifted_controllers(env.action_space, pybullet_sim=pybullet_sim)  # type: ignore # pylint: disable=line-too-long

    # Target object for this episode
    target_object_key = "cube1"

    # Create the pick ground controller.
    lifted_controller = controllers["pick_ground"]
    robot = state.get_object_from_name("robot")
    cube = state.get_object_from_name(target_object_key)
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    params = controller.sample_parameters(state, np.random.default_rng(seed))

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for step_idx in range(400):
        action = controller.step()
        
        
        if show_images:
            robot_name = env.unwrapped._object_centric_env.robot_name  # type: ignore # pylint: disable=protected-access
            env.unwrapped._object_centric_env.set_render_camera("overview")
            overview_image = env.unwrapped._object_centric_env.render()
            env.unwrapped._object_centric_env.set_render_camera(robot_name + "_base")
            base_image = env.unwrapped._object_centric_env.render()
            env.unwrapped._object_centric_env.set_render_camera(robot_name + "_wrist")
            wrist_image = env.unwrapped._object_centric_env.render()
            _visualize_image_in_window(overview_image, "overview")
            _visualize_image_in_window(base_image, "base")
            _visualize_image_in_window(wrist_image, "wrist")
        # Record observation and action before stepping
        
        obs, reward, ep_terminated, ep_truncated, _ = env.step(  # type: ignore
            action
        )

        # Record data for demo (same format as collect_demos_ds.py)
        observations.append(obs)
        actions.append(action)
        rewards.append(float(reward))

        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            print(f"Pick controller terminated after {step_idx + 1} steps")
            break
    else:
        print("Warning: Pick controller did not terminate within 400 steps")

    if not grasping_only:
        # Create the place ground controller.
        lifted_controller = controllers["place_ground"]
        robot = state.get_object_from_name("robot")
        cube = state.get_object_from_name(target_object_key)
        cupboard = state.get_object_from_name("cupboard_1")
        object_parameters = (robot, cube, cupboard)  # type: ignore
        controller = lifted_controller.ground(object_parameters)
        params = controller.sample_parameters(state, np.random.default_rng(seed))

        # Reset and execute the controller until it terminates.
        controller.reset(state, params)
        for step_idx in range(400):
            action = controller.step()
            
            
            if show_images:
                robot_name = env.unwrapped._object_centric_env.robot_name  # type: ignore # pylint: disable=protected-access
                env.unwrapped._object_centric_env.set_render_camera("overview")
                overview_image = env.unwrapped._object_centric_env.render()
                env.unwrapped._object_centric_env.set_render_camera(robot_name + "_base")
                base_image = env.unwrapped._object_centric_env.render()
                env.unwrapped._object_centric_env.set_render_camera(robot_name + "_wrist")
                wrist_image = env.unwrapped._object_centric_env.render()
                _visualize_image_in_window(overview_image, "overview")
                _visualize_image_in_window(base_image, "base")
                _visualize_image_in_window(wrist_image, "wrist")
            # Record observation and action before stepping
            
            obs, reward, ep_terminated, ep_truncated, _ = env.step(  # type: ignore
                action
            )

            # Record data for demo (same format as collect_demos_ds.py)
            observations.append(obs)
            actions.append(action)
            rewards.append(float(reward))

            next_state = env.observation_space.devectorize(obs)
            controller.observe(next_state)
            state = next_state
            if controller.terminated():
                print(f"Place controller terminated after {step_idx + 1} steps")
                break
        else:
            print("Warning: Place controller did not terminate within 400 steps")

    # Save episode data to disk (same format as collect_demos_ds.py)
    if save and len(actions) > 0:
        demo_path = save_demo(
            demo_dir,
            env_id,
            episode_seed,
            observations,
            actions,
            rewards,
            terminated,
            truncated,
        )
        print(f"Episode saved to {demo_path}")
        print(f"  Observations: {len(observations)}, Actions: {len(actions)}")
    elif save:
        print("No actions recorded, episode not saved")


    env.close()  # type: ignore


def main() -> None:
    """Main function to collect demonstration data."""
    parser = argparse.ArgumentParser(description="Collect demonstration data")
    parser.add_argument("--output-dir", default="data/demos", help="Output dir")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--grasping-only", action="store_true", default=True)
    parser.add_argument("--show-images", action="store_true", default=False)
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.add_argument(
        "--n-demos", type=int, default=1, help="Number of demos to collect"
    )
    args = parser.parse_args()
    for demo_idx in range(args.n_demos):
        collect_data(
            output_dir=args.output_dir,
            seed=args.seed + demo_idx,
            save=args.save,
            grasping_only=args.grasping_only,
            show_images=args.show_images,
        )


if __name__ == "__main__":
    main()
