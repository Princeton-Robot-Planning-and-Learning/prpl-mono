"""Dataset collection using bilevel planning parameterized skills for dynamics3d
environments."""

import argparse
import time
from pathlib import Path
from typing import Any

import dill as pkl  # type: ignore[import-untyped]
import kinder
import numpy as np
from relational_structs.spaces import ObjectCentricBoxSpace

from kinder_models.dynamic3d.ground.parameterized_skills import (
    PyBulletSim,
    create_lifted_controllers,
)
from kinder_models.teleop_utils import _visualize_image_in_window

kinder.register_all_environments()

# Default demos directory: ../kinder/demos relative to this script
# Script: prpl-mono/kinder-models/scripts/teleop_dynamics3d_kinder.py
# Demos:  prpl-mono/kinder/demos
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DEMOS_DIR = _SCRIPT_DIR.parent.parent / "kinder" / "demos"


def sanitize_env_id(env_id: str) -> str:
    """Remove unnecessary stuff from the env ID.

    Mirrors the function in kinder/scripts/generate_env_docs.py and collect_demos_ds.py
    for consistent directory naming.
    """
    if env_id.startswith("kinder/"):
        env_id = env_id[len("kinder/") :]
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
    env_name: str = "TidyBot3D-cupboard_real-o1-v0",
):
    """Collect pick and place demonstration data in ground environment.

    Args:
        output_dir: Directory to save episode data.
        seed: Random seed for reproducibility.
        save: Whether to save the episode data to disk.
    """

    env_id = f"kinder/{env_name}"
    demo_dir = Path(output_dir)

    # Create the environment.
    env = kinder.make(f"kinder/{env_name}", render_mode="rgb_array", scene_bg=True)

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=seed)  # type: ignore
    for _ in range(5):
        obs, _, _, _, _ = env.step(np.zeros(11))
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    # Initialize demo collection lists (same format as collect_demos_ds.py)
    observations: list[Any] = [obs]  # Start with initial observation
    actions: list[Any] = []
    rewards: list[float] = []
    terminated = False
    truncated = False

    assert state is not None
    pybullet_sim = PyBulletSim(state, rendering=False)

    controllers = create_lifted_controllers(env.action_space, pybullet_sim=pybullet_sim)  # type: ignore # pylint: disable=line-too-long

    # Target object for this episode
    target_object_key = "cube1"

    try:
        # Create the pick ground controller.
        lifted_controller = controllers["pick_ground"]
        robot = state.get_object_from_name("robot")
        cube = state.get_object_from_name(target_object_key)
        object_parameters = (robot, cube)
        controller = lifted_controller.ground(object_parameters)
        # params = controller.sample_parameters(state, np.random.default_rng(123))
        params = np.array([0.6, 0.0])

        # Reset and execute the controller until it terminates.
        try:
            controller.reset(state, params)
        except ValueError as e:
            print(e)
            print("Pick controller reset failed. Not saving.")
            env.close()  # type: ignore
            return

        for step_idx in range(400):
            action = controller.step()

            if show_images:
                robot_name = env.unwrapped._object_centric_env.robot_name  # type: ignore # pylint: disable=protected-access
                env.unwrapped._object_centric_env.set_render_camera("agentview_1")  # type: ignore # pylint: disable=protected-access
                overview_image = env.unwrapped._object_centric_env.render()  # type: ignore # pylint: disable=protected-access
                env.unwrapped._object_centric_env.set_render_camera(  # type: ignore # pylint: disable=protected-access
                    robot_name + "_base"
                )  # type: ignore # pylint: disable=protected-access
                base_image = env.unwrapped._object_centric_env.render()  # type: ignore # pylint: disable=protected-access
                env.unwrapped._object_centric_env.set_render_camera(  # type: ignore # pylint: disable=protected-access
                    robot_name + "_wrist"
                )  # type: ignore # pylint: disable=protected-access
                wrist_image = env.unwrapped._object_centric_env.render()  # type: ignore # pylint: disable=protected-access
                _visualize_image_in_window(overview_image, "agentview_1")
                _visualize_image_in_window(base_image, "base")
                _visualize_image_in_window(wrist_image, "wrist")
            # Record observation and action before stepping

            obs, reward, _, _, _ = env.step(action)  # type: ignore

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
            raise ValueError("Pick controller did not terminate within 400 steps")

        cube_z = state.get(cube, "z")
        if cube_z < 0.1:
            print("Cube is too low. Not saving.")
            env.close()  # type: ignore
            return
        if not grasping_only:
            # Create the place ground controller.
            lifted_controller = controllers["place_ground"]
            robot = state.get_object_from_name("robot")
            cube = state.get_object_from_name(target_object_key)
            cupboard = state.get_object_from_name("cupboard_1")
            object_parameters = (robot, cube, cupboard)  # type: ignore
            controller = lifted_controller.ground(object_parameters)
            # params = np.array([0.91823519, -0.13385369, -1.57079633])
            # params = np.array([0.88823519, -0.13385369, -1.57079633])
            params = np.array([0.86823519, -0.13385369, -1.57079633])

            # Reset and execute the controller until it terminates.
            try:
                controller.reset(state, params)
            except ValueError as e:
                print(e)
                print("Place controller reset failed. Not saving.")
                env.close()  # type: ignore
                return
            for step_idx in range(400):
                action = controller.step()

                if show_images:
                    robot_name = env.unwrapped._object_centric_env.robot_name  # type: ignore # pylint: disable=protected-access
                    env.unwrapped._object_centric_env.set_render_camera("agentview_1")  # type: ignore # pylint: disable=protected-access
                    overview_image = env.unwrapped._object_centric_env.render()  # type: ignore # pylint: disable=protected-access
                    env.unwrapped._object_centric_env.set_render_camera(  # type: ignore # pylint: disable=protected-access
                        robot_name + "_base"
                    )
                    base_image = env.unwrapped._object_centric_env.render()  # type: ignore # pylint: disable=protected-access
                    env.unwrapped._object_centric_env.set_render_camera(  # type: ignore # pylint: disable=protected-access
                        robot_name + "_wrist"
                    )
                    wrist_image = env.unwrapped._object_centric_env.render()  # type: ignore # pylint: disable=protected-access
                    _visualize_image_in_window(overview_image, "agentview_1")
                    _visualize_image_in_window(base_image, "base")
                    _visualize_image_in_window(wrist_image, "wrist")
                # Record observation and action before stepping

                obs, reward, _, _, _ = env.step(action)  # type: ignore

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
                raise ValueError("Place controller did not terminate within 400 steps")

    except ValueError as e:
        print(e)
        print("Episode not successful. Not saving.")
        env.close()  # type: ignore
        return

    # Save episode data to disk (same format as collect_demos_ds.py)
    if save and len(actions) > 0:
        demo_path = save_demo(
            demo_dir,
            env_id,
            seed,
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
    parser.add_argument(
        "--output-dir",
        default=str(_DEFAULT_DEMOS_DIR),
        help="Directory to save episodes (default: kinder/demos)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--grasping-only", action="store_true", default=False)
    parser.add_argument("--show-images", action="store_true", default=False)
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.add_argument(
        "--n-demos", type=int, default=1, help="Number of demos to collect"
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="TidyBot3D-cupboard_real-o1-v0",
        help="Name of the environment",
    )

    args = parser.parse_args()
    for demo_idx in range(args.n_demos):
        collect_data(
            output_dir=args.output_dir,
            seed=args.seed + demo_idx,
            save=args.save,
            grasping_only=args.grasping_only,
            show_images=args.show_images,
            env_name=args.env_name,
        )


if __name__ == "__main__":
    main()
