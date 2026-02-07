"""Summarize evaluation results from saved pickle files.

This script reads all pickle files from a directory (organized by seed),
and computes summary statistics like success rate, average reward, etc.

Usage:
    python get_summary_shelf.py path/to/results_dir
    python get_summary_shelf.py path/to/results_dir --verbose
    python get_summary_shelf.py path/to/results_dir --show-states
"""

import argparse
import pickle as pkl
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import kinder
import numpy as np
from relational_structs.spaces import ObjectCentricBoxSpace

kinder.register_all_environments()


def discover_all_pickles(results_dir: Path) -> List[Path]:
    """Discover all pickle files in the results directory.

    Returns:
        List of pickle file paths sorted by modification time (newest first).
    """
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist")
        return []

    pickle_files = []
    for pickle_file in results_dir.rglob("*.p"):
        if pickle_file.is_file():
            pickle_files.append(pickle_file)

    # Sort by modification time (newest first)
    pickle_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return pickle_files


def get_pickle_info(pickle_path: Path) -> Tuple[str, int, int]:
    """Extract basic info from pickle path.

    Expected structure: .../seed_XXX/eval_episode_YYY/TIMESTAMP.p

    Returns:
        Tuple of (seed_str, episode_idx, timestamp)
    """
    parts = pickle_path.parts

    # Try to find seed and episode info from path
    seed_str = "unknown"
    episode_idx = -1
    timestamp = int(pickle_path.stem) if pickle_path.stem.isdigit() else 0

    for _, part in enumerate(parts):
        if part.startswith("seed_"):
            seed_str = part
        elif part.startswith("eval_episode_"):
            try:
                episode_idx = int(part.replace("eval_episode_", ""))
            except ValueError:
                pass

    return seed_str, episode_idx, timestamp


def load_pickle(pickle_path: Path) -> Dict[str, Any]:
    """Load a pickle file and return its contents."""
    with open(pickle_path, "rb") as f:
        return pkl.load(f)


def sanitize_env_id(env_id: str) -> str:
    """Fix common env_id issues like double -v0 suffix.

    Args:
        env_id: Environment ID that may have issues.

    Returns:
        Sanitized environment ID.
    """
    # Fix double -v0 suffix (e.g., "kinder/TidyBot3D-cupboard_real-o1-v0-v0")
    if env_id.endswith("-v0-v0"):
        env_id = env_id[:-3]  # Remove trailing "-v0"
    return env_id


def get_observation_space(env_id: str) -> Optional[ObjectCentricBoxSpace]:
    """Create an environment and return its observation space.

    Args:
        env_id: Environment ID (e.g., 'kinder/Transport3D-shelf-o1-v0')

    Returns:
        ObjectCentricBoxSpace or None if failed.
    """
    # Sanitize env_id first
    env_id = sanitize_env_id(env_id)

    try:
        env = kinder.make(env_id, render_mode=None)
        obs_space = env.observation_space
        env.close()  # type: ignore
        if isinstance(obs_space, ObjectCentricBoxSpace):
            return obs_space
        return None
    except Exception as e:
        print(f"Warning: Could not create environment {env_id}: {e}")
        return None


def devectorize_observation(obs: np.ndarray, obs_space: ObjectCentricBoxSpace):
    """Convert vectorized observation to object-centric state."""
    return obs_space.devectorize(obs)


def format_attribute_value(value: Any) -> str:
    """Format an attribute value for printing."""
    if isinstance(value, np.ndarray):
        if value.size <= 10:
            return np.array2string(
                value, precision=4, suppress_small=True, separator=", "
            )
        return f"array(shape={value.shape}, dtype={value.dtype})"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def format_object_state(state) -> bool:
    """Format an object-centric state for printing.

    Uses the same access patterns as planning_data_dynamics3d_kinder.py:
    - state.get_object_from_name(name) to get objects
    - state.get(obj, attr_name) to get attributes
    - state.get_attribute_names(obj) to list all attributes

    Args:
        state: The object-centric state to format.

    Returns:
        Formatted string representation.
    """
    target_cube = state.get_object_from_name("cube1")
    target_z = state.get(target_cube, "z")
    return target_z > 0.15


def get_object_summary(state) -> Dict[str, Dict[str, Any]]:
    """Extract a summary dictionary of all objects and their key attributes.

    This follows the same access pattern as planning_data_dynamics3d_kinder.py:
        robot = state.get_object_from_name("robot")
        cube = state.get_object_from_name("cube1")
        value = state.get(obj, "attribute_name")

    Args:
        state: The object-centric state.

    Returns:
        Dictionary mapping object names to their attributes.
    """
    summary: Dict[str, Dict[str, Any]] = {}

    for obj in state.objects:
        obj_name = obj.name
        obj_attrs: Dict[str, Any] = {"_type": type(obj).__name__}

        for attr_name in state.get_attribute_names(obj):
            value = state.get(obj, attr_name)
            # Convert numpy arrays to lists for easier inspection
            if isinstance(value, np.ndarray):
                obj_attrs[attr_name] = value.tolist()
            else:
                obj_attrs[attr_name] = value

        summary[obj_name] = obj_attrs

    return summary


def organize_by_seed(pickle_files: List[Path]) -> Dict[str, List[Path]]:
    """Organize pickle files by seed.

    Returns:
        Dictionary mapping seed string to list of pickle paths.
    """
    by_seed: Dict[str, List[Path]] = defaultdict(list)

    for pickle_path in pickle_files:
        seed_str, _, _ = get_pickle_info(pickle_path)
        by_seed[seed_str].append(pickle_path)

    # Sort each seed's files by episode index
    for seed_str in by_seed:
        by_seed[seed_str].sort(key=lambda p: get_pickle_info(p)[1])

    return dict(by_seed)


def compute_seed_stats(
    pickle_files: List[Path],
    show_states: bool = False,
    obs_space: Optional[ObjectCentricBoxSpace] = None,
) -> Dict[str, Any]:
    """Compute statistics for a set of pickle files (same seed).

    Args:
        pickle_files: List of pickle file paths.
        verbose: Print detailed info for each episode.
        show_states: Print object-centric states for each episode.
        obs_space: Observation space for devectorizing observations.

    Returns:
        Dictionary with statistics.
    """
    total_rewards = []
    successes = []
    episode_lengths = []
    grasp_success = 0
    final_success_num = 0

    for pickle_path in pickle_files:
        try:
            data = load_pickle(pickle_path)

            # Extract relevant info
            rewards = data.get("rewards", [])
            terminated = data.get("terminated", False)
            truncated = data.get("truncated", False)

            total_reward = sum(rewards) if rewards else 0.0
            total_rewards.append(total_reward)

            # Success is typically when terminated=True (task completed)
            successes.append(terminated and not truncated)

            # Episode length
            actions = data.get("actions", [])
            episode_lengths.append(len(actions))

            # Show object-centric states
            if show_states and obs_space is not None:
                observations = data.get("observations", [])
                if observations:

                    # Initial state - using same devectorize pattern as planning script
                    # env.observation_space.devectorize(obs) -> state
                    grasp = False
                    final_success = False
                    for obs in observations:
                        current_obs = np.array(obs)
                        current_state = devectorize_observation(current_obs, obs_space)
                        if format_object_state(current_state):
                            grasp = True
                            break
                    if grasp:
                        last_observation = observations[-1]
                        last_state = devectorize_observation(
                            last_observation, obs_space
                        )
                        target_cube = last_state.get_object_from_name("cube1")
                        target_z = last_state.get(target_cube, "z")
                        target_x = last_state.get(target_cube, "x")
                        if target_z > 0.5 and abs(target_x - 1.5) < 0.12:
                            final_success = True
                    if grasp:
                        grasp_success += 1
                    if final_success:
                        final_success_num += 1

        except Exception as e:
            print(f"  Warning: Failed to load {pickle_path}: {e}")
            continue

    return {"grasp_success": grasp_success, "final_success": final_success_num}


def main() -> None:
    """Main function to summarize evaluation results from saved pickle files."""
    parser = argparse.ArgumentParser(
        description="Summarize evaluation results from saved pickle files"
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing pickle files (organized by seed)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed info for each episode",
    )
    parser.add_argument(
        "--show-states",
        "-s",
        action="store_true",
        help="Show object-centric states (initial and final) for each episode",
    )

    args = parser.parse_args()

    # Discover all pickle files
    pickle_files = discover_all_pickles(args.results_dir)

    if not pickle_files:
        print(f"No pickle files found in {args.results_dir}")
        return

    print(f"Found {len(pickle_files)} pickle files in {args.results_dir}")

    # Organize by seed
    by_seed = organize_by_seed(pickle_files)
    print(f"Organized into {len(by_seed)} seed(s): {list(by_seed.keys())}")

    # Get observation space if needed for showing states
    obs_space: Optional[ObjectCentricBoxSpace] = None
    if args.show_states:
        # Load first pickle to get env_id
        first_pickle = pickle_files[0]
        first_data = load_pickle(first_pickle)
        env_id = first_data.get("env_id", None)
        if env_id:
            sanitized_id = sanitize_env_id(env_id)
            print(f"Environment: {env_id}")
            if sanitized_id != env_id:
                print(f"  (sanitized to: {sanitized_id})")
            obs_space = get_observation_space(env_id)
            if obs_space is None:
                print(
                    "Warning: Could not get observation space, states will not be shown"
                )
        else:
            print("Warning: env_id not found in pickle, states will not be shown")

    # Compute stats for each seed
    seed_stats = []
    for _, seed_files in sorted(by_seed.items()):
        seed_stats.append(
            compute_seed_stats(
                seed_files,
                show_states=args.show_states,
                obs_space=obs_space,
            )
        )
    print(seed_stats)
    total_episodes = 50
    grasp_success_rate = []
    final_success_rate = []
    for seed_stat in seed_stats:
        grasp_success_rate.append(seed_stat["grasp_success"] / total_episodes)
        final_success_rate.append(seed_stat["final_success"] / total_episodes)
    print(f"Grasp success rates per seed: {grasp_success_rate}")
    print(f"Final success rates per seed: {final_success_rate}")
    print(
        f"Mean grasp success rate: {np.mean(grasp_success_rate):.4f} ± {np.std(grasp_success_rate):.4f}"  # pylint: disable=line-too-long
    )
    print(
        f"Mean final success rate: {np.mean(final_success_rate):.4f} ± {np.std(final_success_rate):.4f}"  # pylint: disable=line-too-long
    )


if __name__ == "__main__":
    main()
