"""Automatically create markdown documents for every registered environment."""

import argparse
import inspect
import subprocess
from pathlib import Path

import gymnasium
import imageio.v2 as iio

import prbench

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "envs"


def get_changed_files() -> set[Path]:
    """Get the set of files that have changed compared to origin/main."""
    # Get the list of changed files from git diff.
    result = subprocess.run(
        ["git", "diff", "origin/main", "--name-only"],
        capture_output=True,
        text=True,
        check=True,
    )

    if not result.stdout.strip():
        return set()

    # Convert to Path objects.
    changed_files = set()
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            changed_files.add(Path(line.strip()).resolve())

    return changed_files


def is_env_changed(env: gymnasium.Env, changed_files: set[Path]) -> bool:
    """Check if the environment has changed based on git diff."""
    module_path = Path(inspect.getfile(env.unwrapped.__class__)).resolve()
    return module_path in changed_files


def sanitize_env_id(env_id: str) -> str:
    """Remove unnecessary stuff from the env ID."""
    assert env_id.startswith("prbench/")
    env_id = env_id[len("prbench/") :]
    env_id = env_id.replace("/", "_")
    assert env_id[-3:-1] == "-v"
    return env_id[:-3]


def create_random_action_gif(
    env_id: str,
    env: gymnasium.Env,
    num_actions: int = 25,
    seed: int = 0,
    default_fps: int = 10,
) -> bool:
    """Create a GIF of taking random actions in the environment.

    Returns:
        bool: True if successful, False if rendering failed.
    """
    try:
        imgs: list = []
        env.reset(seed=seed)
        env.action_space.seed(seed)
        imgs.append(env.render())
        for _ in range(num_actions):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            imgs.append(env.render())
            if terminated or truncated:
                break
        env_filename = sanitize_env_id(env_id)
        outfile = OUTPUT_DIR / "assets" / "random_action_gifs" / f"{env_filename}.gif"
        fps = env.metadata.get("render_fps", default_fps)
        iio.mimsave(outfile, imgs, fps=fps, loop=0)
        return True
    except Exception as e:
        print(f"    Warning: Failed to create random action GIF for {env_id}: {e}")
        return False


def create_initial_state_gif(
    env_id: str,
    env: gymnasium.Env,
    num_resets: int = 25,
    seed: int = 0,
    fps: int = 10,
) -> bool:
    """Create a GIF of different initial states by calling reset().

    Returns:
        bool: True if successful, False if rendering failed.
    """
    try:
        imgs: list = []
        for i in range(num_resets):
            env.reset(seed=seed + i)
            imgs.append(env.render())
        env_filename = sanitize_env_id(env_id)
        outfile = OUTPUT_DIR / "assets" / "initial_state_gifs" / f"{env_filename}.gif"
        iio.mimsave(outfile, imgs, fps=fps, loop=0)
        return True
    except Exception as e:
        print(f"    Warning: Failed to create initial state GIF for {env_id}: {e}")
        return False


def generate_markdown(
    env_id: str,
    env: gymnasium.Env,
    has_random_gif: bool = True,
    has_initial_gif: bool = True,
) -> str:
    """Generate markdown for a given env."""
    md = f"# {env_id}\n"
    env_filename = sanitize_env_id(env_id)

    if has_random_gif:
        md += f"![random action GIF](assets/random_action_gifs/{env_filename}.gif)\n\n"
    else:
        md += "*(Random action GIF could not be generated due to rendering issues)*\n\n"

    description = env.metadata.get("description", "No description defined.")
    md += f"### Description\n{description}\n"
    md += "### Initial State Distribution\n"

    if has_initial_gif:
        md += f"![initial state GIF](assets/initial_state_gifs/{env_filename}.gif)\n\n"
    else:
        md += "*(Initial state GIF could not be generated due to rendering issues)*\n\n"

    md += "### Example Demonstration\n"

    # Use the new subdirectory structure to select the first demo GIF
    demo_subdir = OUTPUT_DIR / "assets" / "demo_gifs" / env_filename
    if demo_subdir.exists():
        gif_files = sorted(
            [f for f in demo_subdir.iterdir() if f.suffix.lower() == ".gif"]
        )
        if gif_files:
            first_gif = gif_files[0].name
            md += f"![demo GIF](assets/demo_gifs/{env_filename}/{first_gif})\n\n"
        else:
            md += "*(No demonstration GIFs available)*\n\n"
    else:
        md += "*(No demonstration GIFs available)*\n\n"

    md += "### Observation Space\n"
    md += env.metadata["observation_space_description"] + "\n\n"
    md += "### Action Space\n"
    md += env.metadata["action_space_description"] + "\n\n"
    md += "### Rewards\n"
    md += env.metadata["reward_description"] + "\n\n"
    if "references" in env.metadata:
        md += "### References\n"
        md += env.metadata["references"] + "\n\n"
    return md.rstrip() + "\n"


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate environment documentation")
    parser.add_argument(
        "--force", action="store_true", help="Force regeneration of all environments"
    )
    args = parser.parse_args()

    print("Regenerating environment docs...")
    if args.force:
        print("Force flag detected - regenerating all environments")
    else:
        print("Checking for changes using git diff origin/main...")

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "assets" / "random_action_gifs").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "assets" / "initial_state_gifs").mkdir(parents=True, exist_ok=True)

    prbench.register_all_environments()

    changed_files = get_changed_files()

    total_envs = 0
    regenerated_envs = 0

    for env_id in prbench.get_all_env_ids():
        total_envs += 1
        env = prbench.make(env_id, render_mode="rgb_array")

        if args.force or is_env_changed(env, changed_files):
            print(f"  Regenerating {env_id}...")
            has_random_gif = create_random_action_gif(env_id, env)
            has_initial_gif = create_initial_state_gif(env_id, env)
            md = generate_markdown(env_id, env, has_random_gif, has_initial_gif)
            assert env_id.startswith("prbench/")
            env_filename = sanitize_env_id(env_id)
            filename = OUTPUT_DIR / f"{env_filename}.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(md)
            regenerated_envs += 1
        else:
            print(f"  Skipping {env_id} (no changes detected)")

    print("Finished generating environment docs.")

    # Add the results.
    subprocess.run(["git", "add", OUTPUT_DIR], check=True)


if __name__ == "__main__":
    _main()
