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
            assert line.startswith("prbench/")
            line = line[len("prbench/") :]
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


def sanitize_class_name(class_name: str) -> str:
    """Sanitize class name for use in filenames."""
    return class_name.replace("/", "_")


def create_random_action_gif(
    class_name: str,
    env: gymnasium.Env,
    num_actions: int = 25,
    seed: int = 0,
    default_fps: int = 10,
) -> bool:
    """Create a GIF of taking random actions in the environment.

    Args:
        class_name: The environment class name (e.g., "ClutteredStorage2D")
        env: The environment instance to use for generating the GIF
        num_actions: Number of random actions to take
        seed: Random seed
        default_fps: Default FPS if not specified in metadata

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
        class_filename = sanitize_class_name(class_name)
        outfile = OUTPUT_DIR / "assets" / "random_action_gifs" / f"{class_filename}.gif"
        fps = env.metadata.get("render_fps", default_fps)
        iio.mimsave(outfile, imgs, fps=fps, loop=0)
        return True
    except Exception as e:
        print(f"    Warning: Failed to create random action GIF for {class_name}: {e}")
        return False


def create_initial_state_gif(
    class_name: str,
    env: gymnasium.Env,
    num_resets: int = 25,
    seed: int = 0,
    fps: int = 10,
) -> bool:
    """Create a GIF of different initial states by calling reset().

    Args:
        class_name: The environment class name (e.g., "ClutteredStorage2D")
        env: The environment instance to use for generating the GIF
        num_resets: Number of resets to show different initial states
        seed: Random seed
        fps: Frames per second for the GIF

    Returns:
        bool: True if successful, False if rendering failed.
    """
    try:
        imgs: list = []
        for i in range(num_resets):
            env.reset(seed=seed + i)
            imgs.append(env.render())
        class_filename = sanitize_class_name(class_name)
        outfile = OUTPUT_DIR / "assets" / "initial_state_gifs" / f"{class_filename}.gif"
        iio.mimsave(outfile, imgs, fps=fps, loop=0)
        return True
    except Exception as e:
        print(f"    Warning: Failed to create initial state GIF for {class_name}: {e}")
        return False


def generate_markdown(
    class_name: str,
    env: gymnasium.Env,
    variants: list[str],
    has_random_gif: bool = True,
    has_initial_gif: bool = True,
) -> str:
    """Generate markdown for a given environment class.

    Args:
        class_name: The environment class name (e.g., "ClutteredStorage2D")
        env: A representative environment instance for extracting metadata
        variants: List of all variant IDs for this class
        has_random_gif: Whether the random action GIF was successfully generated
        has_initial_gif: Whether the initial state GIF was successfully generated

    Returns:
        The markdown content as a string
    """
    md = f"# {class_name}\n\n"
    class_filename = sanitize_class_name(class_name)

    if has_random_gif:
        md += (
            f"![random action GIF](assets/random_action_gifs/{class_filename}.gif)\n\n"
        )
    else:
        md += "*(Random action GIF could not be generated due to rendering issues)*\n\n"

    description = env.metadata.get("description", "No description defined.")
    md += f"## Description\n{description}\n\n"

    # List all available variants
    md += "## Available Variants\n"
    variant_description = env.metadata.get("variant_description", "")
    if variant_description and variant_description != "Variant description not defined":
        md += f"{variant_description}\n\n"
    for variant_id in variants:
        # Extract just the variant suffix (e.g., "b1" from "prbench/ClutteredStorage2D-b1-v0")
        variant_suffix = (
            variant_id.replace("prbench/", "")
            .replace(f"{class_name}-", "")
            .replace("-v0", "")
        )
        md += f"- `{variant_id}` ({variant_suffix})\n"
    md += "\n"

    md += "## Initial State Distribution\n"

    if has_initial_gif:
        md += (
            f"![initial state GIF](assets/initial_state_gifs/{class_filename}.gif)\n\n"
        )
    else:
        md += "*(Initial state GIF could not be generated due to rendering issues)*\n\n"

    md += "## Example Demonstration\n"

    # Search for demo GIFs across all variant subdirectories
    demo_gif_found = False
    # Search backwards, assuming that later variants are "harder" and therefore more
    # interesting to show demonstrations for.
    for variant_id in variants[::-1]:
        # Convert variant ID to the subdirectory name format
        variant_subdir_name = sanitize_env_id(variant_id)
        demo_subdir = OUTPUT_DIR / "assets" / "demo_gifs" / variant_subdir_name
        if demo_subdir.exists():
            gif_files = sorted(
                [f for f in demo_subdir.iterdir() if f.suffix.lower() == ".gif"]
            )
            if gif_files:
                first_gif = gif_files[0].name
                md += f"![demo GIF](assets/demo_gifs/{variant_subdir_name}/{first_gif})\n\n"
                demo_gif_found = True
                break

    if not demo_gif_found:
        md += "*(No demonstration GIFs available)*\n\n"

    md += "## Observation Space\n"
    md += env.metadata["observation_space_description"] + "\n\n"
    md += "## Action Space\n"
    md += env.metadata["action_space_description"] + "\n\n"
    md += "## Rewards\n"
    md += env.metadata["reward_description"] + "\n\n"
    if "references" in env.metadata:
        md += "## References\n"
        md += env.metadata["references"] + "\n\n"
    return md.rstrip() + "\n"


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate environment documentation")
    parser.add_argument(
        "--force", action="store_true", help="Force regeneration of all environments"
    )
    parser.add_argument(
        "--force-tidybot",
        action="store_true",
        help="Force regeneration of all environments for tidybot",
    )
    args = parser.parse_args()

    print("Regenerating environment docs...")
    if args.force:
        print("Force flag detected - regenerating all environment classes")
    elif args.force_tidybot:
        print("Force tidybot flag detected - regenerating all tidybot classes")
    else:
        print("Checking for changes using git diff origin/main...")

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "assets" / "random_action_gifs").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "assets" / "initial_state_gifs").mkdir(parents=True, exist_ok=True)

    prbench.register_all_environments()

    changed_files = get_changed_files()

    total_classes = 0
    regenerated_classes = 0

    env_classes = prbench.get_env_classes()

    for class_name, class_info in env_classes.items():
        total_classes += 1
        variants = class_info["variants"]

        # Use a middle variant as representative (or first if only one variant)
        representative_variant = variants[len(variants) // 2]
        env = prbench.make(representative_variant, render_mode="rgb_array")

        # Check if any variant of this class has changed
        class_changed = any(
            is_env_changed(prbench.make(v, render_mode="rgb_array"), changed_files)
            for v in variants
        )

        if args.force or class_changed:
            print(f"  Regenerating {class_name}...")
            has_random_gif = create_random_action_gif(class_name, env)
            has_initial_gif = create_initial_state_gif(class_name, env)
            md = generate_markdown(
                class_name, env, variants, has_random_gif, has_initial_gif
            )
            class_filename = sanitize_class_name(class_name)
            filename = OUTPUT_DIR / f"{class_filename}.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(md)
            regenerated_classes += 1
        elif args.force_tidybot and "TidyBot3D" in class_name:
            print(f"  Regenerating {class_name}...")
            has_random_gif = create_random_action_gif(class_name, env)
            has_initial_gif = create_initial_state_gif(class_name, env)
            md = generate_markdown(
                class_name, env, variants, has_random_gif, has_initial_gif
            )
            class_filename = sanitize_class_name(class_name)
            filename = OUTPUT_DIR / f"{class_filename}.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(md)
            regenerated_classes += 1
        else:
            print(f"  Skipping {class_name} (no changes detected)")

    print(
        f"Finished generating environment docs. Regenerated {regenerated_classes}/{total_classes} classes."
    )

    # Add the results.
    subprocess.run(["git", "add", OUTPUT_DIR], check=True)


if __name__ == "__main__":
    _main()
