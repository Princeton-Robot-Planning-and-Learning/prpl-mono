#!/usr/bin/env python3
"""Generate images for prbench environments.

Supports both generating images from environments and using existing image files.
To use an existing image, set the `existing_image_path` field in EnvImageConfig.

Usage:
    python scripts/generate_env_images.py
    python scripts/generate_env_images.py --output-dir /path/to/output
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

import prbench


@dataclass
class EnvImageConfig:
    """Configuration for generating an environment image."""

    env_id: str
    seed: int = 0
    # Crop values: (left, top, right, bottom) or None for no cropping
    crop: tuple[int, int, int, int] | None = None
    # Position in combined image: (x, y) from top-left
    position: tuple[int, int] = (0, 0)
    # Scale factor (maintains aspect ratio)
    scale: float = 1.0
    # Path to existing image file to use instead of generating one
    existing_image_path: Path | str | None = None


# Combined image settings
COMBINED_IMAGE_SIZE: tuple[int, int] = (1200, 800)  # (width, height)
COMBINED_IMAGE_BACKGROUND: tuple[int, int, int] = (255, 255, 255)  # RGB

# List of environments to generate images for.
# 4-row layout: 6-6-6-7 = 25 environments on 1200×800 canvas
# Row height: 200 pixels
ENV_CONFIGS = [
    # Row 1: 6 environments (y=0, 200px width each)
    EnvImageConfig(
        env_id="prbench/ClutteredRetrieval2D-o10-v0",
        seed=42,
        crop=None,
        position=(0, 0),
        scale=0.2,
    ),
    EnvImageConfig(
        env_id="prbench/Motion2D-p5-v0",
        seed=42,
        crop=None,
        position=(200, 0),
        scale=0.2,
    ),
    EnvImageConfig(
        env_id="prbench/Obstruction2D-o4-v0",
        seed=42,
        crop=None,
        position=(400, 0),
        scale=0.2,
    ),
    EnvImageConfig(
        env_id="prbench/DynPushPullHook2D-o5-v0",
        seed=42,
        crop=None,
        position=(600, 0),
        scale=0.2,
    ),
    EnvImageConfig(
        env_id="prbench/ClutteredStorage2D-b15-v0",
        seed=42,
        crop=None,
        position=(800, 0),
        scale=0.2,
    ),
    EnvImageConfig(
        env_id="prbench/StickButton2D-b10-v0",
        seed=42,
        crop=None,
        position=(1000, 0),
        scale=0.2,
    ),
    # Row 2: 6 environments (y=200, 200px width each)
    EnvImageConfig(
        env_id="prbench/DynObstruction2D-o3-v0",
        seed=42,
        crop=None,
        position=(0, 200),
        scale=0.2,
    ),
    EnvImageConfig(
        env_id="prbench/DynPushT-t1-v0",
        seed=42,
        crop=None,
        position=(200, 200),
        scale=0.2,
    ),
    EnvImageConfig(
        env_id="prbench/DynScoopPour-o50-v0",
        seed=42,
        crop=None,
        position=(400, 200),
        scale=0.2,
    ),
    EnvImageConfig(
        env_id="prbench/Obstruction3D-o4-v0",
        seed=42,
        crop=None,
        position=(600, 200),
        scale=0.2,
    ),
    EnvImageConfig(
        env_id="prbench/Packing3D-p3-v0",
        seed=42,
        crop=None,
        position=(800, 200),
        scale=0.2,
    ),
    EnvImageConfig(
        env_id="prbench/Table3D-o3-v0",
        seed=42,
        crop=None,
        position=(1000, 200),
        scale=0.2,
    ),
    # Row 3: 6 environments (y=400, 200px width each)
    EnvImageConfig(
        env_id="prbench/Transport3D-o2-v0",
        seed=42,
        crop=None,
        position=(0, 400),
        scale=0.2,
    ),
    EnvImageConfig(
        env_id="prbench/BaseMotion3D-v0",
        seed=42,
        crop=None,
        position=(200, 400),
        scale=0.2,
    ),
    EnvImageConfig(
        env_id="prbench/TidyBot3D-cupboard-o8-v0",
        seed=42,
        crop=None,
        position=(400, 400),
        scale=0.2,
    ),
    EnvImageConfig(
        env_id="prbench/Shelf3D-o10-v0",
        seed=42,
        crop=None,
        position=(600, 400),
        scale=0.2,
    ),
    EnvImageConfig(
        env_id="prbench/TidyBot3D-sort-lab2-o20-sort_the_cluttered_blocks_into_bowls-v0",
        seed=42,
        crop=None,
        position=(800, 400),
        scale=0.2,
    ),
    EnvImageConfig(
        env_id="prbench/TidyBot3D-rearrange-lab2_kitchen-o2-put_the_boxed_drink_and_the_can_next_to_the_bowl-v0",
        seed=42,
        crop=None,
        position=(1000, 400),
        scale=0.2,
    ),
    # Row 4: 7 environments (y=600, ~171px width each)
    EnvImageConfig(
        env_id="prbench/TidyBot3D-tool_use-lab2_kitchen-o50-sweep_the_blocks_to_the_left_side_of_the_kitchen_island-v0",
        seed=42,
        crop=None,
        position=(0, 600),
        scale=0.17,
    ),
    EnvImageConfig(
        env_id="prbench/tidybot-namo-o1-v0",
        seed=42,
        crop=None,
        position=(171, 600),
        scale=0.17,
        existing_image_path="docs/env_images/TidyBot3D-namo-o1-v0.png",
    ),
    EnvImageConfig(
        env_id="prbench/TidyBot3D-dynamic-lab2-o1-toss_the_blocks_into_the_bin-v0",
        seed=42,
        crop=None,
        position=(342, 600),
        scale=0.17,
    ),
    EnvImageConfig(
        env_id="prbench/TidyBot3D-tool_use-lab2_kitchen-o5-scoop_the_blocks_from_the_yellow_bin_to_the_green_bin-v0",
        seed=42,
        crop=None,
        position=(513, 600),
        scale=0.17,
    ),
    EnvImageConfig(
        env_id="prbench/TidyBot3D-dynamic-lab2-o3-balance_beam-v0",
        seed=42,
        crop=None,
        position=(684, 600),
        scale=0.17,
    ),
    EnvImageConfig(
        env_id="prbench/TidyBot3D-tool_use-lab2_kitchen-o5-sweep_the_blocks_into_the_top_drawer_of_the_kitchen_island-v0",
        seed=42,
        crop=None,
        position=(855, 600),
        scale=0.17,
    ),
    EnvImageConfig(
        env_id="prbench/PushPullHook2D-v0",
        seed=42,
        crop=None,
        position=(1026, 600),
        scale=0.17,
    ),
]


def generate_image(
    config: EnvImageConfig, output_dir: Path
) -> tuple[Path, Image.Image]:
    """Generate an image for a single environment or load from existing file."""
    env_name = config.env_id.replace("prbench/", "").replace("/", "_")
    output_path = output_dir / f"{env_name}.png"

    if config.existing_image_path is not None:
        # Load existing image instead of generating
        existing_path = Path(config.existing_image_path)
        if not existing_path.is_absolute():
            # Resolve relative paths from the script's parent directory
            existing_path = Path(__file__).parent.parent / existing_path

        img = Image.open(existing_path)

        if config.crop is not None:
            img = img.crop(config.crop)

        # Save to output directory for consistency
        img.save(output_path)
        return output_path, img

    # Generate from environment
    env = prbench.make(config.env_id, render_mode="rgb_array")
    env.reset(seed=config.seed)
    img_array: NDArray[np.uint8] = env.render()  # type: ignore[assignment]
    env.close()  # type: ignore[no-untyped-call]

    img = Image.fromarray(img_array)

    if config.crop is not None:
        img = img.crop(config.crop)

    img.save(output_path)

    return output_path, img


def combine_images(
    configs: list[EnvImageConfig],
    images: list[Image.Image],
    output_path: Path,
) -> None:
    """Combine multiple images into a single image at specified positions and scales."""
    combined = Image.new("RGB", COMBINED_IMAGE_SIZE, COMBINED_IMAGE_BACKGROUND)

    for config, img in zip(configs, images):
        if config.scale != 1.0:
            new_width = int(img.width * config.scale)
            new_height = int(img.height * config.scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert RGBA to RGB if needed
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, COMBINED_IMAGE_BACKGROUND)
            bg.paste(img, mask=img.split()[3])
            img = bg

        combined.paste(img, config.position)

    combined.save(output_path)
    print(f"Combined image saved to {output_path}")


def main() -> None:
    """Generate images for all configured environments."""
    parser = argparse.ArgumentParser(
        description="Generate images for prbench environments"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "docs" / "env_images",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--combined-name",
        type=str,
        default="combined.png",
        help="Filename for the combined image",
    )
    args = parser.parse_args()

    prbench.register_all_environments()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    images: list[Image.Image] = []
    for config in ENV_CONFIGS:
        if config.existing_image_path is not None:
            print(f"Using existing image for {config.env_id}...")
        else:
            print(f"Generating image for {config.env_id}...")
        output_path, img = generate_image(config, args.output_dir)
        images.append(img)
        print(f"  Saved to {output_path}")

    print(f"\nGenerated {len(ENV_CONFIGS)} images in {args.output_dir}")

    combined_path = args.output_dir / args.combined_name
    combine_images(ENV_CONFIGS, images, combined_path)


if __name__ == "__main__":
    main()
