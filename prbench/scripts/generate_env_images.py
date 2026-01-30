#!/usr/bin/env python3
"""Generate images for prbench environments.

Supports both generating images from environments and using existing image files.
Set USE_EXISTING_IMAGES = True (default) to load from existing images in docs/env_images.
Set USE_EXISTING_IMAGES = False to generate fresh images from environments.

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

# Global toggle: if True, use existing images from docs/env_images instead of generating
USE_EXISTING_IMAGES = True


@dataclass
class EnvImageConfig:
    """Configuration for generating an environment image."""

    env_id: str
    seed: int = 0
    # Crop values: (left_px, top_px, right_px, bottom_px) pixels to crop from each edge, or None for no cropping
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
# 4-row layout: 7-6-6-6 = 25 environments on 1200×800 canvas
# Organized: 2D environments top, 3D environments bottom for visual coherence
# Optimized for camera-ready figure with balanced spacing
ENV_CONFIGS = [
    # Row 1: 7 2D environments (y=1, ~169px width, scale 0.22)
    EnvImageConfig(
        env_id="prbench/ClutteredRetrieval2D-o10-v0",
        seed=42,
        crop=None,
        position=(1, 1),
        scale=0.22,
    ),
    EnvImageConfig(
        env_id="prbench/Motion2D-p5-v0",
        seed=42,
        crop=None,
        position=(170, 3),
        scale=0.22,
    ),
    EnvImageConfig(
        env_id="prbench/Obstruction2D-o4-v0",
        seed=42,
        crop=(20,0,20,0),
        position=(330, -15),
        scale=0.8,
    ),
    EnvImageConfig(
        env_id="prbench/ClutteredStorage2D-b15-v0",
        seed=42,
        crop=None,
        position=(615, 6),
        scale=0.18,
    ),
    EnvImageConfig(
        env_id="prbench/DynPushT-t1-v0",
        seed=42,
        crop=None,
        position=(880, 13),
        scale=0.6,
    ),
    EnvImageConfig(
        env_id="prbench/DynScoopPour-o50-v0",
        seed=42,
        crop=None,
        position=(1030, 12),
        scale=0.5,
    ),
    # Row 2: 6 2D/hybrid environments (y=201, 198px width, scale 0.25)
    EnvImageConfig(
        env_id="prbench/StickButton2D-b10-v0",
        seed=42,
        crop=None,
        position=(1, 201),
        scale=0.2,
    ),
    EnvImageConfig(
        env_id="prbench/PushPullHook2D-v0",
        seed=42,
        crop=None,
        position=(211, 201),
        scale=0.2,
    ),
    EnvImageConfig(
        env_id="prbench/Obstruction3D-o4-v0",
        seed=42,
        crop=(80,0,80,0),
        position=(430, 231),
        scale=0.35,
    ),
    EnvImageConfig(
        env_id="prbench/Packing3D-p3-v0",
        seed=42,
        crop=(80,0,80,0),
        position=(631, 231),
        scale=0.35,
    ),
    EnvImageConfig(
        env_id="prbench/Table3D-o3-v0",
        seed=42,
        crop=(80,0,80,0),
        position=(831, 231),
        scale=0.35,
    ),
    EnvImageConfig(
        env_id="prbench/DynObstruction2D-o3-v0",
        seed=42,
        crop=None,
        position=(1000, 221),
        scale=0.25,
    ),
    # Row 3: 6 3D environments (y=401, 198px width, scale 0.25)
    EnvImageConfig(
        env_id="prbench/Transport3D-o2-v0",
        seed=42,
        crop=(80,0,80,0),
        position=(1, 401),
        scale=0.35,
    ),
    EnvImageConfig(
        env_id="prbench/BaseMotion3D-v0",
        seed=42,
        crop=(80,0,80,0),
        position=(201, 401),
        scale=0.35,
    ),
    EnvImageConfig(
        env_id="prbench/TidyBot3D-cupboard-o8-v0",
        seed=42,
        crop=None,
        position=(401, 381),
        scale=0.35,
    ),
    EnvImageConfig(
        env_id="prbench/Shelf3D-o10-v0",
        seed=42,
        crop=(80,0,80,0),
        position=(580, 401),
        scale=0.35,
    ),
    EnvImageConfig(
        env_id="prbench/TidyBot3D-sort-lab2-o20-sort_the_cluttered_blocks_into_bowls-v0",
        seed=42,
        crop=None,
        position=(780, 381),
        scale=0.35,
    ),
    EnvImageConfig(
        env_id="prbench/TidyBot3D-rearrange-lab2_kitchen-o2-put_the_boxed_drink_and_the_can_next_to_the_bowl-v0",
        seed=42,
        crop=None,
        position=(980, 391),
        scale=0.3,
    ),
    # Row 4: 6 3D TidyBot environments (y=601, 198px width, scale 0.25)
    EnvImageConfig(
        env_id="prbench/TidyBot3D-tool_use-lab2_kitchen-o50-sweep_the_blocks_to_the_left_side_of_the_kitchen_island-v0",
        seed=42,
        crop=None,
        position=(1, 601),
        scale=0.2,
    ),
    EnvImageConfig(
        env_id="prbench/TidyBot3D-namo-o1-v0",
        seed=42,
        crop=None,
        position=(180, 601),
        scale=0.1,
    ),
    EnvImageConfig(
        env_id="prbench/TidyBot3D-dynamic-lab2-o1-toss_the_blocks_into_the_bin-v0",
        seed=42,
        crop=None,
        position=(321, 601),
        scale=0.3,
    ),
    EnvImageConfig(
        env_id="prbench/DynPushPullHook2D-o5-v0",
        seed=42,
        crop=None,
        position=(530, 601),
        scale=0.15,
    ),
    EnvImageConfig(
        env_id="prbench/TidyBot3D-tool_use-lab2_kitchen-o5-scoop_the_blocks_from_the_yellow_bin_to_the_green_bin-v0",
        seed=42,
        crop=None,
        position=(660, 601),
        scale=0.27,
    ),
    EnvImageConfig(
        env_id="prbench/TidyBot3D-dynamic-lab2-o3-balance_beam-v0",
        seed=42,
        crop=None,
        position=(850, 601),
        scale=0.3,
    ),
    EnvImageConfig(
        env_id="prbench/TidyBot3D-tool_use-lab2_kitchen-o5-sweep_the_blocks_into_the_top_drawer_of_the_kitchen_island-v0",
        seed=42,
        crop=None,
        position=(1031, 601),
        scale=0.25,
    ),
]


def generate_image(
    config: EnvImageConfig, output_dir: Path
) -> tuple[Path, Image.Image]:
    """Generate an image for a single environment or load from existing file."""
    env_name = config.env_id.replace("prbench/", "").replace("/", "_")
    output_path = output_dir / f"{env_name}.png"

    # Check if we should use existing images
    use_existing = USE_EXISTING_IMAGES or config.existing_image_path is not None

    if use_existing:
        # Determine existing image path
        if config.existing_image_path is not None:
            # Use explicitly specified path
            existing_path = Path(config.existing_image_path)
        else:
            # Auto-construct path based on env_id
            existing_path = Path(f"{env_name}.png")

        if not existing_path.is_absolute():
            # Resolve relative paths from the script's parent directory
            existing_path = Path(__file__).parent.parent / "docs" / "env_images" / existing_path

        img = Image.open(existing_path)

        if config.crop is not None:
            # Apply cropping in-memory without saving
            # Convert from (left_px, top_px, right_px, bottom_px) edge crops to absolute coordinates
            left, top, right, bottom = config.crop
            crop_box = (left, top, img.width - right, img.height - bottom)
            img = img.crop(crop_box)

        # Use existing image as-is (or cropped in-memory), no need to save
        return existing_path, img

    # Generate from environment
    env = prbench.make(config.env_id, render_mode="rgb_array")
    env.reset(seed=config.seed)
    img_array: NDArray[np.uint8] = env.render()  # type: ignore[assignment]
    env.close()  # type: ignore[no-untyped-call]

    img = Image.fromarray(img_array)

    if config.crop is not None:
        # Convert from (left_px, top_px, right_px, bottom_px) edge crops to absolute coordinates
        left, top, right, bottom = config.crop
        crop_box = (left, top, img.width - right, img.height - bottom)
        img = img.crop(crop_box)

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
        if USE_EXISTING_IMAGES or config.existing_image_path is not None:
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
