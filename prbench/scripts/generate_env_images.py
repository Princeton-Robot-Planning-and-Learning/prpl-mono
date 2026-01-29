#!/usr/bin/env python3
"""Generate images for prbench environments.

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


# Combined image settings
COMBINED_IMAGE_SIZE: tuple[int, int] = (1200, 800)  # (width, height)
COMBINED_IMAGE_BACKGROUND: tuple[int, int, int] = (255, 255, 255)  # RGB

# List of environments to generate images for.
# Configure each with: crop=(left, top, right, bottom), position=(x, y), scale=factor
ENV_CONFIGS = [
    EnvImageConfig(
        env_id="prbench/Obstruction2D-o3-v0",
        seed=42,
        crop=None,
        position=(0, 0),
        scale=0.5,
    ),
    EnvImageConfig(
        env_id="prbench/Motion2D-p3-v0",
        seed=42,
        crop=None,
        position=(600, 0),
        scale=0.5,
    ),
    EnvImageConfig(
        env_id="prbench/Ground3D-o2-v0",
        seed=42,
        crop=None,
        position=(0, 400),
        scale=0.5,
    ),
    EnvImageConfig(
        env_id="prbench/Table3D-o2-v0",
        seed=42,
        crop=None,
        position=(600, 400),
        scale=0.5,
    ),
]


def generate_image(
    config: EnvImageConfig, output_dir: Path
) -> tuple[Path, Image.Image]:
    """Generate an image for a single environment."""
    env = prbench.make(config.env_id, render_mode="rgb_array")
    env.reset(seed=config.seed)
    img_array: NDArray[np.uint8] = env.render()  # type: ignore[assignment]
    env.close()  # type: ignore[no-untyped-call]

    img = Image.fromarray(img_array)

    if config.crop is not None:
        img = img.crop(config.crop)

    env_name = config.env_id.replace("prbench/", "").replace("/", "_")
    output_path = output_dir / f"{env_name}.png"
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
        print(f"Generating image for {config.env_id}...")
        output_path, img = generate_image(config, args.output_dir)
        images.append(img)
        print(f"  Saved to {output_path}")

    print(f"\nGenerated {len(ENV_CONFIGS)} images in {args.output_dir}")

    combined_path = args.output_dir / args.combined_name
    combine_images(ENV_CONFIGS, images, combined_path)


if __name__ == "__main__":
    main()
