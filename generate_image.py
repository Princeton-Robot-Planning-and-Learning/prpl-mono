#!/usr/bin/env python3
"""Generate an image from a prbench environment."""

import prbench
from PIL import Image

prbench.register_all_environments()

env = prbench.make(
    "prbench/Packing3D-p3-v0",
    render_mode="rgb_array",
)

env.reset(seed=42)
image = env.render()
env.close()

Image.fromarray(image).save("output.png")
print(f"Saved output.png ({image.shape[1]}x{image.shape[0]})")
