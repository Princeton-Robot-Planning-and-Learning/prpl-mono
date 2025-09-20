"""Utility functions."""

from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from prpl_perception_utils.structs import DetectedObject2D, RGBImage


def visualize_detections_2d(
    img: RGBImage,
    detections: List[DetectedObject2D],
    mask_rgba: tuple[int, int, int, int] = (0, 100, 255, 100),
) -> RGBImage:
    """Create a new image with detections overlaid on the input image."""

    # Convert numpy array to PIL Image.
    pil_img = Image.fromarray(img)

    # Work with RGBA for compositing.
    pil_img = pil_img.convert("RGBA")
    mask_overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_overlay)

    draw = ImageDraw.Draw(pil_img)

    # Load a font.
    font = ImageFont.load_default()

    for detection in detections:
        bbox = detection.bounding_box

        # Draw mask.
        # Convert mask coordinates to image coordinates.
        mask_array = np.array(detection.mask)
        mask_coords = np.where(mask_array > 0)

        if len(mask_coords[0]) > 0:
            # Get coordinates relative to the bounding box.
            y_coords = mask_coords[0] + bbox.y1
            x_coords = mask_coords[1] + bbox.x1

            # Create a list of points for the mask.
            points = list(zip(x_coords, y_coords))

            # Draw filled polygon for the mask.
            if len(points) > 2:
                mask_draw.polygon(points, fill=mask_rgba)

        # Draw bounding box.
        draw.rectangle([bbox.x1, bbox.y1, bbox.x2, bbox.y2], outline="red", width=3)

        # Prepare label text.
        label = f"{detection.object_id} ({detection.confidence_score:.2f})"

        # Get text size for background.
        bbox_text = draw.textbbox((0, 0), label, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]

        # Draw label background.
        label_x = bbox.x1
        label_y = max(0, bbox.y1 - text_height - 5)

        # Ensure label doesn't go off the top of the image.
        if label_y < 0:
            label_y = bbox.y1 + 5

        draw.rectangle(
            [label_x, label_y, label_x + text_width + 10, label_y + text_height + 5],
            fill="red",
            outline="red",
        )

        # Draw label text
        draw.text((label_x + 5, label_y + 2), label, fill="white", font=font)

    # Composite masks over the main image.
    pil_img = Image.alpha_composite(pil_img, mask_overlay)
    # Convert back to RGB.
    pil_img = pil_img.convert("RGB")

    # Convert back to numpy array.
    return np.array(pil_img)
