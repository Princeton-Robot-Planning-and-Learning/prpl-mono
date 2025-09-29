"""Utility functions."""

from typing import List

import colorsys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from prpl_perception_utils.structs import DetectedObject2D

def hsl_to_rgba(hue: float) -> tuple[int, int, int, int]:
    """Convert an HSL color to RGBA.
    Hue is in degrees, saturation and lightness are 50%."""

    # Convert HSL to RGB using colorsys.
    r, g, b = colorsys.hls_to_rgb(
        h = hue / 360.0,
        l = 0.5,
        s = 1.0
    )

    # Convert to RGBA.
    return (int(r * 255), int(g * 255), int(b * 255), 100)

def get_luminance(r: int, g: int, b: int) -> float:
    """Calculate the luminance of an RGB shape fill color,
    to help determine text color."""

    return 0.299 * r + 0.587 * g + 0.114 * b

def visualize_detections_2d(
    img: np.ndarray,
    detections: List[DetectedObject2D]
) -> np.ndarray:
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

    # Start hue at 0 (red) and step by 30 degrees for each detection.
    hue = 0

    for detection in detections:
        bbox = detection.bounding_box

        # Generate color with hue stepping by 30 degrees.
        detection_color = hsl_to_rgba(hue)

        # Extract RGB values (ignore alpha for luminance calculation).
        r, g, b, _ = detection_color

        # Calculate luminance to decide the text color.
        luminance = get_luminance(r, g, b)

        # Choose text color based on luminance.
        text_color = "black" if luminance > 128 else "white"

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
                mask_draw.polygon(points, fill=detection_color)

        # Draw bounding box with the detection color.
        draw.rectangle(
            [bbox.x1, bbox.y1, bbox.x2, bbox.y2], outline=detection_color, width=3
        )

        # Prepare label text.
        label = f"{detection.object_id} ({detection.confidence_score:.2f})"

        # Get text size for background.
        bbox_text = draw.textbbox((0, 0), label, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]

        # Draw label background with detection color.
        label_x = bbox.x1
        label_y = max(0, bbox.y1 - text_height - 5)

        # Ensure label doesn't go off the top of the image.
        if label_y < 0:
            label_y = bbox.y1 + 5

        draw.rectangle(
            [label_x, label_y, label_x + text_width + 10, label_y + text_height + 5],
            fill=detection_color,
            outline=detection_color,
        )

        # Draw label text with dynamically chosen color.
        draw.text((label_x + 5, label_y + 2), label, fill=text_color, font=font)

        # Step hue for the next detection.
        hue += 30
        if hue >= 360:
            hue = 0  # Wrap hue around if it exceeds 360°

    # Composite masks over the main image.
    pil_img = Image.alpha_composite(pil_img, mask_overlay)
    # Convert back to RGB.
    pil_img = pil_img.convert("RGB")

    # Convert back to numpy array.
    return np.array(pil_img)
