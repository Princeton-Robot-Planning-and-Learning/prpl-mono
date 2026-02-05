#!/usr/bin/env python3
"""Combine GIFs from group_gifs into a grid MP4 video."""

import subprocess
import sys
from pathlib import Path


def get_gif_info(gif_path: Path) -> tuple[int, int, float]:
    """Get width, height, and duration of a GIF using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(gif_path),
        ],
        capture_output=True,
        text=True,
    )
    lines = result.stdout.strip().split("\n")
    w, h = lines[0].split(",")
    duration = float(lines[1])
    return int(w), int(h), duration


def main():
    group_gifs_dir = Path(__file__).parent.parent / "docs/envs/assets/group_gifs"
    output_path = Path(__file__).parent.parent / "docs/envs/assets/group_grid.mp4"

    gif_files = sorted(group_gifs_dir.glob("*.gif"))
    if not gif_files:
        print("No GIFs found in group_gifs directory")
        sys.exit(1)

    print(f"Found {len(gif_files)} GIFs")

    # Gather info for all GIFs
    gif_info = []
    for gif_path in gif_files:
        w, h, duration = get_gif_info(gif_path)
        gif_info.append((gif_path, w, h, duration))
        print(f"  {gif_path.name}: {w}x{h}, {duration:.2f}s")

    # Determine grid dimensions
    n = len(gif_files)
    cols = 4
    rows = (n + cols - 1) // cols  # Ceiling division

    # Target cell size and duration
    cell_size = 360
    speed_multiplier = 4
    max_duration = max(info[3] for info in gif_info)
    target_duration = max_duration / speed_multiplier
    print(f"Target duration: {target_duration:.2f}s")

    # Build ffmpeg filter complex
    inputs = []
    filter_parts = []

    for i, (gif_path, w, h, duration) in enumerate(gif_info):
        # Loop short GIFs twice (if less than 1/3 of max duration)
        if duration < max_duration / 3:
            inputs.extend(["-stream_loop", "1", "-i", str(gif_path)])
            effective_duration = duration * 2
            print(f"  Looping {gif_path.name} (2x)")
        else:
            inputs.extend(["-i", str(gif_path)])
            effective_duration = duration

        # Calculate speed factor to normalize duration
        speed_factor = target_duration / effective_duration

        if w == h:
            # Already square, just scale and adjust duration
            filter_parts.append(
                f"[{i}:v]setpts={speed_factor}*PTS,scale={cell_size}:{cell_size}[v{i}]"
            )
        elif w > h:
            # Wider than tall - add vertical padding
            filter_parts.append(
                f"[{i}:v]setpts={speed_factor}*PTS,scale={cell_size}:-1,pad={cell_size}:{cell_size}:(ow-iw)/2:(oh-ih)/2:color=white[v{i}]"
            )
        else:
            # Taller than wide - add horizontal padding
            filter_parts.append(
                f"[{i}:v]setpts={speed_factor}*PTS,scale=-1:{cell_size},pad={cell_size}:{cell_size}:(ow-iw)/2:(oh-ih)/2:color=white[v{i}]"
            )

    # Add white frames for empty cells if needed
    empty_cells = rows * cols - n
    for i in range(empty_cells):
        idx = n + i
        filter_parts.append(
            f"color=white:s={cell_size}x{cell_size}:d={target_duration}[v{idx}]"
        )

    # Build xstack layout
    total_cells = rows * cols
    layout_parts = []
    for i in range(total_cells):
        row = i // cols
        col = i % cols
        x = col * cell_size
        y = row * cell_size
        layout_parts.append(f"{x}_{y}")

    stream_refs = "".join(f"[v{i}]" for i in range(total_cells))
    layout = "|".join(layout_parts)
    filter_parts.append(
        f"{stream_refs}xstack=inputs={total_cells}:layout={layout}[out]"
    )

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-t",
        str(target_duration),
        str(output_path),
    ]

    print(f"Running ffmpeg to create {output_path}")
    subprocess.run(cmd, check=True)
    print(f"Created {output_path}")


if __name__ == "__main__":
    main()
