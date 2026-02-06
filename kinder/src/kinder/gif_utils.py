"""Utilities for optimizing GIF file sizes."""

import shutil
import subprocess
import tempfile
from pathlib import Path


def optimize_gif(
    input_path: Path,
    output_path: Path | None = None,
    colors: int = 256,
    lossy: int = 80,
) -> Path:
    """Optimize a GIF file to reduce its size.

    Uses gifsicle if available, otherwise returns the original file unchanged.

    Args:
        input_path: Path to the input GIF file.
        output_path: Path for the optimized output. If None, overwrites input.
        colors: Maximum number of colors (2-256). Lower = smaller file.
        lossy: Lossy compression level (0-200). Higher = smaller file but lower quality.
            Recommended range: 30-100. Default 80 is a good balance.

    Returns:
        Path to the optimized GIF file.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)

    # Check if gifsicle is available.
    gifsicle_path = shutil.which("gifsicle")
    if gifsicle_path is None:
        print("Warning: gifsicle not found. GIF optimization skipped.")
        if output_path != input_path:
            shutil.copy(input_path, output_path)
        return output_path

    # Get original size for reporting.
    original_size = input_path.stat().st_size

    # Use a temporary file if we're overwriting the input.
    if output_path == input_path:
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        actual_output = tmp_path
    else:
        actual_output = output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build gifsicle command.
    cmd = [
        gifsicle_path,
        "--optimize=3",  # Maximum optimization
        f"--colors={colors}",
        f"--lossy={lossy}",
        "--no-warnings",
        str(input_path),
        "-o",
        str(actual_output),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: gifsicle failed: {e.stderr.decode()}")
        if output_path != input_path:
            shutil.copy(input_path, output_path)
        return output_path

    # If we used a temp file, move it to the final location.
    if output_path == input_path:
        shutil.move(str(actual_output), str(output_path))

    # Report compression results.
    new_size = output_path.stat().st_size
    reduction = (1 - new_size / original_size) * 100
    orig_str = _format_size(original_size)
    new_str = _format_size(new_size)
    print(f"GIF optimized: {orig_str} -> {new_str} ({reduction:.1f}% reduction)")

    return output_path


def optimize_gif_directory(
    directory: Path,
    colors: int = 256,
    lossy: int = 80,
    recursive: bool = True,
) -> dict[str, tuple[int, int]]:
    """Optimize all GIF files in a directory.

    Args:
        directory: Directory containing GIF files.
        colors: Maximum number of colors (2-256).
        lossy: Lossy compression level (0-200).
        recursive: If True, search subdirectories.

    Returns:
        Dictionary mapping file paths to (original_size, new_size) tuples.
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return {}

    pattern = "**/*.gif" if recursive else "*.gif"
    gif_files = sorted(directory.glob(pattern))

    if not gif_files:
        print(f"No GIF files found in {directory}")
        return {}

    print(f"Found {len(gif_files)} GIF files to optimize...")

    results: dict[str, tuple[int, int]] = {}
    total_original = 0
    total_new = 0

    for i, gif_path in enumerate(gif_files, 1):
        original_size = gif_path.stat().st_size
        print(f"[{i}/{len(gif_files)}] {gif_path.name} ({_format_size(original_size)})")

        optimize_gif(gif_path, colors=colors, lossy=lossy)

        new_size = gif_path.stat().st_size
        results[str(gif_path)] = (original_size, new_size)
        total_original += original_size
        total_new += new_size

    # Summary.
    total_reduction = (
        (1 - total_new / total_original) * 100 if total_original > 0 else 0
    )
    orig_str = _format_size(total_original)
    new_str = _format_size(total_new)
    print(f"\nTotal: {orig_str} -> {new_str} ({total_reduction:.1f}% reduction)")

    return results


def _format_size(size_bytes: int) -> str:
    """Format a file size in human-readable form."""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"
