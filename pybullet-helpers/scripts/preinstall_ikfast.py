#!/usr/bin/env python3
"""Pre-install all IKFast modules to avoid race conditions in parallel tests.

This script discovers all IKFast modules by scanning the third_party/ikfast directory
for subdirectories containing setup.py files, then installs them.
"""

import logging
from pathlib import Path

from pybullet_helpers.ikfast.load import install_ikfast_module
from pybullet_helpers.utils import get_third_party_path


def discover_ikfast_modules() -> list[Path]:
    """Auto-discover all IKFast module directories.

    Returns:
        List of paths to IKFast module directories (those containing setup.py).
    """
    ikfast_base = get_third_party_path() / "ikfast"

    if not ikfast_base.exists():
        return []

    module_dirs = []
    for subdir in ikfast_base.iterdir():
        if subdir.is_dir() and (subdir / "setup.py").exists():
            module_dirs.append(subdir)

    return sorted(module_dirs)


def main() -> None:
    """Pre-install all discovered IKFast modules."""
    logging.basicConfig(level=logging.INFO)

    module_dirs = discover_ikfast_modules()

    if not module_dirs:
        logging.info("No IKFast modules found to pre-install")
        return

    logging.info(f"Discovered {len(module_dirs)} IKFast module(s) to pre-install:")
    for module_dir in module_dirs:
        logging.info(f"  - {module_dir.name}")

    print()

    for module_dir in module_dirs:
        logging.info(f"Installing IKFast module: {module_dir.name}")
        try:
            install_ikfast_module(module_dir)
            logging.info(f"✓ Successfully installed {module_dir.name}")
        except Exception as e:
            logging.error(f"✗ Failed to install {module_dir.name}: {e}")
            raise

    print()
    logging.info(
        f"✓ All {len(module_dirs)} IKFast module(s) pre-installed successfully"
    )


if __name__ == "__main__":
    main()
