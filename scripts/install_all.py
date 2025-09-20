#!/usr/bin/env python3
"""
Simplified install script for the PRPL monorepo.
Optimized for speed, especially in CI environments.
"""

import subprocess
import sys
from pathlib import Path

# Fixed installation order based on dependencies
# This avoids the overhead of dynamic dependency resolution
INSTALL_ORDER = [
    "prpl-utils",
    "toms-geoms-2d", 
    "pybullet-helpers",
    "relational-structs",
    "prpl-llm-utils",
    "prpl-perception-utils",
    "bilevel-planning",
    "prbench",
    "prbench-models",
]


def install_package(package_path: Path) -> bool:
    """Install a single package quickly with minimal output."""
    if not package_path.exists() or not (package_path / "pyproject.toml").exists():
        return True  # Skip missing packages silently
    
    try:
        # Install prpl requirements if they exist
        prpl_requirements = package_path / "prpl_requirements.txt"
        if prpl_requirements.exists():
            subprocess.run(
                ["uv", "pip", "install", "-r", "prpl_requirements.txt"],
                cwd=package_path,
                check=True,
                capture_output=True,
            )
        
        # Install the package in development mode
        subprocess.run(
            ["uv", "pip", "install", "-e", ".[develop]"],
            cwd=package_path,
            check=True,
            capture_output=True,
        )
        return True
        
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package_path.name}", file=sys.stderr)
        return False


def main():
    """Install all packages in the correct order."""
    repo_root = Path(__file__).parents[1]
    
    print(f"Installing {len(INSTALL_ORDER)} packages...")
    
    for package_name in INSTALL_ORDER:
        package_path = repo_root / package_name
        print(f"Installing {package_name}...", end=" ", flush=True)
        
        if install_package(package_path):
            print("✅")
        else:
            print("❌")
            sys.exit(1)
    
    print("🎉 All packages installed successfully!")


if __name__ == "__main__":
    main()
