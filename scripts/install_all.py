#!/usr/bin/env python3
"""Install all dependencies."""

import subprocess
import sys
from pathlib import Path

from generate_topological_order import get_topological_order


def install_package(package_path: Path) -> bool:
    """Install a single package quickly with minimal output."""
    if not package_path.exists() or not (package_path / "pyproject.toml").exists():
        return True  # Skip missing packages silently
    
    try:
        # Install the package in development mode
        subprocess.run(
            ["uv", "pip", "install", "-e", ".[develop]"],
            cwd=package_path,
            check=True,
            capture_output=True,
        )
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_path.name}", file=sys.stderr)
        print(f"   Command: {' '.join(e.cmd)}", file=sys.stderr)
        print(f"   Return code: {e.returncode}", file=sys.stderr)
        if e.stdout:
            print(f"   Stdout:\n{e.stdout.decode()}", file=sys.stderr)
        if e.stderr:
            print(f"   Stderr:\n{e.stderr.decode()}", file=sys.stderr)
        return False


def main():
    """Install all packages in the correct order."""
    repo_root = Path(__file__).parents[1]
    install_order = get_topological_order(repo_root)

    print(f"Installing {len(install_order)} packages...")

    # Build list of package paths for single install command
    package_args = []
    for package_name in install_order:
        package_path = repo_root / package_name
        if package_path.exists() and (package_path / "pyproject.toml").exists():
            package_args.extend(["-e", f"{package_path}[develop]"])

    if not package_args:
        print("⚠ No packages to install")
        return

    # Install all packages in one command
    print(f"Packages: {', '.join(install_order)}")
    try:
        subprocess.run(
            ["uv", "pip", "install"] + package_args,
            check=True,
            cwd=repo_root
        )
        print("🎉 All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages", file=sys.stderr)
        print(f"   Return code: {e.returncode}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
