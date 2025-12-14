#!/usr/bin/env python3
"""Install all dependencies."""

import subprocess
import sys
import time
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

    total_start = time.time()
    times = []

    for package_name in install_order:
        package_path = repo_root / package_name
        print(f"Installing {package_name}...", end=" ", flush=True)

        start = time.time()
        if install_package(package_path):
            elapsed = time.time() - start
            times.append((package_name, elapsed))
            print(f"✅ ({elapsed:.2f}s)")
        else:
            print("❌")
            sys.exit(1)

    total_elapsed = time.time() - total_start
    print(f"\n🎉 All packages installed successfully in {total_elapsed:.2f}s")

    # Show timing breakdown
    print("\nTiming breakdown:")
    for pkg, t in sorted(times, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {pkg}: {t:.2f}s")

    overhead = sum(t for _, t in times)
    print(f"\nTotal package install time: {overhead:.2f}s")
    print(f"Overhead (subprocess, etc.): {total_elapsed - overhead:.2f}s")


if __name__ == "__main__":
    main()
