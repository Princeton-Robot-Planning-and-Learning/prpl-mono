#!/usr/bin/env python3
"""Install all dependencies."""

import os
import subprocess
import sys
import time
from pathlib import Path

from generate_topological_order import get_topological_order


def install_package(package_path: Path, verbose: bool = False, editable: bool = True) -> bool:
    """Install a single package quickly with minimal output."""
    if not package_path.exists() or not (package_path / "pyproject.toml").exists():
        return True  # Skip missing packages silently

    try:
        # Install the package
        # Use --no-deps since dependencies are already installed in topological order
        # Use --verbose in CI to diagnose slow installs
        # Use regular install in CI (not editable) for faster builds
        cmd = ["uv", "pip", "install", "--no-deps"]
        if editable:
            cmd.append("-e")
        cmd.append(".[develop]")

        if verbose:
            cmd.insert(3, "--verbose")  # Insert after "install"

        subprocess.run(
            cmd,
            cwd=package_path,
            check=True,
            capture_output=not verbose,  # Show output if verbose
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

    # Enable verbose mode in CI to diagnose slow installs
    # Use regular (non-editable) installs in CI for speed
    is_ci = os.environ.get('CI') == 'true'
    verbose = is_ci
    editable = not is_ci  # Editable locally, regular in CI

    print(f"Installing {len(install_order)} packages...")
    if is_ci:
        print("ℹ️  CI environment detected")
        print("ℹ️  Using regular installs (not editable) for faster builds")
        print("ℹ️  Using --no-deps to skip dependency re-resolution")
    if verbose:
        print("ℹ️  Verbose mode enabled")

    total_start = time.time()
    times = []

    for package_name in install_order:
        package_path = repo_root / package_name
        print(f"Installing {package_name}...", end=" " if not verbose else "\n", flush=True)

        start = time.time()
        if install_package(package_path, verbose=verbose, editable=editable):
            elapsed = time.time() - start
            times.append((package_name, elapsed))
            if not verbose:
                print(f"✅ ({elapsed:.2f}s)")
            else:
                print(f"✅ {package_name} completed in {elapsed:.2f}s\n")
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
