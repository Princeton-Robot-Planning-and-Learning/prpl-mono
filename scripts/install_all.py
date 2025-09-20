#!/usr/bin/env python3
"""
Install all packages in the PRPL monorepo in topological order.

This script uses the topological ordering from generate_topological_order.py
to install packages in the correct dependency order, ensuring that all
dependencies are installed before their dependents.
"""

import sys
import subprocess
import argparse
from pathlib import Path
from typing import List

# Import the topological ordering function from the same directory
from generate_topological_order import get_topological_order


def run_command(command: List[str], cwd: Path, description: str) -> bool:
    """
    Run a command and return whether it succeeded.
    
    Args:
        command: List of command parts
        cwd: Working directory to run the command in
        description: Description of what the command does for logging
        
    Returns:
        True if the command succeeded, False otherwise
    """
    try:
        print(f"    Running: {' '.join(command)}")
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            # Indent the output for better readability
            for line in result.stdout.strip().split('\n'):
                print(f"      {line}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"    ❌ Error in {description}:")
        print(f"      Command: {' '.join(command)}")
        print(f"      Return code: {e.returncode}")
        if e.stdout:
            print(f"      Stdout: {e.stdout}")
        if e.stderr:
            print(f"      Stderr: {e.stderr}")
        return False


def install_package(package_path: Path, package_name: str, dry_run: bool = False) -> bool:
    """
    Install a single package.
    
    Args:
        package_path: Path to the package directory
        package_name: Name of the package
        dry_run: If True, just show what would be done without executing
        
    Returns:
        True if installation succeeded, False otherwise
    """
    print(f"▶ Installing package: {package_name}")
    
    if not package_path.exists():
        print(f"  ⚠️  Warning: Package directory '{package_path}' not found, skipping")
        return False
    
    if not (package_path / "pyproject.toml").exists():
        print(f"  ⚠️  Warning: No pyproject.toml found in '{package_path}', skipping")
        return False
    
    success = True
    
    # Install prpl requirements if they exist
    prpl_requirements = package_path / "prpl_requirements.txt"
    if prpl_requirements.exists():
        print(f"  📦 Installing prpl requirements...")
        if dry_run:
            print(f"    [DRY RUN] Would run: uv pip install -r prpl_requirements.txt")
        else:
            if not run_command(
                ["uv", "pip", "install", "-r", "prpl_requirements.txt"],
                package_path,
                "installing prpl requirements"
            ):
                success = False
    else:
        print(f"  📦 No prpl_requirements.txt found")
    
    # Install the package in development mode
    print(f"  🔧 Installing package in development mode...")
    if dry_run:
        print(f"    [DRY RUN] Would run: uv pip install -e .[develop]")
    else:
        if not run_command(
            ["uv", "pip", "install", "-e", ".[develop]"],
            package_path,
            "installing package in development mode"
        ):
            success = False
    
    if success:
        print(f"  ✅ Successfully installed {package_name}")
    else:
        print(f"  ❌ Failed to install {package_name}")
    
    return success


def main():
    """Main function to install all packages."""
    parser = argparse.ArgumentParser(
        description="Install all packages in topological order"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually installing"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue installing other packages even if one fails"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output from installation commands"
    )
    
    args = parser.parse_args()
    
    # Get repository root (parent of scripts directory)
    repo_root = Path(__file__).parents[1]
    
    print(f"Installing packages from: {repo_root}")
    if args.dry_run:
        print("🔍 DRY RUN MODE - No actual installation will be performed")
    print()
    
    try:
        # Get the topological ordering
        ordered_packages = get_topological_order(repo_root)
        
        print(f"Found {len(ordered_packages)} packages to install in this order:")
        for i, package in enumerate(ordered_packages, 1):
            print(f"  {i:2d}. {package}")
        print()
        
        failed_packages = []
        
        # Install each package in order
        for package in ordered_packages:
            print("─" * 60)
            package_path = repo_root / package
            
            success = install_package(package_path, package, dry_run=args.dry_run)
            
            if not success:
                failed_packages.append(package)
                if not args.continue_on_error:
                    print(f"\n❌ Installation failed for {package}. Stopping here.")
                    print("Use --continue-on-error to continue with remaining packages.")
                    sys.exit(1)
        
        print("─" * 60)
        
        if failed_packages:
            print(f"⚠️  Some packages failed to install: {', '.join(failed_packages)}")
            if not args.dry_run:
                sys.exit(1)
        else:
            if args.dry_run:
                print("🎉 Dry run complete - all packages would be installed successfully!")
            else:
                print("🎉 All packages installed successfully!")
                
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Installation interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
