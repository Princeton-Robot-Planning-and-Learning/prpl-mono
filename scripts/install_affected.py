#!/usr/bin/env python3
"""
Install only packages affected by changes, plus their dependencies.

This script determines which packages need to be installed based on:
1. Packages that changed
2. Packages that depend on changed packages
3. All dependencies of the above packages

This significantly speeds up CI installation for PRs.
"""

import subprocess
import sys
import os
from pathlib import Path

from generate_topological_order import (
    find_packages,
    build_dependency_graph,
    get_topological_order,
)
from install_all import install_package


def get_all_dependencies(package: str, graph: dict[str, list[str]]) -> set[str]:
    """Get all transitive dependencies of a package."""
    dependencies = set()
    to_visit = list(graph.get(package, []))

    while to_visit:
        dep = to_visit.pop()
        if dep not in dependencies:
            dependencies.add(dep)
            to_visit.extend(graph.get(dep, []))

    return dependencies


def main():
    """Install only affected packages and their dependencies."""
    repo_root = Path(__file__).parents[1]

    # Check if we should do selective installation
    base_sha = os.environ.get('CI_BASE_SHA')

    if not base_sha:
        print("▶ No CI_BASE_SHA set - installing all packages")
        # Fall back to install_all.py
        from install_all import main as install_all_main
        install_all_main()
        return

    print(f"▶ Detecting affected packages (base: {base_sha})...")

    # Get affected packages by calling get_affected_packages.py
    try:
        result = subprocess.run(
            [sys.executable, 'scripts/get_affected_packages.py', base_sha],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root
        )
        affected_packages = result.stdout.strip().split()
    except subprocess.CalledProcessError as e:
        print(f"⚠ Error detecting affected packages: {e}", file=sys.stderr)
        print("⚠ Falling back to installing all packages", file=sys.stderr)
        from install_all import main as install_all_main
        install_all_main()
        return

    if not affected_packages:
        print("⚠ No affected packages detected, falling back to all packages")
        from install_all import main as install_all_main
        install_all_main()
        return

    # Get all packages and dependency graph
    all_packages = find_packages(repo_root)
    graph = build_dependency_graph(repo_root, all_packages)

    # Check if affected packages equals all packages (infrastructure change)
    if set(affected_packages) == set(all_packages):
        print("▶ Infrastructure change detected - installing all packages")
        from install_all import main as install_all_main
        install_all_main()
        return

    # Build installation set: affected packages + all their dependencies
    install_set = set(affected_packages)

    for pkg in affected_packages:
        deps = get_all_dependencies(pkg, graph)
        install_set.update(deps)

    # Get topological order and filter to installation set
    full_order = get_topological_order(repo_root)
    install_order = [pkg for pkg in full_order if pkg in install_set]

    print(f"▶ Installing {len(install_order)}/{len(all_packages)} packages:")
    print(f"  Affected: {', '.join(sorted(affected_packages))}")
    print(f"  Dependencies: {', '.join(sorted(install_set - set(affected_packages)))}")
    print()

    # Install packages in topological order
    for package_name in install_order:
        package_path = repo_root / package_name
        print(f"Installing {package_name}...", end=" ", flush=True)

        if install_package(package_path):
            print("✅")
        else:
            print("❌")
            sys.exit(1)

    skipped_count = len(all_packages) - len(install_order)
    print(f"🎉 Installed {len(install_order)} packages (skipped {skipped_count})")


if __name__ == "__main__":
    main()
