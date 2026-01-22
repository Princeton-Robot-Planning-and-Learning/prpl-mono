#!/usr/bin/env python3
"""
Script to determine which packages need testing based on changed files and dependencies.

This script analyzes git changes and uses the dependency graph to find all packages
that need testing - both changed packages and any packages that depend on them.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from collections import defaultdict

# Reuse functions from generate_topological_order.py
sys.path.insert(0, str(Path(__file__).parent))
from generate_topological_order import (
    find_packages,
    build_dependency_graph,
    parse_prpl_requirements,
)


def build_reverse_dependencies(graph: dict[str, list[str]]) -> dict[str, set[str]]:
    """Build reverse dependency graph (who depends on each package)."""
    reverse = defaultdict(set)
    for package, dependencies in graph.items():
        for dep in dependencies:
            reverse[dep].add(package)
    return reverse


def find_all_dependents(package: str, reverse_graph: dict[str, set[str]]) -> set[str]:
    """Find all packages that transitively depend on the given package."""
    dependents = set()
    to_visit = [package]

    while to_visit:
        current = to_visit.pop()
        for dependent in reverse_graph.get(current, set()):
            if dependent not in dependents:
                dependents.add(dependent)
                to_visit.append(dependent)

    return dependents


def is_infrastructure_file(file_path: str) -> bool:
    """Check if a file is infrastructure that affects all packages."""
    if file_path.startswith('.github/'):
        return True
    if file_path.startswith('scripts/'):
        return True
    if file_path.startswith('run_') and file_path.endswith('.sh'):
        return True
    return False


def file_to_package(file_path: str, packages: list[str]) -> str | None:
    """Map a file path to its package, or None if not in a package."""
    parts = Path(file_path).parts
    if len(parts) > 0 and parts[0] in packages:
        return parts[0]
    return None


def get_changed_files(base_ref: str) -> list[str]:
    """Get list of changed files using git diff."""
    try:
        result = subprocess.run(
            ['git', 'diff', '--name-only', f'{base_ref}..HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        return files
    except subprocess.CalledProcessError as e:
        print(f"Error running git diff: {e}", file=sys.stderr)
        return []


def main():
    """Main function to determine affected packages."""
    parser = argparse.ArgumentParser(
        description="Determine which packages need testing based on changed files"
    )
    parser.add_argument(
        'base_ref',
        nargs='?',
        help='Base git ref to compare against (e.g., commit SHA or branch name)'
    )

    args = parser.parse_args()

    # Determine base ref: from args, environment, or default
    base_ref = args.base_ref or os.environ.get('CI_BASE_SHA')

    if not base_ref:
        print(f"Error: No base ref provided", file=sys.stderr)
        sys.exit(1)

    # Get repository root
    repo_root = Path(__file__).parents[1]

    # Find all packages
    try:
        packages = find_packages(repo_root)
    except Exception as e:
        print(f"Error finding packages: {e}", file=sys.stderr)
        sys.exit(1)

    # Get changed files
    changed_files = get_changed_files(base_ref)

    # Handle empty changed files - safety fallback
    if not changed_files:
        print(' '.join(packages))
        sys.exit(0)

    # Check if any infrastructure files changed
    infrastructure_changed = any(is_infrastructure_file(f) for f in changed_files)

    if infrastructure_changed:
        # Infrastructure changes affect all packages
        print(' '.join(packages))
        sys.exit(0)

    # Map changed files to packages
    changed_packages = set()
    files_outside_packages = False

    for file_path in changed_files:
        pkg = file_to_package(file_path, packages)
        if pkg:
            changed_packages.add(pkg)
        else:
            # File changed outside any package
            files_outside_packages = True

    # If files changed outside packages, run all tests as safety measure
    if files_outside_packages:
        print(' '.join(packages))
        sys.exit(0)

    # If no packages changed (shouldn't happen but handle it), run all
    if not changed_packages:
        print(' '.join(packages))
        sys.exit(0)

    # Build dependency graph
    try:
        graph = build_dependency_graph(repo_root, packages)
        reverse_graph = build_reverse_dependencies(graph)
    except Exception as e:
        print(f"Error building dependency graph: {e}", file=sys.stderr)
        # Fall back to all packages on error
        print(' '.join(packages))
        sys.exit(0)

    # Find all affected packages: changed packages + their dependents
    affected_packages = set(changed_packages)

    for pkg in changed_packages:
        dependents = find_all_dependents(pkg, reverse_graph)
        affected_packages.update(dependents)

    # Output affected packages as space-separated list
    print(' '.join(sorted(affected_packages)))


if __name__ == "__main__":
    main()
