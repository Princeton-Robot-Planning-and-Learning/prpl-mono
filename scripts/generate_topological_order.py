#!/usr/bin/env python3
"""
Script to generate a topological ordering of Python packages in this repository
based on the dependencies listed in prpl_requirements.txt files.

This script analyzes the dependencies between packages in the PRPL monorepo and
produces a topological ordering that respects dependency constraints - packages
are ordered such that dependencies come before their dependents.
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict, deque


def find_packages(repo_root: Path) -> list[str]:
    """
    Find all packages in the repository by looking for directories with pyproject.toml files.
    
    Args:
        repo_root: Path to the repository root
        
    Returns:
        list of package names (directory names)
    """
    packages = []
    for item in repo_root.iterdir():
        if item.is_dir() and (item / "pyproject.toml").exists():
            packages.append(item.name)
    return sorted(packages)


def parse_prpl_requirements(package_path: Path) -> list[str]:
    """
    Parse prpl_requirements.txt file to extract dependencies on other packages in the monorepo.
    
    Args:
        package_path: Path to the package directory
        
    Returns:
        list of package names that this package depends on
    """
    prpl_requirements_file = package_path / "prpl_requirements.txt"
    dependencies = []
    
    if not prpl_requirements_file.exists():
        return dependencies
    
    try:
        with open(prpl_requirements_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Dependencies are specified as relative paths like ../package-name
                if line.startswith('../'):
                    dependency_name = line[3:]  # Remove '../' prefix
                    dependencies.append(dependency_name)
                    
    except Exception as e:
        print(f"Warning: Could not read {prpl_requirements_file}: {e}", file=sys.stderr)
    
    return dependencies


def build_dependency_graph(repo_root: Path, packages: list[str]) -> dict[str, list[str]]:
    """
    Build a dependency graph by parsing prpl_requirements.txt files.
    
    Args:
        repo_root: Path to the repository root
        packages: list of all package names
        
    Returns:
        dictionary mapping package names to their dependencies
    """
    graph = {}
    
    for package in packages:
        package_path = repo_root / package
        dependencies = parse_prpl_requirements(package_path)
        
        # Filter dependencies to only include packages that exist in the repo
        valid_dependencies = [dep for dep in dependencies if dep in packages]
        
        # Warn about invalid dependencies
        invalid_dependencies = [dep for dep in dependencies if dep not in packages]
        if invalid_dependencies:
            print(f"Warning: Package '{package}' has dependencies on non-existent packages: {invalid_dependencies}", 
                  file=sys.stderr)
        
        graph[package] = valid_dependencies
    
    return graph


def topological_sort(graph: dict[str, list[str]]) -> list[str]:
    """
    Perform topological sorting using Kahn's algorithm.
    
    Args:
        graph: dictionary mapping package names to their dependencies
        
    Returns:
        list of packages in topological order (dependencies before dependents)
        
    Raises:
        ValueError: If there is a circular dependency
    """
    # Calculate in-degrees (number of dependencies)
    in_degree = defaultdict(int)
    all_packages = set(graph.keys())
    
    # Initialize in-degrees
    for package in all_packages:
        in_degree[package] = 0
    
    # Count dependencies
    for package, dependencies in graph.items():
        for dep in dependencies:
            in_degree[package] += 1
    
    # Find packages with no dependencies
    queue = deque([pkg for pkg in all_packages if in_degree[pkg] == 0])
    result = []
    
    while queue:
        current = queue.popleft()
        result.append(current)
        
        # Find all packages that depend on the current package
        for package, dependencies in graph.items():
            if current in dependencies:
                in_degree[package] -= 1
                if in_degree[package] == 0:
                    queue.append(package)
    
    # Check for circular dependencies
    if len(result) != len(all_packages):
        remaining = all_packages - set(result)
        raise ValueError(f"Circular dependency detected among packages: {remaining}")
    
    return result


def print_dependency_info(graph: dict[str, list[str]]):
    """Print detailed dependency information."""
    print("=== Dependency Information ===")
    
    # Packages with no dependencies
    no_deps = [pkg for pkg, deps in graph.items() if not deps]
    if no_deps:
        print(f"\nPackages with no dependencies: {', '.join(sorted(no_deps))}")
    
    # Packages with dependencies
    with_deps = [(pkg, deps) for pkg, deps in graph.items() if deps]
    if with_deps:
        print("\nPackages with dependencies:")
        for pkg, deps in sorted(with_deps):
            print(f"  {pkg} -> {', '.join(sorted(deps))}")
    
    print()


def get_topological_order(repo_root: Path) -> list[str]:
    """
    Get the topological ordering of packages.
    
    Args:
        repo_root: Path to the repository root
        
    Returns:
        List of packages in topological order
        
    Raises:
        ValueError: If there is a circular dependency
    """
    packages = find_packages(repo_root)
    graph = build_dependency_graph(repo_root, packages)
    return topological_sort(graph)


def main():
    """Main function to generate topological ordering."""
    parser = argparse.ArgumentParser(
        description="Generate topological ordering of packages based on dependencies"
    )
    parser.add_argument(
        "--list-only", 
        action="store_true", 
        help="Output only the package names in order, one per line"
    )
    
    args = parser.parse_args()
    
    # Get repository root (directory containing this script)
    repo_root = Path(__file__).parents[1]
    
    try:
        if args.list_only:
            # Just output the package names in order
            ordered_packages = get_topological_order(repo_root)
            for package in ordered_packages:
                print(package)
        else:
            # Full detailed output
            print(f"Analyzing packages in: {repo_root}")
            print()
            
            # Find all packages
            packages = find_packages(repo_root)
            print(f"Found {len(packages)} packages: {', '.join(packages)}")
            print()
            
            # Build dependency graph
            graph = build_dependency_graph(repo_root, packages)
            
            # Print dependency information
            print_dependency_info(graph)

            # Generate topological ordering
            ordered_packages = topological_sort(graph)
            
            print("=== Topological Ordering ===")
            print("Install packages in this order (dependencies first):")
            print()
            
            for i, package in enumerate(ordered_packages, 1):
                dependencies = graph[package]
                if dependencies:
                    dep_str = f" (depends on: {', '.join(sorted(dependencies))})"
                else:
                    dep_str = " (no dependencies)"
                print(f"{i:2d}. {package}{dep_str}")
            
            print()
            print("=== Installation Commands ===")
            print("You can install packages in this order using:")
            print()
            
            for package in ordered_packages:
                package_path = repo_root / package
                
                # Check if package has prpl_requirements.txt
                if (package_path / "prpl_requirements.txt").exists():
                    print(f"cd {package} && uv pip install -r prpl_requirements.txt && uv pip install -e .")
                else:
                    print(f"cd {package} && uv pip install -e .")
                    
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
