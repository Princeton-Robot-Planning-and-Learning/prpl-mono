#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <command string to run in each affected package>"
  exit 2
fi

cmd="$1"

# Check if this is a PR context (CI_BASE_SHA is set)
if [[ -z "${CI_BASE_SHA:-}" ]]; then
  echo "▶ No CI_BASE_SHA set - running all packages (not a PR)"
  exec ./run_in_all_packages.sh "$cmd"
fi

# Get affected packages using the Python script
echo "▶ Detecting affected packages (base: $CI_BASE_SHA)..."
affected_packages=$(python scripts/get_affected_packages.py "$CI_BASE_SHA" 2>&1)

# Check if the script succeeded
if [[ $? -ne 0 ]]; then
  echo "⚠ Error detecting affected packages, falling back to all packages"
  echo "$affected_packages" >&2
  exec ./run_in_all_packages.sh "$cmd"
fi

# Check if output is empty
if [[ -z "$affected_packages" ]]; then
  echo "⚠ No affected packages detected, falling back to all packages"
  exec ./run_in_all_packages.sh "$cmd"
fi

# Convert space-separated list to array
IFS=' ' read -r -a packages <<< "$affected_packages"

echo "▶ Affected packages (${#packages[@]}): ${packages[*]}"
echo ""

# Run command in each affected package
for package in "${packages[@]}"; do
  package_dir="./$package"

  if [[ -f "$package_dir/pyproject.toml" ]]; then
    echo "———"
    echo "▶ Running in: $package_dir"
    pushd "$package_dir" >/dev/null
    bash -o pipefail -c "$cmd"
    popd >/dev/null
  else
    echo "⏭ Skipping $package_dir (no pyproject.toml)"
  fi
done

echo ""
echo "✓ Completed running in ${#packages[@]} affected packages"
