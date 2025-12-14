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

# Check if sequential mode is requested
if [[ "${SEQUENTIAL:-false}" == "true" ]]; then
  echo "▶ Running in SEQUENTIAL mode (set SEQUENTIAL=false for parallel execution)"
  echo ""

  # Run command in each affected package sequentially
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
  exit 0
fi

# Parallel mode (default)
echo "▶ Launching commands in PARALLEL (set SEQUENTIAL=true for sequential execution)"
echo ""

# Arrays to track background jobs
declare -a pids
declare -a tmpfiles
declare -a valid_packages

# Launch commands in parallel
for package in "${packages[@]}"; do
  package_dir="./$package"

  if [[ -f "$package_dir/pyproject.toml" ]]; then
    # Create a temporary file to capture output
    tmpfile=$(mktemp)
    tmpfiles+=("$tmpfile")
    valid_packages+=("$package_dir")

    # Run command in background, capturing output
    (
      echo "———" > "$tmpfile"
      echo "▶ Running in: $package_dir" >> "$tmpfile"
      cd "$package_dir"
      if bash -o pipefail -c "$cmd" >> "$tmpfile" 2>&1; then
        echo "✓ Success: $package_dir" >> "$tmpfile"
        exit 0
      else
        echo "✗ Failed: $package_dir" >> "$tmpfile"
        exit 1
      fi
    ) &
    pids+=($!)
  else
    echo "⏭ Skipping $package_dir (no pyproject.toml)"
  fi
done

echo "▶ Waiting for ${#valid_packages[@]} packages to complete..."
echo ""

# Wait for all background jobs and collect exit codes
declare -a exit_codes
for pid in "${pids[@]}"; do
  set +e  # Temporarily disable exit on error
  wait "$pid"
  exit_codes+=($?)
  set -e
done

# Print all outputs in order
for i in "${!valid_packages[@]}"; do
  cat "${tmpfiles[$i]}"
  rm -f "${tmpfiles[$i]}"
done

echo ""
echo "▶ Results:"

# Check if any failed
failed=0
for i in "${!valid_packages[@]}"; do
  if [[ ${exit_codes[$i]} -ne 0 ]]; then
    echo "✗ ${valid_packages[$i]}: FAILED"
    failed=1
  else
    echo "✓ ${valid_packages[$i]}: passed"
  fi
done

if [[ $failed -eq 1 ]]; then
  echo ""
  echo "✗ Some packages failed"
  exit 1
fi

echo ""
echo "✓ All ${#valid_packages[@]} affected packages passed"
