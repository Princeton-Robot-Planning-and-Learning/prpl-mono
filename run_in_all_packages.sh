#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <command string to run in each subdir>"
  exit 2
fi

cmd="$1"

# Check if sequential mode is requested
if [[ "${SEQUENTIAL:-false}" == "true" ]]; then
  echo "▶ Running in SEQUENTIAL mode (set SEQUENTIAL=false for parallel execution)"
  echo ""

  while IFS= read -r -d '' d; do
    if [[ -f "$d/pyproject.toml" ]]; then
      echo "———"
      echo "▶ Running in: $d"
      pushd "$d" >/dev/null
      bash -o pipefail -c "$cmd"
      popd >/dev/null
    else
      echo "⏭ Skipping $d (no pyproject.toml)"
    fi
  done < <(find . -mindepth 1 -maxdepth 1 -type d -print0)

  exit 0
fi

# Parallel mode (default)
# Arrays to track packages and their background jobs
declare -a packages
declare -a pids
declare -a tmpfiles

echo "▶ Launching commands in PARALLEL across all packages (set SEQUENTIAL=true for sequential execution)"
echo ""

# Launch commands in parallel
while IFS= read -r -d '' d; do
  if [[ -f "$d/pyproject.toml" ]]; then
    # Create a temporary file to capture output
    tmpfile=$(mktemp)
    tmpfiles+=("$tmpfile")
    packages+=("$d")

    # Run command in background, capturing output
    (
      echo "———" > "$tmpfile"
      echo "▶ Running in: $d" >> "$tmpfile"
      cd "$d"
      if bash -o pipefail -c "$cmd" >> "$tmpfile" 2>&1; then
        echo "✓ Success: $d" >> "$tmpfile"
        exit 0
      else
        echo "✗ Failed: $d" >> "$tmpfile"
        exit 1
      fi
    ) &
    pids+=($!)
  fi
done < <(find . -mindepth 1 -maxdepth 1 -type d -print0)

echo "▶ Waiting for ${#packages[@]} packages to complete..."
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
for i in "${!packages[@]}"; do
  cat "${tmpfiles[$i]}"
  rm -f "${tmpfiles[$i]}"
done

echo ""
echo "▶ Results:"

# Check if any failed
failed=0
for i in "${!packages[@]}"; do
  if [[ ${exit_codes[$i]} -ne 0 ]]; then
    echo "✗ ${packages[$i]}: FAILED"
    failed=1
  else
    echo "✓ ${packages[$i]}: passed"
  fi
done

if [[ $failed -eq 1 ]]; then
  echo ""
  echo "✗ Some packages failed"
  exit 1
fi

echo ""
echo "✓ All ${#packages[@]} packages passed"
