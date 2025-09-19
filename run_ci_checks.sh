#!/usr/bin/env bash

# Loop through all subdirectories in the current directory
for dir in */ ; do
    # Ensure it's a directory
    if [ -d "$dir" ]; then
        # Check if run_ci_checks.sh exists and is executable
        if [ -f "$dir/run_ci_checks.sh" ]; then
            echo "Running CI checks in $dir"
            (cd "$dir" && ./run_ci_checks.sh)
        else
            echo "Skipping $dir (no run_ci_checks.sh found)"
        fi
    fi
done
