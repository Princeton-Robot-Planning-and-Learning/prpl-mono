#!/usr/bin/env bash

# Loop through all subdirectories in the current directory
for dir in */ ; do
    # Ensure it's a directory
    if [ -d "$dir" ]; then
        # Check if run_autoformat.sh exists and is executable
        if [ -f "$dir/run_autoformat.sh" ]; then
            echo "Running autoformat in $dir"
            (cd "$dir" && ./run_autoformat.sh)
        else
            echo "Skipping $dir (no run_autoformat.sh found)"
        fi
    fi
done
