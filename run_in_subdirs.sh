#!/usr/bin/env bash

# Exit if no command provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <command-to-run>"
    echo "Example: $0 'pytest -q'"
    exit 1
fi

# Collect the command to run (supports spaces)
CMD="$*"

# Loop through all subdirectories in the current directory
for dir in */ ; do
    if [ -d "$dir" ]; then
        echo "Running '$CMD' in $dir"
        (cd "$dir" && eval "$CMD")
    fi
done
