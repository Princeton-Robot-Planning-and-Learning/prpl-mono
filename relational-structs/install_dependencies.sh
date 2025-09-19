#!/usr/bin/env bash
set -e

# Resolve absolute path of this script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# List of projects to install (relative to SCRIPT_DIR)
PROJECTS=(
  "../relational-structs"
  "."  # current project
)

# Loop through and install each one
for proj in "${PROJECTS[@]}"; do
  ABS_PATH="$(realpath "${SCRIPT_DIR}/${proj}")"
  if [[ -d "$ABS_PATH" ]]; then
    echo "Installing $ABS_PATH"
    # Add [develop] only for the current project
    if [[ "$proj" == "." ]]; then
      uv pip install -e "${ABS_PATH}[develop]"
    else
      uv pip install -e "$ABS_PATH"
    fi
  else
    echo "Warning: $ABS_PATH not found, skipping"
  fi
done
