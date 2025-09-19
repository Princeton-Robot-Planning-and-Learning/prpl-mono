#!/usr/bin/env bash
set -e

# Resolve absolute path of this script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# This package does not have any local dependencies, so we can just directly install.
uv pip install -e "${SCRIPT_DIR}[develop]"
