#!/bin/bash
# Rebuild the visualizer frontend bundle and stage it for commit.
#
# The built bundle at
# src/bilevel_planning/visualizer/frontend/dist/ is committed to the repo
# so that users can run `python -m bilevel_planning.visualizer` without
# installing Node/npm. Maintainers run this script after changing any
# frontend source, then commit the updated dist/.
#
# Requires Node.js 18+ and npm.
set -e

FRONTEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src/bilevel_planning/visualizer/frontend"
cd "$FRONTEND_DIR"

npm ci
npm run build

echo
echo "Built $FRONTEND_DIR/dist"
echo "Commit the updated dist/ so users don't need to build it themselves."
