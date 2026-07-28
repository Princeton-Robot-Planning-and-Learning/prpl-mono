#!/usr/bin/env bash
# Build the prpl-agent-sandbox Docker image.
#
# Run from anywhere:
#   bash docker/build.sh
set -euo pipefail

DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Remap the container user's UID/GID to the host's only on Linux, where
# bind-mount ownership is passed through literally. Docker Desktop on macOS
# maps ownership via VirtioFS, and the host GID (20, "staff") collides with
# an existing group in the image.
UID_ARGS=()
if [[ "$(uname)" == "Linux" ]]; then
    UID_ARGS=(--build-arg "USER_UID=$(id -u)" --build-arg "USER_GID=$(id -g)")
fi

echo "Building prpl-agent-sandbox from ${DOCKER_DIR} ..."
docker build \
    --tag prpl-agent-sandbox \
    --file "${DOCKER_DIR}/Dockerfile" \
    ${UID_ARGS[@]+"${UID_ARGS[@]}"} \
    "${DOCKER_DIR}"
echo "Done. Image tagged: prpl-agent-sandbox"
