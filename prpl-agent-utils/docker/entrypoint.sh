#!/bin/bash
# Container entrypoint: init firewall → run the agent CLI.
#
# The firewall requires NET_ADMIN / NET_RAW capabilities:
#   docker run --cap-add=NET_ADMIN --cap-add=NET_RAW ...
#
# The `node` user has passwordless sudo for init-firewall.sh only
# (configured in the Dockerfile via /etc/sudoers.d/node-firewall).
set -e

# Pass PRPL_AGENT_FIREWALL_EXTRA_DOMAINS through sudo (sudo strips env by
# default). PRPL_AGENT_SKIP_FIREWALL=1 disables the firewall for debugging.
if [ "${PRPL_AGENT_SKIP_FIREWALL:-0}" = "1" ]; then
    echo "entrypoint: PRPL_AGENT_SKIP_FIREWALL=1, skipping firewall init" >&2
else
    sudo PRPL_AGENT_FIREWALL_EXTRA_DOMAINS="${PRPL_AGENT_FIREWALL_EXTRA_DOMAINS:-}" \
        /usr/local/bin/init-firewall.sh
fi

exec "$@"
