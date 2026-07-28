#!/bin/bash
# Container entrypoint: init firewall → permanently drop privileges → run the
# agent CLI.
#
# The firewall requires NET_ADMIN / NET_RAW capabilities:
#   docker run --cap-add=NET_ADMIN --cap-add=NET_RAW ...
#
# The container starts as root so the firewall can be installed; before any
# agent code runs, setpriv drops to the unprivileged node user with all
# capability sets cleared and no-new-privs set, so the agent cannot regain
# root or alter the firewall.
set -euo pipefail
IFS=$'\n\t'

# PRPL_AGENT_SKIP_FIREWALL=1 disables the firewall for debugging.
if [ "${PRPL_AGENT_SKIP_FIREWALL:-0}" = "1" ]; then
    echo "entrypoint: PRPL_AGENT_SKIP_FIREWALL=1, skipping firewall init" >&2
else
    if [ "$(id -u)" -ne 0 ]; then
        echo "entrypoint: firewall initialization requires root" >&2
        exit 1
    fi
    /usr/local/bin/init-firewall.sh
fi

unset PRPL_AGENT_FIREWALL_EXTRA_DOMAINS PRPL_AGENT_SKIP_FIREWALL

if [ "$(id -u)" -eq 0 ]; then
    export HOME=/home/node USER=node LOGNAME=node
    exec /usr/bin/setpriv \
        --reuid=node \
        --regid=node \
        --init-groups \
        --bounding-set=-all \
        --inh-caps=-all \
        --ambient-caps=-all \
        --no-new-privs \
        -- "$@"
fi

exec "$@"
