# prpl-agent-sandbox — generic Docker image for running a coding agent in
# isolation.
#
# Based on the official Claude Code devcontainer:
#   https://github.com/anthropics/claude-code/tree/main/.devcontainer
# (via the robocode project), with Python 3.11 + uv added on top of the
# official node:20 base. Unlike robocode's image, nothing project-specific is
# baked in: the agent's entire world is the bind-mounted /sandbox directory.
#
# Build:
#   bash docker/build.sh
#
# Runtime bind-mounts (see sandbox.py):
#   -v <sandbox_dir>:/sandbox                        writable working directory
#   -v <sandbox_dir>/.agent_sessions:/home/node/.claude/projects
#                                                    persistent CLI sessions

FROM node:20

ARG TZ
ENV TZ="$TZ"

ARG CLAUDE_CODE_VERSION=latest

# Remap the node user's UID/GID to match the host's at build time so that
# bind-mounted host files are readable/writable inside the container without
# needing host UID 1000. build.sh passes --build-arg USER_UID=$(id -u) etc.
ARG USER_UID=1000
ARG USER_GID=1000
RUN if [ "$USER_GID" != "1000" ] || [ "$USER_UID" != "1000" ]; then \
        groupmod --gid "$USER_GID" node \
        && usermod --uid "$USER_UID" --gid "$USER_GID" node; \
    fi

# System packages: basic dev tools plus what the firewall script needs
# (jq, dnsutils, aggregate, iptables, ipset, iproute2), util-linux for the
# setpriv root-to-node privilege transition, and a Python runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        util-linux \
        jq \
        iptables \
        ipset \
        iproute2 \
        dnsutils \
        aggregate \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# uv — fast Python package manager, so the agent can set up venvs in /sandbox.
RUN curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR=/usr/local/bin sh

# npm-global directory (mirrors official devcontainer setup).
RUN mkdir -p /usr/local/share/npm-global \
    && chown -R node:node /usr/local/share

ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH="$PATH:/usr/local/share/npm-global/bin"

# Claude Code CLI.
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

# Firewall script — the root entrypoint runs this before permanently dropping
# to the unprivileged node user.
COPY init-firewall.sh /usr/local/bin/init-firewall.sh
RUN chmod 0755 /usr/local/bin/init-firewall.sh

# Config directories for the node user.
RUN mkdir -p /home/node/.claude \
    && chown -R node:node /home/node/.claude

# Entrypoint: init firewall → drop privileges → exec the agent CLI.
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV DEVCONTAINER=true
ENV CLAUDE_CONFIG_DIR=/home/node/.claude
ENV NODE_OPTIONS="--max-old-space-size=4096"

# The entrypoint needs NET_ADMIN while it initializes iptables. It permanently
# drops to node (including all capability sets) before executing agent code.
USER root
WORKDIR /sandbox

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
