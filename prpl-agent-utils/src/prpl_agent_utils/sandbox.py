"""Sandbox directory setup and Docker launch helpers.

Adapted from the robocode project. A sandbox is a persistent host directory that an
agent works in. It is a git repository so the agent's changes are recorded, and it
carries the agent's session state (``.agent_sessions``, ``.agent_home``) so a
conversation can be resumed across processes and containers.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import uuid
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from prpl_agent_utils.claude_auth import (
    get_claude_oauth_token,
    sandbox_claude_session_store,
    throwaway_claude_config,
)

logger = logging.getLogger(__name__)

# Default Docker image name; built by docker/build.sh.
DEFAULT_DOCKER_IMAGE = "prpl-agent-sandbox"

_SANDBOX_GITIGNORE = """\
__pycache__/
*.pyc
.claude/
.agent_sessions/
.agent_home/
.agent_logs/
"""

# PreToolUse hook that blocks Write/Edit outside the sandbox directory. This is
# the write barrier for the local (non-Docker) sandbox; Docker enforces isolation
# at the filesystem level regardless.
_VALIDATE_SANDBOX_SCRIPT = """\
#!/usr/bin/env python3
import json
import os
import sys

data = json.load(sys.stdin)
tool_name = data.get("tool_name", "")
tool_input = data.get("tool_input", {})

if tool_name not in ("Write", "Edit"):
    sys.exit(0)

file_path = tool_input.get("file_path", "")
if not file_path:
    sys.exit(0)

sandbox = os.path.realpath(os.getcwd())
resolved = os.path.realpath(file_path)

if resolved == sandbox or resolved.startswith(sandbox + os.sep):
    sys.exit(0)

json.dump({
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": (
            f"Blocked: {file_path} resolves outside the sandbox directory"
        ),
    }
}, sys.stdout)
"""

_SANDBOX_SETTINGS: dict = {
    "hooks": {
        "PreToolUse": [
            {
                "matcher": "Write|Edit",
                "hooks": [
                    {
                        "type": "command",
                        "command": "python3 .claude/validate_sandbox.py",
                    }
                ],
            }
        ],
    }
}


def setup_sandbox_dir(sandbox_dir: Path, init_files: dict[str, Path]) -> None:
    """Idempotently populate the sandbox: init files, git repo, write hook.

    ``init_files`` maps destination names (relative to the sandbox) to source paths
    on the host; each is copied in once. The directory is git-initialized so the
    agent CLI treats it as the project root and its changes are recorded.
    """
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    for dest_name, source_path in init_files.items():
        dest = sandbox_dir / dest_name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, dest)

    claude_dir = sandbox_dir / ".claude"
    claude_dir.mkdir(exist_ok=True)
    (claude_dir / "settings.json").write_text(
        json.dumps(_SANDBOX_SETTINGS, indent=2) + "\n"
    )
    (claude_dir / "validate_sandbox.py").write_text(_VALIDATE_SANDBOX_SCRIPT)

    gitignore = sandbox_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(_SANDBOX_GITIGNORE)

    if not (sandbox_dir / ".git" / "HEAD").exists():
        subprocess.run(
            ["git", "init"],
            cwd=str(sandbox_dir),
            check=True,
            capture_output=True,
        )
        for key, val in [
            ("user.email", "agent@prpl-agent-utils"),
            ("user.name", "agent"),
        ]:
            subprocess.run(
                ["git", "config", key, val],
                cwd=str(sandbox_dir),
                capture_output=True,
                check=False,
            )
        commit_sandbox_changes(sandbox_dir, "initial sandbox setup")


def commit_sandbox_changes(sandbox_dir: Path, message: str) -> None:
    """Commit any uncommitted changes in the sandbox so nothing is lost."""
    sandbox = str(sandbox_dir)
    subprocess.run(["git", "add", "-A"], cwd=sandbox, capture_output=True, check=False)
    # --porcelain check avoids a no-op commit.
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=sandbox,
        capture_output=True,
        text=True,
        check=False,
    )
    if status.stdout.strip():
        subprocess.run(
            [
                "git",
                "commit",
                "-m",
                message,
                "--author",
                "agent <noreply@prpl-agent-utils>",
            ],
            cwd=sandbox,
            capture_output=True,
            check=False,
        )


@contextmanager
def docker_claude_auth() -> Iterator[tuple[list[str], dict[str, str]]]:
    """Yield Docker CLI args and env vars for Claude authentication.

    Prefers the OAuth token (env var or macOS Keychain); falls back to bind-mounting
    a throwaway, credentials-only copy of the operator's Claude config, so container
    writes and token refreshes never touch the live ``~/.claude``.
    """
    docker_args: list[str] = []
    extra_env: dict[str, str] = {}
    with ExitStack() as stack:
        oauth_token = get_claude_oauth_token()
        if oauth_token:
            docker_args += ["-e", "CLAUDE_CODE_OAUTH_TOKEN"]
            extra_env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token
        else:
            logger.warning(
                "No Claude OAuth token found (env var or Keychain); falling "
                "back to a throwaway credentials-only config. Run `claude "
                "login` on the host if the container cannot authenticate."
            )
            claude_copy = stack.enter_context(throwaway_claude_config())
            docker_args += ["-v", f"{claude_copy}:/home/node/.claude"]
        yield docker_args, extra_env


def build_docker_run_cmd(
    sandbox_dir: Path,
    image: str,
    auth_args: list[str],
    env_vars: dict[str, str],
    extra_network_domains: tuple[str, ...] = (),
) -> list[str]:
    """Build the ``docker run`` prefix for one sandboxed agent query.

    The sandbox directory is bind-mounted writable at ``/sandbox`` (the working
    directory); everything else in the container is untouchable by the agent. The
    NET_ADMIN/NET_RAW capabilities are required by the firewall the entrypoint
    installs. The CLI session store is bind-mounted from the sandbox so a later
    query can ``--continue`` the conversation in a fresh container.
    """
    container_name = f"prpl-agent-sandbox-{uuid.uuid4().hex[:8]}"
    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--cap-add=NET_ADMIN",
        "--cap-add=NET_RAW",
    ]
    for key, val in env_vars.items():
        cmd += ["-e", f"{key}={val}"]
    if extra_network_domains:
        cmd += [
            "-e",
            f"PRPL_AGENT_FIREWALL_EXTRA_DOMAINS={','.join(extra_network_domains)}",
        ]
    cmd += auth_args
    sessions_dir = sandbox_claude_session_store(sandbox_dir)
    cmd += [
        "-v",
        f"{sandbox_dir.resolve()}:/sandbox",
        "-v",
        f"{sessions_dir.resolve()}:/home/node/.claude/projects",
        "-w",
        "/sandbox",
        image,
    ]
    return cmd


def agent_subprocess_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build a clean subprocess environment, stripping ``CLAUDECODE*`` vars.

    The strip matters when the calling process itself runs under Claude Code; the nested
    CLI otherwise refuses to start or inherits the outer session.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDECODE")}
    if extra:
        env.update(extra)
    return env
