"""Isolated Claude configuration for sandboxed agent processes.

Adapted from the robocode project. The goal is that a sandboxed Claude CLI can
authenticate without ever reading or writing the operator's live ``~/.claude``
directory: credentials are copied into a private runtime "broker" directory, and
token refreshes made by sandboxed CLIs persist only in that broker.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_CREDENTIALS = ".credentials.json"
logger = logging.getLogger(__name__)


def get_claude_oauth_token() -> str | None:
    """Resolve the Claude Code OAuth access token.

    Checked in order:
      1. ``CLAUDE_CODE_OAUTH_TOKEN`` env var (works on every platform).
      2. macOS Keychain (service ``"Claude Code-credentials"``, key
         ``claudeAiOauth.accessToken``) populated by ``claude login``.

    Returns ``None`` when neither is available.
    """
    env_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if env_token:
        return env_token
    platform: str = sys.platform
    if platform != "darwin":
        return None
    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-s",
                "Claude Code-credentials",
                "-w",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return None
        creds = json.loads(result.stdout.strip())
        return creds.get("claudeAiOauth", {}).get("accessToken")
    except (subprocess.SubprocessError, json.JSONDecodeError, KeyError):
        return None


def _credential_broker_dir() -> Path:
    """Return the private runtime directory that carries token refreshes."""
    configured = os.environ.get("PRPL_AGENT_CLAUDE_AUTH_DIR")
    path = (
        Path(configured).expanduser()
        if configured
        else Path(tempfile.gettempdir()) / f"prpl-agent-claude-auth-{os.getuid()}"
    )
    if path.is_symlink():
        raise RuntimeError(f"Refusing symlinked Claude auth directory: {path}")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    if info.st_uid != os.getuid() or not stat.S_ISDIR(info.st_mode):
        raise RuntimeError(f"Unsafe Claude auth directory: {path}")
    path.chmod(0o700)
    return path


@contextmanager
def _file_credentials_lock() -> Iterator[None]:
    """Serialize CLI lifetimes that use copied refreshable credentials.

    OAuth refresh tokens rotate. Two independent writable copies can therefore race:
    the first CLI refresh invalidates the token held by the second. A stable
    ``CLAUDE_CODE_OAUTH_TOKEN`` bypasses this fallback entirely and remains parallel.
    Large, operator-supervised sweeps may explicitly accept this risk by setting
    ``PRPL_AGENT_CLAUDE_AUTH_ALLOW_PARALLEL=1``.
    """
    if os.environ.get("PRPL_AGENT_CLAUDE_AUTH_ALLOW_PARALLEL") == "1":
        yield
        return

    lock_path = _credential_broker_dir() / "credentials.lock"
    flags = os.O_CREAT | os.O_RDWR | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(lock_path, flags, 0o600)
    try:
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or not stat.S_ISREG(info.st_mode):
            raise RuntimeError(f"Unsafe Claude auth lock file: {lock_path}")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.warning(
                "Another Claude run is using file-based credentials; waiting to "
                "avoid an OAuth refresh-token race. Set CLAUDE_CODE_OAUTH_TOKEN "
                "to enable safe parallel runs."
            )
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def host_claude_config_dir() -> Path:
    """Return the operator's Claude configuration directory."""
    return Path(
        os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude"))
    ).expanduser()


def _checked_sandbox_subdir(sandbox_dir: Path, name: str) -> Path:
    """Create a real direct child of *sandbox_dir*, rejecting symlink escapes."""
    path = sandbox_dir / name
    if path.is_symlink():
        raise RuntimeError(f"Refusing symlinked Claude state directory: {path}")
    path.mkdir(exist_ok=True)
    if not path.is_dir() or path.resolve().parent != sandbox_dir.resolve():
        raise RuntimeError(f"Claude state directory escaped the sandbox: {path}")
    return path


def sandbox_claude_session_store(sandbox_dir: Path) -> Path:
    """Return the checked, sandbox-local persistent session-store directory."""
    return _checked_sandbox_subdir(sandbox_dir, ".agent_sessions")


def _remove_agent_path(path: Path) -> None:
    """Remove an agent-writable file, symlink, or directory without following links."""
    if path.is_symlink() or not path.is_dir():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path)


def _refresh_broker_from_host() -> Path:
    """Seed or replace the broker when the operator has logged in more recently."""
    broker = _credential_broker_dir() / _CREDENTIALS
    host = host_claude_config_dir() / _CREDENTIALS
    if not host.is_file():
        if broker.is_file():
            return broker
        raise RuntimeError(
            f"No Claude credentials at {host}. Run `claude login` on the host, "
            "or set CLAUDE_CODE_OAUTH_TOKEN."
        )
    host_valid = _valid_claude_credentials(host)
    broker_valid = _valid_claude_credentials(broker)
    if (
        not broker.is_file()
        or (host_valid and not broker_valid)
        or host.stat().st_mtime_ns > broker.stat().st_mtime_ns
    ):
        shutil.copy2(host, broker)
        broker.chmod(0o600)
    return broker


def _valid_claude_credentials(path: Path) -> bool:
    """Return whether *path* has Claude's refreshable OAuth credential shape."""
    try:
        oauth = json.loads(path.read_text(encoding="utf-8"))["claudeAiOauth"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return False
    return all(
        isinstance(oauth.get(key), str) and bool(oauth[key])
        for key in ("accessToken", "refreshToken")
    )


def _persist_broker_credentials(source: Path) -> None:
    """Atomically retain a CLI's rotated credentials for the next process."""
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        source_fd = os.open(source, flags)
    except OSError as err:
        logger.warning(
            "Not persisting unsafe Claude credentials at %s: %s", source, err
        )
        return

    info = os.fstat(source_fd)
    if info.st_uid != os.getuid() or not stat.S_ISREG(info.st_mode):
        os.close(source_fd)
        logger.warning("Not persisting unsafe Claude credentials at %s", source)
        return

    with os.fdopen(source_fd, "rb") as source_file:
        broker = _credential_broker_dir() / _CREDENTIALS
        pending_fd, pending_name = tempfile.mkstemp(
            prefix=f"{_CREDENTIALS}.", suffix=".tmp", dir=broker.parent
        )
        pending = Path(pending_name)
        try:
            with os.fdopen(pending_fd, "wb") as pending_file:
                shutil.copyfileobj(source_file, pending_file)
                os.fchmod(pending_file.fileno(), 0o600)
            pending.replace(broker)
        finally:
            pending.unlink(missing_ok=True)


@contextmanager
def throwaway_claude_config() -> Iterator[Path]:
    """Yield a writable config containing only a copy of host credentials.

    The operator's transcripts, history, memory, settings, and job artifacts never enter
    the sandbox. Token refreshes persist only in a private credentials-only runtime
    broker, not in the operator's live Claude directory.
    """
    with _file_credentials_lock():
        tmp_dir = Path(tempfile.mkdtemp(prefix="prpl-agent-claude-"))
        try:
            config_dir = tmp_dir / ".claude"
            config_dir.mkdir()
            broker = _refresh_broker_from_host()
            copied = config_dir / _CREDENTIALS
            shutil.copy2(broker, copied)
            copied.chmod(0o600)
            try:
                yield config_dir
            finally:
                _persist_broker_credentials(copied)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


@contextmanager
def sandbox_claude_config(sandbox_dir: Path) -> Iterator[Path]:
    """Yield a resumable sandbox-local config without retaining credentials.

    Session state remains under ``.agent_home`` between queries, while the copied
    credential is removed after every CLI process. This prevents either reads from
    or writes to the operator's live config and avoids leaving an authentication
    secret in the sandbox.
    """
    config_dir = _checked_sandbox_subdir(sandbox_dir, ".agent_home")
    if get_claude_oauth_token():
        yield config_dir
        return

    with _file_credentials_lock():
        broker = _refresh_broker_from_host()
        copied = config_dir / _CREDENTIALS
        _remove_agent_path(copied)
        shutil.copy2(broker, copied)
        copied.chmod(0o600)
        try:
            yield config_dir
        finally:
            _persist_broker_credentials(copied)
            _remove_agent_path(copied)
