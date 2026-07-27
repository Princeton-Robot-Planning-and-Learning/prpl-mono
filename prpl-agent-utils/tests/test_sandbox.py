"""Tests for sandbox.py."""

import subprocess

from prpl_agent_utils.sandbox import (
    agent_subprocess_env,
    build_docker_run_cmd,
    commit_sandbox_changes,
    setup_sandbox_dir,
)


def test_setup_sandbox_dir(tmp_path):
    """The sandbox gets init files, a git repo, and the write-validation hook."""
    source = tmp_path / "source.txt"
    source.write_text("hello")
    sandbox_dir = tmp_path / "sandbox"
    setup_sandbox_dir(sandbox_dir, {"nested/dest.txt": source})

    assert (sandbox_dir / "nested" / "dest.txt").read_text() == "hello"
    assert (sandbox_dir / ".git" / "HEAD").exists()
    assert (sandbox_dir / ".gitignore").exists()
    assert (sandbox_dir / ".claude" / "settings.json").exists()
    assert (sandbox_dir / ".claude" / "validate_sandbox.py").exists()

    # The initial state is committed.
    log = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=sandbox_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "initial sandbox setup" in log.stdout

    # Setup is idempotent and does not overwrite existing init files.
    (sandbox_dir / "nested" / "dest.txt").write_text("edited")
    setup_sandbox_dir(sandbox_dir, {"nested/dest.txt": source})
    assert (sandbox_dir / "nested" / "dest.txt").read_text() == "edited"


def test_commit_sandbox_changes(tmp_path):
    """Uncommitted sandbox changes are committed with the given message."""
    sandbox_dir = tmp_path / "sandbox"
    setup_sandbox_dir(sandbox_dir, {})
    (sandbox_dir / "new_file.txt").write_text("content")
    commit_sandbox_changes(sandbox_dir, "test commit")
    log = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=sandbox_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "test commit" in log.stdout


def test_build_docker_run_cmd(tmp_path):
    """The docker command mounts the sandbox and session store."""
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    cmd = build_docker_run_cmd(
        sandbox_dir,
        "prpl-agent-sandbox",
        auth_args=["-e", "CLAUDE_CODE_OAUTH_TOKEN"],
        env_vars={"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "16384"},
        extra_network_domains=("api.openai.com",),
    )
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert "--cap-add=NET_ADMIN" in cmd
    assert "--cap-add=NET_RAW" in cmd
    assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS=16384" in cmd
    assert "PRPL_AGENT_FIREWALL_EXTRA_DOMAINS=api.openai.com" in cmd
    assert f"{sandbox_dir.resolve()}:/sandbox" in cmd
    sessions = sandbox_dir / ".agent_sessions"
    assert f"{sessions.resolve()}:/home/node/.claude/projects" in cmd
    assert cmd[-1] == "prpl-agent-sandbox"
    assert cmd[cmd.index("-w") + 1] == "/sandbox"


def test_agent_subprocess_env(monkeypatch):
    """CLAUDECODE* vars are stripped and extras are merged."""
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDECODE_FOO", "bar")
    monkeypatch.setenv("UNRELATED_VAR", "keep")
    env = agent_subprocess_env({"EXTRA": "value"})
    assert "CLAUDECODE" not in env
    assert "CLAUDECODE_FOO" not in env
    assert env["UNRELATED_VAR"] == "keep"
    assert env["EXTRA"] == "value"
