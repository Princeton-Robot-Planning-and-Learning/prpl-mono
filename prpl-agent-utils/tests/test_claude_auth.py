"""Tests for claude_auth.py."""

import pytest

from prpl_agent_utils.claude_auth import (
    get_claude_oauth_token,
    sandbox_claude_config,
    sandbox_claude_session_store,
)


def test_get_claude_oauth_token_from_env(monkeypatch):
    """The env var takes precedence over any platform credential store."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "env-token")
    assert get_claude_oauth_token() == "env-token"


def test_sandbox_claude_config_with_oauth_token(tmp_path, monkeypatch):
    """With an OAuth token, no credentials are copied into the sandbox."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "env-token")
    with sandbox_claude_config(tmp_path) as config_dir:
        assert config_dir == tmp_path / ".agent_home"
        assert config_dir.is_dir()
        assert not (config_dir / ".credentials.json").exists()


def test_sandbox_claude_session_store(tmp_path):
    """The session store is created inside the sandbox."""
    store = sandbox_claude_session_store(tmp_path)
    assert store == tmp_path / ".agent_sessions"
    assert store.is_dir()


def test_sandbox_state_dir_rejects_symlink(tmp_path):
    """A symlinked state directory is rejected (sandbox escape)."""
    outside = tmp_path / "outside"
    outside.mkdir()
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    (sandbox_dir / ".agent_sessions").symlink_to(outside)
    with pytest.raises(RuntimeError, match="symlink"):
        sandbox_claude_session_store(sandbox_dir)
