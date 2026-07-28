"""Tests for agents.py."""

import json
import stat
import subprocess
from pathlib import Path

import pytest

from prpl_agent_utils.agents import ClaudeCodeAgent

# A stand-in for the Claude CLI: records its argv, appends the prompt to a file
# in the working directory (to exercise sandbox persistence), and emits a
# minimal stream-json transcript.
_FAKE_CLI = """\
#!/usr/bin/env python3
import json
import sys

args = sys.argv[1:]
with open("cli_args.json", "w") as f:
    json.dump(args, f)

prompt = args[args.index("-p") + 1]
with open("prompts.txt", "a") as f:
    f.write(prompt + "\\n")

resumed = "--continue" in args
print(json.dumps({
    "type": "assistant",
    "message": {"content": [
        {"type": "text", "text": "thinking out loud"},
        {"type": "tool_use", "name": "Write", "input": {"file_path": "x.py"}},
    ]},
}))
print(json.dumps({
    "type": "result",
    "is_error": False,
    "result": f"done resumed={resumed}",
    "total_cost_usd": 0.01,
    "modelUsage": {"claude-sonnet": {"inputTokens": 10}},
    "subtype": "success",
}))
"""

_FAILING_CLI = """\
#!/usr/bin/env python3
import sys
print("something went wrong", file=sys.stderr)
sys.exit(1)
"""

# Writes far more to stderr than a PIPE buffer holds before emitting its
# result, which would deadlock a PIPE-backed stderr against the stdout parser.
_NOISY_STDERR_CLI = """\
#!/usr/bin/env python3
import json
import sys

sys.stderr.write("x" * 1_000_000)
sys.stderr.flush()
print(json.dumps({"type": "result", "is_error": False, "result": "ok"}))
"""


def _install_fake_cli(tmp_path: Path, monkeypatch, script: str) -> Path:
    cli = tmp_path / "fake_claude"
    cli.write_text(script)
    cli.chmod(cli.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PRPL_AGENT_CLAUDE_CMD", str(cli))
    # Bypass the credential-copying auth flow in tests.
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "test-token")
    return cli


def test_claude_code_agent_query_and_persistence(tmp_path, monkeypatch):
    """Files and the conversation persist between queries."""
    _install_fake_cli(tmp_path, monkeypatch, _FAKE_CLI)
    sandbox_dir = tmp_path / "sandbox"
    agent = ClaudeCodeAgent(sandbox_dir, use_docker=False)
    assert agent.get_id() == "claude-code-sonnet"
    assert agent.sandbox_dir == sandbox_dir

    response = agent.query("First prompt.")
    assert response.text == "done resumed=False"
    assert response.metadata["total_cost_usd"] == 0.01
    assert response.metadata["num_turns"] == 1
    assert response.metadata["num_tool_calls"] == 1
    assert not response.metadata["is_error"]

    # The fake CLI wrote into the sandbox, and the sandbox is a git repo.
    assert (sandbox_dir / "prompts.txt").read_text() == "First prompt.\n"
    assert (sandbox_dir / ".git").is_dir()

    # The second query resumes the conversation and sees the earlier files.
    response = agent.query("Second prompt.")
    assert response.text == "done resumed=True"
    assert (
        sandbox_dir / "prompts.txt"
    ).read_text() == "First prompt.\nSecond prompt.\n"

    # The agent's changes were auto-committed.
    log = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=sandbox_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "auto-commit after agent query" in log.stdout


def test_claude_code_agent_reset(tmp_path, monkeypatch):
    """Reset() starts a new conversation but keeps sandbox files."""
    _install_fake_cli(tmp_path, monkeypatch, _FAKE_CLI)
    sandbox_dir = tmp_path / "sandbox"
    agent = ClaudeCodeAgent(sandbox_dir, use_docker=False)
    agent.query("First prompt.")
    agent.reset()
    response = agent.query("Second prompt.")
    assert response.text == "done resumed=False"
    assert (sandbox_dir / "prompts.txt").exists()


def test_claude_code_agent_cli_options(tmp_path, monkeypatch):
    """Constructor options are forwarded to the CLI."""
    _install_fake_cli(tmp_path, monkeypatch, _FAKE_CLI)
    sandbox_dir = tmp_path / "sandbox"
    agent = ClaudeCodeAgent(
        sandbox_dir,
        model="opus",
        use_docker=False,
        system_prompt="Be terse.",
        max_budget_usd_per_query=2.5,
        init_files={"data/input.txt": tmp_path / "fake_claude"},
    )
    agent.query("Prompt.")
    args = json.loads((sandbox_dir / "cli_args.json").read_text())
    assert args[args.index("--model") + 1] == "opus"
    assert args[args.index("--system-prompt") + 1] == "Be terse."
    assert args[args.index("--max-budget-usd") + 1] == "2.5"
    assert (sandbox_dir / "data" / "input.txt").exists()


def test_claude_code_agent_cli_failure(tmp_path, monkeypatch):
    """A CLI run that produces no result raises."""
    _install_fake_cli(tmp_path, monkeypatch, _FAILING_CLI)
    agent = ClaudeCodeAgent(tmp_path / "sandbox", use_docker=False)
    with pytest.raises(RuntimeError, match="no result"):
        agent.query("Prompt.")


def test_claude_code_agent_large_stderr_no_deadlock(tmp_path, monkeypatch):
    """A CLI that floods stderr cannot deadlock the stream parser."""
    _install_fake_cli(tmp_path, monkeypatch, _NOISY_STDERR_CLI)
    agent = ClaudeCodeAgent(tmp_path / "sandbox", use_docker=False)
    response = agent.query("Prompt.")
    assert response.text == "ok"


def test_real_claude_code_agent(tmp_path, request):
    """Query the real Claude Code CLI on the host (opt in with --runagents)."""
    if not request.config.getoption("--runagents"):
        pytest.skip("Use --runagents to run tests with real coding agents.")
    agent = ClaudeCodeAgent(tmp_path / "sandbox", use_docker=False)
    response = agent.query("Write hello.txt containing exactly: hello world")
    assert (agent.sandbox_dir / "hello.txt").read_text().strip() == "hello world"
    assert response.text
