"""Interfaces for sandboxed coding agents.

The design mirrors ``prpl_llm_utils.models``, with one deliberate difference: an
agent is stateful. Each agent owns a persistent sandbox directory that it works in,
and its conversation continues across ``query()`` calls, so later queries can build
on earlier ones (files written, context established). There is no response cache
for the same reason.
"""

from __future__ import annotations

import abc
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import IO, Any

from prpl_agent_utils.claude_auth import (
    get_claude_oauth_token,
    sandbox_claude_config,
)
from prpl_agent_utils.sandbox import (
    DEFAULT_DOCKER_IMAGE,
    agent_subprocess_env,
    build_docker_run_cmd,
    commit_sandbox_changes,
    docker_claude_auth,
    setup_sandbox_dir,
)
from prpl_agent_utils.structs import AgentResponse

logger = logging.getLogger(__name__)


class Agent(abc.ABC):
    """A general-purpose coding agent with a persistent sandbox directory.

    The sandbox directory is where the agent reads and writes files; it persists
    between queries and between agent processes, so an ``Agent`` can be created
    once and reused as a member of another class (e.g. an approach that queries
    the agent at every decision step).
    """

    def __init__(self, sandbox_dir: Path) -> None:
        self._sandbox_dir = sandbox_dir

    @property
    def sandbox_dir(self) -> Path:
        """The persistent working directory for this agent."""
        return self._sandbox_dir

    @abc.abstractmethod
    def get_id(self) -> str:
        """Get a string identifier for this agent (e.g. backend and model)."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def query(self, prompt: str) -> AgentResponse:
        """Run one prompt to completion in the sandbox.

        The conversation and the sandbox files persist to the next query.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def reset(self) -> None:
        """Start a new conversation.

        Files in the sandbox are kept; delete the sandbox directory itself to start
        fully fresh.
        """
        raise NotImplementedError("Override me!")


class ClaudeCodeAgent(Agent):
    """An agent backed by the Claude Code CLI.

    With ``use_docker=True`` (the default), each query runs in a fresh container
    of the ``prpl-agent-sandbox`` image: the sandbox directory is the only
    writable host path, and an in-container firewall restricts network access to
    the Anthropic API, GitHub, and PyPI (plus ``extra_network_domains``). Build
    the image once with ``bash docker/build.sh``.

    With ``use_docker=False``, the CLI runs directly on the host with
    ``--dangerously-skip-permissions``; a hook blocks Write/Edit outside the
    sandbox, but shell commands can still read (not write) the host filesystem,
    so prefer Docker for anything untrusted.

    Persistence works in both modes: the sandbox directory keeps the agent's
    files, and the CLI session is stored under the sandbox
    (``.agent_sessions`` / ``.agent_home``) so the next query resumes the same
    conversation via ``--continue``, even in a new container.
    """

    def __init__(
        self,
        sandbox_dir: Path,
        model: str = "sonnet",
        use_docker: bool = True,
        system_prompt: str = "",
        init_files: dict[str, Path] | None = None,
        max_budget_usd_per_query: float = 5.0,
        max_output_tokens: int = 16384,
        tools: str = "Bash,Read,Write,Edit,Glob,Grep,Task",
        docker_image: str = DEFAULT_DOCKER_IMAGE,
        extra_network_domains: tuple[str, ...] = (),
    ) -> None:
        super().__init__(sandbox_dir)
        self._model = model
        self._use_docker = use_docker
        self._system_prompt = system_prompt
        self._init_files = dict(init_files or {})
        self._max_budget_usd_per_query = max_budget_usd_per_query
        self._max_output_tokens = max_output_tokens
        self._tools = tools
        self._docker_image = docker_image
        self._extra_network_domains = extra_network_domains
        self._resume_session = False

    def get_id(self) -> str:
        return f"claude-code-{self._model}"

    def reset(self) -> None:
        self._resume_session = False

    def query(self, prompt: str) -> AgentResponse:
        setup_sandbox_dir(self._sandbox_dir, self._init_files)
        cli_cmd = self._build_cli_cmd(prompt)
        if self._use_docker:
            response = self._query_docker(cli_cmd)
        else:
            response = self._query_local(cli_cmd)
        commit_sandbox_changes(self._sandbox_dir, "auto-commit after agent query")
        self._resume_session = True
        return response

    def _build_cli_cmd(self, prompt: str) -> list[str]:
        """Build the Claude CLI command for one query."""
        claude_cmd = (
            "claude"
            if self._use_docker
            else os.environ.get("PRPL_AGENT_CLAUDE_CMD", "claude")
        )
        cmd = [
            claude_cmd,
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            self._model,
            "--dangerously-skip-permissions",
            "--tools",
            self._tools,
            "--setting-sources",
            "project",
        ]
        # The session is persisted under the sandbox, so --continue resumes the
        # most recent conversation in this working directory, which is exactly
        # this agent's conversation.
        if self._resume_session:
            cmd.append("--continue")
        if self._system_prompt:
            cmd += ["--system-prompt", self._system_prompt]
        if self._max_budget_usd_per_query > 0:
            cmd += ["--max-budget-usd", str(self._max_budget_usd_per_query)]
        return cmd

    def _query_local(self, cli_cmd: list[str]) -> AgentResponse:
        env = agent_subprocess_env(
            {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(self._max_output_tokens)}
        )
        # The CLI checks the token env var before any filesystem credential
        # store, so resolving it here (env var or macOS Keychain) lets
        # sandbox_claude_config skip credential copying.
        oauth_token = get_claude_oauth_token()
        if oauth_token:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token
        with sandbox_claude_config(self._sandbox_dir) as config_dir:
            env["CLAUDE_CONFIG_DIR"] = str(config_dir)
            return self._run_and_parse(cli_cmd, env, cwd=self._sandbox_dir)

    def _query_docker(self, cli_cmd: list[str]) -> AgentResponse:
        with docker_claude_auth() as (auth_args, auth_env):
            docker_cmd = build_docker_run_cmd(
                self._sandbox_dir,
                self._docker_image,
                auth_args,
                env_vars={
                    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(self._max_output_tokens)
                },
                extra_network_domains=self._extra_network_domains,
            )
            env = agent_subprocess_env(auth_env)
            return self._run_and_parse(docker_cmd + cli_cmd, env, cwd=None)

    def _run_and_parse(
        self, cmd: list[str], env: dict[str, str], cwd: Path | None
    ) -> AgentResponse:
        log_dir = self._sandbox_dir / ".agent_logs"
        log_dir.mkdir(exist_ok=True)
        logger.info("Running agent: %s ...", " ".join(cmd[:8]))
        # stderr goes to a file, not a PIPE: stdout is parsed to EOF before
        # stderr is read, so a CLI that fills a stderr PIPE buffer first would
        # deadlock against the parser.
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_file:
            with subprocess.Popen(
                cmd,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                text=True,
            ) as proc:
                return _parse_stream(proc, log_dir / "stream.jsonl", stderr_file)


def _parse_stream(
    proc: subprocess.Popen[str], stream_log_path: Path, stderr_file: IO[str]
) -> AgentResponse:
    """Parse ``stream-json`` output from a Claude CLI process into a response.

    Assistant messages and tool calls are logged as they stream; the final
    ``result`` message provides the response text and usage metadata. Raises
    ``RuntimeError`` if the CLI exits without producing a result.
    """
    result_text: str | None = None
    is_error = False
    num_turns = 0
    num_tool_calls = 0
    total_cost: float | None = None
    model_usage: dict[str, Any] = {}
    stop_reason: str | None = None

    assert proc.stdout is not None
    with open(stream_log_path, "a", encoding="utf-8") as stream_log:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            stream_log.write(line + "\n")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Non-JSON output: %s", line[:200])
                continue

            msg_type = msg.get("type", "")
            if msg_type == "assistant":
                num_turns += 1
                for block in msg.get("message", {}).get("content", []):
                    if block.get("type") == "text":
                        logger.info("Agent: %s", block["text"])
                    elif block.get("type") == "tool_use":
                        num_tool_calls += 1
                        input_str = json.dumps(block.get("input", {}))
                        if len(input_str) > 300:
                            input_str = input_str[:300] + "..."
                        logger.info("Tool call: %s(%s)", block.get("name"), input_str)
            elif msg_type == "result":
                # A single query can span several CLI sessions (autocompaction
                # re-inits the CLI, each session emitting its own result), so
                # keep the latest values of the cumulative fields.
                is_error = msg.get("is_error", False)
                result_text = msg.get("result", result_text)
                total_cost = msg.get("total_cost_usd", total_cost)
                model_usage = msg.get("modelUsage") or model_usage
                stop_reason = msg.get("subtype") or stop_reason

    proc.wait()
    stderr_file.seek(0)
    stderr_output = stderr_file.read()

    if result_text is None:
        raise RuntimeError(
            "Agent CLI produced no result "
            f"(exit code {proc.returncode}): {stderr_output[:1000]}"
        )

    metadata: dict[str, Any] = {
        "is_error": is_error,
        "num_turns": num_turns,
        "num_tool_calls": num_tool_calls,
        "total_cost_usd": total_cost,
        "model_usage": model_usage,
        "stop_reason": stop_reason,
    }
    return AgentResponse(result_text, metadata)
