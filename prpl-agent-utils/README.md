# PRPL Agent Utils

Coding agent utilities from the Princeton Robot Planning and Learning group.

The main feature is a plug-and-play `Agent` interface, in the spirit of
[prpl-llm-utils](../prpl-llm-utils), for running a general-purpose coding agent
(currently Claude Code) inside a sandbox. Two properties distinguish an agent
from an LLM query:

- **Sandboxing.** By default, each query runs in a Docker container where the
  only writable host path is the agent's sandbox directory and an in-container
  firewall restricts network access to the Anthropic API, GitHub, and PyPI.
- **Persistence.** The sandbox directory and the agent's conversation both
  persist between queries, so an agent can be created once and reused as a
  member of another class, with later queries building on earlier ones.

Much of the sandbox machinery descends from the robocode project.

## Usage Example

Build the sandbox image once (requires Docker):

```bash
bash docker/build.sh
```

Then:

```python
from pathlib import Path
from prpl_agent_utils import ClaudeCodeAgent

agent = ClaudeCodeAgent(Path("my_sandbox"), model="sonnet")
response = agent.query(
    "Write count_vowels.py containing a function count_vowels(s: str) -> int."
)
print(response.text)  # The agent's final message.
print(response.metadata["total_cost_usd"])
# The file persists in the sandbox...
assert (agent.sandbox_dir / "count_vowels.py").exists()
# ...and the conversation continues across queries.
response = agent.query("Now add a test file for it and run the tests.")
```

A typical pattern from research code, where the agent is a member of a method
class and queried repeatedly:

```python
class PureAgentMethod(Method):
    def __init__(self, sandbox_dir: Path) -> None:
        self._agent = ClaudeCodeAgent(sandbox_dir)

    def train(self, train_problems):
        self._agent.query(f"Study these problems and record notes: ...")

    def step(self, obs):
        return parse_action(self._agent.query(f"Current observation: {obs}. Next action?").text)
```

Useful constructor options:

- `use_docker=False` runs the CLI directly on the host (a hook still blocks
  writes outside the sandbox, but shell commands can read the host filesystem;
  prefer Docker for anything untrusted).
- `init_files={"data.json": Path("...")}` copies files into the sandbox before
  the first query.
- `system_prompt`, `max_budget_usd_per_query`, `extra_network_domains`.

Authentication comes from `claude login` on the host (or
`CLAUDE_CODE_OAUTH_TOKEN`); the operator's live `~/.claude` directory is never
read or written by the sandboxed CLI.

## Example

`examples/pendulum_pure_agent.ipynb` works through the recipe in [*Now what? A
recipe for after the problem
setting*](https://tomsilver.github.io/blog/2026/now-whats-your-solution/) on
gymnasium's `Pendulum-v1`: a random baseline, a pure agent method that writes and
revises its own policy inside the sandbox, and an energy-shaping oracle. Install
the extra dependencies with `pip install -e ".[examples]"`. Running it spends a
few tens of cents of API budget.

## Red teaming the sandbox

`integration_tests/red_team_sandbox.py` runs adversarial prompts against a real
agent and checks a canary file outside the sandbox after each one. It covers
path escapes through the file tools and bash, network restriction, the
privilege drop (the agent user must not regain root or re-run the firewall
script), and host-config persistence (container writes under `~/.claude` must
not survive on the host).

```bash
python integration_tests/red_team_sandbox.py --docker           # full suite
python integration_tests/red_team_sandbox.py --firewall-only    # no API cost
python integration_tests/red_team_sandbox.py --home-persist-only
```

The full suite spends real API budget (capped at $1 per prompt, typically much
less). Without `--docker` it tests the host sandbox, where the bash read
escapes are reported as known limitations rather than breaches.

## Requirements

- Python 3.10+
- The [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code), logged in
- Docker (unless `use_docker=False`)

## Installation

1. Recommended: create and source a virtualenv.
2. `pip install -e ".[develop]"`

## Check Installation

Run `./run_ci_checks.sh`. It should complete with all green successes.
