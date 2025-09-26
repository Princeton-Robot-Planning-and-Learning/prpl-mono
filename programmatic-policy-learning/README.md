# Programmatic Policy Learning

This is a codebase that the PRPL lab is using for multiple projects related to programmatic policy learning.

## Installation

Requirements: Python >=3.10 and <=3.12.

We strongly recommend [uv](https://docs.astral.sh/uv/getting-started/installation/). The steps below assume that you have `uv` installed. If you do not, just remove `uv` from the commands and the installation should still work.

```
# Install PRPL dependencies.
uv pip install -r prpl_requirements.txt
# Install this package and third-party dependencies.
uv pip install -e ".[develop]"
```

Check the installation: ```./run_ci_checks.sh```

If you want to use an OpenAI LLM, make sure you have an `OPENAI_API_KEY` set (e.g., see [here](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety))

## Usage Example

```python
from pathlib import Path

import gymnasium
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel

from programmatic_policy_learning.approaches.ppl_approach import (
    ProgrammaticPolicyLearningApproach,
)

env = gymnasium.make("LunarLander-v3")
env.action_space.seed(123)
environment_description = (
    "The well-known LunarLander in gymnasium, i.e., "
    'env = gymnasium.make("LunarLander-v3")'
)

cache = SQLite3PretrainedLargeModelCache(Path("llm_cache.db"))
llm = OpenAIModel("gpt-4o-mini", cache)

approach = ProgrammaticPolicyLearningApproach(
    environment_description,
    env.observation_space,
    env.action_space,
    seed=123,
    llm=llm,
)

obs, info = env.reset()
approach.reset(obs, info)

print(approach._policy)

for _ in range(5):
    action = approach.step()
    assert env.action_space.contains(action)
    obs, reward, terminated, _, info = env.step(action)
    approach.update(obs, reward, terminated, info)
```

## Running Experiments

We use [hydra](https://hydra.cc/) to run experiments at scale. See `experiments/run_experiment.py`. For example:

```
python experiments/run_experiment.py -m env=lunar_lander llm=openai seed='range(0,2)'
```


## Notes

### Box2D Installation on macOS

If you encounter an error when installing dependencies (e.g., `box2d-py`) that looks like this:

```
Box2D/Box2D_wrap.cpp:3378:10: fatal error: 'string' file not found
3378 | #include <string>
      |          ^~~~~~~~
1 error generated.
error: command '/usr/bin/clang++' failed with exit code 1
```

This might mean that your macOS Command Line Tools (CLT) or SDK isn’t installed or selected correctly, and the compiler (`clang++`) cannot find the C++ standard library headers.

To fix this issue, try these steps:

1. **Reinstall or point to the correct Command Line Tools (CLT):**

   - Remove any broken or partial CLT installations:
     ```bash
     sudo rm -rf /Library/Developer/CommandLineTools
     ```

   - Reinstall the CLT (a GUI prompt will appear):
     ```bash
     xcode-select --install
     ```

2. After completing the installation, try installing the dependencies again:
   ```bash
   uv pip install -e ".[develop]"
   ```

If you are using `uv` to manage your virtual environment, you can also try installing `box2d-py` directly to verify the fix:

```bash
uv pip install box2d-py
```
---
# Adding a New Environment to PPL

You can add environments in two ways:

1. **Plain Gymnasium env** (already registered via `gymnasium.make`)  
2. **Provider-based env** (env lives in a separate repo and needs a small adapter)


## 1. Plain Gymnasium Env (no provider)

If the env is already registered with Gymnasium, just add a YAML under `conf/env/` and you’re done.
**Example:** `conf/env/lunarlander.yaml`

```yaml
# Passed into gymnasium.make() to create the environment.
make_kwargs:
  id: "LunarLander-v3"
  render_mode: null  # "human", "rgb_array", or null

# Optional, purely descriptive.
description: "The well-known LunarLander in gymnasium, i.e., env = gymnasium.make('LunarLander-v3')"
```


**How it’s used in code:**

```python
from programmatic_policy_learning.env.registry import EnvRegistry

registry = EnvRegistry()
env = registry.load(cfg.env)  # default fallback is gymnasium.make(**make_kwargs)
```
> If you don’t specify a `provider`, `EnvRegistry` falls back to `gymnasium.make(**make_kwargs)`.

## 2. Provider-Based Env (from a separate repo)

Use this when your env lives in another repo (e.g., PRBench, GGG, custom maze env).  
You’ll: (a) create a YAML with a `provider`, (b) add a provider function, and (c) (if needed) pin the external repo in `pyproject.toml`.

### 2.1 Create the YAML (under `conf/env/`)

**Example:** `conf/env/prbench_motion2d_p1.yaml`

```yaml
make_kwargs:
  id: "prbench/Motion2D-p1-v0"
  render_mode: null

provider: prbench  # <--- important

description: "PRBench Motion2D-p1. Gymnasium-style env registered by PRBench"
```
### 2.2 Register the Provider

**Edit:** `programmatic_policy_learning/env/registry.py`

Add an entry to the provider map:
```python
self._providers: dict[str, Callable[[Any], Any]] = {
    "ggg": create_ggg_env,
    "prbench": create_prbench_env,
    # "gym_maze": create_maze_env,  # example for your own provider
}
```

### 2.3 Implement the Provider Function

**File structure:**
```cpp
programmatic_policy_learning/
  env/
    providers/
      prbench_provider.py      # define create_prbench_env(cfg)
      ggg_provider.py          # define create_ggg_env(cfg)
      maze_provider.py         # define create_maze_env(cfg)  (example)
```

**Example:** `programmatic_policy_learning/env/providers/prbench_provider.py`

```python
from __future__ import annotations
from typing import Any
import gymnasium as gym

def create_prbench_env(cfg: Any):
    """Create and return a PRBench env using cfg.env.make_kwargs."""
    make_kwargs = dict(cfg.env.make_kwargs)
    env = gym.make(**make_kwargs)
    return env
```

> Your provider can do anything needed (import the external package, wrap the env, set seeds, apply wrappers, etc.). Just return the final `env`.
### 2.4 Add the External Repo to the dependencies (if you import it)

If your provider imports an external repo, put it in `pyproject.toml` so CI and collaborators get the same version.

**Example (GGG):**
```toml
[project.optional-dependencies]
ggg = [
  "generalization_grid_games @ git+https://github.com/zahraabashir/generalization_grid_games.git"
]
```

**Example (your own repo):**
```toml
my_env = [
  "my_cool_env_pkg @ git+https://github.com/your-org/my_cool_env_pkg.git@<commit-hash>"
]
```

After this, you only need to run the following command to install that dependency:

```bash
uv pip install -e ".[my_env]"
```

## 3) How to Instantiate in Code

Same pattern for both plain and provider-based envs:
```python
from programmatic_policy_learning.env.registry import EnvRegistry

registry = EnvRegistry()
env = registry.load(cfg.env)  # uses provider if present, else gymnasium.make
```

-   If your YAML has `provider: ...`, `EnvRegistry` routes to the matching provider function.
    
-   If there’s **no** `provider`, it calls `gymnasium.make(**make_kwargs)`.



## Minimal Checklist

-   Add `conf/env/<your_env>.yaml`
- If external repo:
    
	-  Add dependency pin in `pyproject.toml` under `[project.optional-dependencies]`
	    
	-   Add provider entry in `EnvRegistry` (provider name → function)
	    
	-   Implement `create_<provider>_env(cfg)` in `env/providers/<provider>_provider.py`
	    
-   Instantiate with `EnvRegistry().load(cfg.env)`
That's it!
---

## Contributing

* Ask an owner of the repository to add your GitHub username to the collaborators list
* All checks must pass before code is merged (see `./run_ci_checks.sh`)
* All code goes through the pull request review process on GitHub
