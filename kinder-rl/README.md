# Reinforcement Learning Baselines for KinDER

![workflow](https://github.com/Jaraxxus-Me/kinder-rl/actions/workflows/ci.yml/badge.svg)

This package provides RL baselines (PPO, SAC) for the KinDER physical reasoning benchmark environments.

## Installation

1. Recommended: create and source a virtualenv (perhaps with [uv](https://github.com/astral-sh/uv))
2. Install this repo: `pip install -e ".[develop]"`

For monorepo installation, install dependencies first:
```bash
pip install -r prpl_requirements.txt
pip install -e ".[develop]"
```

## Usage

### Inspect Results

Download [PPO results](https://drive.google.com/file/d/1lxE4IRT2fQiGWLSdmegQAJJvq68UDKLg/view?usp=drive_link) or 
[SAC results](https://drive.google.com/file/d/16llJIk3TYx40KwgW2Pa3WYD1QItVL7Ai/view?usp=drive_link).
You should expect folder structure:
```
prpl-mono/
  kinder-rl/
    outputs/
    runs/
```
Run:
```
cd prpl-mono
python kinder-rl/experiments/gen_results.py --outputs_dir kinder-rl/outputs --runs_dir kinder-rl/runs
```
You should see: `kinder-rl/results.csv`.

### Running Experiments

Experiments are configured using [Hydra](https://hydra.cc/). Run from the `kinder-rl` directory:

```bash
cd kinder-rl
python experiments/run_experiment.py agent=<agent_config> env_id=<environment_id> [options]
```

See all of the scripts in `scripts/` for running different agents on different environments.

### Available Agents

| Agent | Config Name | Description |
|-------|-------------|-------------|
| PPO | `ppo_*` | Proximal Policy Optimization |
| SAC | `sac_*` | Soft Actor-Critic |
| Random | `random` | Random action baseline |

### Supported Environments

#### 2D Environments
| Environment | ID | Agent Configs |
|-------------|-----|---------------|
| Motion2D | `kinder/Motion2D-p{0,1,2}-v0` | `ppo_motion2d_*`, `sac_motion2d_*` |
| StickButton2D | `kinder/StickButton2D-b{1,2,3}-v0` | `ppo_stickbutton2d_*`, `sac_stickbutton2d_*` |
| DynObstruction2D | `kinder/DynObstruction2D-o{1,2,3}-v0` | `ppo_dynobstruction2d`, `sac_dynobstruction2d` |
| DynPushPullHook2D | `kinder/DynPushPullHook2D-o{1,2,3}-v0` | `ppo_dynpushpullhook2d`, `sac_dynpushpullhook2d` |

#### 3D Environments
| Environment | ID | Agent Configs |
|-------------|-----|---------------|
| BaseMotion3D | `kinder/BaseMotion3D-v0` | `ppo_basemotion3d`, `sac_basemotion3d` |
| Transport3D | `kinder/Transport3D-o{1,2,3}-v0` | `ppo_transport3d`, `sac_transport3d` |
| Shelf3D | `kinder/Shelf3D-o{1,2,3}-v0` | `ppo_shelf3d`, `sac_shelf3d` |

### Example Commands

**Train PPO on BaseMotion3D:**
```bash
python experiments/run_experiment.py \
    agent=ppo_basemotion3d \
    env_id="kinder/BaseMotion3D-v0" \
    agent.args.total_timesteps=50000 \
    max_episode_steps=100 \
    eval_episodes=50 \
    seed=0
```

**Train SAC on DynObstruction2D:**
```bash
python experiments/run_experiment.py \
    agent=sac_dynobstruction2d \
    env_id="kinder/DynObstruction2D-o1-v0" \
    max_episode_steps=200 \
    eval_episodes=50 \
    seed=42
```

**Train PPO on Transport3D with custom hyperparameters:**
```bash
python experiments/run_experiment.py \
    agent=ppo_transport3d \
    env_id="kinder/Transport3D-o2-v0" \
    agent.args.total_timesteps=2000000 \
    agent.args.hidden_size=256 \
    agent.args.learning_rate=1e-4 \
    seed=0
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `agent` | Agent configuration file (without .yaml) | `ppo_motion2d_0_passage` |
| `env_id` | KinDER environment ID | `kinder/Motion2D-p0-v0` |
| `max_episode_steps` | Maximum steps per episode | `300` |
| `eval_episodes` | Number of evaluation episodes | `50` |
| `seed` | Random seed | `0` |

### Agent-Specific Options

**PPO:**
| Option | Description | Default |
|--------|-------------|---------|
| `agent.args.total_timesteps` | Total training timesteps | `1000000` |
| `agent.args.learning_rate` | Learning rate | `3e-4` |
| `agent.args.hidden_size` | Hidden layer size | `128` |
| `agent.args.num_steps` | Steps per rollout | `2048` |
| `agent.args.num_minibatches` | Number of minibatches | `32` |
| `agent.args.update_epochs` | PPO update epochs | `10` |

**SAC:**
| Option | Description | Default |
|--------|-------------|---------|
| `agent.args.total_timesteps` | Total training timesteps | `1000000` |
| `agent.args.policy_lr` | Policy learning rate | `3e-4` |
| `agent.args.q_lr` | Q-network learning rate | `1e-3` |
| `agent.args.hidden_size` | Hidden layer size | `256` |
| `agent.args.buffer_size` | Replay buffer size | `1000000` |
| `agent.args.batch_size` | Training batch size | `256` |

### Output

After training completes, results are saved to `outputs/<date>/<time>/`:
- `agent.pkl` - Trained agent checkpoint
- `train_results.csv` - Training episode returns
- `eval_results.csv` - Evaluation episode returns
- `config.yaml` - Experiment configuration

TensorBoard logs are saved to `runs/<exp_name>/`.

A summary of results is printed at the end:
```
============================================================
EXPERIMENT RESULTS SUMMARY
============================================================
Agent: ppo
Environment: kinder/BaseMotion3D-v0
Seed: 0
------------------------------------------------------------
TRAINING:
  Total episodes: 100
  Mean return: -150.32
  Std return: 45.21
  ...
------------------------------------------------------------
EVALUATION:
  Episodes: 50
  Mean return: -120.45
  Std return: 38.12
  Success rate (return > -150): 35/50 (70.0%)
============================================================
```

## Reproducing RL Baseline Experiments

The `scripts/` directory contains shell scripts to reproduce all RL baseline experiments from the paper. Each script runs training across multiple random seeds (typically 5) for statistical significance.

### Quick Start

```bash
cd kinder-rl

# Run all experiments with a single command
./scripts/run_all_experiments.sh [seed]  # default seed=0
```

### Individual Environment Scripts

All scripts should be run from the `kinder-rl` directory.

#### 2D Geometric Environments

| Script | Environment | Seeds | Description |
|--------|-------------|-------|-------------|
| `./scripts/run_ppo_motion2d.sh` | Motion2D-p0, Motion2D-p2 | 0-4 | PPO on 0 and 2 passages |
| `./scripts/run_sac_motion2d.sh` | Motion2D-p0, Motion2D-p2 | 0-4 | SAC on 0 and 2 passages |
| `./scripts/run_ppo_stickbutton2d.sh` | StickButton2D-b1, StickButton2D-b3 | 0-4 | PPO on 1 and 3 buttons |
| `./scripts/run_sac_stickbutton2d.sh` | StickButton2D-b1, StickButton2D-b3 | 0-4 | SAC on 1 and 3 buttons |

#### 2D Dynamic Environments

| Script | Environment | Seeds | Description |
|--------|-------------|-------|-------------|
| `./scripts/run_ppo_dynobstruction2d.sh` | DynObstruction2D-o1 | 0-4 | PPO training |
| `./scripts/run_sac_dynobstruction2d.sh [n] [seed]` | DynObstruction2D-o{n} | custom | SAC (n=num obstructions, default 1) |
| `./scripts/run_ppo_dynpushpullhook2d.sh` | DynPushPullHook2D-o5 | 301-305 | PPO training |
| `./scripts/run_sac_dynpushpullhook2d.sh [n] [seed]` | DynPushPullHook2D-o{n} | custom | SAC (n=num cubes, default 1) |

#### 3D Geometric Environments

| Script | Environment | Seeds | Description |
|--------|-------------|-------|-------------|
| `./scripts/run_ppo_basemotion3d.sh` | BaseMotion3D | 0-4 | PPO training |
| `./scripts/run_sac_basemotion3d.sh` | BaseMotion3D | 0-4 | SAC training |
| `./scripts/run_ppo_basemotion3d_dense.sh` | BaseMotion3D | 0-4 | PPO with dense reward |
| `./scripts/run_sac_basemotion3d_dense.sh` | BaseMotion3D | 0-4 | SAC with dense reward |
| `./scripts/run_ppo_transport3d.sh [n] [seed]` | Transport3D-o{n} | custom | PPO (n=num cubes, default 1) |
| `./scripts/run_sac_transport3d.sh [n] [seed]` | Transport3D-o{n} | custom | SAC (n=num cubes, default 1) |

#### TidyBot3D Environments

| Script | Environment | Seeds | Description |
|--------|-------------|-------|-------------|
| `./scripts/run_ppo_shelf3d.sh` | TidyBot3D-cupboard_real-o1 | 300-304 | PPO on cupboard task |
| `./scripts/run_sac_shelf3d.sh` | TidyBot3D-tool_use (sweep) | 300-304 | SAC on sweep blocks task |
| `./scripts/run_ppo_sweep3d.sh` | TidyBot3D-tool_use (sweep) | 300-304 | PPO on sweep blocks task |
| `./scripts/run_sac_sweep3d.sh` | TidyBot3D-tool_use (sweep) | 300-304 | SAC on sweep blocks task |

### Script Details

Most scripts automatically loop over 5 random seeds. Scripts with `[n] [seed]` arguments run a single experiment with configurable parameters:

```bash
# Scripts with automatic seed loops (just run directly):
./scripts/run_ppo_motion2d.sh
./scripts/run_ppo_basemotion3d.sh
./scripts/run_ppo_shelf3d.sh

# Scripts with configurable arguments:
./scripts/run_sac_dynobstruction2d.sh 2 42    # 2 obstructions, seed 42
./scripts/run_ppo_transport3d.sh 1 0          # 1 cube, seed 0
./scripts/run_sac_transport3d.sh 3 123        # 3 cubes, seed 123
```

### Benchmarking Environment Speed

To test environment parallelization performance:

```bash
python scripts/test_env_speed.py
```

This compares single-env, SyncVectorEnv, and AsyncVectorEnv speeds across TidyBot3D environments.

## Running Tests

```bash
cd kinder-rl
pytest tests/ -v
```

## Development

Run CI checks:
```bash
./run_ci_checks.sh
```
