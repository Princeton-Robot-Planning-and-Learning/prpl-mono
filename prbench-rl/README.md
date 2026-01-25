# Reinforcement Learning Baselines for PRBench

![workflow](https://github.com/Jaraxxus-Me/prbench-rl/actions/workflows/ci.yml/badge.svg)

This package provides RL baselines (PPO, SAC) for the PRBench physical reasoning benchmark environments.

## Installation

1. Recommended: create and source a virtualenv (perhaps with [uv](https://github.com/astral-sh/uv))
2. Install this repo: `pip install -e ".[develop]"`

For monorepo installation, install dependencies first:
```bash
pip install -r prpl_requirements.txt
pip install -e ".[develop]"
```

## Usage

### Running Experiments

Experiments are configured using [Hydra](https://hydra.cc/). Run from the `prbench-rl` directory:

```bash
cd prbench-rl
python experiments/run_experiment.py agent=<agent_config> env_id=<environment_id> [options]
```

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
| Motion2D | `prbench/Motion2D-p{0,1,2}-v0` | `ppo_motion2d_*`, `sac_motion2d_*` |
| StickButton2D | `prbench/StickButton2D-b{1,2,3}-v0` | `ppo_stickbutton2d_*`, `sac_stickbutton2d_*` |
| DynObstruction2D | `prbench/DynObstruction2D-o{1,2,3}-v0` | `ppo_dynobstruction2d`, `sac_dynobstruction2d` |
| DynPushPullHook2D | `prbench/DynPushPullHook2D-o{1,2,3}-v0` | `ppo_dynpushpullhook2d`, `sac_dynpushpullhook2d` |

#### 3D Environments
| Environment | ID | Agent Configs |
|-------------|-----|---------------|
| BaseMotion3D | `prbench/BaseMotion3D-v0` | `ppo_basemotion3d`, `sac_basemotion3d` |
| Transport3D | `prbench/Transport3D-o{1,2,3}-v0` | `ppo_transport3d`, `sac_transport3d` |
| Shelf3D | `prbench/Shelf3D-o{1,2,3}-v0` | `ppo_shelf3d`, `sac_shelf3d` |

### Example Commands

**Train PPO on BaseMotion3D:**
```bash
python experiments/run_experiment.py \
    agent=ppo_basemotion3d \
    env_id="prbench/BaseMotion3D-v0" \
    max_episode_steps=300 \
    eval_episodes=50 \
    seed=0
```

**Train SAC on DynObstruction2D:**
```bash
python experiments/run_experiment.py \
    agent=sac_dynobstruction2d \
    env_id="prbench/DynObstruction2D-o1-v0" \
    max_episode_steps=200 \
    eval_episodes=50 \
    seed=42
```

**Train PPO on Transport3D with custom hyperparameters:**
```bash
python experiments/run_experiment.py \
    agent=ppo_transport3d \
    env_id="prbench/Transport3D-o2-v0" \
    agent.args.total_timesteps=2000000 \
    agent.args.hidden_size=256 \
    agent.args.learning_rate=1e-4 \
    seed=0
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `agent` | Agent configuration file (without .yaml) | `ppo_motion2d_0_passage` |
| `env_id` | PRBench environment ID | `prbench/Motion2D-p0-v0` |
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
Environment: prbench/BaseMotion3D-v0
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

## Running Tests

```bash
cd prbench-rl
pytest tests/ -v
```

## Development

Run CI checks:
```bash
./run_ci_checks.sh
```
