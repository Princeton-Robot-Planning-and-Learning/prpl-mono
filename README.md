# KinDER: A Physical Reasoning Benchmark for Robot Learning and Planning

This repository contains code for the paper: ``KinDER: A Physical Reasoning Benchmark for Robot Learning and Planning``. 

KinDER, short for Kinematic and Dynamic Embodied Reasoning, targets physical reasoning challenges arising in robot learning and planning. It comprises 25 procedurally generated environments, a Gymnasium-compatible Python library with parameterized skills and demonstrations, and a standardized evaluation suite with 8 implemented baselines spanning task and motion planning, imitation learning, reinforcement learning, and foundation-model-based approaches.

The environments are designed to isolate five core physical reasoning challenges: basic spatial relations, nonprehensile multi-object manipulation, tool use, combinatorial geometric constraints, and dynamic constraints, disentangled from perception, language understanding, and application-specific complexity.

## Repository Structure

This repository consists of multiple Python packages that can be installed separately. The basic structure of this repo is:

```
kinder-mono/
  .github/workflows/ci.yml
  our-utils/
    pyproject.toml
    src/our_utils
    tests/
  our-llm-utils/
    pyproject.toml
    src/our-llm-utils
    tests/
  kinder/
    pyproject.toml
    src/kinder
    tests/
  ...
```

## Instructions for Usage

### Installation

We recommend installing using `uv`:

```
uv run python scripts/install_all.py
```

If you encounter issues, see the Troubleshooting section below.

## Task Demonstrations

We provide 100+ demonstrations for 10 tasks. Demonstration files are omitted due to space restrictions.

## Reproducing Baselines

### Reinforcement Learning

See details for running RL baselines in kinder-rl/README.md

### Bilevel Planning

Use following commands to run bilevel planning baselines:

```
cd kinder-bilevel-planning
python experiments/run_experiment.py -m seed='range(300,305)' env=Motion2D-p0-v0,StickButton2D-b1-v0,BaseMotion3D-v0,Transport3D-o2-v0,Shelf3D-o1-v0  hydra/launcher=joblib
```

### VLM Planning

Use following commands to run VLM planning baselines:

```
cd kinder-vlm-planning
python experiments/run_experiment.py -m seed='range(300,305)' \
    env=Motion2D-p0-v0,StickButton2D-b1-v0,BaseMotion3D-v0,Transport3D-o2-v0,Shelf3D-o1-v0 \
    vlm_model=gpt-5 rgb_observation=true,false temperature=1 \
    hydra/launcher=joblib
```

### Imitation Learning

We do not include code for running imitation learning baselines. This repository however does contain various tools that were critical to data collection for all tasks.

#### Teleoperation

Dynamic3D: 
```
cd kinder-models
python scripts/teleop_dynamics3d_kinder.py --env-name <env_name>
```

Dynamic2D and Geom2D: 
```
cd kinder
python scripts/collect_demos.py --env_id <your_env_id>
```

Geom3D: 
```
cd kinder-ds-policies
python experiments/collect_demos_ds.py
```

Planning to generate data: 
```
cd kinder-models
python scripts/planning_data_dynamics3d_kinder.py
```


## Troubleshooting

### Installing PyBullet on Recent MacOS

If you encounter issues installing PyBullet on recent versions of MacOS, try this workaround (adapted from [here](github.com/phospho-app/phosphobot/issues/174)):
1. Make sure you are in the virtual environment where you are installing the mono repo.
2. Clone PyBullet: `git clone https://github.com/bulletphysics/bullet3`
3. In `bullet3`, open `examples/ThirdPartyLibs/zlib/zutil.h` and comment out this line by adding `//` at the beginning:
```
#define fdopen(fd, mode) NULL
```
4. Install from source:
```
uv pip install setuptools
python setup.py build
python setup.py install
```

### Installing `gymnasium[box2d]` on MacOS

If you encounter issues installing `gymnasium[box2d]` on MacOS, do `brew install swig` first and then retry.

