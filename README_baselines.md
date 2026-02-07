# Reproducing Baselines

## Reinforcement Learning

See details for running RL baselines in kinder-rl/README.md

## Bi-level Planning

Use following commands to run bi-level planning baselines:

```
$ cd kinder-bilevel-planning
$ python experiments/run_experiment.py -m seed='range(300,305)' env=Motion2D-p0-v0,StickButton2D-b1-v0,BaseMotion3D-v0,Transport3D-o2-v0,Shelf3D-o1-v0  hydra/launcher=joblib
```

## VLM Planning

Use following commands to run VLM planning baselines:

```
$ cd kinder-vlm-planning
$ python experiments/run_experiment.py -m seed='range(300,305)' \
    env=Motion2D-p0-v0,StickButton2D-b1-v0,BaseMotion3D-v0,Transport3D-o2-v0,Shelf3D-o1-v0 \
    vlm_model=gpt-5 rgb_observation=true,false temperature=1 \
    hydra/launcher=joblib
```

## Imitation Learning

We do not include code for running imitation learning baselines. This repository however does contain various tools that were critical to data collection for all tasks.

### Teleoperation

Dynamic3D: kinder-models/scripts/teleop_dynamics3d_prbench.pyGeom2D 

Dynamic2D: kinder/scripts/collect_demos.py

Geom3D: kinder-ds-policies/experiments/collect_demos_ds.py 

Planning to generate data: kinder-models/scripts/planning_data_dynamics3d_kinder.py
