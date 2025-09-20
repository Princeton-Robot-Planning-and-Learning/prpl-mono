# PRBench

A **p**hysical **r**easoning **bench**mark for robotics.

There's growing excitement around large language models and their ability to "reason"—but reasoning isn't just about tokens and text. **Robots must reason too**: over long horizons, under uncertainty, and with sparse feedback. And unlike purely symbolic systems, **robotic reasoning is physical**: it's grounded in low-level, continuous state and action spaces. It requires understanding kinematics, geometry, dynamics, contact, force, tool use, and more.

This benchmark is designed for this kind of physical reasoning with robots. We invite researchers to try their best task and motion planning, reinforcement learning, imitation learning, and foundation models approaches. We hope that PRBench will bridge perspectives and foster shared progress toward physically intelligent robots.

## :clock1: Status

For currently implemented environments, see `docs/envs`.

Unless you have had a direct conversation with a maintainer, this code is not ready for you! But check back soon.

## :zap: Usage Example

### Basic Usage (Gym API)

```python
import prbench
prbench.register_all_environments()
env = prbench.make("prbench/Obstruction2D-o3-v0")  # 3 obstructions
obs, info = env.reset()  # procedural generation
action = env.action_space.sample()
next_obs, reward, terminated, truncated, info = env.step(action)
img = env.render()  
```

### Object-Centric States

All environments in PRBench use object-centric states. For example:

```python
from prbench.envs.geom2d.obstruction2d import ObjectCentricObstruction2DEnv
env = ObjectCentricObstruction2DEnv(num_obstructions=3)
obs, _ = env.reset(seed=123)
print(obs.pretty_str())
```
Here, `obs` is an [ObjectCentricState](https://github.com/tomsilver/relational-structs/blob/main/src/relational_structs/object_centric_state.py#L25), and the printout is:
```
############################################################### STATE ###############################################################
type: crv_robot           x         y    theta    base_radius    arm_joint    arm_length    vacuum    gripper_height    gripper_width
-----------------  --------  --------  -------  -------------  -----------  ------------  --------  ----------------  ---------------
robot              0.885039  0.803795  -1.5708            0.1          0.1           0.2         0              0.07             0.01

type: rectangle           x         y    theta    static    color_r    color_g    color_b    z_order      width     height
-----------------  --------  --------  -------  --------  ---------  ---------  ---------  ---------  ---------  ---------
obstruction0       0.422462  0.100001        0         0       0.75        0.1        0.1        100  0.132224   0.0766399
obstruction1       0.804663  0.100001        0         0       0.75        0.1        0.1        100  0.0805652  0.0955062
obstruction2       0.559246  0.100001        0         0       0.75        0.1        0.1        100  0.12608    0.180172

type: target_block          x         y    theta    static    color_r    color_g    color_b    z_order     width    height
--------------------  -------  --------  -------  --------  ---------  ---------  ---------  ---------  --------  --------
target_block          1.20082  0.100001        0         0   0.501961          0   0.501961        100  0.138302  0.155183

type: target_surface           x    y    theta    static    color_r    color_g    color_b    z_order     width    height
----------------------  --------  ---  -------  --------  ---------  ---------  ---------  ---------  --------  --------
target_surface          0.499675    0        0         1   0.501961          0   0.501961        101  0.180286       0.1
#####################################################################################################################################
```

For compatibility with baselines, the observations provided by the main environments are vectors. It is easy to convert between vectors and object-centric states. For example:
```python
import prbench
prbench.register_all_environments()
env = prbench.make("prbench/Obstruction2D-o3-v0")
vec_obs, _ = env.reset(seed=123)
object_centric_obs = env.observation_space.devectorize(vec_obs)
recovered_vec_obs = env.observation_space.vectorize(object_centric_obs)
```

## :muscle: Challenges for Existing Approaches

What makes PRBench challenging?

### For Reinforcement Learning

Environments have long horizons and sparse rewards. Users are welcome to engineer dense rewards, but doing so may be nontrivial. Environments also have very diverse task distributions (as in the `reset()` function), so learned policies must generalize.

### For Imitation Learning

As with RL, generalization across tasks is a major challenge for imitation learning. Furthermore, we supply some demonstrations, but they are typically suboptimal, multimodal, and limited in quantity. Users are welcome to collect their own demonstrations.

### For Language Models

The physical reasoning required in PRBench is not easy to represent in natural language alone. Vision-language and vision-language-action models may fare better, but the tasks in PRBench are beyond the capabilities of current VLMs and VLAs.* (*This is an empirical claim that we will test!)

### For Hierarchical Approaches

Approaches that first decide "what to do" and then decide "how to do it" will run into difficulties in PRBench when there are couplings between these high-level and low-level decisions. For example, the exact grasp of an object may determine whether the object can later be placed into a tight space.

### For Task and Motion Planning

PRBench does not provide any models for TAMP. Users are welcome to engineer their own, but doing so may be nontrivial. Furthermore, some environments in PRBench are meant to strain the assumptions that are sometimes made in TAMP. Finally, some environments contain many objects, which may make planning slow even when models are available.

## :octocat: Contributing

### :ballot_box_with_check: Requirements
1. Python >=3.10, <3.13
2. Tested on MacOS Monterey and Ubuntu 22.04 (but we aim to support most platforms)

### :wrench: Installation
We strongly recommend [uv](https://docs.astral.sh/uv/getting-started/installation/). The steps below assume that you have `uv` installed. If you do not, just remove `uv` from the commands and the installation should still work.

Then, choose one of the following based on you need:
- `uv pip install -r optional_prpl_requirements/core.txt && uv pip install -e .` - Installs only core dependencies (matplotlib, numpy, relational_structs, prpl_utils)
- `uv pip install -r prpl_requirements.txt && uv pip install -e .uv pip install -e ".[all]"` - Installs everything (excluding develop)
- `uv pip install -r optional_prpl_requirements/geom2d.txt && uv pip install -e ".[geom2d]"` - Installs only core + geom2d dependencies (no pybullet)
- `uv pip install -r optional_prpl_requirements/dynamic2d.txt && uv pip install -e ".[dynamic2d]"` - Installs only core + dynamic2d dependencies
- `uv pip install -e ".[tidybot]"` - Installs only core + tidybot dependencies
- `uv pip install -r optional_prpl_requirements/geom3d.txt && uv pip install -e ".[geom3d]"` - Installs only core + geom3d dependencies
- `uv pip install -r prpl_requirements.txt && uv pip install -e ".[develop]"` - Installs all + development tools
- Compositionally install the dependencies like `[geom2d,geom3d]`

### :microscope: Check Installation
Run `./run_ci_checks.sh`. It should complete with all green successes.

### :mag: General Guidelines
* All checks must pass before code is merged (see `./run_ci_checks.sh`)
* All code goes through the pull request review process

### :new: Adding New Environments
Some new environment requests are in Issues. To add a new environment, please see the examples in `src/prbench/env`. Also consider:
* Environments are registered in `src/prbench/__init__.py`
* Each environment should have at least one demonstration (see `scripts/collect_demos.py`)
* After collecting a demonstraction, create a video with `scripts/generate_demo_video.py`, which will be used in the autogenerated documentation

### :video_camera: Demonstration Collection
Here is an example of demo collection.
<video src='https://github.com/user-attachments/assets/265b0401-6615-47be-8fca-cb9f409b6945' />

### :oncoming_automobile: Roadmap
For specific environments, we will use issue tracking. Here are higher level TODOs:

- [x] Decide which simulator(s) to use for 3D environments
- [ ] Determine what metrics we want to record, and how
- [ ] Run reinforcement learning baselines
- [ ] Run VLM/VLA baselines
- [ ] Create interface to PDDLStream
- [x] Create interface to "task then motion planning"
- [ ] Run "code as policies" type baselines
- [ ] Collect at least 100 demonstrations per environment
- [ ] Run imitation learning baselines
- [ ] Create website
