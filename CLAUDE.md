# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the **Princeton Robot Planning and Learning (PRPL) monorepo**, containing multiple interdependent Python packages for physical reasoning benchmarks and robotics research. The central package is **PRBench**, a physical reasoning benchmark for robotics that tests planning, reinforcement learning, and foundation model approaches on tasks requiring understanding of kinematics, geometry, dynamics, contact, and tool use.

## Monorepo Structure

The repository contains ~20 Python packages organized as:
- **Core Infrastructure**: `relational-structs`, `prpl-utils`, `prpl-llm-utils`, `prpl-perception-utils`
- **Benchmark**: `prbench` (main benchmark), `prbench-models`, `prbench-bilevel-planning`, `prbench-rl`, `prbench-vlm-planning`
- **Planning**: `bilevel-planning` (core planning algorithms)
- **Utilities**: `pybullet-helpers`, `toms-geoms-2d`
- **Others**: `alphatamp`, `pr2s2r`, `programmatic-policy-learning`

Packages with dependencies on other monorepo packages include a `prpl_requirements.txt` file listing local dependencies (as relative paths like `../relational-structs`).

## Installation & Development

### Initial Setup
```bash
# Install all packages in topological order
uv run python scripts/install_all.py

# Alternative: Install individual package
cd <package-name>
uv pip install -r prpl_requirements.txt  # if exists
uv pip install -e ".[develop]"
```

### Running Tests and Checks

**For a single package:**
```bash
cd <package-name>
./run_ci_checks.sh  # Runs autoformat, mypy, pylint, pytest
```

**For all packages:**
```bash
./run_all_ci_checks.sh  # Runs CI checks across all packages
```

**Individual tools:**
```bash
# Autoformat (black, docformatter, isort)
./run_autoformat.sh

# Type checking
mypy .

# Linting
pytest . --pylint -m pylint --pylint-rcfile=.pylintrc

# Tests
pytest tests/

# Run single test
pytest tests/test_file.py::test_function_name -v
```

## Key Architecture Concepts

### Object-Centric States (`relational-structs`)

All PRBench environments use **object-centric state representations** instead of flat vectors. The core data structure is `ObjectCentricState` from `relational_structs.object_centric_state`:

```python
from relational_structs.object_centric_state import ObjectCentricState
# States are dictionaries mapping Object -> numpy array of features
# Objects have types (Type) which define their feature schema
```

**Conversion between vector and object-centric:**
```python
env = prbench.make("prbench/Obstruction2D-o3-v0")
vec_obs, _ = env.reset()
object_centric_obs = env.observation_space.devectorize(vec_obs)
recovered_vec = env.observation_space.vectorize(object_centric_obs)
```

Key methods:
- `state.get(obj, "feature_name")` - get feature value
- `state.set(obj, "feature_name", value)` - set feature value
- `state.get_objects(object_type)` - get all objects of a type
- `state.pretty_str()` - human-readable table format

### Bilevel Planning (`bilevel-planning`)

Bilevel planning uses **abstractions** to decompose planning into high-level (abstract) and low-level (concrete) search:

- **Abstract state space** (discrete, symbolic) vs **state space** (continuous, geometric)
- **Abstract actions** (e.g., "grasp cup") vs **actions** (robot joint trajectories)
- **Abstract successor function**: returns valid (abstract_action, next_abstract_state) pairs

**Key planners:**
1. **Abstract BFS**: Breadth-first search in abstract space with trajectory sampling
2. **SeSamE** (Search, Sample, Execute): Multi-abstract plan generation with backtracking refinement
3. **Relational abstractions**: Uses PDDL for efficient abstract planning

### PRBench Environments (`prbench`)

**Environment categories** (in `src/prbench/envs/`):
- `geom2d/`: Geometric 2D planning (Obstruction2D, Motion2D, ClutteredStorage2D, etc.)
- `dynamic2d/`: Dynamic 2D physics (DynObstruction2D, DynPushT, etc.) using Pymunk
- `geom3d/`: Geometric 3D planning (Motion3D, Obstruction3D) using PyBullet
- `tidybot/`: Dynamic 3D manipulation tasks using MuJoCo

**Environment registration** happens in `src/prbench/__init__.py`:
```python
import prbench
prbench.register_all_environments()
env = prbench.make("prbench/Obstruction2D-o3-v0")
```

Environments follow Gymnasium API: `reset()`, `step()`, `render()`, with sparse rewards and procedural generation.

### PyBullet Helpers (`pybullet-helpers`)

Utilities for PyBullet-based 3D environments:
- `camera.py`: Camera configuration and rendering
- `inverse_kinematics.py`, `motion_planning.py`: IK and motion planning
- `manipulation.py`: Grasping and manipulation primitives
- `robots/`: Robot-specific utilities (PR2, Fetch, etc.)

### Optional Dependencies

PRBench uses optional dependency groups for different environment types:
- `[geom2d]`: No extra dependencies
- `[dynamic2d]`: pygame, pymunk
- `[tidybot]`: mujoco, opencv-python
- `[geom3d]`: pybullet
- `[all]`: All environment dependencies
- `[develop]`: All + development tools (black, mypy, pylint, pytest)

Install specific groups: `uv pip install -e ".[geom2d,dynamic2d]"`

## Code Quality Standards

- **Python 3.10-3.12** required
- **Formatting**: black (line-length 88), isort (profile black), docformatter
- **Type checking**: mypy with strict equality, disallow_untyped_calls
- **Linting**: pylint
- **Testing**: pytest with coverage
- All checks must pass before merging (enforced by GitHub Actions CI)

## Common Issues

### PyBullet on macOS
If installation fails, manually build from source:
```bash
git clone https://github.com/bulletphysics/bullet3
# Edit examples/ThirdPartyLibs/zlib/zutil.h: comment out line with fdopen
uv pip install setuptools
cd bullet3 && python setup.py build && python setup.py install
```

### gymnasium[box2d] on macOS
```bash
brew install swig
uv pip install gymnasium[box2d]
```

### Headless Rendering
The code auto-detects headless mode and sets appropriate OpenGL backends (OSMesa for Linux, GLFW for macOS).

## Git Workflow

- Main branch: `main`
- Current development often happens on feature branches
- Use `./run_all_ci_checks.sh` before creating PRs
- CI runs autoformat, mypy, pylint, and pytest on all packages
