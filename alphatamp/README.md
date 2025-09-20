# AlphaTAMP

![workflow](https://github.com/tomsilver/alphatamp/actions/workflows/ci.yml/badge.svg)

This is a codebase that the PRPL lab is using for multiple projects related to accelerating TAMP through learning.

## Installation

1. Requirements: Python >=3.11 and <=3.12.
2. Recommended: use a virtual environment. For example, we like [uv](https://github.com/astral-sh/uv).
    - Install `uv`:  ```curl -LsSf https://astral.sh/uv/install.sh | sh```
    - Create the virtual environment: `uv venv --python=3.11`
    - Activate the environment (every time you start a new terminal): `source .venv/bin/activate`
3. Clone this repository and `cd` into it.
4. Install this repository and its dependencies:
    - If you are using `uv`, do ```uv pip install -e ".[develop]"```
    - Otherwise, just do ```pip install -e ".[develop]"```
5. Check the installation: ```./run_ci_checks.sh```
