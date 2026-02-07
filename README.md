# KinDER: Kinematic and Dynamic Environments for Reasoning

This repository contains code under active development.

There are multiple Python packages that can be installed and developed separately. They are included in a monorepo because some are interdependent and we want to make sure that changes in one package do not break code in another.

The basic structure is:
```
prpl-mono/
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

Packages that depend on other packages in this repo should include a `our_requirements.txt` file.

## Task Demonstrations

We collected demonstrations for all 25 tasks, and provide videos in the media attachment. We omit demonstration files due to space restrictions.

## Instructions for Usage

### Using an Existing Package
1. Clone this repository.
2. Installing all packages in this repository, `uv run python scripts/install_all.py`.
3. Follow the README instructions in the package or packages that you want to edit.
4. Open a pull request on this repository.

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
