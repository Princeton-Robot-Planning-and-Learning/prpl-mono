# Princeton Robot Planning and Learning Monorepo

This repository contains code under active development by the Princeton Robot Planning and Learning group.

There are multiple Python packages that can be installed and developed separately. They are included in a monorepo because some are interdependent and we want to make sure that changes in one package do not break code in another.

The basic structure is:
```
prpl-mono/
  .github/workflows/ci.yml
  prpl-utils/
    pyproject.toml
    src/prpl_utils
    tests/
  prpl-llm-utils/
    pyproject.toml
    src/prpl-llm-utils
    tests/
  prbench/
    pyproject.toml
    src/prbench
    tests/
  ...
```

Packages that depend on other packages in this repo should include a `prpl_requirements.txt` file.

## Instructions for Contributing

### Contributing to an Existing Package
1. Clone this repository.
2. Installing all packages in this repository, `uv run python scripts/install_all.py`.
3. Follow the README instructions in the package or packages that you want to edit.
4. Open a pull request on this repository.

### Adding a New Package
Instructions coming later. In the meantime, use one of the existing packages as a reference.

## Using a Package Externally
You can use any individual package externally. For example:
```
uv pip install "prpl_utils@git+https://github.com/Princeton-Robot-Planning-and-Learning/prpl-mono.git#subdirectory=prpl-utils"
```
But beware that things are changing. Pinning commits is a good idea if you need stable code.

## Troubleshooting

We are doing our best to make installation easy across platforms, but here are some known issues and workarounds.

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
