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
  kinder/
    pyproject.toml
    src/kinder
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

## Publishing Packages to PyPI

Some packages in this monorepo are published to PyPI. To publish a new version:

1. Update the `version` in the package's `pyproject.toml`.
2. From the package directory, build and publish:
```bash
cd <package-dir>
rm -rf dist/ build/ src/*.egg-info/
uv build
uv publish dist/*
```

`uv publish` requires a PyPI token. Set it via:
```bash
export UV_PUBLISH_TOKEN=pypi-YOUR_TOKEN_HERE
```

### Packages currently on PyPI

| Package | PyPI name | Directory |
|---------|-----------|-----------|
| prpl_utils | `prpl_utils` | `prpl-utils/` |
| relational_structs | `relational_structs` | `relational-structs/` |
| tomsgeoms2d | `tomsgeoms2d` | `toms-geoms-2d/` |
| pybullet_helpers | `pybullet_helpers` | `pybullet-helpers/` |

### When to publish

Publish a new version whenever you make changes to a package that external repos (e.g. `kinder`) depend on. Remember to bump the version number each time — PyPI does not allow re-uploading the same version.

### Installing `gymnasium[box2d]` on MacOS

If you encounter issues installing `gymnasium[box2d]` on MacOS, do `brew install swig` first and then retry.
