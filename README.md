# Princeton Robot Planning and Learning Monorepo

This repository contains code under active development by the Princeton Robot Planning and Learning group.

There are multiple Python packages that can be installed and developed separately. They are included in a monorepo because some are interdependent and we want to make sure that changes in one repository does not break code in another.

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
2. Follow the README instructions in the package or packages that you want to edit.
3. Open a pull request on this repository.

### Adding a New Package
Instructions coming later. In the meantime, use one of the existing packages as a reference.

## Using a Package Externally
You can use any individual package externally. For example:
```
uv pip install "prpl_utils@git+https://github.com/Princeton-Robot-Planning-and-Learning/prpl-mono.git#subdirectory=prpl-utils"
```
But beware that things are changing. Pinning commits is a good idea if you need stable code.

