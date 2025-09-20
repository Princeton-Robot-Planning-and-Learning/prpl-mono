# relational-structs

Data structures for relational state and action spaces in robot environments.

Derived from [predicators](https://github.com/Learning-and-Intelligent-Systems/predicators).

## Requirements

- Python 3.10+
- Tested on MacOS Catalina

## Installation

We strongly recommend [uv](https://docs.astral.sh/uv/getting-started/installation/). The steps below assume that you have `uv` installed. If you do not, just remove `uv` from the commands and the installation should still work.

```
# Install PRPL dependencies.
uv pip install -r prpl_requirements.txt
# Install this package and third-party dependencies.
uv pip install -e ".[develop]"
```

## Check Installation

Run `./run_ci_checks.sh`. It should complete with all green successes in 5-10 seconds.
