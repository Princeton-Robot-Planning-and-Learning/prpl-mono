# AlphaTAMP

This is a codebase that the PRPL lab is using for multiple projects related to accelerating TAMP through learning.

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
