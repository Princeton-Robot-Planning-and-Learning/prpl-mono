# PRPL Utils

![workflow](https://github.com/Princeton-Robot-Planning-and-Learning/prpl-utils/actions/workflows/ci.yml/badge.svg)

Miscellaneous Python utilities from the Princeton Robot Planning and Learning group.
- **Motion planning**: RRT, BiRRT
- **PDDL planning**: interfaces to pyperplan, Fast Downward
- **Heuristic search**: A*, GBFS, hill-climbing, policy-guided A*
- **Gymnasium**: agent interface, helper spaces
- **Other**: a few other miscellaneous utils

## Requirements

- Python 3.10+
- Tested on MacOS Monterey and Ubuntu 22.04

## Installation

1. Recommended: create and source a virtualenv.
2. `pip install -e ".[develop]"`

## Check Installation

Run `./run_ci_checks.sh`. It should complete with all green successes in 5-10 seconds.

## Contributing

Pull requests welcome. Note that this is meant to be a **lightweight** package. For example, it should be safe to use in homework assignments with minimal assumptions about the user's setup. Do not add heavy-duty dependencies.
