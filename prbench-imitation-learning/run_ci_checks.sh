#!/bin/bash
./run_autoformat.sh
mypy src/
pytest . --pylint -m pylint --pylint-rcfile=.pylintrc
pytest tests/
