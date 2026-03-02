#!/bin/bash
./run_autoformat.sh
mypy .
pytest . --pylint -m pylint --pylint-rcfile=.pylintrc --ignore=notebooks
pytest tests/
pytest notebooks/ --nbmake --nbmake-timeout=120
