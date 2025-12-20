#!/bin/bash
./run_autoformat.sh
mypy .
pytest . --pylint -m pylint --pylint-rcfile=.pylintrc

# Pre-install IKFast modules to avoid race conditions in parallel tests
if [ -f scripts/preinstall_ikfast.py ]; then
    echo "Pre-installing IKFast modules..."
    python scripts/preinstall_ikfast.py
fi

pytest tests/
