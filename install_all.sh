#!/usr/bin/env bash
./run_in_all_packages.sh 'if [ -f prpl_requirements.txt ]; then uv pip install -r prpl_requirements.txt; fi'
./run_in_all_packages.sh 'uv pip install -e ."[develop]"'
