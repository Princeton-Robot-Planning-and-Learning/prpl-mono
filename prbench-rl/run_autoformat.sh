#!/bin/bash
python -m black .
docformatter -i -r .
isort .
