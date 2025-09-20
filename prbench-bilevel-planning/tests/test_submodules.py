"""Tests that we can import code from submodules."""

import importlib


def test_submodule_imports():
    """Dynamically test that submodules can be imported."""

    for module in ["bilevel_planning", "prbench"]:
        importlib.import_module(module)
