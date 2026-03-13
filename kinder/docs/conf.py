"""Configuration file for the Sphinx documentation builder.

This file contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))


# -- Project information -----------------------------------------------------

project = 'KinDER'

_AUTHORS = (
    'Yixuan Huang, Bowen Li, Vaibhav Saxena, Utkarsh Aashu Mishra, '
    'Yichao Liang, Liang Ji, Lihan Zha, Jimmy Wu, Nishanth Kumar, '
    'Sebastian Scherer, Danfei Xu, Tom Silver'
)

copyright = f'2026, {_AUTHORS}'  # pylint: disable=redefined-builtin
author = _AUTHORS

# The full version, including alpha/beta/rc tags
release = '1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',  # Automatically generate documentation from docstrings
    'sphinx.ext.napoleon',  # Support for Google-style and NumPy-style docstrings
    'myst_parser',  # Support for Markdown files
]

# Mock imports for packages that are not available during doc build
autodoc_mock_imports = [
    'relational_structs',
    'mujoco',
    'dm_control',
    'pybullet',
    'pybullet_helpers',
    'gymnasium',
    'gym',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
