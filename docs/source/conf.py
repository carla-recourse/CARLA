# flake8: noqa
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys

sys.path.insert(0, os.path.abspath("../../carla"))


# -- Project information -----------------------------------------------------

project = "CARLA"
copyright = "2021, Martin Pawelczyk, Sascha Bielawski, Johannes van den Heuvel, Tobias Richter, Gjergji Kasneci"
author = "Martin Pawelczyk, Sascha Bielawski, Johannes van den Heuvel, Tobias Richter, Gjergji Kasneci"

# The full version, including alpha/beta/rc tags
release = "0.0.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",
    "sphinx_gallery.load_style",
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
    "numpydoc",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
]

numpydoc_show_class_members = True

autodoc_mock_imports = ["recourse", "cplex", "carla"]

# generate autosummary even if no references
autosummary_generate = True
autosummary_imported_members = True

pygments_style = "sphinx"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []  # type: ignore


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
