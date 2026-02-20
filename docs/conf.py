# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('./docs'))

project = 'shipp'
copyright = '2026, Jenna Iori'
author = 'Jenna Iori'
release = '0.1'

# The master toctree document.
master_doc = "index"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
    "sphinx_tabs.tabs",
    "autodoc2"
]

# Napoleon settings
napoleon_google_docstring = True

myst_enable_extensions = [
    "linkify",
    "dollarmath",
]

autodoc2_packages = [
    "../src/shipp/components.py",
    "../src/shipp/kernel.py",
    "../src/shipp/kernel_pyomo.py",
    "../src/shipp/io_functions.py",
    "../src/shipp/timeseries.py",
]

autodoc2_docstring_parser_regexes = [
    # this will render all docstrings as Markdown
    (r".*", "docstrings_parser"),
]

# Ensure the docs directory is in the Python path
sys.path.insert(0, os.path.abspath('.'))

# Register the custom parser

from docstrings_parser import NapoleonParser

autodoc2_parser = {
    "docstrings_parser": NapoleonParser,
}

myst_enable_extensions = [
    "dollarmath",
    "fieldlist",
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

# html_theme = "sphinx_rtd_theme"
# html_theme = "sphinx_book_theme"
# html_theme = "pydata_sphinx_theme"
html_theme = "shibuya"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    "accent_color": "violet",
    "light_logo": "_static/shipp_logo_small.png",
    "dark_logo": "_static/shipp_logo_small_dark.png",
}

html_favicon = "_static/shipp_logo_small_dark.png"

html_title = "SHIPP"



# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_static_path = ['_static']
