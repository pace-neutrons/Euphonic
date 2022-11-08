# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

# Ensure the euphonic source directory is on the path. Otherwise it can
# quietly document the current Python's installed Euphonic, which we don't want!
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'Euphonic'
copyright = '2020, Rebecca Fair'
author = 'Rebecca Fair'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.napoleon',
        'sphinxarg.ext',
        'sphinx_autodoc_typehints',
        'sphinx.ext.doctest'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

master_doc = 'index' # Otherwise readthedocs searches for contents.rst


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Autodoc settings
autodoc_member_order = 'bysource'
add_module_names = False

# Napoleon docstring style settings
napoleon_use_rtype = False
napoleon_use_param = True
napoleon_use_ivar = True

# Configure Autodoc's autodoc-skip-member event to not skip __init__ methods
def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip

def setup(app):
    app.connect("autodoc-skip-member", skip)

rst_prolog = """
.. |q| replace:: \|\ q\ \|
"""

# -- Doctest configuration ---------------------------------------------------

# Global setup, import required modules and initialise fnames
# fnames are files that are required by each doctest, and will be
# copied from Euphonic's test directory in the setup for that test.
# Initialise fnames here so it can safely be used in cleanup, even if
# a specific test setup doesn't define it.
doctest_global_setup = """
import os, shutil
from tests_and_analysis.test.utils import get_castep_path, get_data_path
fnames = []
"""
# Cleanup, ensure figures are closed and any copied files are removed
# If a test only requires one file, fnames may not be a list.
doctest_global_cleanup = """
import matplotlib.pyplot as plt
plt.close('all')
if not isinstance(fnames, list):
    fnames = [fnames]
for fname in fnames:
    os.remove(fname)
"""