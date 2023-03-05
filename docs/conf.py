# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyBKTR'
copyright = '2023, Julien Lanthier, Mengying Lei, Aurelie Labbe, Lijun Sun'
author = 'Julien Lanthier, Mengying Lei, Aurelie Labbe, Lijun Sun'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc', 
    'sphinx.ext.mathjax', 
    'sphinx.ext.napoleon', 
    'sphinx.ext.viewcode',
    'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autoclass_content = 'both'
autodoc_typehints_format = 'short'
python_use_unqualified_type_names = True

nbsphinx_custom_formats = {
    ".md": ["jupytext.reads", {"fmt": "mystnb"}],
}
nbsphinx_execute = 'never'
