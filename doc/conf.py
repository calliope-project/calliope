# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

from sphinx.builders.singlehtml import SingleFileHTMLBuilder, StandaloneHTMLBuilder

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("_themes"))

from helpers import generate_readable_schema  # noqa: E402

# Generates readable markdown files from YAML schema
generate_readable_schema.process()

# Redefine supported_image_types for the HTML builder to prefer PNG over SVG
image_types = ["image/png", "image/svg+xml", "image/gif", "image/jpeg"]
StandaloneHTMLBuilder.supported_image_types = image_types
SingleFileHTMLBuilder.supported_image_types = image_types


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Calliope"
copyright = "Since 2013 Calliope contributors listed in ><a href='https://github.com/calliope-project/calliope/blob/main/AUTHORS'>AUTHORS</a> (Apache 2.0 licensed)<br><a href='https://joss.theoj.org/papers/10.21105/joss.00825'>Academic reference</a>"

__version__ = ""
# Sets the __version__ variable
exec(open("../src/calliope/_version.py").read())

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
version = release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "sphinx_search.extension",
    "myst_parser",
]

# The suffix of source filenames.
source_suffix = ".rst"

# A string of reStructuredText that will be included at the beginning of every
# source file that is read
rst_prolog = """
.. role:: python(code)
   :language: python

.. role:: yaml(code)
   :language: yaml

.. role:: sh(code)
   :language: sh
"""

# The root toctree document.
root_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "flask_theme_support.FlaskyStyle"

# -- Options for extensions --------------------------------------------------

# Ensure that mathjax will render large equations
mathjax3_config = {"tex": {"MAXBUFFER": 25 * 1024}}

nbviewer_url = "https://nbviewer.org/url/calliope.readthedocs.io/"

# Generate RTD base url: if a dev version, point to "latest", else "v..."
if "dev" in __version__:
    docs_base_url = "en/latest/"
else:
    docs_base_url = "en/v{}/".format(__version__)
extlinks = {"nbviewer_docs": (nbviewer_url + docs_base_url + "%s", None)}

# Mock modules for Read The Docs autodoc generation

MOCK_MODULES = [
    "xarray",
    "pandas",
    "numpy",
    "pyomo",
    "pyparsing",
    "netCDF4",
    "geographiclib",
]
autodoc_mock_imports = MOCK_MODULES
autodoc_typehints = "both"
autodoc_member_order = "bysource"
autoclass_content = "both"

intersphinx_mapping = {
    "python": ("http://docs.python.org/3", None),
    "xarray": ("https://docs.xarray.dev/en/v2022.03.0", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme_path = ["_themes"]
html_theme = "flask_calliope"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    "index": [
        "sidebar_title.html",
        "sidebar_search.html",
        "sidebar_downloads.html",
        "sidebar_toc.html",
    ],
    "**": ["sidebar_title.html", "sidebar_search.html", "sidebar_toc.html"],
}

# Output file base name for HTML help builder.
htmlhelp_basename = "Calliopedoc"

# -- Options for LaTeX output --------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-latex-output

latex_elements = {
    "papersize": "letterpaper",  # The paper size ('letterpaper' or 'a4paper').
    "pointsize": "10pt",  # The font size ('10pt', '11pt' or '12pt').
    "preamble": "",  # Additional stuff for the LaTeX preamble.
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    (
        "index",
        "Calliope.tex",
        "Calliope Documentation",
        "Calliope contributors",
        "manual",
    )
]
imgmath_latex_preamble = r"\usepackage{breqn}"

# -- Options for manual page output --------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-manual-page-output

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ("index", "calliope", "Calliope Documentation", ["Calliope contributors"], 1)
]

# -- Options for Texinfo output ------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-texinfo-output

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        "index",
        "Calliope",
        "Calliope Documentation",
        "Calliope contributors",
        "Calliope",
        "One line description of project.",
        "Miscellaneous",
    )
]

# Edit linkcheck config
linkcheck_anchors = False
