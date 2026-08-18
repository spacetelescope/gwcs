# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Sphinx documentation build configuration file.
#
# The documentation is built from the installed package, so no source-tree path
# needs to be added here.

import tomllib
from datetime import datetime
from importlib.metadata import distribution
from pathlib import Path

# -- Extensions and general options -------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "numpydoc",
    "pytest_doctestplus.sphinx.doctestplus",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "sphinx_inline_tabs",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "astropy.org": ("https://www.astropy.org/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
}

exclude_patterns = ["_build", "_templates"]
rst_epilog = "\n.. _Astropy: https://www.astropy.org\n"

# -- Project information ------------------------------------------------------

# Read the package metadata so the documentation title and version stay in sync.
with (Path(__file__).parent.parent / "pyproject.toml").open("rb") as metadata_file:
    configuration = tomllib.load(metadata_file)
    metadata = configuration["project"]

project = metadata["name"]
author = metadata["authors"][0]["name"]
copyright = f"{datetime.now().year}, {author}"  # noqa: A001

release = distribution(project).version
version = ".".join(release.split(".")[:2])

# -- HTML output ---------------------------------------------------------------
html_title = f"{project} v{release}"

htmlhelp_basename = f"{project}doc"

# -- LaTeX output --------------------------------------------------------------
latex_documents = [
    ("index", project + ".tex", project + " Documentation", author, "manual")
]
latex_logo = "_static/images/logo-light-mode.png"

# -- Manual page output --------------------------------------------------------
man_pages = [("index", project.lower(), project + " Documentation", [author], 1)]

# -- Autodoc and automodapi options -------------------------------------------
autoclass_content = "both"
automodapi_toctreedirnm = "api"
autosummary_generate = True
default_role = "obj"
numpydoc_show_class_members = False
numpydoc_xref_aliases = {
    "function": ":term:`python:function`",
    "iterator": ":term:`python:iterator`",
    "mapping": ":term:`python:mapping`",
}
numpydoc_xref_ignore = {
    "optional",
    "or",
    "keyword-only",
    "instance",
    "default",
    "type",
    "thereof",
    "subclass",
    "method",
    "of",
    "class",
}

# -- Cross-reference options ---------------------------------------------------
nitpicky = True
nitpick_ignore = [
    ("py:obj", "astropy.modeling.projections.projcodes"),
    ("py:attr", "gwcs.WCS.bounding_box"),
    ("py:meth", "gwcs.WCS.footprint"),
]

# -- HTML theme -----------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
# Do not add the theme's default sidebar alongside the custom layout.
html_sidebars = {}
html_logo = ""

html_theme_options = {
    "light_logo": "images/stsci_logo.png",
    "dark_logo": "images/stsci_logo.png",
}

pygments_style = "monokai"
# Furo uses this style for dark-mode code blocks.
pygments_dark_style = "monokai"

# Render inheritance diagrams as SVG files.
graphviz_output_format = "svg"

graphviz_dot_args = [
    "-Nfontsize=10",
    "-Nfontname=Helvetica Neue, Helvetica, Arial, sans-serif",
    "-Efontsize=10",
    "-Efontname=Helvetica Neue, Helvetica, Arial, sans-serif",
    "-Gbgcolor=white",
    "-Gfontsize=10",
    "-Gfontname=Helvetica Neue, Helvetica, Arial, sans-serif",
]
