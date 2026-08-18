# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Sphinx documentation build configuration file.
#
# The documentation is built from the installed package, so no source-tree path
# needs to be added here.

import importlib
import inspect
import re
import tomllib
import warnings
from datetime import datetime
from importlib.metadata import distribution
from pathlib import Path

from astropy.utils.exceptions import AstropyDeprecationWarning
from sphinx.ext.intersphinx import missing_reference as intersphinx_missing_reference

# Import the deprecated astropy.samp here so that when sphinx_autodoc_type_hints
# imports all of astropy to resolve type hints, it doesn't raise a deprecation
# warning.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", AstropyDeprecationWarning)
    import astropy.samp  # noqa: F401

# -- Extensions and general options -------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "pytest_doctestplus.sphinx.doctestplus",
    "sphinx_inline_tabs",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

exclude_patterns = ["_build", "_templates"]
templates_path = ["_templates"]

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

# -- Autodoc options -----------------------------------------------------------
autoclass_content = "both"
autosummary_generate = True
default_role = "obj"

# Document members re-exported via a module's __all__ (e.g. gwcs.wcs)
autosummary_ignore_module_all = False

# -- Napoleon options ----------------------------------------------------------
# Parse NumPy style docstrings into ``:param:``/``:type:`` fields so that
# sphinx_autodoc_typehints can fill in the types from the annotations.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
# Emit ``:ivar:`` fields rather than ``.. attribute::`` directives, which would
# otherwise duplicate the members autodoc already documents.
napoleon_use_ivar = True

# -- Type hint options ---------------------------------------------------------
# Fill in types from the annotations even when the docstring omits them.
always_document_param_types = True
typehints_defaults = "comma"


# -- Cross-reference options ---------------------------------------------------
nitpicky = True
nitpick_ignore = [
    ("py:obj", "astropy.modeling.projections.projcodes"),
    ("py:obj", "n_inputs"),
    ("py:data", "typing.Union"),
    ("py:attr", "gwcs.WCS.bounding_box"),
    ("py:meth", "gwcs.WCS.footprint"),
    # Unqualified names left by `from __future__ import annotations`
    ("py:class", "WorldAxisObjectClasses"),
]

# Prose words that appear in hand-written docstring types (here and in astropy)
# which Sphinx tries, and fails, to resolve as cross-references.
nitpick_ignore += [
    ("py:class", name)
    for name in (
        "array-like",
        "default",
        "iterable",
        "ndarray",
        "np.nan",
        "optional",
        "scalar",
    )
]

suppress_warnings = ["config.cache"]

# -- HTML theme -----------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
# Do not add the theme's default sidebar alongside the custom layout.
html_sidebars = {}

html_theme_options = {
    "light_logo": "images/stsci_logo.png",
    "dark_logo": "images/stsci_logo.png",
}

pygments_style = "monokai"
# Furo uses this style for dark-mode code blocks.
pygments_dark_style = "monokai"
# Render inheritance diagrams in SVG
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

# `from __future__ import annotations` leaves these names unqualified in the
# rendered type hints, so rewrite them to their fully qualified names, which
# intersphinx can resolve.
unqualified_type_names = {
    "Model": "astropy.modeling.Model",
    "ModelBoundingBox": "astropy.modeling.bounding_box.ModelBoundingBox",
    "CompoundBoundingBox": "astropy.modeling.bounding_box.CompoundBoundingBox",
}

# Type hints report the defining private submodule, e.g.
# ``gwcs.coordinate_frames._base.WorldAxisObjectClass``.
private_module = re.compile(r"\._\w+(?=\.)")


def resolve_missing_reference(app, env, node, contnode):
    target = node.get("reftarget", "")

    if target.rsplit(".", 1)[-1].startswith("_"):
        return contnode.deepcopy()

    if external := unqualified_type_names.get(target):
        node["reftarget"] = external
        return intersphinx_missing_reference(app, env, node, contnode)

    public = private_module.sub("", target)
    if target.startswith("gwcs.") and public != target:
        node["reftarget"] = public
        return env.get_domain("py").resolve_xref(
            env,
            node.get("refdoc"),
            app.builder,
            node["reftype"],
            public,
            node,
            contnode,
        )

    return None


def skip_excluded_inherited_members(*event_args):
    obj = event_args[3]
    owner = getattr(obj, "__objclass__", None)
    wrapped = getattr(obj, "__func__", obj)
    if (
        getattr(wrapped, "__qualname__", "").startswith("str.")
        or owner is str
        or (
            owner is not None
            and owner.__name__ == "Model"
            and owner.__module__ == "astropy.modeling.core"
        )
    ):
        return True

    return None


def setup(app):
    app.connect("autodoc-skip-member", skip_excluded_inherited_members)
    app.connect("missing-reference", resolve_missing_reference)


def is_property(modname, qualname, attr):
    """Used by the autosummary class template to pick autoproperty vs autoattribute."""
    obj = importlib.import_module(modname)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    try:
        member = inspect.getattr_static(obj, attr)
    except AttributeError:
        return False
    return isinstance(member, property)


def is_inherited_model_name(modname, qualname, attr):
    if attr != "name":
        return False

    obj = importlib.import_module(modname)
    for part in qualname.split("."):
        obj = getattr(obj, part)

    return "name" not in obj.__dict__ and any(
        "name" in base.__dict__ and base.__module__.startswith("astropy.modeling")
        for base in obj.__mro__[1:]
    )


autosummary_generate = True
# Document members re-exported via a module's __all__ (e.g. gwcs.wcs)
autosummary_ignore_module_all = False
autosummary_context = {
    "is_inherited_model_name": is_inherited_model_name,
    "is_property": is_property,
}
