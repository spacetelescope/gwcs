# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Sphinx documentation build configuration file.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
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
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

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

# -- sphinx.ext.intersphinx configuration ------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

exclude_patterns = ["_build", "_templates"]
templates_path = ["_templates"]

# -- Project information ------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

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
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
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
    "light_logo": "https://github.com/spacetelescope/branding/blob/81b7021231e885bda9c022e6c082b3cf26009b3b/stsci_pri_combo_mark_light_bkgd.png?raw=true",
    "dark_logo": "https://github.com/spacetelescope/branding/blob/81b7021231e885bda9c022e6c082b3cf26009b3b/stsci_pri_combo_mark_dark_bkgd.png?raw=true",
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

typing_type_aliases = frozenset(
    f"gwcs.typing.{name}"
    for name in (
        "AstropyBuiltInFrame",
        "AxesType",
        "ForwardTransform",
        "LowLevelArray",
        "LowLevelInput",
        "Mdl",
        "StepTuple",
        "WorldAxisObjectClasses",
    )
)

# PEP 695 type parameters are local to their generic declarations and do not
# have Python-domain documentation targets. Preserve their rendered text.
local_type_parameters = frozenset({"Wcs"})

# Type hints report the defining private submodule, e.g.
# ``gwcs.coordinate_frames._base.WorldAxisObjectClass``.
private_module = re.compile(r"\._\w+(?=\.)")


def resolve_missing_reference(app, env, node, contnode):
    """Resolve type references that Sphinx cannot find through its normal lookup.

    Private names are rendered as plain text. Unqualified external types are
    retried through intersphinx, PEP 695 aliases emitted as classes are retried
    as Python ``type`` objects, and GWCS private-module paths are rewritten to
    their public import paths before resolving them in the Python domain.
    """
    # Sphinx provides the unresolved target on the reference node.
    target = node.get("reftarget", "")

    # Generic type parameters are not importable objects, so they cannot link
    # to an API page; preserve their type-hint text without a cross-reference.
    if target in local_type_parameters:
        return contnode.deepcopy()

    # Private implementation details have no public documentation target.
    if target.rsplit(".", 1)[-1].startswith("_"):
        return contnode.deepcopy()

    # Resolve postponed, unqualified third-party annotations with intersphinx.
    if external := unqualified_type_names.get(target):
        node["reftarget"] = external
        return intersphinx_missing_reference(app, env, node, contnode)

    # sphinx_autodoc_typehints labels PEP 695 aliases as classes; use their
    # documented Python-domain type targets instead.
    if target in typing_type_aliases and node["reftype"] == "class":
        return env.get_domain("py").resolve_xref(
            env,
            node.get("refdoc"),
            app.builder,
            "type",
            target,
            node,
            contnode,
        )

    # Replace a defining private GWCS module with its public re-export before
    # retrying Python-domain resolution.
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
    """Exclude inherited ``str`` and Astropy ``Model`` members from autodoc.

    The ``autodoc-skip-member`` event passes the candidate object as its fourth
    positional argument. Its owner and wrapped implementation identify members
    inherited from the two noisy base classes; returning ``True`` omits them,
    while ``None`` leaves Sphinx's default decision unchanged.
    """
    # Extract the candidate object and normalize bound methods to functions.
    obj = event_args[3]
    owner = getattr(obj, "__objclass__", None)
    wrapped = getattr(obj, "__func__", obj)
    # Suppress inherited string methods and Astropy Model members that do not
    # describe the GWCS subclass's API.
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

    # Defer all other members to Sphinx's standard inclusion rules.
    return None


def setup(app):
    """Register the custom autodoc and missing-reference event handlers."""
    # Filter inherited implementation details while autodoc gathers members.
    app.connect("autodoc-skip-member", skip_excluded_inherited_members)
    # Repair type-hint cross-references after normal resolution fails.
    app.connect("missing-reference", resolve_missing_reference)


def is_property(modname, qualname, attr):
    """Return whether an autosummary class member is a property descriptor.

    The class template uses this to choose ``autoproperty`` rather than
    ``autoattribute``. Static lookup avoids triggering descriptors while the
    class and requested member are inspected.
    """
    # Walk from the module to the nested class requested by the template.
    obj = importlib.import_module(modname)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    # Missing members are attributes rather than properties for template use.
    try:
        member = inspect.getattr_static(obj, attr)
    except AttributeError:
        return False
    # Only descriptor instances declared as properties need autoproperty.
    return isinstance(member, property)


def is_inherited_model_name(modname, qualname, attr):
    """Return whether ``name`` is inherited from an Astropy modeling base class.

    The autosummary template uses this to avoid documenting an inherited model
    name as though it were declared by the current GWCS class. Members other
    than ``name`` cannot match this special case.
    """
    # This workaround applies only to the inherited Model ``name`` attribute.
    if attr != "name":
        return False

    # Walk from the module to the nested class requested by the template.
    obj = importlib.import_module(modname)
    for part in qualname.split("."):
        obj = getattr(obj, part)

    # Report a match only when this class lacks ``name`` and an Astropy modeling
    # base class supplies it.
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
    "typing_type_aliases": typing_type_aliases,
}
