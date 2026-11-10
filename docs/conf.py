# Configuration file for the Sphinx documentation builder.
# `scverse_doc` sets the theme and shared defaults, see https://scverse-doc.readthedocs.io
# For the full list of options, see https://www.sphinx-doc.org/page/usage/configuration.html

from datetime import datetime
from importlib.metadata import metadata

# NOTE: If you installed your project in editable mode, this might be stale.
#       If this is the case, reinstall it to refresh the metadata
info = metadata("hotspotsc")
project = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}."
version = release = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
source_repository = urls["Source"]

extensions = [
    "scverse_doc",
    "scverse_misc.sphinx_ext",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "IPython.sphinxext.ipython_console_highlighting",
]

bibtex_bibfiles = ["references.bib"]
nb_execution_mode = "off"  # tutorials need external data; outputs are committed
typehints_defaults = "braces"
always_use_bars_union = True  # use `|` instead of `Union` in types even when building with Python ≤3.14

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

xref_overrides = {
    # If building the documentation fails because a type annotation links to a private path
    # (e.g. `igraph._igraph.GraphBase`), map it to the documented name here.
    # For a missing link that can’t be fixed this way, add it to `nitpick_ignore` instead.
    #     "igraph._igraph.GraphBase": "igraph.GraphBase",
    "numpy._typing._array_like.NDArray": ("py:data", "numpy.typing.NDArray"),  # documented as data, not class
    "numpy.float64": "numpy.double",  # numpy documents float64 under its `double` alias
}
