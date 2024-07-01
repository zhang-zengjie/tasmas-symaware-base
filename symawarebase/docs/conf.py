# This file is execfile()d with the current directory set to its containing dir.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import os
import sys
import shutil

# -- Path setup --------------------------------------------------------------

__location__ = os.path.dirname(__file__)

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.join(__location__, "../src"))

# -- Project information -----------------------------------------------------

# Get the version from the package itself
try:
    from symaware.base._version import __version__

    version = __version__
except ImportError as e:
    version = ""

if not version or version.lower() == "unknown":
    version = os.getenv("READTHEDOCS_VERSION", "unknown")  # automatically set by RTD
release = version

copyright = "2024"

from sphinx_pyproject import SphinxConfig

project = ""
author = ""
# Override the above variables with values from pyproject.toml
config = SphinxConfig("../pyproject.toml", globalns=globals(), config_overrides={"version": version})


# -- Run sphinx-apidoc -------------------------------------------------------
# This hack is necessary since RTD does not issue `sphinx-apidoc` before running
# `sphinx-build -b html . _build/html`. See Issue:
# https://github.com/readthedocs/readthedocs.org/issues/1139
# DON'T FORGET: Check the box "Install your project inside a virtualenv using
# setup.py install" in the RTD Advanced Settings.
# Additionally it helps us to avoid running apidoc manually

from sphinx.ext import apidoc

autodoc_mock_imports = ["symaware.simulators.prescan"]

# Import and document the current base package
output_dir = os.path.join(__location__, "api")
module_dir = os.path.join(__location__, "../src/symaware")
try:
    shutil.rmtree(output_dir)
except FileNotFoundError:
    pass
try:
    import sphinx

    cmd_line = f"sphinx-apidoc --implicit-namespaces -f -o {output_dir} {module_dir}"

    args = cmd_line.split(" ")
    if tuple(sphinx.__version__.split(".")) >= ("1", "7"):
        # This is a rudimentary parse_version to avoid external dependencies
        args = args[1:]

    apidoc.main(args)
except Exception as e:
    print(f"Running `sphinx-apidoc` failed!\n{e}", file=sys.stderr)

# Import and document all the external packages
output_dir = os.path.join(__location__, "api")
module_dir = os.path.join(__location__, "external/symaware")
try:
    import requests
    import tarfile
    from pathlib import Path

    def document_branch(branch_name: str, path: str):
        try:
            os.mkdir("external")
        except FileExistsError:
            pass

        # Download the latest version of the package
        url = f"https://gitlab.mpi-sws.org/api/v4/projects/sadegh%2Feicsymaware/repository/archive.tar.gz?sha={branch_name}&path={path}"
        print(url)

        def get_tar_members_stripped(tar: tarfile.TarFile, n_folders_stripped: int = 1):
            members = []
            for member in tar.getmembers():
                p = Path(member.path)
                member.path = p.relative_to(*p.parts[:n_folders_stripped])
                members.append(member)
            return members

        response = requests.get(url)
        with open("symaware.tar.gz", "wb") as f:
            f.write(response.content)
            print("Downloaded the package")
        with tarfile.open("symaware.tar.gz", mode="r:gz") as f:
            f.extractall(path="external", members=get_tar_members_stripped(f, 2))
        os.remove("symaware.tar.gz")

    for branch, path in (("prescan", "python/symaware"), ("pybullet", "src/symaware"), ("pymunk", "src/symaware")):
        document_branch(branch, path)

    cmd_line = f"sphinx-apidoc --implicit-namespaces -f -o {output_dir} {module_dir}"

    args = cmd_line.split(" ")
    if tuple(sphinx.__version__.split(".")) >= ("1", "7"):
        # This is a rudimentary parse_version to avoid external dependencies
        args = args[1:]

    apidoc.main(args)

    with open("api/symaware.rst", "a") as f:
        f.write(
            """.. toctree::
   :maxdepth: 4

   symaware.base
   symaware.simulators
"""
        )

except Exception as e:
    print(f"Running `sphinx-apidoc` on external failed!\n{e}", file=sys.stderr)

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.ifconfig",
    "sphinx.ext.mathjax",
    "sphinxcontrib.mermaid",
    "sphinx_rtd_theme",
    "m2r2",
]

def autodoc_skip_member_handler(app, what, name: str, obj, skip, options):
    # Basic approach; you might want a regex instead
    import re
    return name.startswith("_abc_impl") or re.match(r"^__.+__$", name) is not None

# Automatically called by sphinx at startup
def setup(app):
    # Connect the autodoc-skip-member event from apidoc to the callback
    app.connect('autodoc-skip-member', autodoc_skip_member_handler)

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = [".rst", ".md"]

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# General information about the project.


# -- Configuration of "napoleon" ---------------------------------------------
napoleon_use_param = True

# -- Configuration of "sphinx_autodoc" ---------------------------------------
autodoc_default_options = {"members": True, "undoc-members": True, "private-members": True}


# -- Configuration of "sphinx_autodoc_typehints" -----------------------------
typehints_use_rtype = False
typehints_defaults = "comma"
always_document_param_types = False

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
# language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".venv"]

# The reST default role (used for this markup: `text`) to use for all documents.
# default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
# keep_warnings = False

# If this is True, todo emits a warning for each TODO entries. The default is False.
todo_emit_warnings = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# html_theme_options = {"sidebar_width": "300px", "page_width": "1200px"}

# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
# html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/logo.png"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
# html_domain_indices = True

# If false, no index is generated.
# html_use_index = True

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, links to the reST sources are added to the pages.
# html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
# html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
# html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = "symaware-mpi-doc"

html_context = {
    "display_gitlab": True,  # Integrate Gitlab
    "gitlab_user": "sadegh",  # Username
    "gitlab_repo": "eicsymaware",  # Repo name
    "gitlab_version": "base",  # Version
    "conf_py_path": "/docs/",  # Path in the checkout to the docs root
    "gitlab_host": "gitlab.mpi-sws.org",
}


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ("letterpaper" or "a4paper").
    # "papersize": "letterpaper",
    # The font size ("10pt", "11pt" or "12pt").
    # "pointsize": "10pt",
    # Additional stuff for the LaTeX preamble.
    # "preamble": "",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [("index", "user_guide.tex", "symaware-mpi Documentation", "SymAware team", "manual")]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = ""

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_domain_indices = True

# -- External mapping --------------------------------------------------------
python_version = ".".join(map(str, sys.version_info[0:2]))
intersphinx_mapping = {
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
    "python": ("https://docs.python.org/" + python_version, None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "setuptools": ("https://setuptools.pypa.io/en/stable/", None),
    "pyscaffold": ("https://pyscaffold.org/en/stable", None),
    "pytest": ("https://docs.pytest.org/en/stable", None),
    "chaospy": ("https://chaospy.readthedocs.io/en/master", None),
    "numpoly": ("https://numpoly.readthedocs.io/en/latest", None),
    "stlpy": ("https://stlpy.readthedocs.io/en/latest/", None),
}

# -- Options for sphinxcontrib.mermaid ---------------------------------------
mermaid_init_js = """
mermaid.initialize({
    startOnLoad: true,
});
"""

# -- Options for linkcheck --------------------------------------------------
linkcheck_ignore = [r"https://doi.org/", r"https://gitlab.mpi-sws.org/sadegh/eicsymaware/-/issues"]

print(f"loading configurations for {project} {version} ...", file=sys.stderr)
