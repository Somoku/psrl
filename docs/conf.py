# -- PSRL Documentation Configuration ------------------------------------
# Built with Sphinx using pydata-sphinx-theme (Ray-style).
# Supports both Markdown (.md) and reStructuredText (.rst).

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information --------------------------------------------------

project = "PSRL"
copyright = "2025, PKUDAIR Lab PSRL Team"
author = "PKUDAIR Lab PSRL Team"
release = "0.1.0"

# -- General configuration ------------------------------------------------

extensions = [
    # Markdown support via MyST
    "myst_parser",
    # Sphinx built-ins
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    # UI enhancements
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "sphinxcontrib.video",
]

# -- MyST configuration ---------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
    "tasklist",
    "attrs_inline",
    "dollarmath",
    "amsmath",
]
myst_heading_anchors = 3

# -- Source file suffixes -------------------------------------------------

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output ----------------------------------------------

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "logo": {
        "text": "PSRL",
    },
    "github_url": "https://github.com/lhy101/psrl",
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navigation_depth": 3,
    "show_nav_level": 1,
    "navbar_align": "left",
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "header_links_before_dropdown": 6,
    "pygments_light_style": "default",
    "pygments_dark_style": "monokai",
}

html_context = {
    "github_user": "lhy101",
    "github_repo": "psrl",
    "github_version": "main",
    "doc_path": "docs/",
}

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_js_files = ["js/relabel.js"]

# -- Copybutton configuration --------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ |> "
copybutton_prompt_is_regexp = True

# -- Intersphinx mapping --------------------------------------------------
# Disabled: intersphinx is unavailable in restricted-network environments.
# Re-enable by adding "sphinx.ext.intersphinx" to extensions and uncommenting below.
# intersphinx_mapping = {
#     "python": ("https://docs.python.org/3/", None),
#     "torch": ("https://pytorch.org/docs/stable/", None),
# }

# -- Napoleon settings (Google/NumPy docstrings) --------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Mermaid configuration ------------------------------------------------

mermaid_output_format = "raw"
