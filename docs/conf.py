import os
import sys

# Only needed for local builds without an install; harmless otherwise.
sys.path.insert(0, os.path.abspath(".."))


project = "nDspec"
copyright = "2023, Matteo Lucchini, Phil Uttley"
author = "Matteo Lucchini, Phil Uttley"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",       
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

language = "en"          

autodoc_mock_imports = ["xspec"]   # only package that can't be installed no readthedocs
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

#annoying warnings to ignore while building
nitpick_ignore = [
    ("py:class", "np.array"),
    ("py:class", "array_like"),  
    ("py:class", "Fit"), 
]
nitpick_ignore_regex = [
    (r"py:.*", r"^\s*(object|objects|optional|None\.?)\s*$"),
]
#alias notation to numpy.ndarray; it doesn't catch everything but close enough
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = {
    "np.array": "numpy.ndarray",
    "np.ndarray": "numpy.ndarray",
    "array_like": "numpy.ndarray",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "astropy": ("https://docs.astropy.org/en/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "lmfit": ("https://lmfit.github.io/lmfit-py/", None),
}

#Dont re-run the notebooks at build time, instead execute each once and
#commit the executed outputs instead.
nbsphinx_execute = "never"
nbsphinx_allow_errors = True

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "github_url": "https://github.com/nDspec/nDspec",
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "header_links_before_dropdown": 5,
    "show_nav_level": 1,        # how deep the left nav is expanded initially
    "navigation_depth": 4,      # how deep it can go
    "show_toc_level": 2,        # right-hand page TOC depth
    "use_edit_page_button": False,
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}

html_context = {
    "github_user": "nDspec",
    "github_repo": "nDspec",
    "github_version": "main",
    "doc_path": "docs",
    "default_mode": "auto",     # follow OS light/dark preference
}

html_sidebars = {
    "**": ["sidebar-nav-bs"],
}
