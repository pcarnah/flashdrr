import os, sys
sys.path.insert(0, os.path.abspath('../'))
from flashdrr import __version__

# -- Project information -----------------------------------------------------

project = 'FlashDRR'
copyright = '2026, Patrick Carnahan'
author = 'Patrick Carnahan'
# Allow CI to override the banner version (e.g. when building an immutable
# tag snapshot where flashdrr.__version__ already advanced on main).
release = os.environ.get('FLASHDRR_DOCS_VERSION') or __version__

# -- General configuration --------------------------------------------------

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
exclude_patterns = ['_build', '_build_versions.py', 'Thumbs.db', '.DS_Store']

master_doc = 'index'
language = 'en'

myst_enable_extensions = [
    'colon_fence',
]

myst_heading_anchors = 2

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# Skip inherited members so that nn.Module subclasses do not pull in the
# hundreds of torch.nn.Module boilerplate methods (train, eval, parameters,
# apply, cuda, to, ...). Class-level inheritance information is still shown
# via :show-inheritance:.
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'inherited-members': False,
    'show-inheritance': True,
}

autodoc_class_signature = 'separated'
autodoc_typehints = 'description'

# Autodoc will try to import every public object it documents. The full
# flashdrr.rendering stack needs torch, numpy, and triton at import time, but
# for *rendering the docs* we only need the API surface (signatures, classes,
# docstrings). Mocking these heavy / platform-specific deps lets the docs job
# run on a vanilla CPython without pulling the CUDA stack.
autodoc_mock_imports = ['torch', 'numpy', 'triton', 'triton.language']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_baseurl = 'https://pcarnah.github.io/flashdrr/'
# Inject the version-switcher dropdown on every page. The script queries the
# GitHub Releases API at runtime and prepends a banner with a dropdown of
# stable/latest/<tag> builds. No per-build data file is needed: the release
# list is the single source of truth.
html_js_files = ['version-switcher.js']
templates_path = ['_templates']
# Inline <script> emitted by _templates/layout.html. Stamped at config load
# time so sphinx can cache the config (callables are not picklable). Holds
# this build's own version and whether it represents the moving main build.
import json as _json
_FLASHDRR_DOCS_PAYLOAD = _json.dumps(
    {
        'version': release,
        'is_latest': bool(os.environ.get('FLASHDRR_DOCS_IS_LATEST')),
    },
    separators=(',', ':'),
)
html_context = {
    'flashdrr_docs_context_js':
        '<script>window.FLASHDRR_DOCS=' + _FLASHDRR_DOCS_PAYLOAD + ';</script>',
}
