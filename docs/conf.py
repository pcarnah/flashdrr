import os, sys
sys.path.insert(0, os.path.abspath('../'))
from flashdrr import __version__

# -- Project information -----------------------------------------------------

project = 'FlashDRR'
copyright = '2026, Patrick Carnahan'
author = 'Patrick Carnahan'
# sphinx-multiversion stamps each build with the current ref's name (e.g.
# ``v0.5.1`` for a tag, ``main`` for a branch). On a plain ``sphinx-build``
# invocation there is no ref, so fall back to the package version.
release = os.environ.get('FLASHDRR_DOCS_VERSION') or __version__

# -- General configuration --------------------------------------------------

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
    'sphinx_multiversion',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

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

# Custom templates live here. The sphinx-rtd-theme auto-picks up a
# ``versions.html`` from the theme's own template path; by dropping one
# with the same name into ``templates_path`` we override the empty stub
# that ships with rtd and feed sphinx-multiversion's ``versions`` /
# ``current_version`` Jinja context into the flyout menu.
templates_path = ['_templates']

# -- sphinx-multiversion -----------------------------------------------------
# Build documentation for every git tag matching the version pattern below
# plus the main branch. Each version lands in its own subdirectory under
# ``_build/html/<refname>/`` and the ``versioning.html`` sidebar template
# produces a flyout menu linking them together.
#
# NOTE: sphinx-multiversion 0.2.4 (latest on PyPI) still calls
# ``Config.read(path, overrides)`` with the two-arg form that Sphinx 9.0
# removed (sphinx-doc/sphinx#13633). The [docs] extra in pyproject.toml
# pins ``sphinx<9`` to keep this compatible.
# Build a version selector over release tags + the main branch. The
# standard recipe (whitelist both local branches and remotes) collides on
# ``main`` because ``refs/heads/main`` and ``refs/remotes/origin/main``
# both map to the same outputdir. Restricting ``smv_remote_whitelist`` to
# ``None`` (use local branches only) avoids the duplicate: CI's
# ``actions/checkout`` always creates a local branch for the current ref,
# so ``main`` is the canonical development build, and release tags are
# still discovered via ``refs/tags/*`` regardless of remote settings.
# With ``smv_outputdir_format = '{ref.name}'`` the dev build lands at
# ``main/`` and released tags at ``<version>/`` (e.g. ``0.5.1/``).
smv_tag_whitelist = r'^v?\d+\.\d+\.\d+$'
smv_branch_whitelist = r'^(main|master)$'
smv_remote_whitelist = None
smv_released_pattern = r'^refs/tags/v?\d+\.\d+\.\d+$'
smv_outputdir_format = '{ref.name}'
# When invoked through ``sphinx-multiversion`` we have a real ref; for a
# plain ``sphinx-build`` invocation (e.g. local previews, RTD's single-version
# build) there isn't one, so we don't fail the build for a missing refname.
smv_vcs_url = 'https://github.com/pcarnah/flashdrr'
