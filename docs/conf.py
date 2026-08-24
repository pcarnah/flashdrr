import os, sys
sys.path.insert(0, os.path.abspath('../'))
from flashdrr import __version__

# -- Project information -----------------------------------------------------

project = 'FlashDRR'
copyright = '2026, Patrick Carnahan'
author = 'Patrick Carnahan'
release = __version__

# -- General configuration --------------------------------------------------

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

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

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
