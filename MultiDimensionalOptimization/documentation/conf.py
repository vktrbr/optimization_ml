import os
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(os.path.abspath('.')).parents[1]))

# -- Project information -----------------------------------------------------

project = 'Multidimensional optimization'
copyright = '2022, Victor Barbarich'
author = 'Victor Barbarich, Adelina Tsoi'

# The full version, including alpha/beta/rc tags
release = '1.0.0'

extensions = [
    'sphinxcontrib.pseudocode', 'sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode'
]

templates_path = ['_templates']
exclude_patterns = ['_build']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'press'

add_module_names = False
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
autodoc_typehints = "description"
autodoc_class_signature = "separated"
latex_elements = {'extraclassoptions': 'openany,oneside',
                  'extrapackages': r'\usepackage{tikz}'
                                   r'\usetikzlibrary{shapes,positioning}'
                                   r'\usepackage{amsmath}'}

math_number_all = True
math_numfig = False
latex_use_xindy = False
