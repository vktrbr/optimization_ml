import os
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(os.path.abspath('.')).parents[1]))

import OneDimensionalOptimization

project = 'One-Dimensional-Optimization'
copyright = '2022, Victor Barbarich'
author = 'Victor Barbarich'
release = '1.0.0'

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode']
templates_path = ['_templates']
exclude_patterns = ['test*.py']
html_theme = 'press'
add_module_names = False
html_static_path = ['_static']
autodoc_typehints = "description"
autodoc_class_signature = "separated"
