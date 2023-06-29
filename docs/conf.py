# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys


CURR_PATH = os.path.abspath(os.path.dirname(__file__))
LIB_PATH = os.path.join(CURR_PATH, os.path.pardir, "src")
sys.path.insert(0, LIB_PATH)

project = "AutoTM"
copyright = "2023, Strong AI Lab"
author = "Khodorchenko Maria, Butakov Nikolay"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autosummary"]

# Delete external references
autosummary_mock_imports = [
    "numpy",
    "pandas",
    "artm",
    "gensim",
    "billiard",
    "plotly",
    "scipy",
    "spacy_langdetect",
    "sklearn",
    "spacy",
    "pymystem3",
    # "nltk"
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
