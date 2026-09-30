import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

# Make the project root importable during docs build.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

project = "LabFit"
with (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb") as metadata_file:
    version = tomllib.load(metadata_file)["project"]["version"]
release = version
author = "matt@matthewd0yle.com"
copyright = "2026, " + author

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]

autodoc_default_options = {
    "members": True,
    "special-members": "__init__",
    "undoc-members": True,
}
autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_return = True

html_theme = "sphinx_rtd_theme"
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_static_path = ["_static"]
html_theme_options = {
    "collapse_navigation": False,
    "style_external_links": True,
}

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
