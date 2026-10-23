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
    "sphinx_copybutton",
    "sphinx_design",
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

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

html_theme = "furo"
html_title = f"LabFit {version}"
html_short_title = "LabFit"
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2c3e50",
        "color-brand-content": "#16a085",
        "color-api-name": "#2c3e50",
        "color-api-pre-name": "#7f8c8d",
        "color-highlight-on-target": "#e8f8f5",
    },
    "dark_css_variables": {
        "color-brand-primary": "#1abc9c",
        "color-brand-content": "#48c9b0",
        "color-api-name": "#1abc9c",
        "color-api-pre-name": "#95a5a6",
        "color-highlight-on-target": "#164e63",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/matthewjdoyle/labfit",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/matthewjdoyle/labfit",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="
                        M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
                        0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01
                        1.08.58 1.23.82.72 1.21 1.87.87
                        2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95
                        0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27
                        2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82
                        1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0
                        .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z
                    "></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
