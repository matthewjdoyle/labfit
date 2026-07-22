# Contributing to LabFit

Thanks for your interest in improving LabFit. This document covers the basics of
getting set up and submitting changes.

## Development setup

LabFit targets Python >= 3.10 and depends on NumPy, SciPy, and Matplotlib.

```bash
git clone https://github.com/matthewjdoyle/labfit.git
cd labfit
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The `dev` extra installs `pytest`, `pytest-cov`, `ruff`, and `mypy`. The `docs`
extra installs Sphinx and the Read the Docs theme.

## Running the checks

Before opening a pull request, make sure all three pass:

```bash
pytest -q --cov=labfit --cov-report=term-missing
ruff check .
ruff format --check .
mypy labfit
```

These are the same commands the CI workflow runs. Coverage must stay at or above
90%.

## Style and conventions

- Line length is 110 characters (configured in `pyproject.toml`).
- Code is formatted with `ruff format`. Run `ruff format .` to fix formatting.
- Lint rules (`E`, `F`, `W`, `I`, `UP`, `B`, `SIM`) are selected in
  `pyproject.toml`.
- Type annotations are required on all public functions (`disallow_untyped_defs`).
- Do not add comments unless they explain non-obvious logic.
- Follow the existing NumPy-style docstrings (parsed by Sphinx Napoleon).

## Adding a built-in model

1. Define the model function in `labfit/models.py` with a NumPy-style docstring.
   The first parameter is always `x`; subsequent parameters are the fit
   parameters.
2. Register it in `MODEL_REGISTRY`.
3. Add an initial-guess function to the guess registry in `fitter_impl.py`.
4. Add the equation and variable definitions to
   `docs/fitting-functions.rst`.
5. Add a test in `tests/test_labfit_branches.py` covering param names, shape, and
   an end-to-end fit.

## Pull request checklist

- [ ] Tests pass locally (`pytest`, `ruff check`, `ruff format --check`, `mypy`).
- [ ] New code is covered by tests.
- [ ] Public API changes are documented in `CHANGELOG.md`.
- [ ] Docs updated if behaviour or signatures changed.
