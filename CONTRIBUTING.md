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

Before opening a pull request, make sure all checks pass:

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
- Fully annotate function arguments and returns, including private helpers and
  nested functions. `mypy labfit` enforces `disallow_untyped_defs`; keep
  `check_untyped_defs` enabled so bodies are also checked during migrations.
- Use concrete types for public API inputs and outputs. Reserve `Any` for dynamic
  keyword forwarding (such as Matplotlib options), and preserve decorated model
  parameter names and types. Dataclass constructors accept array-like inputs;
  their stored data/error fields are normalized NumPy arrays.
- The package ships `labfit/py.typed` (PEP 561) so installed-package consumers can
  use inline annotations. Keep the marker in setuptools package data.
  `tests/test_typing_contract.py` checks supported calls, inferred return types,
  invalid model arguments, and rejection of untyped definitions with mypy.
  Extend its consumer fixture in `tests/typing/consumer_valid.py` for API additions.
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
