# LabFit practical tutorial

Start with [the illustrated tutorial](tutorial.md) or [the printable PDF](tutorial.pdf).
The [LaTeX source](tutorial.tex) is editable and uses the accompanying vector figures.

The six examples progress from sensor calibration and exponential decay to spectral
backgrounds, repeated oscillations, overlapping peaks, and correlated thermal measurements.
All observations are synthetic. Generation details, true parameters, units and seeds are
in [the dataset manifest](data/manifest.json). Numerical output is in [results.json](results.json).

## Run the examples

From the repository root, install the checkout:

~~~sh
python -m venv .venv
# Activate .venv using your platform's usual command.
python -m pip install -e .
python examples/tutorial/run_examples.py
~~~

Each example also runs independently:

~~~sh
python examples/tutorial/example_01_calibration.py
python examples/tutorial/example_02_decay.py
python examples/tutorial/example_03_single_peak.py
python examples/tutorial/example_04_oscillators.py
python examples/tutorial/example_05_double_peak.py
python examples/tutorial/example_06_correlated.py
~~~

Paths are resolved relative to the scripts, so the current working directory does not
change which data are loaded. Scripts use Matplotlib's non-interactive Agg backend and
save PNG and vector PDF figures. No interactive window is required.

## Rebuild the tutorial

~~~sh
python examples/tutorial/generate_data.py
python examples/tutorial/build_tutorial.py
~~~

Data generation is optional: the build always uses the existing CSVs and never replaces
them. The first command explicitly recreates the supplied observations using fixed seeds.

The document build needs **Pandoc** and **Tectonic**, in addition to LabFit's normal
Python dependencies. Install these executables using their official distributions.
An existing bundled Codex Tectonic is detected automatically. Otherwise:

~~~sh
python examples/tutorial/build_tutorial.py --tectonic /path/to/tectonic
python examples/tutorial/build_tutorial.py --markdown-only
~~~

Set the environment variable `TECTONIC` as an alternative to `--tectonic`.
A missing executable produces a clear error after the Markdown has been generated.
Build commands fail if a fit is unreliable, a template token is unresolved, or an
asset is missing.

## Edit the source

Edit `tutorial.template.md` for prose, `example_*.py` for analysis, and `style.tex`
for print styling. Do not hand-edit computed tables in `tutorial.md` or `tutorial.tex`;
rebuild them. Code blocks come from named markers in the example scripts, and result
tables come from `results.json`. The Lua filter selects vector images, numbers display
equations, and starts each main chapter on a new PDF page.

The downloadable document source requires `style.tex` only when regenerating through
Pandoc; the generated `tutorial.tex` already contains the style. It requires the PDF
files in `figures/` when compiling. Keep the whole tutorial directory together.

The integration tests check reproducibility, uncertainty propagation, truth recovery
within stated tolerances, and the generalized least-squares objective. From the root:

~~~sh
python -m pytest tests/test_tutorial.py -q
~~~
