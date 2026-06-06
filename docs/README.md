# Building the docs

1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. `python generate_gallery_assets.py` to refresh the committed gallery plots
4. `make html`

Output is in `_build/html/`.
