#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent

cmd = [
    "sphinx-build",
    "-b",
    "html",
    str(DOCS),
    str(DOCS / "_build" / "html"),
]
result = subprocess.run(cmd)
sys.exit(result.returncode)
