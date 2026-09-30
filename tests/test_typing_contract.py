"""Exercise types as a downstream user, including decorated model signatures."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mypy(source: Path, cache: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            str(ROOT / "pyproject.toml"),
            "--cache-dir",
            str(cache),
            str(source),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_supported_consumer_types(tmp_path):
    checked = _mypy(ROOT / "tests/typing/consumer_valid.py", tmp_path / "cache")
    assert checked.returncode == 0, checked.stdout + checked.stderr


def test_invalid_consumer_types_and_untyped_definitions_are_rejected(tmp_path):
    source = tmp_path / "consumer_invalid.py"
    source.write_text(
        "from labfit import FitResult, fit, fit_multi\n"
        "from labfit.models import linear\n"
        "fit([0.0, 1.0], [1.0, 2.0], model=42)\n"
        "fit([0.0, 1.0], [1.0, 2.0], absolute_sigma='yes')\n"
        "linear([0.0, 1.0], slope='bad', intercept=1.0)\n"
        "linear([0.0, 1.0], gradient=2.0, intercept=1.0)\n"
        "single: FitResult = fit_multi([])\n"
        "def missing_annotations(value):\n"
        "    return value\n",
        encoding="utf-8",
    )
    checked = _mypy(source, tmp_path / "cache")
    assert checked.returncode == 1, checked.stdout + checked.stderr
    for line, code in [
        (3, "arg-type"),
        (4, "arg-type"),
        (5, "arg-type"),
        (6, "call-arg"),
        (7, "assignment"),
        (8, "no-untyped-def"),
    ]:
        assert any(
            f":{line}: error:" in diagnostic and f"[{code}]" in diagnostic
            for diagnostic in checked.stdout.splitlines()
        ), checked.stdout
