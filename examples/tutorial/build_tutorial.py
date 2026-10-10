"""Build Markdown, editable LaTeX, and PDF from the same tutorial source."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import textwrap
import zipfile
from pathlib import Path

import numpy as np
from common import DATA, ROOT
from run_examples import run_all

TOKEN = re.compile(r"\{\{([^{}]+)\}\}")
DATASETS = {
    "example_01_calibration": "calibration",
    "example_02_decay": "decay",
    "example_03_single_peak": "single_peak",
    "example_05_double_peak": "double_peak",
    "example_06_correlated": "thermal",
}


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def render_markdown() -> str:
    records = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    examples = records["examples"]

    def replace(match: re.Match[str]) -> str:
        parts = match.group(1).split(":")
        kind, *args = parts
        if kind == "versions":
            return ", ".join(f"{name} {version}" for name, version in records["versions"].items())
        if kind == "code":
            example, marker = args
            source = (ROOT / f"{example}.py").read_text(encoding="utf-8")
            pattern = rf"^[ \t]*# tutorial:begin {marker}\n(.*?)^[ \t]*# tutorial:end {marker}"
            snippet = re.search(pattern, source, re.MULTILINE | re.DOTALL)
            if snippet is None:
                raise ValueError(f"Missing code excerpt: {example}/{marker}")
            return "~~~python\n" + textwrap.dedent(snippet.group(1)).strip() + "\n~~~"
        if kind == "preview":
            name = args[0]
            values = np.loadtxt(DATA / f"{name}.csv", delimiter=",", skiprows=1)[:5]
            return markdown_table(["x", "y", "sigma"], [[f"{value:.6g}" for value in row] for row in values])
        if kind == "table":
            example, label = args
            fitted = examples[example]["fits"][label]
            dataset = f"oscillator_{label[-1]}" if example == "example_04_oscillators" else DATASETS[example]
            units = manifest["datasets"][dataset]["units"]
            rows = [
                [
                    f"`{name}`",
                    f"{fitted['params'][name]:.6g}",
                    f"{fitted['uncertainties'][name]:.3g}",
                    units.get(name, ""),
                ]
                for name in fitted["param_names"]
            ]
            table = markdown_table(["Parameter", "Estimate", "Standard error", "Unit"], rows)
            return (
                table + f"\n\nDegrees of freedom: **{fitted['dof']}**; "
                f"reduced chi-square: **{fitted['reduced_chi2']:.4g}**; "
                f"chi-square p-value: **{fitted['p_value']:.4g}**. "
                f"Converged: **{fitted['success']}**; identifiable: **{fitted['identifiable']}**."
            )
        if kind == "derived":
            return markdown_table(
                ["Derived quantity", "Estimate", "Standard error", "Unit"],
                [
                    [name, f"{value:.6g}", f"{error:.3g}", unit]
                    for name, (value, error, unit) in examples[args[0]]["derived"].items()
                ],
            )
        if kind == "value":
            example, key = args
            return f"{examples[example]['metrics'][key]:.6g}"
        raise ValueError(f"Unknown template token: {match.group(1)}")

    result = TOKEN.sub(replace, (ROOT / "tutorial.template.md").read_text(encoding="utf-8"))
    if TOKEN.search(result):
        raise ValueError("Unresolved tutorial placeholders")
    for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", result):
        if "://" not in target and not target.startswith("#") and not (ROOT / target.split("#")[0]).exists():
            raise FileNotFoundError(f"Broken tutorial asset or link: {target}")
    return result


def find_tectonic(explicit: str | None) -> str:
    candidate = explicit or os.environ.get("TECTONIC") or shutil.which("tectonic")
    if candidate:
        return candidate
    plugin = Path.home() / ".codex" / "plugins" / "cache" / "openai-bundled" / "latex-tectonic"
    matches = sorted(plugin.glob("*/bin/tectonic.exe" if os.name == "nt" else "*/bin/tectonic"))
    if matches:
        return str(matches[-1])
    raise FileNotFoundError("Tectonic was not found. Pass --tectonic /path/to/tectonic or set TECTONIC.")


def build(markdown_only: bool = False, tectonic: str | None = None) -> None:
    run_all()
    (ROOT / "tutorial.md").write_text(render_markdown(), encoding="utf-8")
    if markdown_only:
        return
    pandoc = shutil.which("pandoc")
    if not pandoc:
        raise FileNotFoundError("Pandoc was not found. Install it or use --markdown-only.")
    compiler = find_tectonic(tectonic)
    subprocess.run(
        [
            pandoc,
            "tutorial.md",
            "--from=markdown+tex_math_dollars+implicit_figures",
            "--standalone",
            "--to=latex",
            "--listings",
            "--toc",
            "--number-sections",
            "--lua-filter=print.lua",
            "--include-in-header=style.tex",
            "-V",
            "documentclass:article",
            "-V",
            "papersize:a4",
            "-V",
            "fontsize:10pt",
            "-V",
            "geometry:margin=20mm",
            "-V",
            "colorlinks:true",
            "--output=tutorial.tex",
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run([compiler, "--outdir", str(ROOT), "tutorial.tex"], cwd=ROOT, check=True)
    with zipfile.ZipFile(ROOT / "tutorial_bundle.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob("*")):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix in {".py", ".md", ".tex", ".lua", ".json", ".csv", ".png", ".pdf"}
            ):
                archive.write(path, Path("tutorial") / path.relative_to(ROOT))
    print(f"Built Markdown, LaTeX, PDF and complete bundle in {ROOT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown-only", action="store_true")
    parser.add_argument("--tectonic", help="Path to the Tectonic executable")
    args = parser.parse_args()
    build(args.markdown_only, args.tectonic)
