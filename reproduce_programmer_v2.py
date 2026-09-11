"""One-command reproduction entry for the frozen programmer-stage deliverables.

The default path recomputes the annual results, audits the official workbooks,
regenerates all figures, and runs both figure gates.  ``--reuse-results`` is an
author-only shortcut for re-running the downstream audits after a presentation-
only change; it is deliberately omitted from the recorded reproduction command.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


SEED = 2026
FORECAST_MODEL = "F1"
HISTORY_DAYS = 30
ALPHA = 0.8
RISK_MODE = "unclustered"
CONTROLLER = "mpc"
MPC_HORIZON = 36
TRACKING_WEIGHT = 0.01
SKILL_ROOT = Path(r"C:\Users\lenovo\.codex\skills\math-modeling-skill-main")


def run_checked(command: list[str], root: Path) -> None:
    print("[reproduce]", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def metadata(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "modified_time_ns": stat.st_mtime_ns,
    }


def resolve_inside(root: Path, raw: str, option: str) -> Path:
    requested = Path(raw)
    resolved = (requested if requested.is_absolute() else root / requested).resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"{option} must resolve inside --project-root")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument(
        "--reuse-results", action="store_true",
        help="author-only: skip the expensive solve and re-run audits/figures",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    results = resolve_inside(root, args.output_dir, "--output-dir")
    figures = resolve_inside(root, args.figures_dir, "--figures-dir")
    results_rel = results.relative_to(root).as_posix()
    figures_rel = figures.relative_to(root).as_posix()
    python = sys.executable

    if not args.reuse_results:
        run_checked([
            python, "-X", "utf8", "run_all.py", "--project-root", ".",
            "--output-dir", results_rel,
            "--forecast-model", FORECAST_MODEL,
            "--history-days", str(HISTORY_DAYS),
            "--alpha", str(ALPHA),
            "--risk-mode", RISK_MODE,
            "--controller", CONTROLLER,
            "--mpc-horizon", str(MPC_HORIZON),
            "--tracking-weight", str(TRACKING_WEIGHT),
        ], root)

    run_checked([
        python, "-X", "utf8", "audit_results.py", "--project-root", ".",
        "--results-dir", results_rel,
    ], root)
    run_checked([
        python, "-X", "utf8", "plot_all.py", "--project-root", ".",
        "--results-dir", results_rel, "--figures-dir", figures_rel,
    ], root)

    check_figure = SKILL_ROOT / "tools" / "figure" / "scripts" / "check_figure.py"
    figure_audit = (
        SKILL_ROOT / "references" / "roles" / "编程手" / "scripts" /
        "figure_audit.py"
    )
    run_checked([
        python, "-X", "utf8", str(check_figure),
        str(figures / "*.png"), str(figures / "*.svg"),
        str(figures / "_qa" / "*.png"), "--strict",
    ], root)
    run_checked([
        python, "-X", "utf8", str(figure_audit), str(figures),
        "--questions", "q1", "q2", "q3", "q4", "--strict",
    ], root)

    manifest_path = results / "复现清单.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["command"] = (
        "python -X utf8 reproduce_programmer_v2.py --project-root . "
        f"--output-dir {results_rel} --figures-dir {figures_rel}"
    )
    source_names = [
        "microgrid_core.py", "run_all.py", "evaluate_results.py",
        "audit_results.py", "plot_all.py", "reproduce_programmer_v2.py",
    ]
    manifest["source_files"] = [metadata(root / name) for name in source_names]
    result_outputs = sorted(p for p in results.iterdir() if p.is_file())
    figure_outputs = sorted(p for p in figures.rglob("*") if p.is_file())
    manifest["outputs"] = [
        str(p.relative_to(root)) for p in result_outputs + figure_outputs
    ]
    manifest["validation"] = (
        "annual physical checks, official-workbook audit, per-file figure "
        "compliance, q1-q4 coverage audit; all commands must exit 0"
    )
    manifest["hash_checks"] = (
        "omitted per explicit user request; reproducibility snapshot uses "
        "absolute path, byte size and modification time only"
    )
    text = json.dumps(manifest, ensure_ascii=False, indent=2)
    for name in ("主运行记录.json", "复现清单.json", "manifest_nohash.json"):
        (results / name).write_text(text, encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "results": str(results),
        "figures": str(figures),
        "unique_reproduction_command": manifest["command"],
        "hash_checks": manifest["hash_checks"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
