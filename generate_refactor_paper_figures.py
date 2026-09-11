from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_MAIN = ROOT / "results" / "q34_risk_contract_refactor"
DEFAULT_BASELINE = ROOT / "results" / "q34_strict_baselines"


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.constrained_layout.use": True,
    })


def save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out / f"{name}.{suffix}", facecolor="white", dpi=300)
    plt.close(fig)


def comparison_figure(
    names: list[str], metrics: list[dict[str, float]], title: str,
    filename: str, out: Path,
) -> None:
    panels = [
        ("cash_cost_yuan", 1e4, "现金费用（万元）"),
        ("emergency_kwh", 1e4, "应急购电（万 kWh）"),
        ("unused_contract_kwh", 1e4, "闲置合同（万 kWh）"),
        ("pv_curtailed_kwh", 1e4, "弃光（万 kWh）"),
    ]
    colors = ["#D55E00", "#0072B2"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.7))
    for panel, (ax, (key, scale, ylabel)) in enumerate(zip(axes.flat, panels)):
        values = np.array([row[key] for row in metrics], dtype=float) / scale
        bars = ax.bar(np.arange(2), values, width=0.58, color=colors,
                      edgecolor="#333333", linewidth=0.45)
        ax.set_xticks(np.arange(2), names)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.margins(y=0.18)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, linestyle=":")
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.2f}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)
        ax.text(-0.13, 1.04, chr(ord("a") + panel), transform=ax.transAxes,
                fontsize=9, fontweight="bold")
    fig.suptitle(title, fontsize=9.5, fontweight="bold")
    save(fig, out, filename)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-results", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--baseline-results", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    main_results = args.main_results.resolve()
    baseline_results = args.baseline_results.resolve()
    out = (args.output_dir or (main_results / "figures")).resolve()
    configure_style()
    q34 = json.loads(
        (main_results / "核心指标.json").read_text(encoding="utf-8")
    )["annual_feb_dec"]
    baseline = json.loads(
        (baseline_results / "核心指标.json").read_text(encoding="utf-8")
    )["annual_feb_dec"]
    comparison_figure(
        ["冻结0时合同", "日内调账"], [baseline["q3_no_adjust"], q34["q3"]],
        "固定电价：同0时信息下的调账净效应", "q3_greedy_adjustment_comparison", out,
    )
    comparison_figure(
        ["冻结0时合同", "日内调账"], [baseline["q4_3_no_adjust"], q34["q4_3"]],
        "变动电价：同0时信息下的调账净效应", "q4_greedy_adjustment_comparison", out,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
