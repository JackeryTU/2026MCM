"""Plot the national-award evidence added after the P1 model review.

The inputs are deterministic back-test summaries.  The figures therefore show
the individual model results directly and deliberately omit confidence bands,
error bars, and significance markers.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from utils.plot_style import PALETTE, add_panel_labels, audit_design, audit_layout


DEFAULT_SKILL_ROOT = Path(r"C:\Users\lenovo\.codex\skills\math-modeling-skill-main")
SKILL_ROOT = Path(os.environ.get("MATH_MODELING_SKILL_ROOT", DEFAULT_SKILL_ROOT))
FIGURE_SCRIPTS = SKILL_ROOT / "tools" / "figure" / "scripts"
sys.path.insert(0, str(FIGURE_SCRIPTS))
from export_figure import export_figure  # noqa: E402
from setup_style import setup_style  # noqa: E402


COLORS = {
    "q2": PALETTE["primary"],
    "q3": PALETTE["contrast"],
    "neutral": PALETTE["neutral"],
    "positive": PALETTE["positive"],
    "accent": PALETTE["accent"],
}


def export_checked(fig, figures_dir: Path, name: str, size: tuple[float, float]) -> None:
    """Run the local design checks and export exact-size SVG/PNG/gray PNG."""
    layout_issues = audit_layout(fig)
    design_issues = audit_design(fig)
    if layout_issues or design_issues:
        raise RuntimeError(f"{name} 图形预检失败: {layout_issues + design_issues}")
    export_figure(
        fig,
        str(figures_dir / name),
        formats=["svg", "png"],
        dpi=300,
        size_inches=size,
        grayscale_preview=False,
        tight=False,
        pad_inches=0.0,
    )
    qa_dir = figures_dir / "_qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(figures_dir / f"{name}.png") as source:
        expected = (round(size[0] * 300), round(size[1] * 300))
        if source.size != expected:
            raise RuntimeError(f"{name} PNG 尺寸 {source.size}，预期 {expected}")
        source.convert("L").save(
            qa_dir / f"{name}_grayscale.png", dpi=(300, 300)
        )
    plt.close(fig)


def plot_risk_ablation(data: pd.DataFrame, figures_dir: Path) -> None:
    """Directly compare clustered and unclustered risk trajectories."""
    fig, axes = plt.subplots(
        1, 2, figsize=(6.3, 3.25), layout="constrained",
        gridspec_kw={"wspace": 0.12},
    )
    branch_specs = [("q2_fixed", "问题二", COLORS["q2"]),
                    ("q3_fixed", "问题三", COLORS["q3"])]

    for strategy, label, color in branch_specs:
        branch = data.loc[data["strategy"] == strategy].set_index("risk_mode")
        for mode, marker in [("clustered", "X"), ("unclustered", "o")]:
            row = branch.loc[mode]
            axes[0].scatter(
                row["pinball_mean_kw"], 100 * row["safe_net_coverage"],
                color=color, marker=marker, s=43, zorder=3,
            )
            axes[1].scatter(
                row["emergency_kwh"] / 1e4, row["cash_cost_yuan"] / 1e4,
                color=color, marker=marker, s=43, zorder=3,
            )
        axes[0].plot(
            branch.loc[["clustered", "unclustered"], "pinball_mean_kw"],
            100 * branch.loc[["clustered", "unclustered"], "safe_net_coverage"],
            color=color, lw=1.15, label=label,
        )
        axes[1].plot(
            branch.loc[["clustered", "unclustered"], "emergency_kwh"] / 1e4,
            branch.loc[["clustered", "unclustered"], "cash_cost_yuan"] / 1e4,
            color=color, lw=1.15, label=label,
        )

    axes[0].set(
        xlabel="Pinball 损失 (kW，越低越好)",
        ylabel="安全净负荷覆盖率 (%，越高越好)",
    )
    axes[1].set(
        xlabel="应急购电 (万 kWh，越低越好)",
        ylabel="现金费用 (万元，越低越好)",
    )
    axes[0].legend(loc="lower left")
    # Separate marker legend explains the risk construction without duplicating colors.
    marker_handles = [
        plt.Line2D([], [], color=COLORS["neutral"], marker="X", ls="none", label="聚类轨迹"),
        plt.Line2D([], [], color=COLORS["neutral"], marker="o", ls="none", label="未聚类轨迹"),
    ]
    axes[1].legend(handles=marker_handles, loc="upper right")
    add_panel_labels(axes, x_offset_pt=-8)
    export_checked(fig, figures_dir, "result_q2_q3_risk_ablation", (6.3, 3.25))


def plot_information_value(annual: pd.DataFrame, marginal: pd.DataFrame, figures_dir: Path) -> None:
    """Show all eight release coalitions and the exact Shapley decomposition."""
    label_map = {
        "S_none": "仅0时（冻结）",
        "S_6": "0+6时",
        "S_12": "0+12时",
        "S_18": "0+18时",
        "S_6_12": "0+6+12时",
        "S_6_18": "0+6+18时",
        "S_12_18": "0+12+18时",
        "S_6_12_18": "0+6+12+18时",
    }
    ordered = annual.sort_values("saving_vs_S_none_yuan", ascending=True).copy()
    y = np.arange(len(ordered))
    savings = ordered["saving_vs_S_none_yuan"].to_numpy() / 1e4

    fig, axes = plt.subplots(
        2, 1, figsize=(6.3, 4.65), layout="constrained",
        height_ratios=[2.2, 1.0],
    )
    axes[0].hlines(y, 0, savings, color=COLORS["neutral"], lw=0.85)
    axes[0].scatter(savings, y, color=COLORS["q3"], marker="o", s=31, zorder=3)
    axes[0].set_yticks(y, [label_map[x] for x in ordered["release_set"]])
    axes[0].set(xlabel="相对冻结0时基线的年度节省 (万元)", ylabel="可读发布集合")
    axes[0].axvline(0, color=COLORS["neutral"], lw=0.75)
    for yi, value in zip(y, savings, strict=True):
        axes[0].text(value + 0.55, yi, f"{value:.2f}", va="center", fontsize=7)
    full = annual.loc[annual["release_set"] == "S_6_12_18"].iloc[0]
    axes[0].text(
        0.99, 0.05,
        f"全集节省 {full['saving_vs_S_none_yuan']/1e4:.2f} 万元"
        f"（{full['saving_vs_S_none_percent']:.2f}%）",
        transform=axes[0].transAxes, ha="right", va="bottom", fontsize=7.2,
    )

    shapley = marginal.loc[marginal["record_type"] == "shapley"].copy()
    shapley["release_hour"] = shapley["release_hour"].astype(int)
    shapley = shapley.sort_values("release_hour")
    shapley_values = shapley["shapley_saving_yuan"].to_numpy() / 1e4
    bars = axes[1].barh(
        np.arange(3), shapley_values,
        color=[COLORS["q2"], COLORS["positive"], COLORS["accent"]],
        hatch=["///", "...", "xx"], edgecolor="white", linewidth=0.5,
    )
    axes[1].set_yticks(np.arange(3), [f"{h}时" for h in shapley["release_hour"]])
    axes[1].set(xlabel="Shapley 费用贡献 (万元)", ylabel="新增发布")
    for bar, value in zip(bars, shapley_values, strict=True):
        axes[1].text(value + 0.3, bar.get_y() + bar.get_height()/2,
                     f"{value:.2f}", va="center", fontsize=7)
    add_panel_labels(axes, x_offset_pt=-8)
    export_checked(fig, figures_dir, "result_q3_information_value", (6.3, 4.65))


def plot_monthly_low_pv(monthly: pd.DataFrame, strata: pd.DataFrame, figures_dir: Path) -> None:
    """Describe monthly heterogeneity and the within-month low-PV strata."""
    specs = [("q2_fixed", "问题二", COLORS["q2"], "o", "-"),
             ("q3_fixed", "问题三", COLORS["q3"], "s", "--")]
    fig, axes = plt.subplots(
        2, 1, figsize=(6.3, 4.45), layout="constrained", sharex=True,
        height_ratios=[1.35, 1.0],
    )
    months = np.arange(2, 13)
    for variant, label, color, marker, linestyle in specs:
        subset = monthly.loc[monthly["variant"] == variant].set_index("month").loc[months]
        axes[0].plot(
            months, subset["emergency_kwh"] / subset["days"],
            color=color, marker=marker, ls=linestyle, label=label, lw=1.15, ms=4.2,
        )
        branch = strata.loc[strata["variant"] == variant]
        pivot = branch.pivot(index="month", columns="group", values="emergency_day_rate").loc[months]
        difference = 100 * (pivot["low_pv_q20"] - pivot["other"])
        axes[1].plot(
            months, difference, color=color, marker=marker, ls=linestyle,
            label=label, lw=1.15, ms=4.2,
        )
    axes[0].set(ylabel="月内日均应急购电 (kWh/日)")
    axes[0].legend(loc="upper right", ncol=2)
    axes[1].axhline(0, color=COLORS["neutral"], lw=0.75)
    axes[1].set(
        xlabel="月份（正式评价期）",
        ylabel="低光伏组−其他组\n应急日率 (百分点)",
    )
    axes[1].set_xticks(months, [f"{m}月" for m in months])
    axes[1].text(
        0.99, 0.95, "Q20为月内经验分层；每月低光伏组6--7日",
        transform=axes[1].transAxes, ha="right", va="top", fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )
    add_panel_labels(axes, x_offset_pt=-8)
    export_checked(fig, figures_dir, "result_monthly_low_pv_robustness", (6.3, 4.45))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    results_dir = root / "results" / "national_award"
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    setup_style(journal="general", lang="zh", use_sciplots=True, serif_for_zh=True)
    plt.rcParams.update({
        "font.size": 7.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
    })

    risk = pd.read_csv(results_dir / "risk_ablation_summary.csv")
    information = pd.read_csv(results_dir / "information_value_summary.csv")
    marginal = pd.read_csv(results_dir / "information_value_marginal.csv")
    monthly = pd.read_csv(results_dir / "monthly_operations.csv")
    strata = pd.read_csv(results_dir / "low_pv_stratification.csv")

    plot_risk_ablation(risk, figures_dir)
    plot_information_value(information, marginal, figures_dir)
    plot_monthly_low_pv(monthly, strata, figures_dir)
    print("National-award evidence figures generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
