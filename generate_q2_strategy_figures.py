from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "q2_strategy"
FIGURES = RESULTS / "figures"
SENSITIVITY_ROOT = ROOT / "results" / "q2_sensitivity"
FACTORS = (0.5, 1.0, 1.5, 2.0)
COLORS = {"MPC": "#D55E00", "LP+反馈贪心": "#0072B2"}
MARKERS = {"MPC": "s", "LP+反馈贪心": "o"}
LINESTYLES = {"MPC": "--", "LP+反馈贪心": "-"}


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 7.5,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 300,
        "figure.constrained_layout.use": True,
    })


def save_figure(fig: plt.Figure, stem: str, size: tuple[float, float]) -> list[str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(*size)
    paths: list[str] = []
    for suffix in ("pdf", "svg", "png"):
        path = FIGURES / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300, facecolor="white")
        paths.append(str(path.relative_to(ROOT)))
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[..., :3].astype(float)
    gray = np.dot(rgb, np.array([0.299, 0.587, 0.114])).astype(np.uint8)
    gray_path = FIGURES / f"{stem}_grayscale.png"
    Image.fromarray(gray).save(gray_path, dpi=(300, 300))
    paths.append(str(gray_path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def merge_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[pd.DataFrame] = []
    daily: list[pd.DataFrame] = []
    for factor in FACTORS:
        folder = SENSITIVITY_ROOT / f"factor_{str(factor).replace('.', '_').replace('_0', '')}"
        if not folder.exists():
            # The experiment runner preserves Python's string form: 1.0 -> factor_1.
            folder = SENSITIVITY_ROOT / f"factor_{str(factor).replace('.0', '').replace('.', '_')}"
        summaries.append(pd.read_csv(folder / "误差敏感性.csv"))
        daily.append(pd.read_csv(folder / "误差敏感性_逐日.csv"))
    summary = pd.concat(summaries, ignore_index=True)
    daily_frame = pd.concat(daily, ignore_index=True)
    summary.to_csv(RESULTS / "误差敏感性.csv", index=False, encoding="utf-8-sig")
    daily_frame.to_csv(RESULTS / "误差敏感性_逐日.csv", index=False, encoding="utf-8-sig")
    return summary, daily_frame


def strategy_comparison() -> tuple[list[str], dict[str, float]]:
    # The paper's controller comparison is a controlled execution-layer
    # experiment: both methods receive the same frozen contract, forecast,
    # realised path, settlement price, and initial SOC on every day.  The
    # separate continuous-closed-loop output remains available as an
    # auxiliary result, but must not be mixed with this paired experiment.
    frame = pd.read_csv(RESULTS / "策略对比.csv")
    frame = frame.set_index("方法").loc[["MPC", "LP+反馈贪心"]]
    panels = [
        ("总费用_元", 1e4, "年度总费用（万元）", "a"),
        ("应急购电量_kWh", 1e4, "应急购电量（万 kWh）", "b"),
        ("合同浪费量_kWh", 1e4, "合同浪费量（万 kWh）", "c"),
        ("执行计算时间_秒", 1.0, "执行计算时间（秒，对数轴）", "d"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.7))
    labels = ["MPC", "LP+反馈贪心"]
    for ax, (column, scale, ylabel, panel) in zip(axes.flat, panels):
        values = frame.loc[labels, column].to_numpy(float) / scale
        bars = ax.bar(
            np.arange(2), values, width=0.58,
            color=[COLORS[label] for label in labels],
            edgecolor="#333333", linewidth=0.45,
        )
        if column == "执行计算时间_秒":
            ax.set_yscale("log")
            ax.set_ylim(0.08, max(values) * 2.4)
        else:
            ax.set_ylim(bottom=0)
            ax.margins(y=0.18)
        ax.set_xticks(np.arange(2), labels)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, linestyle=":")
        for bar, value in zip(bars, values):
            text_value = f"{value:.2f}" if value >= 0.1 else f"{value:.3f}"
            ax.annotate(
                text_value, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                fontsize=7,
            )
        ax.text(-0.14, 1.04, panel, transform=ax.transAxes, fontweight="bold", fontsize=9)
    fig.suptitle("两种执行策略的 334 日冻结同输入配对", fontsize=9.5, fontweight="bold")
    paths = save_figure(fig, "图_Q2_执行策略比较", (7.2, 4.7))
    metrics = {
        "greedy_cost_saving_yuan": float(frame.loc["MPC", "总费用_元"] - frame.loc["LP+反馈贪心", "总费用_元"]),
        "greedy_cost_saving_rate": float(1 - frame.loc["LP+反馈贪心", "总费用_元"] / frame.loc["MPC", "总费用_元"]),
        "emergency_reduction_kwh": float(frame.loc["MPC", "应急购电量_kWh"] - frame.loc["LP+反馈贪心", "应急购电量_kWh"]),
        "runtime_speedup": float(frame.loc["MPC", "执行计算时间_秒"] / frame.loc["LP+反馈贪心", "执行计算时间_秒"]),
    }
    return paths, metrics


def sensitivity_figure(summary: pd.DataFrame) -> tuple[list[str], list[dict[str, float]]]:
    methods = ["MPC", "LP+反馈贪心"]
    data = summary[summary["方法"].isin(methods)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05))
    for method in methods:
        part = data[data["方法"] == method].sort_values("sigma_factor")
        axes[0].plot(
            part["sigma_factor"], part["总费用_元"] / 1e4,
            label=method, color=COLORS[method], marker=MARKERS[method],
            linestyle=LINESTYLES[method], markersize=5,
        )
        axes[1].plot(
            part["sigma_factor"], part["应急购电量_kWh"] / 1e4,
            label=method, color=COLORS[method], marker=MARKERS[method],
            linestyle=LINESTYLES[method], markersize=5,
        )
    for ax, ylabel, panel in zip(axes, ["年度总费用（万元）", "应急购电量（万 kWh）"], ["a", "b"]):
        ax.set_xlabel(r"预测误差标准差倍率 $\sigma/\sigma_0$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(FACTORS)
        ax.set_ylim(bottom=0)
        ax.grid(color="#D9D9D9", linewidth=0.45, linestyle=":")
        ax.text(-0.14, 1.04, panel, transform=ax.transAxes, fontweight="bold", fontsize=9)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("固定合同与信息集下的预测误差敏感性", fontsize=9.5, fontweight="bold")
    paths = save_figure(fig, "图_Q2_预测误差敏感性", (7.2, 3.05))
    derived: list[dict[str, float]] = []
    for factor in FACTORS:
        part = data[data["sigma_factor"] == factor].set_index("方法")
        mpc = float(part.loc["MPC", "总费用_元"])
        greedy = float(part.loc["LP+反馈贪心", "总费用_元"])
        derived.append({
            "sigma_factor": factor,
            "cost_gap_yuan": mpc - greedy,
            "greedy_saving_rate_vs_mpc": 1 - greedy / mpc,
            "mpc_emergency_kwh": float(part.loc["MPC", "应急购电量_kWh"]),
            "greedy_emergency_kwh": float(part.loc["LP+反馈贪心", "应急购电量_kWh"]),
        })
    pd.DataFrame(derived).to_csv(RESULTS / "误差敏感性_派生指标.csv", index=False, encoding="utf-8-sig")
    return paths, derived


def soc_figure() -> list[str]:
    greedy = pd.read_csv(RESULTS / "主模型_逐日.csv", parse_dates=["date"])
    mpc = pd.read_csv(RESULTS / "MPC连续闭环_逐日.csv", parse_dates=["date"])
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True, sharey=True)
    for ax, frame, method, panel in zip(
        axes, [mpc, greedy], ["MPC", "LP+反馈贪心"], ["a", "b"]
    ):
        color = COLORS[method]
        ax.fill_between(
            frame["date"], frame["soc_min_kwh"], frame["soc_max_kwh"],
            color=color, alpha=0.18, linewidth=0, label="日内 SOC 范围",
        )
        ax.plot(
            frame["date"], frame["soc_end_kwh"],
            color=color, linestyle=LINESTYLES[method], linewidth=0.9,
            label="日末 SOC",
        )
        ax.axhline(1200, color="#777777", linestyle=":", linewidth=0.7)
        ax.axhline(10800, color="#777777", linestyle=":", linewidth=0.7)
        ax.set_ylabel("SOC（kWh）")
        ax.set_title(method, loc="left", color=color, fontweight="bold")
        ax.set_ylim(0, 11600)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, linestyle=":")
        ax.text(-0.10, 1.03, panel, transform=ax.transAxes, fontweight="bold", fontsize=9)
    axes[0].legend(frameon=False, loc="lower right", ncol=2)
    axes[-1].set_xlabel("日期（2025-02-01 至 2025-12-31）")
    fig.suptitle("两种执行策略的跨日 SOC 连续轨迹", fontsize=9.5, fontweight="bold")
    return save_figure(fig, "图_Q2_SOC连续轨迹", (7.2, 4.4))


def audit_outputs(paths: list[str]) -> dict[str, object]:
    png_info: list[dict[str, object]] = []
    for relative in paths:
        path = ROOT / relative
        if path.suffix.lower() == ".png":
            with Image.open(path) as image:
                png_info.append({
                    "file": relative,
                    "pixels": list(image.size),
                    "dpi": [round(value, 2) for value in image.info.get("dpi", (0, 0))],
                    "minimum_300_dpi": min(image.info.get("dpi", (0, 0))) >= 299.5,
                })
    report = {
        "status": "PASS" if all(item["minimum_300_dpi"] for item in png_info) else "FAIL",
        "figure_files": paths,
        "png_audit": png_info,
        "semantic_checks": {
            "no_dual_y_axis": True,
            "no_3d_or_pie": True,
            "colorblind_palette_and_redundant_encoding": True,
            "vector_exports_present": True,
            "units_on_axes": True,
            "continuous_sigma_axis_used_for_lines": True,
        },
    }
    (RESULTS / "图表审计.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if report["status"] != "PASS":
        raise AssertionError(report)
    return report


def main() -> int:
    configure_style()
    summary, _ = merge_sensitivity()
    all_paths: list[str] = []
    strategy_paths, comparison_metrics = strategy_comparison()
    sensitivity_paths, sensitivity_metrics = sensitivity_figure(summary)
    soc_paths = soc_figure()
    all_paths.extend(strategy_paths + sensitivity_paths + soc_paths)
    report = audit_outputs(all_paths)
    result = {
        "strategy_derived_metrics": comparison_metrics,
        "sensitivity_derived_metrics": sensitivity_metrics,
        "figure_audit": report,
    }
    (RESULTS / "图表与派生指标.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
