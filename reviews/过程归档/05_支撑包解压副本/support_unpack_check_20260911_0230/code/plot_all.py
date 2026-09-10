"""Generate the figures declared in figures/图表契约-重算版.md."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr

from microgrid_core import DT_HOURS, ISSUES, build_analog_forecasts, load_inputs, official_pv_residuals
from utils.plot_style import PALETTE, add_panel_labels, audit_design, audit_layout


DEFAULT_SKILL_ROOT = Path(r"C:\Users\lenovo\.codex\skills\math-modeling-skill-main")
SKILL_ROOT = Path(os.environ.get("MATH_MODELING_SKILL_ROOT", DEFAULT_SKILL_ROOT))
FIGURE_SCRIPTS = SKILL_ROOT / "tools" / "figure" / "scripts"
sys.path.insert(0, str(FIGURE_SCRIPTS))
from export_figure import export_figure  # noqa: E402
from setup_style import setup_style  # noqa: E402


COLORS = {
    "load": PALETTE["dark"],
    "pv": PALETTE["positive"],
    "purchase": PALETTE["primary"],
    "charge": PALETTE["secondary"],
    "discharge": PALETTE["contrast"],
    "adjusted": PALETTE["accent"],
    "neutral": PALETTE["neutral"],
    "positive": PALETTE["positive"],
    "contrast": PALETTE["contrast"],
}


def export_checked(fig, figures_dir: Path, name: str, size: tuple[float, float]) -> None:
    # Matplotlib colorbars rasterize automatically; preserve editable vectors.
    for ax in fig.axes:
        for collection in ax.collections:
            collection.set_rasterized(False)
    layout = audit_layout(fig)
    design = audit_design(fig)
    if layout or design:
        raise RuntimeError(f"{name} 图形预检失败: {layout + design}")
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
    # The supplied preview helper overwrites the exact-size PNG with a tight
    # crop. Keep its export path, but convert the final PNG without recropping.
    qa_dir = figures_dir / "_qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(figures_dir / f"{name}.png") as source:
        assert source.size == (round(size[0]*300), round(size[1]*300))
        source.convert("L").save(qa_dir / f"{name}_grayscale.png", dpi=(300,300))
    plt.close(fig)


def month_tick_positions(dates: pd.DatetimeIndex) -> tuple[list[int], list[str]]:
    positions, labels = [], []
    for month in range(1, 13):
        idx = np.flatnonzero(dates.month == month)
        positions.append(int(idx[len(idx) // 2]))
        labels.append(f"{month}月")
    return positions, labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    setup_style(journal="general", lang="zh", use_sciplots=True, serif_for_zh=True)
    plt.rcParams["legend.frameon"] = False

    data = load_inputs(root)
    q1 = pd.read_csv(root / "results" / "问题1_逐时结果.csv")
    q2d = pd.read_csv(root / "results" / "q2_每日汇总.csv", parse_dates=["date"])
    q3d = pd.read_csv(root / "results" / "q3_每日汇总.csv", parse_dates=["date"])
    q42d = pd.read_csv(root / "results" / "q4_2_每日汇总.csv", parse_dates=["date"])
    q43d = pd.read_csv(root / "results" / "q4_3_每日汇总.csv", parse_dates=["date"])
    q2i = pd.read_csv(root / "results" / "q2_逐时结果.csv", parse_dates=["date"])
    q3i = pd.read_csv(root / "results" / "q3_逐时结果.csv", parse_dates=["date"])
    q43i = pd.read_csv(root / "results" / "q4_3_逐时结果.csv", parse_dates=["date"])
    metrics = json.loads((root / "results" / "核心指标.json").read_text(encoding="utf-8"))
    hours = np.arange(144) / 6.0

    # raw q1: typical profiles
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 4.1), sharex=True, layout="constrained", height_ratios=[1.7, 1.0])
    axes[0].plot(hours, data.typical_load_kw / 1000, color=COLORS["load"], label="负荷", lw=1.25)
    axes[0].plot(hours, data.typical_pv_kw / 1000, color=COLORS["pv"], ls="--", label="光伏", lw=1.25)
    axes[0].set_ylabel("功率 (MW)")
    axes[0].legend(ncol=2, loc="upper left")
    axes[1].step(hours, data.typical_price, where="post", color=COLORS["purchase"], lw=1.15)
    axes[1].set(xlabel="时刻 (h)", ylabel="电价 (元/kWh)", xlim=(0, 24))
    axes[1].set_xticks([0, 4, 8, 12, 16, 20, 24])
    add_panel_labels(axes, x_offset_pt=-7)
    export_checked(fig, figures, "raw_q1_typical_profiles", (6.3, 4.1))

    # process q1: price-dispatch relationship
    net_discharge = q1["discharge_kwh"].to_numpy() - q1["charge_kwh"].to_numpy()
    rho = spearmanr(q1["price_yuan_per_kwh"], net_discharge).statistic
    fig, ax = plt.subplots(figsize=(4.4, 3.35), layout="constrained")
    sc = ax.scatter(q1["price_yuan_per_kwh"], net_discharge, c=hours, cmap="viridis", s=17, alpha=0.78, edgecolors="none")
    ax.axhline(0, color=COLORS["neutral"], lw=0.8, ls="--")
    ax.set(xlabel="电价 (元/kWh)", ylabel="净放电量 (kWh/10 min)")
    ax.text(0.03, 0.95, rf"描述性 Spearman $\rho$={rho:.2f}", transform=ax.transAxes, va="top")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("时刻 (h)")
    export_checked(fig, figures, "process_q1_price_dispatch", (4.4, 3.35))

    # result q1: optimal dispatch and SOC
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 4.4), sharex=True, layout="constrained", height_ratios=[1.65, 1.0])
    axes[0].plot(hours, q1["plan_purchase_kwh"], color=COLORS["purchase"], label="计划购电", lw=1.2)
    axes[0].fill_between(hours, 0, q1["charge_kwh"], color=COLORS["charge"], alpha=0.45, label="充电")
    axes[0].fill_between(hours, 0, -q1["discharge_kwh"], color=COLORS["discharge"], alpha=0.45, label="放电")
    axes[0].axhline(0, color=COLORS["neutral"], lw=0.6)
    axes[0].set_ylabel("时段电量 (kWh)")
    axes[0].legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    axes[1].plot(np.arange(145)/6, np.r_[q1["soc_start_kwh"], q1["soc_end_kwh"].iloc[-1]], color=COLORS["positive"], lw=1.25)
    axes[1].axhline(1200, color=COLORS["neutral"], ls="--", lw=0.8)
    axes[1].axhline(10800, color=COLORS["neutral"], ls="--", lw=0.8)
    axes[1].set(xlabel="时刻 (h)", ylabel="储能电量 (kWh)", xlim=(0, 24), ylim=(0, 12000))
    axes[1].set_xticks([0, 4, 8, 12, 16, 20, 24])
    add_panel_labels(axes, x_offset_pt=-7)
    export_checked(fig, figures, "result_q1_optimal_schedule", (6.3, 4.4))

    # raw q2: full-year net-load heatmap
    net_kw = data.actual_load_kw - data.actual_pv_kw
    limit = max(abs(float(net_kw.min())), abs(float(net_kw.max())))
    fig, ax = plt.subplots(figsize=(6.3, 3.75), layout="constrained")
    # Monotone luminance preserves the sign ordering in grayscale as well.
    image = ax.pcolormesh(np.arange(145)/6, np.arange(366), net_kw, cmap="cividis", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), rasterized=False)
    ax.invert_yaxis()
    ax.set(xlabel="时刻 (h)", ylabel="日期")
    ax.set_xticks([0, 6, 12, 18, 24])
    ypos, ylabel = month_tick_positions(data.dates)
    ax.set_yticks(ypos, ylabel)
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label("净负荷 (kW)")
    export_checked(fig, figures, "raw_q2_annual_netload", (6.3, 3.75))

    fig, ax = plt.subplots(figsize=(4.65,3.65), layout="constrained")
    hb = ax.hexbin(data.actual_pv_kw.ravel()/1000, data.actual_load_kw.ravel()/1000, gridsize=55, bins="log", mincnt=1, cmap="viridis")
    ax.set(xlabel="实际光伏 (MW)", ylabel="实际负荷 (MW)")
    fig.colorbar(hb, ax=ax, pad=.02).set_label("时段数 (log)")
    export_checked(fig, figures, "raw_q2_pv_load_relation", (4.65,3.65))

    # process q2: strict-causal load forecast validation
    load_point, _ = build_analog_forecasts(data.actual_load_kw, data.typical_load_kw, weekday_weight=True)
    start = int(np.flatnonzero(data.dates == pd.Timestamp("2025-02-01"))[0])
    truth = data.actual_load_kw[start:].reshape(-1)
    pred = load_point[start:].reshape(-1)
    mae = float(np.mean(np.abs(truth - pred)))
    rmse = float(np.sqrt(np.mean((truth - pred) ** 2)))
    r2 = float(1.0 - np.sum((truth - pred) ** 2) / np.sum((truth - truth.mean()) ** 2))
    lo, hi = float(min(truth.min(), pred.min())), float(max(truth.max(), pred.max()))
    fig, ax = plt.subplots(figsize=(4.6, 3.7), layout="constrained")
    hb = ax.hexbin(pred, truth, gridsize=55, bins="log", mincnt=1, cmap="viridis")
    ax.plot([lo, hi], [lo, hi], color=COLORS["contrast"], ls="--", lw=1.0)
    padding = .03 * (hi - lo)
    ax.set(xlabel="预测负荷 (kW)", ylabel="实际负荷 (kW)", xlim=(lo-padding, hi+padding), ylim=(lo-padding, hi+padding))
    ax.text(0.03, 0.96, f"MAE={mae:.0f} kW\nRMSE={rmse:.0f} kW\n$R^2$={r2:.3f}", transform=ax.transAxes, va="top")
    cbar = fig.colorbar(hb, ax=ax, pad=0.02)
    cbar.set_label("点密度 (log)")
    export_checked(fig, figures, "process_q2_forecast_validation", (4.6, 3.7))

    # result q2: annual cost and emergency risk
    roll_cost = q2d["cash_cost_yuan"].rolling(7, center=True, min_periods=1).mean()
    roll_emg = q2d["emergency_kwh"].rolling(7, center=True, min_periods=1).mean()
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 4.25), sharex=True, layout="constrained", height_ratios=[1.2, 1.0])
    axes[0].plot(q2d["date"], q2d["cash_cost_yuan"] / 10000, color=COLORS["neutral"], alpha=0.35, lw=0.55, label="每日")
    axes[0].plot(q2d["date"], roll_cost / 10000, color=COLORS["purchase"], lw=1.3, label="7日均值")
    axes[0].set_ylabel("费用 (万元/日)")
    axes[0].legend(ncol=2, loc="upper left")
    axes[1].fill_between(q2d["date"], 0, q2d["emergency_kwh"] / 1000, color=COLORS["contrast"], alpha=0.22)
    axes[1].plot(q2d["date"], roll_emg / 1000, color=COLORS["contrast"], lw=1.15)
    axes[1].set(xlabel="日期", ylabel="应急电量 (MWh/日)")
    add_panel_labels(axes, x_offset_pt=-7)
    export_checked(fig, figures, "result_q2_annual_risk", (6.3, 4.25))

    # raw q3: official forecast error by issue and lead
    residual = official_pv_residuals(data)
    mae_matrix = np.nanmean(np.abs(residual[31:]), axis=0)
    fig, ax = plt.subplots(figsize=(6.3, 2.75), layout="constrained")
    image = ax.pcolormesh(np.arange(25)+.5, np.arange(5)-.5, mae_matrix, cmap="viridis", rasterized=False)
    ax.invert_yaxis()
    ax.set(xlabel="预测提前期 (h)", ylabel="发布时间")
    ax.set_xticks([1, 6, 12, 18, 24])
    ax.set_yticks(np.arange(4), ["0:00", "6:00", "12:00", "18:00"])
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label("光伏预测 MAE (kW)")
    export_checked(fig, figures, "raw_q3_forecast_error", (6.3, 2.75))

    # process q3: representative-day contract updates
    rep = pd.Timestamp("2025-06-21")
    sub = q3i[q3i["date"] == rep].sort_values("slot")
    h = (sub["slot"].to_numpy() - 1) / 6.0
    net_actual_e = (sub["actual_load_kw"].to_numpy() - sub["actual_pv_kw"].to_numpy()) * DT_HOURS
    adjustment_mwh = float(np.abs(sub["final_contract_kwh"] - sub["initial_plan_kwh"]).sum() / 1000)
    fig, ax = plt.subplots(figsize=(6.3, 3.55), layout="constrained")
    ax.plot(h, sub["initial_plan_kwh"], color=COLORS["purchase"], lw=1.0, ls="--", label="0时计划")
    ax.plot(h, sub["final_contract_kwh"], color=COLORS["adjusted"], lw=1.3, label="最终合同")
    ax.plot(h, net_actual_e, color=COLORS["load"], lw=0.9, alpha=0.75, label="实际净负荷")
    for issue in ISSUES[1:]:
        ax.axvline(issue, color=COLORS["neutral"], lw=0.7, ls=":")
    ax.axhline(0, color=COLORS["neutral"], lw=0.5)
    ax.set(xlabel="时刻 (h)", ylabel="时段电量 (kWh)", xlim=(0, 24))
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.text(0.02, 0.04, f"累计绝对调整量={adjustment_mwh:.2f} MWh", transform=ax.transAxes)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(.5,1.01))
    export_checked(fig, figures, "process_q3_contract_updates", (6.3, 3.55))

    # result q3: paired daily comparison
    q30 = pd.read_csv(root/"results/experiments/q3_0h_daily.csv", parse_dates=["date"])
    q36 = pd.read_csv(root/"results/experiments/q3_6h_daily.csv", parse_dates=["date"])
    merged = q30[q30.date>="2025-02-01"].merge(q36[q36.date>="2025-02-01"], on="date", suffixes=("_q2", "_q3"))
    assert len(merged)==334
    x = merged["cash_cost_yuan_q2"] / 10000
    y = merged["cash_cost_yuan_q3"] / 10000
    emg_drop = (merged["emergency_kwh_q2"] - merged["emergency_kwh_q3"]) / 1000
    low, high = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    fig, ax = plt.subplots(figsize=(4.65, 3.75), layout="constrained")
    sc = ax.scatter(x, y, c=emg_drop, cmap="cividis", s=15, alpha=0.72, edgecolors="none")
    ax.plot([low, high], [low, high], color=COLORS["contrast"], ls="--", lw=1.0)
    better = float(np.mean(y < x) * 100)
    ax.text(0.03, 0.96, f"费用下降天数占比={better:.1f}%", transform=ax.transAxes, va="top")
    ax.set(xlabel="相同0时信息、仅0时规划 (万元/日)", ylabel="每6小时更新 (万元/日)", xlim=(low*.95, high*1.05), ylim=(low*.95, high*1.05))
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("应急电量降幅 (MWh/日)")
    export_checked(fig, figures, "result_q3_adjustment_benefit", (4.65, 3.75))

    # raw q4: all monthly observations, including every outlier.
    month_price = [data.variable_price[data.dates.month == month].ravel() for month in range(1,13)]
    fig, ax = plt.subplots(figsize=(6.3, 3.05), layout="constrained")
    ax.boxplot(month_price, widths=.55, patch_artist=True, boxprops=dict(facecolor=COLORS["purchase"],alpha=.45), medianprops=dict(color=COLORS["load"]), flierprops=dict(marker=".",markersize=1.5,alpha=.35))
    ax.set(xlabel="月份", ylabel="实际电价 (元/kWh)")
    export_checked(fig, figures, "raw_q4_price_variability", (6.3, 3.05))

    # process q4: price-directed storage action
    net_action = q43i["discharge_kwh"].to_numpy() - q43i["charge_kwh"].to_numpy()
    price_all = q43i["price_yuan_per_kwh"].to_numpy()
    rho4 = spearmanr(price_all, net_action).statistic
    fig, ax = plt.subplots(figsize=(4.65, 3.65), layout="constrained")
    hb = ax.hexbin(price_all, net_action, gridsize=58, bins="log", mincnt=1, cmap="viridis")
    ax.axhline(0, color=COLORS["contrast"], ls="--", lw=0.8)
    ax.text(0.98, 0.04, rf"事后描述 $\rho$={rho4:.2f}", transform=ax.transAxes, ha="right", va="bottom")
    ax.set(xlabel="实际结算电价 (元/kWh)", ylabel="已执行净放电量 (kWh/10 min)")
    cbar = fig.colorbar(hb, ax=ax, pad=0.02)
    cbar.set_label("点密度 (log)")
    export_checked(fig, figures, "process_q4_price_dispatch", (4.65, 3.65))

    # result q4: annual strategy comparison
    annual = metrics["annual_feb_dec"]
    labels = ["固定价\n无调整", "固定价\n滚动调整", "变价\n无调整", "变价\n滚动调整"]
    keys = ["q2", "q3", "q4_2", "q4_3"]
    costs = np.asarray([annual[k]["cash_cost_yuan"] for k in keys]) / 1e6
    emergency = np.asarray([annual[k]["emergency_kwh"] for k in keys]) / 1000
    bar_colors = [COLORS["neutral"], COLORS["purchase"], COLORS["neutral"], COLORS["purchase"]]
    hatches = ["///", "", "///", ""]
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 4.55), layout="constrained", height_ratios=[1.15, 1.0])
    bars0 = axes[0].bar(labels, costs, color=bar_colors, edgecolor="white", linewidth=0.6)
    bars1 = axes[1].bar(labels, emergency, color=bar_colors, edgecolor="white", linewidth=0.6)
    for bars in (bars0, bars1):
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
    axes[0].set_ylabel("总费用 (百万元)")
    axes[1].set_ylabel("应急电量 (MWh)")
    axes[1].set_xlabel("价格与调整策略")
    fixed_saving = 100 * (costs[0] - costs[1]) / costs[0]
    variable_saving = 100 * (costs[2] - costs[3]) / costs[2]
    axes[0].text(0.5, costs[:2].max() * 1.03, f"节约 {fixed_saving:.1f}%", ha="center")
    axes[0].text(2.5, costs[2:].max() * 1.03, f"节约 {variable_saving:.1f}%", ha="center")
    axes[0].set_ylim(0, costs.max() * 1.16)
    axes[1].set_ylim(0, emergency.max() * 1.12)
    add_panel_labels(axes, x_offset_pt=-7)
    export_checked(fig, figures, "result_q4_strategy_comparison", (6.3, 4.55))

    experiments=pd.read_csv(root/"results/experiments/实验总表.csv").set_index("name")
    b=experiments.loc[["q2_safe","q2_point","q2_typical","q2_no_storage"]]
    fig,ax=plt.subplots(figsize=(6.3,3.5),layout="constrained")
    labs=["主策略","点预测","典型日","无储能"]
    ax.bar(labs,b.settlement_yuan/1e6,color=COLORS["purchase"],label="合同结算")
    ax.bar(labs,b.emergency_yuan/1e6,bottom=b.settlement_yuan/1e6,color=COLORS["neutral"],label="应急购电")
    ax.set(xlabel="日前策略",ylabel="334日费用 (百万元)",ylim=(0,32))
    ax.legend(ncol=2,loc="upper left")
    export_checked(fig,figures,"result_q2_baselines",(6.3,3.5))

    fig,axes=plt.subplots(2,1,figsize=(6.3,4.5),layout="constrained",sharex=True,height_ratios=[1.2,1])
    for prefix,label,color,marker,offset in (("q3","固定价",COLORS["purchase"],"o",-.08),("q43","变价",COLORS["neutral"],"s",.08)):
        b=experiments.loc[[f"{prefix}_{k}" for k in ("0h","12h","6h","2h")]]
        for ax,col in zip(axes,("cash_cost_yuan","alternative_cash_cost_yuan")):
            ax.scatter(np.arange(4)+offset,b[col]/1e6,color=color,marker=marker,s=28,label=label)
            ax.set_ylabel("费用 (百万元)")
    axes[0].legend(ncol=2,loc="upper right")
    axes[0].text(.02,.1,"主结算规则",transform=axes[0].transAxes)
    axes[1].text(.02,.1,"替代规则：同策略重计费",transform=axes[1].transAxes)
    axes[1].set_xticks(range(4),["仅0时","0/12时","每6小时","每2小时*"])
    axes[1].set_xlabel("更新集合（*中间时刻为因果预测修正）")
    add_panel_labels(axes,x_offset_pt=-7)
    export_checked(fig,figures,"result_q3_update_frequency",(6.3,4.5))

    fig,axes=plt.subplots(2,1,figsize=(6.3,4.5),layout="constrained",sharex=True,height_ratios=[1.2,1])
    for prefix,base,label,color,style in (("q2","q2_safe","固定价日前",COLORS["purchase"],"-o"),("q43","q43_6h","变价滚动",COLORS["neutral"],"--s")):
        names=[base if a==.8 else f"{prefix}_alpha_{a:.2f}" for a in (.7,.75,.8,.85,.9,.95)]
        b=experiments.loc[names]
        axes[0].plot(b.alpha,b.cash_cost_yuan/1e6,style,color=color,label=label,markersize=3)
        axes[1].plot(b.alpha,b.emergency_kwh/1000,style,color=color,markersize=3)
    axes[0].set_ylabel("费用 (百万元)")
    axes[1].set(xlabel="保守分位数参数 alpha",ylabel="应急电量 (MWh)")
    axes[0].legend(ncol=2,loc="upper right")
    for ax in axes: ax.axvline(.8,color=COLORS["neutral"],ls=":",lw=.8)
    add_panel_labels(axes,x_offset_pt=-7)
    export_checked(fig,figures,"result_q4_alpha_sensitivity",(6.3,4.5))

    stats = pd.DataFrame(
        [
            {"figure": "process_q1_price_dispatch", "metric": "spearman_rho", "value": rho},
            {"figure": "process_q2_forecast_validation", "metric": "mae_kw", "value": mae},
            {"figure": "process_q2_forecast_validation", "metric": "rmse_kw", "value": rmse},
            {"figure": "process_q2_forecast_validation", "metric": "r2", "value": r2},
            {"figure": "process_q3_contract_updates", "metric": "absolute_adjustment_mwh", "value": adjustment_mwh},
            {"figure": "result_q3_adjustment_benefit", "metric": "days_cost_reduced_percent", "value": better},
            {"figure": "process_q4_price_dispatch", "metric": "spearman_rho", "value": rho4},
            {"figure": "result_q4_strategy_comparison", "metric": "fixed_price_saving_percent", "value": fixed_saving},
            {"figure": "result_q4_strategy_comparison", "metric": "variable_price_saving_percent", "value": variable_saving},
        ]
    )
    stats.to_csv(root / "results" / "图表统计.csv", index=False, encoding="utf-8-sig")
    print(f"Generated 16 logical figures in {figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
