# -*- coding: utf-8 -*-
"""问题二：周期预测 + 残差分位安全轨迹 + 日前合同 LP + 滚动 MPC 执行。

主配置：F1 中心预测、W=30 日残差窗、alpha=0.9；2025-01-01 0:00 由 6000 kWh
连续预运行，正式输出 2025-02-01—12-31 共 334 天。
输出：results/result2.xlsx、results/csv/、figures/问题2/。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import microgrid_core as mc
import solution_lib as sl
import utils.plot_style as ps

FIG = sl.FIG / "问题2"

_SENS_G = {}


def _sens_init(data: dict) -> None:
    _SENS_G["data"] = data


def _sens_worker(cfg: tuple) -> dict:
    kind, W, alpha, tag = cfg
    data = _SENS_G["data"]
    resid = sl.net_residual_matrix(kind, data)
    run = sl.run_year_q2(data, kind, alpha, W, resid=resid)
    out = {k: v for k, v in sl.year_metrics(run).items() if k != "daily_cash"}
    out.update({"方案": tag, "预测器": kind, "W": W, "alpha": alpha})
    if tag.startswith("主配置"):
        out["_run"] = run
    return out


def sensitivity(data: dict, resid_cache: dict, workers: int = 1) -> pd.DataFrame:
    configs = [
        ("F1", 30, 0.9, "主配置 F1-W30-a0.9"),
        ("F1", 30, 0.8, "分位 a=0.8"),
        ("F1", 30, 0.95, "分位 a=0.95"),
        ("F1", 21, 0.9, "窗口 W=21"),
        ("F1", 45, 0.9, "窗口 W=45"),
        ("F2", 30, 0.9, "预测器 F2"),
        ("F0", 30, 0.9, "预测器 F0"),
    ]
    if workers > 1:
        with Pool(workers, initializer=_sens_init, initargs=(data,)) as pool:
            packed = pool.map(_sens_worker, configs)
        base = next(row.pop("_run") for row in packed if "_run" in row)
        return pd.DataFrame(packed), base
    rows = []
    base = None
    for kind, W, alpha, tag in configs:
        if kind not in resid_cache:
            resid_cache[kind] = sl.net_residual_matrix(kind, data)
        run = sl.run_year_q2(data, kind, alpha, W, resid=resid_cache[kind])
        m = {k: v for k, v in sl.year_metrics(run).items() if k != "daily_cash"}
        m.update({"方案": tag, "预测器": kind, "W": W, "alpha": alpha})
        rows.append(m)
        if tag.startswith("主配置"):
            base = run
    return pd.DataFrame(rows), base


def figures(data: dict, run: dict, base: dict, sens: pd.DataFrame, day: int = 265) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    ps.apply_publication_style("zh", "report")
    t = (np.arange(mc.T) + 0.5) * mc.DT
    ticks = list(range(0, 25, 4))

    # 图 1：F1 中心预测 vs 实测
    lhat, vhat = sl.predict_center("F1", day, data)
    fig, axes = ps.publication_subplots(1, 3, aspect=0.95)
    axes[0].plot(t, data["load_act"][day], color=ps.PALETTE["dark"], label="实测")
    axes[0].plot(t, lhat, color=ps.PALETTE["primary"], label="F1 预测")
    axes[0].set_title("负荷")
    axes[0].set_ylabel("功率/kW")
    axes[1].plot(t, data["pv_act"][day], color=ps.PALETTE["dark"], label="实测")
    axes[1].plot(t, vhat, color=ps.PALETTE["positive"], label="F1 预测")
    axes[1].set_title("光伏")
    axes[2].plot(t, data["load_act"][day] - data["pv_act"][day],
                 color=ps.PALETTE["dark"], label="实测")
    axes[2].plot(t, lhat - vhat, color=ps.PALETTE["contrast"], label="F1 预测")
    axes[2].set_title("净负荷")
    for ax in axes:
        ax.set_xlabel("时刻/h")
        ax.set_xticks(ticks)
        ax.legend()
    ps.add_panel_labels(axes)
    fig.suptitle(f"F1 中心预测与实测（{mc.date_of(day)}）", fontsize=8.5)
    ps.export_figure(fig, FIG / "问题2_中心预测对比")
    plt.close(fig)

    # 图 2：安全轨迹与实测净负荷
    ltilde, vtilde, ntilde = sl.safe_trajectory(day, lhat, vhat, run["resid"], 0.9, 30)
    nbar = lhat - vhat
    nact = data["load_act"][day] - data["pv_act"][day]
    ncontract = run["P"][day] / mc.DT
    fig, ax = ps.publication_subplots(1, 1, aspect=0.6)
    ax.fill_between(t, nbar, ntilde, color=ps.PALETTE["sky"], alpha=0.35,
                    label="分位风险余量")
    ax.plot(t, nbar, color=ps.PALETTE["primary"], label="中心净负荷")
    ax.plot(t, nact, color=ps.PALETTE["dark"], label="实测净负荷")
    ax.plot(t, ncontract, color=ps.PALETTE["contrast"], lw=0.9, label="合同购电")
    ax.set_xlabel("时刻/h")
    ax.set_ylabel("功率/kW")
    ax.set_xticks(ticks)
    ax.legend(ncols=2)
    ax.set_title("安全轨迹、合同与实测净负荷")
    fig.suptitle(f"{mc.date_of(day)}", fontsize=8.5)
    ps.export_figure(fig, FIG / "问题2_安全轨迹与合同")
    plt.close(fig)

    # 图 3：年度逐日现金费与应急电量
    days = np.arange(31, 365)
    dates = [mc.date_of(int(d)) for d in days]
    fig, axes = ps.publication_subplots(2, 1, aspect=1.15, height_ratios=[1.3, 1])
    axes[0].plot(dates, run["cash"][31:365], color=ps.PALETTE["primary"], lw=0.8)
    axes[0].set_ylabel("日现金费/元")
    axes[0].set_title("逐日购电现金费")
    axes[1].bar(dates, run["U"][31:365].sum(axis=1), color=ps.PALETTE["contrast"],
                width=0.9)
    axes[1].set_ylabel("应急购电量/kWh")
    axes[1].set_title("逐日应急购电量")
    ps.add_panel_labels(axes)
    ps.export_figure(fig, FIG / "问题2_年度费用与应急")
    plt.close(fig)

    # 图 4：重点日实际充放电与储能状态
    fig, axes = ps.publication_subplots(2, 1, aspect=1.3, height_ratios=[1, 1])
    pos = (np.arange(mc.T) + 0.5) * mc.DT
    axes[0].bar(pos, run["X"][day], 0.15, color=ps.PALETTE["primary"], label="充电量")
    axes[0].bar(pos, -run["Y"][day], 0.15, color=ps.PALETTE["contrast"], label="放电量")
    axes[0].axhline(0, color=ps.PALETTE["neutral"], lw=0.6)
    axes[0].set_ylabel("电量/kWh")
    axes[0].set_title("实际充放电")
    axes[0].legend()
    axes[1].plot(pos, run["S"][day], color=ps.PALETTE["secondary"], label="实际储电量")
    axes[1].axhline(mc.S_MIN, color=ps.PALETTE["neutral"], ls="--", lw=0.7, label="SOC 上下限")
    axes[1].axhline(mc.S_MAX, color=ps.PALETTE["neutral"], ls="--", lw=0.7)
    axes[1].set_ylabel("储电量/kWh")
    axes[1].set_xlabel("时刻/h")
    axes[1].set_title("储能状态")
    axes[1].legend()
    for ax in axes:
        ax.set_xlim(0, 24)
        ax.set_xticks(ticks)
    ps.add_panel_labels(axes)
    fig.suptitle(f"{mc.date_of(day)}", fontsize=8.5)
    ps.export_figure(fig, FIG / "问题2_重点日充放电与储能")
    plt.close(fig)

    # 图 5：敏感性与信息价值
    fig, axes = ps.publication_subplots(1, 3, aspect=0.8)
    cfg = sens.set_index("方案")
    MAIN = "主配置 F1-W30-a0.9"
    groups = [
        ("分位 alpha", ["分位 a=0.8", MAIN, "分位 a=0.95"],
         ["0.8", "0.9(主)", "0.95"]),
        ("残差窗口 W", ["窗口 W=21", MAIN, "窗口 W=45"],
         ["21", "30(主)", "45"]),
    ]
    for ax, (name, keys, labs) in zip(axes[:2], groups):
        ax.bar(labs, [float(cfg.loc[k, "cash_total"]) / 1e4 for k in keys],
               color=ps.PALETTE["primary"], width=0.55)
        ax.set_title(name)
        ax.set_ylabel("年度现金费/万元")
    lab = ["F0", "F1", "F2"]
    # 主配置行即 F1；F0/F2 由“预测器 X”行给出。按预测器列直取，避免用方案名过滤漏掉 F1。
    vals = [float(sens.loc[sens["预测器"] == k, "cash_total"].iloc[0]) / 1e4 for k in lab]
    axes[2].bar(lab, vals, color=ps.PALETTE["primary"], width=0.55)
    axes[2].set_title("预测器")
    axes[2].set_ylabel("年度现金费/万元")
    ps.add_panel_labels(axes)
    ps.export_figure(fig, FIG / "问题2_配置敏感性")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-only", action="store_true",
                    help="只运行 F1-W30-alpha0.9 主配置，不跑敏感性和绘图")
    ap.add_argument("--executor", choices=("mpc", "greedy"), default="mpc",
                    help="执行器；greedy 为不读取当前槽实测的预测驱动贪心")
    ap.add_argument("--days", type=int, default=365,
                    help="运行前 N 天；小于 365 时只作 Lite 烟雾验证")
    ap.add_argument("--no-figures", action="store_true",
                    help="完成全部计算与表格输出，但不生成图件")
    ap.add_argument("--workers", type=int, default=0,
                    help="完整敏感性计算的并行进程数")
    args = ap.parse_args()
    t0 = time.perf_counter()
    data = mc.load_inputs()
    sl.CSV.mkdir(parents=True, exist_ok=True)

    if args.main_only:
        resid = sl.net_residual_matrix("F1", data)
        run = sl.run_year_q2(data, "F1", 0.9, 30, resid=resid,
                             n_days=args.days, executor=args.executor)
        gates = sl.check_run(run)
        with open(sl.CSV / "问题2_门禁.json", "w", encoding="utf-8") as fh:
            json.dump(gates, fh, ensure_ascii=False, indent=2)
        daily = sl.monthly_summary(run)
        daily.to_csv(sl.CSV / "问题2_逐日汇总.csv", index=False,
                     encoding="utf-8-sig")
        if args.days == 365:
            sl.write_workbook("result2.xlsx", "result2.xlsx", {
                "计划购电量": lambda ws: sl._fill_plan_sheet(ws, run),
                "充放电量": lambda ws: sl._fill_cycle_sheet(ws, run),
                "紧急购电量": lambda ws: sl._fill_emergency_sheet(ws, run),
            })
            np.savez_compressed(sl.CSV / "q2_bundle.npz",
                                **{k: run[k] for k in
                                   ("P", "X", "Y", "U", "W", "V", "S",
                                    "cash", "plan_x", "plan_y", "plan_s_next")})
        metrics = {k: v for k, v in sl.year_metrics(run).items()
                   if k != "daily_cash"}
        print(json.dumps(dict(gates=gates, **metrics), ensure_ascii=False,
                         indent=2))
        print("elapsed_s", round(time.perf_counter() - t0, 1))
        return

    # 点预测误差对比
    err = pd.DataFrame([dict(预测器=k, **sl.point_error_metrics(k, data))
                        for k in ("F0", "F1", "F2")])
    err.to_csv(sl.CSV / "问题2_预测误差对比.csv", index=False, encoding="utf-8-sig")

    resid_cache = {k: sl.net_residual_matrix(k, data) for k in ("F0", "F1", "F2")}
    workers = args.workers or min(4, max(1, (os.cpu_count() or 4) - 2))
    sens, run = sensitivity(data, resid_cache, workers=workers)
    sens.to_csv(sl.CSV / "问题2_配置敏感性.csv", index=False, encoding="utf-8-sig")

    pf = sl.perfect_foresight_q2(data)
    nb = sl.run_year_q2(data, no_battery=True)
    ref = pd.DataFrame([
        dict(方案="主配置", **{k: v for k, v in sl.year_metrics(run).items()
                              if k != "daily_cash"}),
        dict(方案="理想信息参照", **{k: v for k, v in sl.year_metrics(pf).items()
                                  if k != "daily_cash"}),
        dict(方案="无储能", **{k: v for k, v in sl.year_metrics(nb).items()
                              if k != "daily_cash"}),
    ])
    ref.to_csv(sl.CSV / "问题2_参照方案.csv", index=False, encoding="utf-8-sig")

    # 门禁
    gates = sl.check_run(run)
    with open(sl.CSV / "问题2_门禁.json", "w", encoding="utf-8") as fh:
        json.dump(gates, fh, ensure_ascii=False, indent=2)

    # 典型日结果（表 1/2/3）
    day_idx = [sl.DAY_INDEX[d] for d in sl.REP_DATES]
    t1 = sl.slot_table(run, data["price1"], day_idx)
    t1.to_csv(sl.CSV / "问题2_表1_指定时段.csv", index=False, encoding="utf-8-sig")
    t2 = sl.energy_block_table(run, day_idx)
    t2.to_csv(sl.CSV / "问题2_表2_四小时聚合.csv", index=False, encoding="utf-8-sig")
    t3 = sl.emergency_table(run, day_idx)
    t3.to_csv(sl.CSV / "问题2_表3_紧急购电.csv", index=False, encoding="utf-8-sig")

    daily = sl.monthly_summary(run)
    daily.to_csv(sl.CSV / "问题2_逐日汇总.csv", index=False, encoding="utf-8-sig")

    # 模板副本
    sl.write_workbook("result2.xlsx", "result2.xlsx", {
        "计划购电量": lambda ws: sl._fill_plan_sheet(ws, run),
        "充放电量": lambda ws: sl._fill_cycle_sheet(ws, run),
        "紧急购电量": lambda ws: sl._fill_emergency_sheet(ws, run),
    })

    if not args.no_figures:
        figures(data, run, pf, sens)
    print(json.dumps(dict(gates=gates, **{k: v for k, v in sl.year_metrics(run).items()
                                         if k != "daily_cash"}), ensure_ascii=False, indent=2))
    print("elapsed_s", round(time.perf_counter() - t0, 1))


if __name__ == "__main__":
    main()
