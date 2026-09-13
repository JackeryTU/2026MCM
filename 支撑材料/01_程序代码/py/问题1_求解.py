# -*- coding: utf-8 -*-
"""问题一：典型日确定性线性规划。

数据：附件 1（典型日 144 时段电价、小区负载、光伏发电预测功率）。
模型：第 4 节储能—功率平衡 LP，令 g=p、S_1=S_145=6000 kWh，
      目标 min Σ c_t p_t；随后在费用不变容差内最小化充放电吞吐。
输出：results/result1.xlsx（官方模板副本）、results/csv/、figures/问题1/。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd

import microgrid_core as mc
import utils.plot_style as ps

ROOT = mc.ROOT
TPL_DIR = ROOT / "data" / "附件" / "附件5"
RES = ROOT / "results"
CSV = RES / "csv"
FIG = ROOT / "figures" / "问题1"


def solve() -> dict:
    data = mc.load_inputs()
    c = data["price1"]                       # 元/kWh
    load_kwh = data["load1"] * mc.DT         # kWh/槽
    pv_kwh = data["pv1"] * mc.DT
    res = mc.solve_day_lp(c, load_kwh, pv_kwh, mc.S0, s_end=mc.S0)
    # 末槽储能电量序列 S_t（t=0 为 0:00，即 S0；S[t] 为第 t+1 槽末）
    S = np.concatenate([[mc.S0], res["S"]])
    g = np.maximum(res["g"], 0.0)
    x = np.maximum(res["x"], 0.0)
    y = np.maximum(res["y"], 0.0)
    v = np.maximum(res["v"], 0.0)
    out = dict(
        c=c, load_kwh=load_kwh, pv_kwh=pv_kwh,
        g=g, x=x, y=y, v=v, S=S,
        curtail=np.maximum(res["curtail"], 0.0),
        cash_cost=float(np.sum(c * g)),
        solver_objective=res["objective"],
    )
    return out


def verify(sol: dict) -> dict:
    """§11.4 数值门禁：母线平衡、边界、互斥、日电量恒等式。"""
    g, x, y, v, S = sol["g"], sol["x"], sol["y"], sol["v"], sol["S"]
    L, V = sol["load_kwh"], sol["pv_kwh"]
    balance_res = float(np.abs(g + v + y - x - L).max())
    soc_res = float(np.abs(np.diff(S) - (mc.ETA_C * x - y / mc.ETA_D)).max())
    bound_res = float(max(
        max(np.maximum(S - mc.S_MAX, 0).max(), np.maximum(mc.S_MIN - S, 0).max()),
        max(np.maximum(x - mc.B_MAX, 0).max(), np.maximum(y - mc.B_MAX, 0).max()),
        max(np.maximum(v - V, 0).max(), -min(v.min(), 0.0)),
        -min(g.min(), 0.0),
    ))
    overlap = float(np.minimum(x, y).max())
    X, Y, G = float(x.sum()), float(y.sum()), float(g.sum())
    ident = {
        "Y_minus_0.81X": Y - mc.ETA_C * mc.ETA_D * X,
        "G_identity": G - (float(L.sum()) - float(v.sum()) + (1 - mc.ETA_C * mc.ETA_D) * X),
        "no_battery_purchase": float((L - V).clip(min=0).sum()),
    }
    return dict(
        balance_max=balance_res, soc_res_max=soc_res, bound_max=bound_res,
        overlap_max=overlap, X=X, Y=Y, G=G,
        S_min=float(S.min()), S_max=float(S.max()),
        curtail_total=float(sol["curtail"].sum()),
        cash_cost=float(sol["cash_cost"]),
        solver_objective=float(sol["solver_objective"]),
        **ident,
    )


def write_template(sol: dict) -> None:
    """按官方模板行序逐列填入（不改表头）。"""
    RES.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.load_workbook(TPL_DIR / "result1.xlsx")
    ws = wb["计划购电量"]
    for i in range(mc.T):
        ws.cell(row=2 + i, column=2).value = float(sol["g"][i])
    ws2 = wb["充放电量"]
    ch = mc.four_hour_blocks(sol["x"])
    dis = mc.four_hour_blocks(sol["y"])
    for k in range(6):
        ws2.cell(row=2 + k, column=2).value = float(ch[k])
        ws2.cell(row=2 + k, column=3).value = float(dis[k])
    ws2.cell(row=2, column=5).value = float(sol["S"][0])
    ws2.cell(row=3, column=5).value = float(sol["S"][-1])
    wb.save(RES / "result1.xlsx")


def write_csv(sol: dict, met: dict) -> None:
    CSV.mkdir(parents=True, exist_ok=True)
    idx = np.arange(mc.T)
    df = pd.DataFrame({
        "槽序号": idx + 1,
        "自然区间": [f"{mc.slot_label_start(i)}-{mc.slot_label_end(i)}" for i in idx],
        "电价_元每kWh": sol["c"],
        "负荷_kWh": sol["load_kwh"],
        "光伏可用_kWh": sol["pv_kwh"],
        "光伏利用_kWh": sol["v"],
        "弃光_kWh": sol["curtail"],
        "计划购电量_kWh": sol["g"],
        "充电量_kWh": sol["x"],
        "放电量_kWh": sol["y"],
        "槽末储电量_kWh": sol["S"][1:],
    })
    df.to_csv(CSV / "问题1_逐槽计划.csv", index=False, encoding="utf-8-sig")

    blocks = pd.DataFrame({
        "时间段": ["0:00-4:00", "4:00-8:00", "8:00-12:00", "12:00-16:00",
                 "16:00-20:00", "20:00-24:00"],
        "充电量_kWh": mc.four_hour_blocks(sol["x"]),
        "放电量_kWh": mc.four_hour_blocks(sol["y"]),
    })
    blocks.loc[len(blocks)] = ["全天", blocks["充电量_kWh"].sum(), blocks["放电量_kWh"].sum()]
    blocks.to_csv(CSV / "问题1_四小时聚合.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame([{
        "项目": "数值", "全天计划购电量_kWh": met["G"],
        "全天购电费_元": met["cash_cost"],
        "含吞吐正则的求解目标值": met["solver_objective"],
        "全天充电量_kWh": met["X"], "全天放电量_kWh": met["Y"],
        "0时储电量_kWh": sol["S"][0], "24时储电量_kWh": sol["S"][-1],
        "弃光量_kWh": met["curtail_total"],
        "无储能购电量_kWh": met["no_battery_purchase"],
    }])
    summary.to_csv(CSV / "问题1_汇总.csv", index=False, encoding="utf-8-sig")
    with open(CSV / "问题1_校验.json", "w", encoding="utf-8") as fh:
        json.dump(met, fh, ensure_ascii=False, indent=2)


def figures(sol: dict, met: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    ps.apply_publication_style("zh", "report")
    t = (np.arange(mc.T) + 0.5) * mc.DT          # 小时
    L = sol["load_kwh"] / mc.DT
    V = sol["pv_kwh"] / mc.DT
    g = sol["g"] / mc.DT

    # 图 1：电价、功率平衡与储能状态
    fig, axes = ps.publication_subplots(3, 1, aspect=1.55, height_ratios=[1, 1.35, 1.25])
    ax = axes[0]
    ax.plot(t, sol["c"], color=ps.PALETTE["secondary"])
    ax.set_ylabel("电价/(元/kWh)")
    ax.set_title("典型日电价")
    ax.set_xlim(0, 24)

    ax = axes[1]
    ax.plot(t, L, color=ps.PALETTE["dark"], label="负荷")
    ax.plot(t, V, color=ps.PALETTE["positive"], label="可用光伏")
    ax.plot(t, g, color=ps.PALETTE["primary"], label="计划购电")
    ax.plot(t, sol["v"] / mc.DT, color=ps.PALETTE["sky"], lw=0.9, label="光伏利用")
    ax.set_ylabel("功率/kW")
    ax.set_title("负荷、光伏与计划购电")
    ax.legend(ncols=2)
    ax.set_xlim(0, 24)

    ax = axes[2]
    ax.plot(t, sol["S"][1:], color=ps.PALETTE["contrast"], label="储电量")
    ax.axhline(mc.S_MIN, color=ps.PALETTE["neutral"], lw=0.7, ls="--", label="SOC 上下限")
    ax.axhline(mc.S_MAX, color=ps.PALETTE["neutral"], lw=0.7, ls="--")
    ax.set_ylabel("储电量/kWh")
    ax.set_title("储能状态轨迹")
    ax.set_xlabel("时刻/h")
    ax.legend()
    ax.set_xlim(0, 24)
    ps.add_panel_labels(axes)
    ps.export_figure(fig, FIG / "问题1_典型日最优计划")
    plt.close(fig)

    # 图 2：四小时聚合充放电量
    fig, ax = ps.publication_subplots(1, 1, aspect=0.55)
    labels = ["0-4", "4-8", "8-12", "12-16", "16-20", "20-24"]
    pos = np.arange(6)
    ch = mc.four_hour_blocks(sol["x"])
    dis = mc.four_hour_blocks(sol["y"])
    ax.bar(pos - 0.18, ch, 0.36, color=ps.PALETTE["primary"], label="充电量")
    ax.bar(pos + 0.18, dis, 0.36, color=ps.PALETTE["contrast"], label="放电量")
    ax.set_xticks(pos, labels)
    ax.set_xlabel("时段/h")
    ax.set_ylabel("电量/kWh")
    ax.set_title("四小时聚合充放电量")
    ax.legend()
    ps.export_figure(fig, FIG / "问题1_四小时聚合充放电")
    plt.close(fig)

    # 图 3：有储能与无储能对比
    fig, axes = ps.publication_subplots(1, 2, aspect=0.72)
    g0 = (sol["load_kwh"] - sol["pv_kwh"]).clip(min=0)
    labels = ["有储能", "无储能"]
    axes[0].bar(labels, [met["G"], met["no_battery_purchase"]],
                color=[ps.PALETTE["primary"], ps.PALETTE["neutral"]], width=0.5)
    axes[0].set_ylabel("全天购电量/kWh")
    axes[0].set_title("全天购电量")
    axes[1].bar(labels, [met["objective"], float((sol["c"] * g0).sum())],
                color=[ps.PALETTE["primary"], ps.PALETTE["neutral"]], width=0.5)
    axes[1].set_ylabel("全天购电费/元")
    axes[1].set_title("全天购电费")
    ps.export_figure(fig, FIG / "问题1_储能效益对比")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figures", action="store_true",
                    help="只生成数值结果和模板，不生成或修改图片")
    args = ap.parse_args()
    sol = solve()
    met = verify(sol)
    write_template(sol)
    write_csv(sol, met)
    if not args.no_figures:
        figures(sol, met)
    print(json.dumps(met, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
