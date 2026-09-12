# -*- coding: utf-8 -*-
"""不画图的年度补充计算：储能结构敏感性与预测误差倍率压力测试。"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from multiprocessing import Pool

import pandas as pd

import microgrid_core as mc
import solution_lib as sl


BASE = {
    "ETA_C": mc.ETA_C, "ETA_D": mc.ETA_D, "E_MAX": mc.E_MAX,
    "S_MIN": mc.S_MIN, "S_MAX": mc.S_MAX, "P_MAX": mc.P_MAX,
    "B_MAX": mc.B_MAX, "S0": mc.S0,
}


def _set_parameters(cfg: dict) -> None:
    """在独立工作进程中设置单因素情景参数。"""
    for key, value in BASE.items():
        setattr(mc, key, value)
    if "eta" in cfg:
        mc.ETA_C = mc.ETA_D = float(cfg["eta"])
    if "capacity_factor" in cfg:
        factor = float(cfg["capacity_factor"])
        mc.E_MAX = BASE["E_MAX"] * factor
        mc.S_MIN = 0.1 * mc.E_MAX
        mc.S_MAX = 0.9 * mc.E_MAX
    if "power_factor" in cfg:
        factor = float(cfg["power_factor"])
        mc.P_MAX = BASE["P_MAX"] * factor
        mc.B_MAX = mc.P_MAX * mc.DT
    if "soc_ratio" in cfg:
        mc.S0 = float(cfg["soc_ratio"]) * mc.E_MAX


def _worker(cfg: dict) -> dict:
    _set_parameters(cfg)
    data = mc.load_inputs()
    resid = sl.net_residual_matrix("F1", data)
    gamma = float(cfg.get("gamma", 1.0))
    run = sl.run_year_q2(data, "F1", 0.9, 30, resid=resid * gamma,
                         n_days=365, executor="mpc")
    metrics = sl.year_metrics(run)
    gates = sl.check_run(run)
    return {
        **cfg,
        **{k: v for k, v in metrics.items() if k != "daily_cash"},
        "eta_c": mc.ETA_C, "eta_d": mc.ETA_D,
        "capacity_kWh": mc.E_MAX, "power_kW": mc.P_MAX,
        "S_min_kWh": mc.S_MIN, "S_max_kWh": mc.S_MAX,
        "S0_kWh": mc.S0,
        "numeric_ok": bool(
            gates["balance_max"] <= 1e-6
            and gates["overlap_max"] <= 1e-5
            and gates["bound_S"] <= 1e-7
        ),
    }


def configs() -> list[dict]:
    """唯一计算情景；与基准相同的三种标注在汇总阶段复制。"""
    return [
        {"group": "基准", "scenario": "基准"},
        {"group": "效率", "scenario": "eta=sqrt(0.9)", "eta": math.sqrt(0.9)},
        {"group": "初始SOC", "scenario": "初始SOC=20%", "soc_ratio": 0.2},
        {"group": "初始SOC", "scenario": "初始SOC=80%", "soc_ratio": 0.8},
        {"group": "容量", "scenario": "容量-20%", "capacity_factor": 0.8},
        {"group": "容量", "scenario": "容量+20%", "capacity_factor": 1.2},
        {"group": "功率", "scenario": "功率-20%", "power_factor": 0.8},
        {"group": "功率", "scenario": "功率+20%", "power_factor": 1.2},
        {"group": "误差倍率", "scenario": "gamma=0.5", "gamma": 0.5},
        {"group": "误差倍率", "scenario": "gamma=1.5", "gamma": 1.5},
        {"group": "误差倍率", "scenario": "gamma=2.0", "gamma": 2.0},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    cfgs = configs()
    workers = args.workers or min(4, max(1, (os.cpu_count() or 4) - 2))
    t0 = time.perf_counter()
    print(f"[SENS] scenarios={len(cfgs)} workers={workers}", flush=True)
    if workers <= 1:
        rows = [_worker(c) for c in cfgs]
    else:
        with Pool(workers) as pool:
            rows = pool.map(_worker, cfgs)
    base = next(r for r in rows if r["scenario"] == "基准")
    for group, scenario, extra in [
        ("效率", "eta=0.9/0.9", {"eta": 0.9}),
        ("初始SOC", "初始SOC=50%", {"soc_ratio": 0.5}),
        ("误差倍率", "gamma=1.0", {"gamma": 1.0}),
    ]:
        alias = dict(base)
        alias.update(group=group, scenario=scenario, **extra)
        rows.append(alias)
    df = pd.DataFrame(rows)
    base_cost = float(df.loc[df["scenario"] == "基准", "cash_total"].iloc[0])
    df["相对基准费用变化_元"] = df["cash_total"] - base_cost
    df["相对基准费用变化_pct"] = (df["cash_total"] / base_cost - 1.0) * 100.0
    sl.CSV.mkdir(parents=True, exist_ok=True)
    common = ["group", "scenario", "eta_c", "eta_d", "capacity_kWh",
              "power_kW", "S0_kWh", "cash_total", "相对基准费用变化_元",
              "相对基准费用变化_pct", "contract_kwh", "emergency_kwh",
              "emergency_slots", "charge_kwh", "discharge_kwh",
              "unused_contract_kwh", "S_min", "S_max", "S_end", "numeric_ok"]
    structural = df[df["group"] != "误差倍率"][common].copy()
    pressure = df[df["group"] == "误差倍率"][["gamma"] + common].copy()
    structural.to_csv(sl.CSV / "模型评价_结构敏感性.csv", index=False,
                      encoding="utf-8-sig")
    pressure.to_csv(sl.CSV / "模型评价_误差倍率压力测试.csv", index=False,
                    encoding="utf-8-sig")
    with pd.ExcelWriter(mc.RESULT_DIR / "补充分析结果.xlsx", engine="openpyxl") as writer:
        structural.to_excel(writer, sheet_name="结构敏感性", index=False)
        pressure.to_excel(writer, sheet_name="误差倍率压力测试", index=False)
    summary = {
        "scenario_count": len(df),
        "all_numeric_ok": bool(df["numeric_ok"].all()),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    with open(sl.CSV / "模型评价_补充计算门禁.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
