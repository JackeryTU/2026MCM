
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import microgrid_core as mc
import solution_lib as sl

KEYS = ("P", "Q", "X", "Y", "U", "W", "V", "S", "cash", "cash_plan",
        "cash_contract", "plan_x", "plan_y", "plan_s_next")
BASES = ("prev_day", "mean7", "mean30", "median30")
BASE_LABEL = {"prev_day": "昨日同时段", "mean7": "近7日均值",
              "mean30": "近30日均值", "median30": "近30日中位数",
              "oracle": "0时已知真价（理想信息）"}
DEFAULT_BASE = "median30"
FULL_SET = (0, 1, 2, 3)
UPDATES2 = (1, 2, 3)
SPIKE_Q = 0.75
OFFSET = 31
NTOT = 365


def build_configs():
    cfgs = []
    for base in BASES:
        cfgs.append({"tag": "Q4-2_" + base, "mode_note": "模式一(固定合同)",
                     "base": base, "updates": (), "pv": "archive",
                     "price_mode": "forecast", "keep": base == DEFAULT_BASE})
    for base in BASES:
        cfgs.append({"tag": "Q4-3_" + base, "mode_note": "模式二(可调合同)",
                     "base": base, "updates": UPDATES2, "pv": "archive",
                     "price_mode": "forecast", "keep": base == DEFAULT_BASE})
    cfgs.append({"tag": "Q4-2_oracle", "mode_note": "模式一(固定合同)",
                 "base": "oracle", "updates": (), "pv": "archive",
                 "price_mode": "oracle", "keep": True})
    cfgs.append({"tag": "Q4-3_oracle", "mode_note": "模式二(可调合同)",
                 "base": "oracle", "updates": UPDATES2, "pv": "archive",
                 "price_mode": "oracle", "keep": True})
    cfgs.append({"tag": "Q4-2_center_" + DEFAULT_BASE,
                 "mode_note": "模式一(固定合同)", "base": DEFAULT_BASE,
                 "updates": (), "pv": "center", "price_mode": "forecast",
                 "keep": True})
    return cfgs





_G = {}


def _init(data, prof, n_days=NTOT):
    _G["data"] = data
    _G["prof"] = prof
    _G["n_days"] = n_days


def _worker(cfg):
    data, prof = _G["data"], _G["prof"]
    n_days = _G["n_days"]
    run = sl.run_year_q34(data, prof, "F1", 0.7725, 35, "stratified", 6, FULL_SET,
                          cfg["updates"], cfg["pv"], cfg["price_mode"],
                          cfg["base"], "actual", 5.0, n_days)
    out = dict(cfg)
    out["metrics"] = sl.year_metrics_q34(run, OFFSET)
    out["gates"] = sl.check_run_q34(run, OFFSET)
    if cfg["keep"]:
        out["arrays"] = {k: run[k] for k in KEYS}
        out["stats"] = run["stats"]
        out["S_end"] = run["S_end"]
    return out





def price_error_table(data, offset=OFFSET, end=NTOT):
    A = data["price_act"]
    days = list(range(offset, end))
    spike = np.vstack([A[d] >= np.quantile(A[d], SPIKE_Q) for d in days])
    rows = []
    for base in BASES:
        e0, e12 = [], []
        for d in days:
            bp = mc.price_forecast_0h(d, A, data["price1"], base)
            e0.append(bp - A[d])
            e12.append(mc.price_intraday_correct(bp, A[d], 72) - A[d])
        E0 = np.vstack(e0)
        E12 = np.vstack(e12)
        rows.append({
            "价格基线": base, "基线说明": BASE_LABEL[base],
            "MAE0_元每kWh": float(np.abs(E0).mean()),
            "RMSE0_元每kWh": float(np.sqrt((E0 ** 2).mean())),
            "Bias0_元每kWh": float(E0.mean()),
            "尖峰MAE0_元每kWh": float(np.abs(E0[spike]).mean()),
            "MAE12_元每kWh": float(np.abs(E12).mean()),
            "尖峰MAE12_元每kWh": float(np.abs(E12[spike]).mean()),
            "最大绝对误差0": float(np.abs(E0).max()),
        })
    return pd.DataFrame(rows)


def coverage_of(data, prof, alpha=0.7725, W=35, mode="stratified", d0=OFFSET,
                d1=NTOT):
    resid, pv_ver = sl.q3_net_residual(data, prof, FULL_SET, "F1", "archive")
    L, V = data["load_act"], data["pv_act"]
    R = np.empty((d1 - d0, mc.T))
    for i, d in enumerate(range(d0, d1)):
        lhat, _ = sl.predict_center("F1", d, data)
        res_kw, _ = sl.q3_reserve(d, resid, alpha, W, mode, FULL_SET)
        R[i] = (L[d] - V[d]) - (lhat - pv_ver[d] + res_kw)
    return {"Ncov": float((R <= 0.0).mean()),
            "pinball": float((R * (alpha - (R < 0.0))).mean()),
            "resid_MAE_kW": float(np.abs(resid[d0:d1]).mean())}





def slot_table(run, price_act, day_indices):
    rows = []
    for d in day_indices:
        c = price_act[d]
        for t in sl.REP_SLOTS:
            rows.append({
                "日期": str(mc.date_of(d)),
                "时段": "%s-%s" % (mc.slot_label_start(t), mc.slot_label_end(t)),
                "实际结算电价_元每kWh": float(c[t]),
                "负荷_kW": float(run["load_kw"][d][t]),
                "光伏实际_kW": float(run["pv_kw"][d][t]),
                "原计划合同_p_kWh": float(run["P"][d][t]),
                "最终合同_q_kWh": float(run["Q"][d][t]),
                "实际购电_kWh": float(run["Q"][d][t] - run["W"][d][t]
                                   + run["U"][d][t]),
                "应急购电_u_kWh": float(run["U"][d][t]),
                "充电量_kWh": float(run["X"][d][t]),
                "放电量_kWh": float(run["Y"][d][t]),
                "时段末储电量_kWh": float(run["S"][d][t]),
                "合同调整量_kWh": float(run["Q"][d][t] - run["P"][d][t]),
            })
    return pd.DataFrame(rows)


def daily_table(run, offset=OFFSET):
    rows = []
    for d in range(offset, run["P"].shape[0]):
        dt = mc.date_of(d)
        rows.append({
            "日期": dt, "月份": dt.strftime("%Y-%m"),
            "现金费_元": run["cash"][d],
            "计划购电量_kWh": run["P"][d].sum(),
            "最终合同量_kWh": run["Q"][d].sum(),
            "调整量_kWh": (run["Q"][d] - run["P"][d]).sum(),
            "应急购电量_kWh": run["U"][d].sum(),
            "充电量_kWh": run["X"][d].sum(),
            "放电量_kWh": run["Y"][d].sum(),
            "弃购量_kWh": run["W"][d].sum(),
            "弃光量_kWh": (run["pv_kwh"][d] - run["V"][d]).sum(),
            "日末储电量_kWh": run["S"][d][-1]})
    return pd.DataFrame(rows)


def _jsonable(rec):
    out = {}
    for k, v in rec.items():
        if k in ("arrays", "stats"):
            continue
        if k == "metrics":
            v = {kk: vv for kk, vv in v.items() if kk != "daily_cash"}
        out[k] = v
    return out


def _numeric_ok(g: dict) -> bool:
    return bool(
        g["balance_max_kWh"] <= 1e-6 and g["overlap_max_kWh"] <= 1e-5
        and g["charge_emergency_max_kWh"] <= 1e-5
        and max(g[k] for k in ("bound_p", "bound_q", "bound_x", "bound_y",
                               "bound_v", "bound_u", "bound_w", "bound_S")) <= 1e-7
        and g["soc_chain_max_kWh"] <= 1e-6
        and g["soc_end_mismatch_kWh"] <= 1e-6
        and g["settle_flow_max_yuan"] <= 1e-5)





def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--days", type=int, default=NTOT,
                    help="运行前 N 天；小于 365 时只作 Lite 烟雾验证")
    ap.add_argument("--main-only", action="store_true",
                    help="只运行 Q4-2/Q4-3 的 median30 主配置")
    args = ap.parse_args()
    if not OFFSET < args.days <= NTOT:
        ap.error("--days 必须在 32..365 之间")
    t0 = time.perf_counter()
    sl.CSV.mkdir(parents=True, exist_ok=True)

    data = mc.load_inputs()
    prof = sl.build_pv_archive(data, 35)
    cfgs = build_configs()
    if args.main_only:
        keep_tags = {"Q4-2_" + DEFAULT_BASE, "Q4-3_" + DEFAULT_BASE}
        cfgs = [c for c in cfgs if c["tag"] in keep_tags]
    workers = args.workers or min(len(cfgs), max(1, (os.cpu_count() or 4) - 4))
    print("[Q4] configs=%d workers=%d" % (len(cfgs), workers), flush=True)

    if workers <= 1:
        _init(data, prof, args.days)
        records = [_worker(c) for c in cfgs]
    else:
        with Pool(workers, initializer=_init, initargs=(data, prof, args.days)) as pool:
            records = pool.map(_worker, cfgs)
    print("[Q4] runs done in %.0fs" % (time.perf_counter() - t0), flush=True)

    err = price_error_table(data, end=args.days)
    err.to_csv(sl.CSV / "问题4_价格基线误差.csv", index=False,
               encoding="utf-8-sig")
    err_map = dict(zip(err["价格基线"], err["MAE0_元每kWh"]))
    cov = coverage_of(data, prof, d1=args.days)

    rows = []
    for rec in records:
        m = rec["metrics"]
        row = {"方案": rec["mode_note"], "tag": rec["tag"],
               "价格基线": BASE_LABEL.get(rec["base"], rec["base"]),
               "光伏预报源": rec["pv"], "定价信息": rec["price_mode"]}
        row.update({k: v for k, v in m.items() if k != "daily_cash"})
        row["价格MAE0_元每kWh"] = float(
            err_map.get(rec["base"], 0.0 if rec["base"] == "oracle" else np.nan))
        row["安全轨迹覆盖率"] = cov["Ncov"]
        rows.append(row)
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(sl.CSV / "问题4_方案对比.csv", index=False,
                  encoding="utf-8-sig")

    gate_report = {r["tag"]: dict(r["gates"], coverage_Ncov=cov["Ncov"],
                                  all_numeric_ok=_numeric_ok(r["gates"]))
                   for r in records}
    gate_report["_all_numeric_ok"] = bool(
        all(v["all_numeric_ok"] for k, v in gate_report.items()
            if not k.startswith("_")))
    with open(sl.CSV / "问题4_门禁.json", "w", encoding="utf-8") as fh:
        json.dump(gate_report, fh, ensure_ascii=False, indent=2, default=float)

    by_tag = {r["tag"]: r for r in records}
    prim2 = by_tag["Q4-2_" + DEFAULT_BASE]
    prim3 = by_tag["Q4-3_" + DEFAULT_BASE]
    common = {"load_kwh": data["load_act"] * mc.DT,
              "pv_kwh": data["pv_act"] * mc.DT,
              "load_kw": data["load_act"], "pv_kw": data["pv_act"],
              "cash_plan": None, "cash_contract": None}
    day_idx = [sl.DAY_INDEX[d] for d in sl.REP_DATES
               if sl.DAY_INDEX[d] < args.days]
    price_act = data["price_act"]
    for prim, key in ((prim2, "问题4-2"), (prim3, "问题4-3")):
        run = dict(prim["arrays"])
        for k, v in common.items():
            if v is not None:
                run[k] = v
        slot_table(run, price_act, day_idx).to_csv(
            sl.CSV / (key + "_表1_指定时段.csv"), index=False,
            encoding="utf-8-sig")
        sl.energy_block_table(run, day_idx).to_csv(
            sl.CSV / (key + "_表2_四小时聚合.csv"), index=False,
            encoding="utf-8-sig")
        sl.emergency_table(run, day_idx).to_csv(
            sl.CSV / (key + "_表3_紧急购电.csv"), index=False,
            encoding="utf-8-sig")
        daily_table(run).to_csv(sl.CSV / (key + "_逐日汇总.csv"), index=False,
                                encoding="utf-8-sig")

    run2 = dict(prim2["arrays"])
    run2.update({k: v for k, v in common.items() if v is not None})
    run3 = dict(prim3["arrays"])
    run3.update({k: v for k, v in common.items() if v is not None})
    if args.days == NTOT:
        sl.write_workbook("result4-2.xlsx", "result4-2.xlsx", {
            "计划购电量": lambda ws: sl._fill_plan_sheet(ws, run2, key="P",
                                                     fee_key="cash_plan"),
            "充放电量": lambda ws: sl._fill_cycle_sheet(ws, run2),
            "紧急购电量": lambda ws: sl._fill_emergency_sheet(ws, run2),
        })
        sl.write_workbook("result4-3.xlsx", "result4-3.xlsx", {
            "计划购电量": lambda ws: sl._fill_plan_sheet(ws, run3, key="P",
                                                     fee_key="cash_plan"),
            "调整购电量": lambda ws: sl._fill_plan_sheet(ws, run3, key="Q",
                                                     fee_key="cash_contract"),
            "充放电量": lambda ws: sl._fill_cycle_sheet(ws, run3),
            "紧急购电量": lambda ws: sl._fill_emergency_sheet(ws, run3),
        })

    np.savez_compressed(sl.CSV / "q4_bundle.npz",
                        **{("%s_%s" % (r["tag"], k)): r["arrays"][k]
                           for r in records if "arrays" in r for k in KEYS})
    with open(sl.CSV / "q4_metrics.json", "w", encoding="utf-8") as fh:
        json.dump({"configs": [_jsonable(r) for r in records],
                   "price_error": err.to_dict("records"), "coverage": cov},
                  fh, ensure_ascii=False, indent=2, default=float)

    print(cmp_df[["tag", "cash_total", "emergency_kwh", "updates"
                  ]].to_string())
    summary = {"main_base": DEFAULT_BASE,
               "Q4-2_cash": float(prim2["metrics"]["cash_total"]),
               "Q4-3_cash": float(prim3["metrics"]["cash_total"]),
               "Q4-2_emergency_kWh": float(prim2["metrics"]["emergency_kwh"]),
               "Q4-3_emergency_kWh": float(prim3["metrics"]["emergency_kwh"]),
               "all_numeric_ok": gate_report["_all_numeric_ok"]}
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    print("elapsed_s", round(time.perf_counter() - t0, 1))


if __name__ == "__main__":
    main()
