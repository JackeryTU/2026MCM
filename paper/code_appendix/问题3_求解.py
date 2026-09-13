
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
FULL_SET = (0, 1, 2, 3)


def _cfg(tag, mode, H, allowed, updates, note, keep=False):
    return {"tag": tag, "mode": mode, "H": H, "allowed": tuple(allowed),
            "updates": tuple(updates), "note": note, "keep": keep}


def build_configs() -> list:
    cfgs = [
        _cfg("main", "stratified", 6, FULL_SET, (1, 2, 3),
             "主配置：分层储备 + H=6 + 三次调整", keep=True),
        _cfg("uniform", "uniform", 6, FULL_SET, (1, 2, 3),
             "对照：统一逐时段分位储备 + H=6", keep=True),
        _cfg("h72", "stratified", 72, FULL_SET, (1, 2, 3),
             "对照：分层储备 + H=72", keep=True),
    ]
    for allowed, updates, note in [
        ((0,), (), "消融 空集：仅 0:00 预报，仍共享每 10 min 重算机会"),
        ((0, 1), (1,), "消融 {6}"),
        ((0, 2), (2,), "消融 {12}"),
        ((0, 3), (3,), "消融 {18}"),
        ((0, 1, 2), (1, 2), "消融 {6,12}"),
        ((0, 1, 3), (1, 3), "消融 {6,18}"),
        ((0, 2, 3), (2, 3), "消融 {12,18}"),
    ]:
        tag = "abl_" + ("_".join(str(h) for h in updates) or "none")
        cfgs.append(_cfg(tag, "stratified", 6, allowed, updates, note,
                         keep=True))
    return cfgs





_G = {}


def _init(data, prof, n_days):
    _G["data"] = data
    _G["prof"] = prof
    _G["n_days"] = n_days


def _coverage(data, prof, allowed, mode, alpha, W, d0=31, d1=365):
    resid, pv_ver = sl.q3_net_residual(data, prof, allowed, "F1", "archive")
    L, V = data["load_act"], data["pv_act"]
    R = np.empty((d1 - d0, mc.T))
    for i, d in enumerate(range(d0, d1)):
        lhat, _ = sl.predict_center("F1", d, data)
        res_kw, _ = sl.q3_reserve(d, resid, alpha, W, mode, allowed)
        R[i] = (L[d] - V[d]) - (lhat - pv_ver[d] + res_kw)
    resid_f1 = resid[d0:d1]
    return {
        "Ncov": float((R <= 0.0).mean()),
        "pinball": float((R * (alpha - (R < 0.0))).mean()),
        "resid_MAE_kW": float(np.abs(resid_f1).mean()),
        "resid_RMSE_kW": float(np.sqrt((resid_f1 ** 2).mean())),
    }


def _worker(cfg):
    data, prof, n_days = _G["data"], _G["prof"], _G["n_days"]
    run = sl.run_year_q34(data, prof, "F1", 0.7725, 35, cfg["mode"], cfg["H"],
                          cfg["allowed"], cfg["updates"], "archive", "fixed",
                          "median30", "fixed", 5.0, n_days)
    out = dict(cfg)
    if n_days > 31:
        out["metrics"] = sl.year_metrics_q34(run, 31)
        out["coverage"] = _coverage(data, prof, cfg["allowed"], cfg["mode"], 0.7725, 35)
        out["gates"] = sl.check_run_q34(run, 31)
    if cfg["keep"]:
        out["arrays"] = {k: run[k] for k in KEYS}
        out["stats"] = run["stats"]
        out["S_end"] = run["S_end"]
        out["resid"] = run["resid"]
        out["pv_ver"] = run["pv_ver"]
    return out





def slot_table_q3(run, price_day, day_indices):
    rows = []
    for d in day_indices:
        c = price_day if np.ndim(price_day) == 1 else price_day[d]
        for t in sl.REP_SLOTS:
            rows.append({
                "日期": str(mc.date_of(d)),
                "时段": "%s-%s" % (mc.slot_label_start(t), mc.slot_label_end(t)),
                "电价_元每kWh": c[t],
                "负荷_kW": float(run["load_kw"][d][t]),
                "光伏实际_kW": float(run["pv_kw"][d][t]),
                "原计划合同_p_kWh": run["P"][d][t],
                "最终合同_q_kWh": run["Q"][d][t],
                "实际购电_kWh": run["Q"][d][t] - run["W"][d][t] + run["U"][d][t],
                "应急购电_u_kWh": run["U"][d][t],
                "充电量_kWh": run["X"][d][t],
                "放电量_kWh": run["Y"][d][t],
                "时段末储电量_kWh": run["S"][d][t],
                "合同调整量_kWh": run["Q"][d][t] - run["P"][d][t],
            })
    return pd.DataFrame(rows)


def ablation_table(records):
    by_tag = {r["tag"]: r for r in records if "metrics" in r}
    base = by_tag["abl_none"]["metrics"]
    rows = []
    for rec in records:

        if "metrics" not in rec or not (rec["tag"].startswith("abl_")
                                         or rec["tag"] == "main"):
            continue
        m = rec["metrics"]
        dn = int(m["updates"] - base["updates"])
        dc = float(base["cash_total"] - m["cash_total"])
        rows.append({
            "获准集合S": ("{6,12,18}" if rec["tag"] == "main" else
                           ("{}" if rec["tag"] == "abl_none" else
                            rec["note"].replace("消融 ", ""))),
            "tag": rec["tag"],
            "允许发布时刻": ",".join(str(sl.ISSUE_HOURS[r]) for r in rec["allowed"]),
            "现金总费_元": m["cash_total"],
            "计划购电量_kWh": m["plan_kwh"],
            "最终合同量_kWh": m["contract_kwh"],
            "调整事件数": m["updates"],
            "dn_相对空集": dn,
            "应急购电量_kWh": m["emergency_kwh"],
            "应急时段数": m["emergency_slots"],
            "弃购量_kWh": m["unused_contract_kwh"],
            "期末储电量_kWh": m["S_end"],
            "相对空集节约_元": dc,
            "kappa_star_元每次": (dc / dn) if dn > 0 else float("nan"),
        })
    df = pd.DataFrame(rows)
    cash = {r["tag"]: (r["metrics"]["cash_total"], r["allowed"])
            for r in records if "metrics" in r and
            (r["tag"].startswith("abl_") or r["tag"] == "main")}
    for r in (1, 2, 3):
        diffs = []
        for tag, (c_s, allowed) in cash.items():
            if r in allowed:
                continue
            enlarged = tuple(sorted((set(allowed) | {r}) - {0}))
            tag2 = "main" if enlarged == (1, 2, 3) else (
                "abl_" + ("_".join(str(h) for h in enlarged) or "none"))
            if tag2 in cash:
                diffs.append(c_s - cash[tag2][0])
        df["版本%d时点_平均边际收益_元" % sl.ISSUE_HOURS[r]] = (
            float(np.mean(diffs)) if diffs else float("nan"))
    return df


def lead_scale_table(data, prof, offset=31):
    rows = []
    for r in range(sl.N_ISSUE):
        for h in range(1, 25):
            pos = sl.ISSUE_SLOTS[r] + 6 * h
            if pos > mc.T:
                continue
            err = np.array([data["pv_act"][d, pos - 1] - prof[d, r, pos]
                            for d in range(offset, 365)])
            rows.append({
                "发布时刻_h": sl.ISSUE_HOURS[r], "提前期_h": h,
                "MAE_kW": float(np.abs(err).mean()),
                "RMSE_kW": float(np.sqrt((err ** 2).mean())),
                "Bias_kW": float(err.mean()),
                "q90_kW": mc.empirical_quantile(err, 0.9),
                "尺度_s_kW": float(np.sqrt((err ** 2).mean())),
            })
    df = pd.DataFrame(rows)
    agg = df.groupby("提前期_h").agg(
        MAE_kW=("MAE_kW", "mean"), RMSE_kW=("RMSE_kW", "mean"),
        Bias_kW=("Bias_kW", "mean"), q90_kW=("q90_kW", "mean")).reset_index()
    return df, agg


def monthly_table(run, offset=31):
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
            "弃光量_kWh": float((run["pv_kwh"][d] - run["V"][d]).sum()),
            "日末储电量_kWh": run["S"][d][-1],
        })
    return pd.DataFrame(rows)





def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--main-only", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()
    mc.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    sl.CSV.mkdir(parents=True, exist_ok=True)

    data = mc.load_inputs()
    prof = sl.build_pv_archive(data, 35)
    cfgs = build_configs()
    if args.main_only:
        cfgs = cfgs[:1]
    n_cpu = os.cpu_count() or 4
    workers = args.workers or min(len(cfgs), max(1, n_cpu - 4))
    print("[Q3] configs=%d workers=%d days=%d" % (len(cfgs), workers, args.days),
          flush=True)

    if workers <= 1:
        _init(data, prof, args.days)
        records = [_worker(c) for c in cfgs]
    else:
        with Pool(workers, initializer=_init, initargs=(data, prof, args.days)) as pool:
            records = pool.map(_worker, cfgs)
    print("[Q3] runs done in %.0fs" % (time.perf_counter() - t0), flush=True)

    specs = []
    for rec in records:
        if "metrics" not in rec:
            continue
        row = {"方案": rec["note"], "tag": rec["tag"], "mode": rec["mode"],
               "H": rec["H"], "updates": rec["updates"]}
        row.update({k: v for k, v in rec["metrics"].items() if k != "daily_cash"})
        row.update({"cov_" + k: v for k, v in rec["coverage"].items()})
        specs.append(row)
    pd.DataFrame(specs).to_csv(sl.CSV / "问题3_方案对比.csv", index=False,
                               encoding="utf-8-sig")

    if any(r["tag"].startswith("abl_") for r in records):
        ablation_table(records).to_csv(sl.CSV / "问题3_信息消融.csv", index=False,
                                       encoding="utf-8-sig")

    lead_df, lead_agg = lead_scale_table(data, prof)
    lead_df.to_csv(sl.CSV / "问题3_预见期误差_分来源.csv", index=False,
                   encoding="utf-8-sig")
    lead_agg.to_csv(sl.CSV / "问题3_预见期尺度.csv", index=False, encoding="utf-8-sig")

    main_rec = next(r for r in records if r["tag"] == "main")
    run = dict(main_rec["arrays"])
    run["stats"] = main_rec["stats"]
    run["S_end"] = main_rec["S_end"]
    run["resid"] = main_rec["resid"]
    run["pv_ver"] = main_rec["pv_ver"]
    run["load_kwh"] = data["load_act"] * mc.DT
    run["pv_kwh"] = data["pv_act"] * mc.DT
    run["load_kw"] = data["load_act"]
    run["pv_kw"] = data["pv_act"]
    run["price_settle"] = np.tile(data["price1"], (run["P"].shape[0], 1))


    n_run = run["P"].shape[0]
    day_idx = [sl.DAY_INDEX[d] for d in sl.REP_DATES if sl.DAY_INDEX[d] < n_run]
    if day_idx:
        slot_table_q3(run, data["price1"], day_idx).to_csv(
            sl.CSV / "问题3_表1_指定时段.csv", index=False, encoding="utf-8-sig")
        sl.energy_block_table(run, day_idx).to_csv(
            sl.CSV / "问题3_表2_四小时聚合.csv", index=False, encoding="utf-8-sig")
        sl.emergency_table(run, day_idx).to_csv(
            sl.CSV / "问题3_表3_紧急购电.csv", index=False, encoding="utf-8-sig")
    monthly_table(run).to_csv(sl.CSV / "问题3_逐日汇总.csv", index=False,
                              encoding="utf-8-sig")

    gates = dict(main_rec["gates"])
    gates["coverage"] = main_rec["coverage"]
    gates["all_numeric_ok"] = bool(
        gates["balance_max_kWh"] <= 1e-6 and gates["overlap_max_kWh"] <= 1e-5
        and gates["charge_emergency_max_kWh"] <= 1e-5
        and max(gates[k] for k in ("bound_p", "bound_q", "bound_x", "bound_y",
                                   "bound_v", "bound_u", "bound_w", "bound_S")) <= 1e-7
        and gates["settle_flow_max_yuan"] <= 1e-5)
    with open(sl.CSV / "问题3_门禁.json", "w", encoding="utf-8") as fh:
        json.dump(gates, fh, ensure_ascii=False, indent=2, default=float)

    sl.write_workbook("result3.xlsx", "result3.xlsx", {
        "计划购电量": lambda ws: sl._fill_plan_sheet(ws, run, key="P", fee_key="cash_plan"),
        "调整购电量": lambda ws: sl._fill_plan_sheet(ws, run, key="Q", fee_key="cash_contract"),
        "充放电量": lambda ws: sl._fill_cycle_sheet(ws, run),
        "紧急购电量": lambda ws: sl._fill_emergency_sheet(ws, run),
    })

    np.savez_compressed(sl.CSV / "q3_bundle.npz",
                        **{"%s_%s" % (r["tag"], k): r["arrays"][k]
                           for r in records if "arrays" in r for k in KEYS})
    def _jsonable(rec):
        out = {}
        for k, v in rec.items():
            if k in ("arrays", "resid", "pv_ver", "stats"):
                continue
            if k == "metrics":
                v = {kk: vv for kk, vv in v.items() if kk != "daily_cash"}
            out[k] = v
        return out

    meta = {"days": args.days, "configs": [_jsonable(r) for r in records]}
    with open(sl.CSV / "q3_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2, default=float)

    summary = {k: v for k, v in main_rec["metrics"].items() if k != "daily_cash"}
    print(json.dumps({"summary": summary, "gates": gates}, ensure_ascii=False,
                     indent=2, default=float))
    print("elapsed_s", round(time.perf_counter() - t0, 1))


if __name__ == "__main__":
    main()
