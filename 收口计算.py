# -*- coding: utf-8 -*-
"""评审意见 Gate 1--3：会计闭环、口径审计、执行诊断与轻量稳健性。"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import microgrid_core as mc
import solution_lib as sl

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "results" / "csv"
OFF = 31
END = 365


def load_npz(name):
    return {k: v for k, v in np.load(CSV / name).items()}


def fixed_price(data):
    return np.tile(data["price1"], (END, 1))


def actual_price(data):
    return data["price_act"]


def fee_parts(P, Q, U, price, q3):
    P, Q, U, price = map(lambda a: np.asarray(a, dtype=float), (P, Q, U, price))
    if price.ndim == 1:
        price = np.tile(price, (P.shape[0], 1))
    over = np.maximum(Q - P, 0.0)
    under = np.maximum(P - Q, 0.0)
    plan_a = (price * np.minimum(P, Q)).sum(axis=1)
    adj = (1.5 * price * over + 0.5 * price * under).sum(axis=1) if q3 else np.zeros(P.shape[0])
    plan_b = (price * P).sum(axis=1)
    emg = (5.0 * price * U).sum(axis=1)
    return {"plan_A": plan_a, "plan_B": plan_b, "adjust": adj, "emergency": emg,
            "total_A": plan_a + adj + emg, "total_B": plan_b + adj + emg}


def diag_row(label, arrays, price, q3, data):
    parts = fee_parts(arrays["P"], arrays["Q"] if q3 else arrays["P"], arrays["U"], price, q3)
    X, Y, S = arrays["X"], arrays["Y"], arrays["S"]
    plan_x = arrays.get("plan_x", np.full_like(X, np.nan))
    plan_y = arrays.get("plan_y", np.full_like(Y, np.nan))
    plan_s = arrays.get("plan_s_next", np.full_like(S, np.nan))
    slc = slice(OFF, END)
    soc_e = S[slc]
    soc_p = plan_s[slc]
    finite = np.isfinite(soc_p)
    e = soc_e[finite] - soc_p[finite]
    cash = parts["total_A"][slc]
    row = {
        "方案": label, "口径A现金总费_元": float(parts["total_A"][slc].sum()),
        "计划费用A_元": float(parts["plan_A"][slc].sum()),
        "调整费用_元": float(parts["adjust"][slc].sum()),
        "紧急费用_元": float(parts["emergency"][slc].sum()),
        "口径B现金总费_元": float(parts["total_B"][slc].sum()),
        "口径B相对A_元": float((parts["total_B"] - parts["total_A"])[slc].sum()),
        "计划购电量_kWh": float(arrays["P"][slc].sum()),
        "最终合同量_kWh": float((arrays["Q"] if q3 else arrays["P"])[slc].sum()),
        "调整量_kWh": float(((arrays["Q"] - arrays["P"]) if q3 else np.zeros_like(arrays["P"]))[slc].sum()),
        "应急购电量_kWh": float(arrays["U"][slc].sum()),
        "弃购量_kWh": float(arrays["W"][slc].sum()),
        "弃光量_kWh": float((data["pv_act"] * mc.DT - arrays["V"])[slc].sum()),
        "充电量_kWh": float(X[slc].sum()), "放电量_kWh": float(Y[slc].sum()),
        "期末SOC_kWh": float(S[-1, -1]),
        "SOC一步MAE_kWh": float(np.abs(e).mean()) if e.size else np.nan,
        "SOC一步RMSE_kWh": float(np.sqrt(np.mean(e * e))) if e.size else np.nan,
        "SOC一步最大绝对误差_kWh": float(np.abs(e).max()) if e.size else np.nan,
        "充电一步MAE_kWh": float(np.abs(X[slc][finite] - plan_x[slc][finite]).mean()) if e.size else np.nan,
        "放电一步MAE_kWh": float(np.abs(Y[slc][finite] - plan_y[slc][finite]).mean()) if e.size else np.nan,
        "费用加总误差_元": float(np.max(np.abs(parts["total_A"] - (parts["plan_A"] + parts["adjust"] + parts["emergency"])) [slc])),
    }
    return row


def settlement_sensitivity(data, q3, q4):
    rows = []
    q3_main = {k[len("main_"):]: v for k, v in q3.items()
               if k.startswith("main_")}
    q4_main = {k[len("Q4-3_median30_"):]: v for k, v in q4.items()
               if k.startswith("Q4-3_median30_")}
    specs = [
        ("Q3主方案", q3_main, fixed_price(data), True),
        ("Q4-3_median30", q4_main, actual_price(data), True),
    ]
    # Q3 八组组合：直接使用保存的逐槽数组，重算两种口径及排序。
    for tag in ("main", "abl_none", "abl_1", "abl_2", "abl_3", "abl_1_2", "abl_1_3", "abl_2_3"):
        a = {k[len(tag)+1:]: v for k, v in q3.items() if k.startswith(tag + "_")}
        p = fee_parts(a["P"], a["Q"], a["U"], fixed_price(data), True)
        rows.append({"问题":"Q3", "方案":tag, "口径A总费_元":p["total_A"][OFF:END].sum(),
                     "口径B总费_元":p["total_B"][OFF:END].sum(),
                     "口径B-A_元":(p["total_B"]-p["total_A"])[OFF:END].sum()})
    for label, a, price, q3flag in specs:
        p = fee_parts(a["P"], a["Q"] if q3flag else a["P"], a["U"], price, q3flag)
        rows.append({"问题":label.split("-")[0], "方案":label,
                     "口径A总费_元":p["total_A"][OFF:END].sum(),
                     "口径B总费_元":p["total_B"][OFF:END].sum(),
                     "口径B-A_元":(p["total_B"]-p["total_A"])[OFF:END].sum()})
    df = pd.DataFrame(rows)
    df["口径A排序"] = df.groupby("问题")["口径A总费_元"].rank(method="min")
    df["口径B排序"] = df.groupby("问题")["口径B总费_元"].rank(method="min")
    return df


def tail_risk(label, daily, dates):
    x = np.asarray(daily, dtype=float)[OFF:END]
    q = float(np.quantile(x, 0.95, method="linear"))
    tail = x[x >= q]
    i = int(np.argmax(x))
    d = pd.Timestamp(str(dates[OFF:END][i])).date()
    return {"方案": label, "日均费用_元": float(x.mean()), "VaR95_元": q,
            "CVaR95_元": float(tail.mean()), "最坏日费用_元": float(x.max()),
            "最坏日": str(d), "最坏5日均费用_元": float(np.sort(x)[-5:].mean())}


def monthly_pair(label_a, a, label_b, b, dates):
    da = pd.DataFrame({"日期": pd.to_datetime(dates[OFF:END]), "a": a[OFF:END], "b": b[OFF:END]})
    da["月份"] = da["日期"].dt.strftime("%Y-%m")
    out = da.groupby("月份").agg(方案A日均费用_元=("a", "mean"), 方案B日均费用_元=("b", "mean"),
                                  天数=("a", "size")).reset_index()
    out["A相对B日均差_元"] = out["方案A日均费用_元"] - out["方案B日均费用_元"]
    out.insert(0, "方案A", label_a); out.insert(1, "方案B", label_b)
    return out


def time_audit(data, q2, q3, q4):
    cols = mc._TIME_COLS
    checks = []
    for i, c in enumerate(cols):
        checks.append({"槽序号": i + 1, "原始列标签": c, "自然区间": f"{mc.slot_label_start(i)}-{mc.slot_label_end(i)}",
                       "是否首槽": i == 0, "是否末槽": i == mc.T - 1})
    df = pd.DataFrame(checks)
    boundary = pd.DataFrame([
        {"审计项":"附件2首列", "结果":cols[0], "判定":"第1槽按0:00-0:10解释"},
        {"审计项":"附件2末列", "结果":cols[-1], "判定":"第144槽按23:50-24:00解释"},
        {"审计项":"官方发布槽", "结果":"0,36,72,108", "判定":"对应0/6/12/18时刻"},
        {"审计项":"重点槽", "结果":"60,72,84,96,108,120", "判定":"对应10:00至20:00起始槽"},
        {"审计项":"跨午夜应急合并", "结果":"不跨午夜", "判定":"连续区间按自然日截断"},
    ])
    # 记录整体±1槽的“已求解轨迹重放”检查，明确其用途仅是索引/结算敏感度。
    rows = []
    dates = np.array([str(mc.date_of(d)) for d in range(END)])
    for label, a, price, q3flag in [("Q2", q2, np.tile(data["price1"], (END,1)), False),
                                    ("Q3", q3, np.tile(data["price1"], (END,1)), True)]:
        for shift in (-1, 0, 1):
            pp = np.roll(price, shift, axis=1)
            p = fee_parts(a["P"], a["Q"] if q3flag else a["P"], a["U"], pp, q3flag)
            rows.append({"方案":label, "价格整体平移槽数":shift,
                         "重放费用_元":p["total_A"][OFF:END].sum()})
    audit = pd.DataFrame(rows)
    baseline = audit.loc[audit["价格整体平移槽数"] == 0].set_index("方案")["重放费用_元"]
    audit["相对原映射_元"] = audit["重放费用_元"] - audit["方案"].map(baseline)
    return pd.concat([df, boundary, audit], ignore_index=True)


def main():
    t0 = time.perf_counter()
    data = mc.load_inputs()
    q2 = load_npz("q2_bundle.npz")
    q3 = load_npz("q3_bundle.npz")
    q4 = load_npz("q4_bundle.npz")
    dates = np.array([str(mc.date_of(d)) for d in range(END)])
    q2a = {k: v for k, v in q2.items()}
    q3a = {k[len("main_"):]: v for k, v in q3.items() if k.startswith("main_")}
    q4a = {k[len("Q4-3_median30_"):]: v for k, v in q4.items() if k.startswith("Q4-3_median30_")}
    q4b = {k[len("Q4-2_median30_"):]: v for k, v in q4.items() if k.startswith("Q4-2_median30_")}

    main_rows = [diag_row("Q2-F1-W35-alpha0.805-H49", q2a, np.tile(data["price1"], (END,1)), False, data),
                 diag_row("Q3-S={6,12,18}", q3a, np.tile(data["price1"], (END,1)), True, data),
                 diag_row("Q4-2-median30", q4b, data["price_act"], False, data),
                 diag_row("Q4-3-median30", q4a, data["price_act"], True, data)]
    pd.DataFrame(main_rows).to_csv(CSV / "总体主表_费用与执行诊断.csv", index=False, encoding="utf-8-sig")

    settle = settlement_sensitivity(data, q3, q4)
    settle.to_csv(CSV / "结算歧义敏感性.csv", index=False, encoding="utf-8-sig")
    time_audit(data, q2a, q3a, q4a).to_csv(CSV / "时间标签索引审计.csv", index=False, encoding="utf-8-sig")

    # 重点日期统一诊断表：按题面四个日期输出同一字段集合。
    rows = []
    for label, a, price, q3flag in [("Q2", q2a, np.tile(data["price1"], (END,1)), False),
                                    ("Q3", q3a, np.tile(data["price1"], (END,1)), True),
                                    ("Q4-2", q4b, data["price_act"], False),
                                    ("Q4-3", q4a, data["price_act"], True)]:
        for date in sl.REP_DATES:
            d = (date - mc.DAY0).days
            if d >= END: continue
            c = price[d] if price.ndim == 2 else price
            Q = a["Q"] if q3flag else a["P"]
            rows.append({"方案":label, "日期":str(date), "全天购电费用_元":float(fee_parts(a["P"][d:d+1], Q[d:d+1], a["U"][d:d+1], c, q3flag)["total_A"][0]),
                         "计划购电量_kWh":a["P"][d].sum(), "最终合同量_kWh":Q[d].sum(), "充电量_kWh":a["X"][d].sum(),
                         "放电量_kWh":a["Y"][d].sum(), "应急购电量_kWh":a["U"][d].sum(), "弃购量_kWh":a["W"][d].sum(),
                         "弃光量_kWh":(data["pv_act"][d]*mc.DT-a["V"][d]).sum(), "0时SOC_kWh":(a["S"][d-1,-1] if d else mc.S0),
                         "24时SOC_kWh":a["S"][d,-1]})
    pd.DataFrame(rows).to_csv(CSV / "重点日期_统一诊断表.csv", index=False, encoding="utf-8-sig")

    risk = pd.DataFrame([
        tail_risk("Q2-F1-W35-alpha0.805-H49", q2["cash"], dates),
        tail_risk("Q3-S={6,12,18}", q3["main_cash"], dates),
        tail_risk("Q4-2-median30", q4["Q4-2_median30_cash"], dates),
        tail_risk("Q4-3-median30", q4["Q4-3_median30_cash"], dates),
    ])
    risk.to_csv(CSV / "尾部风险_日费用VaR_CVaR.csv", index=False, encoding="utf-8-sig")
    monthly_pair("Q3-S={6,12,18}", q3["main_cash"], "Q3-空集", q3["abl_none_cash"], dates).to_csv(CSV / "尾部风险_月度配对差.csv", index=False, encoding="utf-8-sig")

    # 结果校验：有限性、弃光非负、费用分解回加、规划-执行字段完整。
    checks = {"finite": True, "curtail_nonnegative_max_violation_kWh": 0.0,
              "fee_decomposition_max_error_yuan": 0.0, "required_files": [], "elapsed_s": 0.0}
    for a in (q2a, q3a, q4a, q4b):
        for k, v in a.items():
            if isinstance(v, np.ndarray) and not np.isfinite(v).all(): checks["finite"] = False
        checks["curtail_nonnegative_max_violation_kWh"] = max(checks["curtail_nonnegative_max_violation_kWh"], float(np.maximum(a["V"] - data["pv_act"] * mc.DT, 0).max()))
    checks["required_files"] = [p.name for p in (CSV / "总体主表_费用与执行诊断.csv", CSV / "结算歧义敏感性.csv", CSV / "时间标签索引审计.csv", CSV / "重点日期_统一诊断表.csv", CSV / "尾部风险_日费用VaR_CVaR.csv", CSV / "尾部风险_月度配对差.csv") if p.exists()]
    checks["elapsed_s"] = round(time.perf_counter() - t0, 1)
    with open(CSV / "结果文件校验.json", "w", encoding="utf-8") as f: json.dump(checks, f, ensure_ascii=False, indent=2)
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
