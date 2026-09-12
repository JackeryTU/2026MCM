# -*- coding: utf-8 -*-
"""问题三增量：3/9/15/21 h 候选预报时刻的单次边际代理回放。

附件3没有这些时刻的独立预报版本，因此候选版本采用“此前最近官方版本”的
已校正预报曲线作为代理；结果只用于代理情景估计，不能称为实测预报结论。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from multiprocessing import Pool

# 四个进程各跑一个候选情景；限制每个 C/C++ 数值内核为单线程，避免过度争抢。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import pandas as pd

import microgrid_core as mc
import solution_lib as sl


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "csv"
KEYS = ("P", "Q", "X", "Y", "U", "W", "V", "S", "cash", "cash_plan",
        "cash_contract", "plan_x", "plan_y", "plan_s_next")


def make_proxy_profiles(data, W=35):
    """在原四版本后追加四个候选版本，使用此前官方版本作为代理。"""
    old_slots, old_hours, old_n = sl.ISSUE_SLOTS, sl.ISSUE_HOURS, sl.N_ISSUE
    base = sl.build_pv_archive(data, W)
    candidate_hours = (3, 9, 15, 21)
    candidate_slots = tuple(6 * h for h in candidate_hours)
    # 候选点分别位于 0/6、6/12、12/18、18/24 区间，代理取左侧官方版本。
    source = (0, 1, 2, 3)
    proxy = np.stack([base[:, r, :] for r in source], axis=1)
    prof = np.concatenate([base, proxy], axis=1)
    sl.ISSUE_SLOTS = old_slots + candidate_slots
    sl.ISSUE_HOURS = old_hours + candidate_hours
    sl.N_ISSUE = old_n + 4
    return prof, old_slots, old_hours, old_n


def restore(state):
    sl.ISSUE_SLOTS, sl.ISSUE_HOURS, sl.N_ISSUE = state


def run_case(data, prof, allowed, updates, n_days):
    run = sl.run_year_q34(data, prof, "F1", 0.7725, 35, "stratified", 6,
                          allowed, updates, "archive", "fixed", "median30",
                          "fixed", 5.0, n_days)
    return run, sl.year_metrics_q34(run, 31)


def load_existing_baseline(n_days):
    """复用问题三已有主配置，避免重复跑 {0,6,12,18}。"""
    path = OUT / "q3_bundle.npz"
    if not path.exists():
        raise FileNotFoundError("缺少已有问题三结果: %s" % path)
    with np.load(path) as z:
        cash = np.asarray(z["main_cash"][:n_days], dtype=float)
        emergency = np.asarray(z["main_U"][:n_days], dtype=float)
    if n_days <= 31:
        raise ValueError("--days 必须大于 31")
    return {
        "daily_cash": cash[31:],
        "cash_total": float(cash[31:].sum()),
        "emergency_kwh": float(emergency[31:].sum()),
    }


def candidate_worker(job):
    """Windows spawn 安全的候选情景工作进程。"""
    hour, r, n_days = job
    data = mc.load_inputs()
    prof, *state = make_proxy_profiles(data, 35)
    try:
        _, metrics = run_case(
            data, prof, (0, 1, 2, 3, r), (1, 2, 3, r), n_days
        )
        return hour, metrics
    finally:
        restore(state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    base_m = load_existing_baseline(args.days)
    jobs = [(hour, r, args.days)
            for hour, r in zip((3, 9, 15, 21), (4, 5, 6, 7))]
    workers = max(1, min(args.workers, len(jobs)))
    print("[Q3-extra] C++ HiGHS scenarios=%d workers=%d days=%d"
          % (len(jobs), workers, args.days), flush=True)
    if workers == 1:
        results = [candidate_worker(job) for job in jobs]
    else:
        with Pool(workers) as pool:
            results = pool.map(candidate_worker, jobs)

    rows = []
    for hour, m in results:
            daily_base = np.asarray(base_m["daily_cash"])
            daily_new = np.asarray(m["daily_cash"])
            saving = daily_base - daily_new
            rows.append({
                "候选时刻_h": hour,
                "对照方案": "{0,6,12,18}",
                "增强方案": "{0,6,12,18,%d}" % hour,
                "代理来源": "此前最近官方版本",
                "delta_n": 1,
                "现金总费_对照_元": base_m["cash_total"],
                "现金总费_增强_元": m["cash_total"],
                "全年节省_元": float(base_m["cash_total"] - m["cash_total"]),
                # 全年新增事件数为验证日数；等价地，日均口径下 delta_n=1。
                "kappa_star_元每次": float(saving.mean()),
                "日均节省_元": float(saving.mean()),
                "节省为正天数": int((saving > 1e-8).sum()),
                "验证天数": int(saving.size),
                "多数验证日节省为正": bool((saving > 1e-8).sum() > saving.size / 2),
                "应急购电_对照_kWh": base_m["emergency_kwh"],
                "应急购电_增强_kWh": m["emergency_kwh"],
                "调整事件数_增强": m["updates"],
                "结论": (
                    "数值正收益，但夜间无光伏，不建议实际引入" if hour == 21 else
                    "代理情景下可谨慎考虑" if (
                        base_m["cash_total"] > m["cash_total"] and
                        (saving > 1e-8).sum() > saving.size / 2
                    ) else "代理情景下不建议引入"
                ),
            })
    df = pd.DataFrame(rows).sort_values("候选时刻_h")
    df.to_csv(OUT / "问题3_其他时刻单次边际测试.csv", index=False, encoding="utf-8-sig")
    meta = {
        "status": "proxy_scenario_only",
        "engine": "SciPy linprog + compiled C++ HiGHS, scenarios parallelized",
        "workers": workers,
        "candidate_hours": [3, 9, 15, 21],
        "proxy_rule": "each candidate uses the corrected profile of the previous official issue time",
        "base_metrics": {k: v for k, v in base_m.items() if k != "daily_cash"},
        "results": rows,
    }
    with open(OUT / "问题3_其他时刻单次边际测试.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=float)
    print(df.to_string(index=False))
    print("output:", OUT / "问题3_其他时刻单次边际测试.csv")


if __name__ == "__main__":
    main()
