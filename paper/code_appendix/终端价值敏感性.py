
from __future__ import annotations

import json
import time

import pandas as pd

import microgrid_core as mc
import solution_lib as sl


def main() -> None:
    t0 = time.perf_counter()
    data = mc.load_inputs()
    resid = sl.net_residual_matrix("F1", data)
    rows = []
    for factor in (0.0, 0.5, 1.0, 1.5):
        run = sl.run_year_q2(data, "F1", 0.805, 35, resid=resid,
                             horizon=49, n_days=365, executor="mpc",
                             terminal_value_factor=factor)
        m = sl.year_metrics(run)
        g = sl.check_run(run)
        rows.append({
            "终端价值倍率": factor,
            "现金费用_元": m["cash_total"],
            "合同购电量_kWh": m["contract_kwh"],
            "应急购电量_kWh": m["emergency_kwh"],
            "充电量_kWh": m["charge_kwh"],
            "放电量_kWh": m["discharge_kwh"],
            "弃购量_kWh": m["unused_contract_kwh"],
            "期末SOC_kWh": m["S_end"],
            "SOC下限触及比例": float((run["S"] <= mc.S_MIN + 1e-7).mean()),
            "SOC上限触及比例": float((run["S"] >= mc.S_MAX - 1e-7).mean()),
            "数值门禁": bool(g["balance_max"] <= 1e-6 and
                            g["overlap_max"] <= 1e-5 and
                            g["bound_S"] <= 1e-7 and
                            g["soc_chain_max_kWh"] <= 1e-6),
        })
    df = pd.DataFrame(rows)
    base = float(df.loc[df["终端价值倍率"] == 1.0, "现金费用_元"].iloc[0])
    df["相对倍率1费用变化_元"] = df["现金费用_元"] - base
    df["相对倍率1费用变化_pct"] = (df["现金费用_元"] / base - 1.0) * 100.0
    sl.CSV.mkdir(parents=True, exist_ok=True)
    df.to_csv(sl.CSV / "终端价值敏感性.csv", index=False, encoding="utf-8-sig")
    report = {"scenario_count": len(df), "all_numeric_ok": bool(df["数值门禁"].all()),
              "elapsed_s": round(time.perf_counter() - t0, 1)}
    with open(sl.CSV / "终端价值敏感性_门禁.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
