"""P1 vertical-slice test on real competition inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from microgrid_core import (
    SOC_INITIAL,
    build_analog_forecasts,
    load_inputs,
    official_pv_residuals,
    run_q1,
    run_q2_day,
    run_q3_day,
    summarize_day,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--date", default="2025-03-20")
    parser.add_argument("--output", default="results/p1_smoke.json")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    data = load_inputs(root)
    day = int(np.flatnonzero(data.dates == np.datetime64(args.date))[0])
    load_point, load_residual = build_analog_forecasts(
        data.actual_load_kw, data.typical_load_kw, weekday_weight=True
    )
    pv_point, pv_residual = build_analog_forecasts(
        data.actual_pv_kw, data.typical_pv_kw, weekday_weight=False
    )
    issued_residual = official_pv_residuals(data)

    q1 = run_q1(data)
    q2 = run_q2_day(
        data,
        day,
        SOC_INITIAL,
        load_point,
        pv_point,
        load_residual,
        pv_residual,
        variable_price=False,
    )
    q3 = run_q3_day(
        data,
        day,
        SOC_INITIAL,
        load_point,
        load_residual,
        issued_residual,
        variable_price=False,
    )
    q4_2 = run_q2_day(
        data,
        day,
        SOC_INITIAL,
        load_point,
        pv_point,
        load_residual,
        pv_residual,
        variable_price=True,
    )
    q4_3 = run_q3_day(
        data,
        day,
        SOC_INITIAL,
        load_point,
        load_residual,
        issued_residual,
        variable_price=True,
    )

    report = {
        "command": f"python smoke_test.py --project-root {args.project_root} --date {args.date} --output {args.output}",
        "date": args.date,
        "time_contract": "row 1 is natural interval 00:00-00:10; official result header is retained verbatim",
        "units": {"power": "kW", "interval_energy": "kWh", "soc": "kWh", "price": "yuan/kWh"},
        "q1": {
            "status": q1["status"],
            "purchase_kwh": float(np.sum(q1["grid_kwh"])),
            "cash_cost_yuan": q1["cash_contract_cost"],
            "soc_start_kwh": float(q1["soc_kwh"][0]),
            "soc_end_kwh": float(q1["soc_kwh"][-1]),
            "soc_min_kwh": float(np.min(q1["soc_kwh"])),
            "soc_max_kwh": float(np.max(q1["soc_kwh"])),
            "max_eq_residual_kwh": q1["max_eq_residual"],
            "simultaneous_flow_max_kwh": q1["simultaneous_flow_max"],
        },
        "q2": summarize_day(q2, adjusted=False),
        "q3": summarize_day(q3, adjusted=True),
        "q4_2": summarize_day(q4_2, adjusted=False),
        "q4_3": summarize_day(q4_3, adjusted=True),
        "causality": {
            "q2_history_end_exclusive": q2["forecast_meta"]["history_end_exclusive"],
            "decision_day_index": day,
            "q3_updates": [{k: v for k, v in m.items() if k not in ("pv_point_kw", "bias_kw")} for m in q3["update_meta"]],
        },
    }
    for key in ("q1", "q2", "q3", "q4_2", "q4_3"):
        values = report[key]
        if values.get("max_balance_residual_kwh", values.get("max_eq_residual_kwh", 0.0)) > 1e-6:
            raise AssertionError(f"{key}平衡残差超阈值")
        if values["soc_min_kwh"] < 1_200 - 1e-7 or values["soc_max_kwh"] > 10_800 + 1e-7:
            raise AssertionError(f"{key} SOC越界")

    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
