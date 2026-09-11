"""Audit the frozen Q3/Q4 risk-contract runs without recomputing the year."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "results" / "q34_risk_contract_refactor"
BASE = ROOT / "results" / "q34_strict_baselines"
OUTPUT_START = pd.Timestamp("2025-02-01")
OUTPUT_END = pd.Timestamp("2025-12-31")


def load_metrics(path: Path) -> dict:
    return json.loads((path / "核心指标.json").read_text(encoding="utf-8"))


def audit_daily(path: Path, variant: str) -> dict:
    frame = pd.read_csv(path / f"{variant}_每日汇总.csv")
    dates = pd.to_datetime(frame["date"])
    continuity = np.abs(
        frame["soc_start_kwh"].to_numpy()[1:]
        - frame["soc_end_kwh"].to_numpy()[:-1]
    )
    return {
        "rows": int(len(frame)),
        "start": dates.min().date().isoformat(),
        "end": dates.max().date().isoformat(),
        "dates_unique": bool(dates.is_unique),
        "dates_complete": bool(
            len(frame) == 334
            and dates.min() == OUTPUT_START
            and dates.max() == OUTPUT_END
            and dates.diff().dropna().eq(pd.Timedelta(days=1)).all()
        ),
        "soc_min_kwh": float(frame["soc_min_kwh"].min()),
        "soc_max_kwh": float(frame["soc_max_kwh"].max()),
        "soc_cross_day_breaks": int(np.count_nonzero(continuity > 1e-7)),
        "max_cross_day_soc_gap_kwh": float(continuity.max(initial=0.0)),
        "max_balance_residual_kwh": float(frame["max_balance_residual_kwh"].max()),
        "cash_cost_yuan": float(frame["cash_cost_yuan"].sum()),
        "emergency_kwh": float(frame["emergency_kwh"].sum()),
        "unused_contract_kwh": float(frame["unused_contract_kwh"].sum()),
        "pv_curtailed_kwh": float(frame["pv_curtailed_kwh"].sum()),
    }


def main() -> int:
    main_metrics = load_metrics(MAIN)
    base_metrics = load_metrics(BASE)
    sources = {
        "q3": MAIN,
        "q4_2": MAIN,
        "q4_3": MAIN,
        "q3_no_adjust": BASE,
        "q4_3_no_adjust": BASE,
    }
    daily = {name: audit_daily(path, name) for name, path in sources.items()}
    all_metrics = {
        **main_metrics["annual_feb_dec"],
        **base_metrics["annual_feb_dec"],
    }
    for name, row in daily.items():
        frozen = all_metrics[name]
        for key in (
            "cash_cost_yuan", "emergency_kwh", "unused_contract_kwh",
            "pv_curtailed_kwh", "max_balance_residual_kwh",
        ):
            if not np.isclose(row[key], frozen[key], atol=1e-6, rtol=1e-12):
                raise AssertionError(f"{name}.{key} does not match 核心指标.json")
        if not row["dates_complete"] or not row["dates_unique"]:
            raise AssertionError(f"{name} annual dates are incomplete")
        if row["soc_min_kwh"] < 1200 - 1e-7 or row["soc_max_kwh"] > 10800 + 1e-7:
            raise AssertionError(f"{name} SOC is out of bounds")
        if row["soc_cross_day_breaks"] or row["max_balance_residual_kwh"] > 1e-6:
            raise AssertionError(f"{name} violates continuity or bus balance")

    pure = {
        "fixed_price": {
            "baseline": "q3_no_adjust",
            "adjusted": "q3",
        },
        "variable_price": {
            "baseline": "q4_3_no_adjust",
            "adjusted": "q4_3",
        },
    }
    for row in pure.values():
        base = all_metrics[row["baseline"]]
        adjusted = all_metrics[row["adjusted"]]
        row.update({
            "cash_saving_yuan": base["cash_cost_yuan"] - adjusted["cash_cost_yuan"],
            "cash_saving_percent": 100.0 * (
                base["cash_cost_yuan"] - adjusted["cash_cost_yuan"]
            ) / base["cash_cost_yuan"],
            "emergency_reduction_kwh": base["emergency_kwh"] - adjusted["emergency_kwh"],
            "unused_contract_reduction_kwh": base["unused_contract_kwh"] - adjusted["unused_contract_kwh"],
            "pv_curtailment_reduction_kwh": base["pv_curtailed_kwh"] - adjusted["pv_curtailed_kwh"],
            "information_contract": (
                "same 00:00 official forecast archive, F1 load forecast, 30 completed-day "
                "residual window, alpha=0.8 risk LP, initial SOC and greedy executor; "
                "only 06:00/12:00/18:00 contract adjustment permission differs"
            ),
        })

    report = {
        "status": "PASS",
        "daily_audit": daily,
        "pure_adjustment_comparisons": pure,
        "scope_note": (
            "q2-vs-q3 and q4_2-vs-q4_3 remain complete-strategy comparisons because "
            "their 00:00 forecast sources differ; only the pairs above identify the "
            "effect of intraday adjustment permission."
        ),
    }
    output = BASE / "年度风险合同审计.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
