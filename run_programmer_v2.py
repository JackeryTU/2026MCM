"""Versioned programmer-stage experiments for the revised causal model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import microgrid_core as m
from evaluate_results import validate_day


DEVELOPMENT_START = pd.Timestamp("2025-02-01")
SELECTION_START = pd.Timestamp("2025-08-01")
FREEZE_DATE = pd.Timestamp("2025-09-30")
TEST_START = pd.Timestamp("2025-10-01")
REPRESENTATIVE_DATES = {"2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"}


def prediction_table(data: m.Inputs, archive: dict[str, dict]) -> pd.DataFrame:
    rows = []
    start = int(np.flatnonzero(data.dates == DEVELOPMENT_START)[0])
    for name, values in archive.items():
        load_error = data.actual_load_kw[start:] - values["load_kw"][start:]
        pv_error = data.actual_pv_kw[start:] - values["pv_kw"][start:]
        net_error = load_error - pv_error
        for scope, error in (("load", load_error), ("pv", pv_error), ("net_load", net_error)):
            rows.append({
                "forecast_model": name, "scope": scope,
                "mae_kw": float(np.mean(np.abs(error))),
                "rmse_kw": float(np.sqrt(np.mean(error ** 2))),
                "bias_kw": float(np.mean(error)),
                "evaluation_start": str(DEVELOPMENT_START.date()),
                "evaluation_end": "2025-12-31",
                "description": values["description"],
            })
    return pd.DataFrame(rows)


def price_prediction_table(data: m.Inputs) -> pd.DataFrame:
    errors: dict[str, list[np.ndarray]] = {
        key: [] for key in ("yesterday", "mean_7", "mean_30", "median_30")
    }
    for day in range(31, len(data.dates)):
        candidates = m.price_forecast_candidates(data, day)
        for name, prediction in candidates.items():
            errors[name].append(data.variable_price[day] - prediction)
    return pd.DataFrame([
        {
            "price_model": name, "mae_yuan_per_kwh": float(np.mean(np.abs(np.asarray(values)))),
            "rmse_yuan_per_kwh": float(np.sqrt(np.mean(np.asarray(values) ** 2))),
            "evaluation_start": str(DEVELOPMENT_START.date()), "evaluation_end": "2025-12-31",
        }
        for name, values in errors.items()
    ])


def quantile_pinball(error: np.ndarray, correction: np.ndarray, alpha: float) -> float:
    miss = error - correction
    return float(np.mean(np.maximum(alpha * miss, (alpha - 1.0) * miss)))


def simulate_q2_case(
    data: m.Inputs, forecast: dict, *, name: str, history_days: int, alpha: float,
    horizon: int, tracking_weight: float, controller: str,
    output: Path, end_day: int = 365,
) -> dict:
    started = time.perf_counter()
    lp, lr = forecast["load_kw"], forecast["load_residual_kw"]
    vp, vr = forecast["pv_kw"], forecast["pv_residual_kw"]
    soc = m.SOC_INITIAL
    rows = []
    intervals = []
    solve_times = []
    fallback_count = 0
    clip_count = 0
    for day in range(end_day):
        safe = m.safe_analog_trajectory(
            day, lp[day], vp[day], lr, vr, alpha=alpha,
            history_days=history_days, risk_mode="unclustered",
        )
        result = m.run_q2_day(
            data, day, soc, lp, vp, lr, vr, variable_price=False, alpha=alpha,
            safe_pair=safe, history_days=history_days, controller=controller,
            mpc_config=m.MpcConfig(horizon_slots=horizon, tracking_weight=tracking_weight),
        )
        result["date"] = data.dates[day]
        validate_day(result, False, soc)
        actual = result["actual"]
        diagnostics = actual.get("solver_diagnostics")
        if diagnostics is not None:
            if diagnostics["median_seconds"] is not None:
                solve_times.append(diagnostics["median_seconds"])
            fallback_count += diagnostics["fallback_count"]
            clip_count += diagnostics["clip_count"]
        correction = np.asarray(safe[2]["net_residual_quantile_kw"])
        net_error = lr[day] - vr[day]
        summary = m.summarize_day(result, adjusted=False)
        rows.append({
            "date": data.dates[day].date().isoformat(), **summary,
            "net_mae_kw": float(np.mean(np.abs(net_error))),
            "pinball_loss_kw": quantile_pinball(net_error, correction, alpha),
            "covered_slot_rate": float(np.mean(net_error <= correction)),
            "daily_cost_var_source": "realized rolling-origin cash cost",
        })
        for slot in range(m.N_SLOTS):
            intervals.append({
                "date": data.dates[day].date().isoformat(), "slot": slot + 1,
                "contract_kwh": float(result["plan"]["grid_kwh"][slot]),
                "planned_charge_kwh": float(actual["planned_charge_kwh"][slot]),
                "planned_discharge_kwh": float(actual["planned_discharge_kwh"][slot]),
                "executed_charge_kwh": float(actual["charge_kwh"][slot]),
                "executed_discharge_kwh": float(actual["discharge_kwh"][slot]),
                "emergency_kwh": float(actual["emergency_kwh"][slot]),
                "soc_end_kwh": float(actual["soc_kwh"][slot + 1]),
                "actual_load_kw": float(data.actual_load_kw[day, slot]),
                "actual_pv_kw": float(data.actual_pv_kw[day, slot]),
                "point_net_forecast_kw": float(lp[day, slot] - vp[day, slot]),
                "safe_net_forecast_kw": float(safe[0][slot] - safe[1][slot]),
            })
        soc = float(actual["soc_kwh"][-1])
        if day % 30 == 0 or day == end_day - 1:
            print(f"[{name}] {day + 1}/{end_day} SOC={soc:.2f}", flush=True)
    daily = pd.DataFrame(rows)
    detail = pd.DataFrame(intervals)
    daily.to_csv(output / f"{name}_daily.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(output / f"{name}_interval.csv", index=False, encoding="utf-8-sig")
    evaluated = daily[daily.date >= "2025-02-01"]
    costs = evaluated.cash_cost_yuan.to_numpy()
    var90 = float(np.quantile(costs, .9, method="inverted_cdf"))
    cvar90 = float(costs[costs >= var90].mean())
    by_period = {}
    for period, begin, end in (
        ("development", "2025-02-01", "2025-07-31"),
        ("selection", "2025-08-01", "2025-09-30"),
        ("frozen_test", "2025-10-01", "2025-12-31"),
    ):
        part = daily[(daily.date >= begin) & (daily.date <= end)]
        by_period[period] = {
            "days": int(len(part)), "cash_cost_yuan": float(part.cash_cost_yuan.sum()),
            "emergency_kwh": float(part.emergency_kwh.sum()),
            "unused_contract_kwh": float(part.unused_contract_kwh.sum()),
            "net_mae_kw": float(part.net_mae_kw.mean()),
            "pinball_loss_kw": float(part.pinball_loss_kw.mean()),
        }
    report = {
        "name": name, "forecast_model": forecast["description"],
        "parameters": {"history_days": history_days, "alpha": alpha, "controller": controller, "horizon_slots": horizon, "tracking_weight": tracking_weight},
        "evaluated_days": int(len(evaluated)), "periods": by_period,
        "totals_feb_dec": {
            "cash_cost_yuan": float(evaluated.cash_cost_yuan.sum()),
            "emergency_kwh": float(evaluated.emergency_kwh.sum()),
            "unused_contract_kwh": float(evaluated.unused_contract_kwh.sum()),
            "pv_curtailed_kwh": float(evaluated.pv_curtailed_kwh.sum()),
            "var90_daily_cost_yuan": var90, "cvar90_daily_cost_yuan": cvar90,
        },
        "solver": {
            "fallback_count": fallback_count, "clip_count": clip_count,
            "median_day_median_seconds": float(np.median(solve_times)) if solve_times else None,
            "p95_day_median_seconds": float(np.quantile(solve_times, .95)) if solve_times else None,
        },
        "representative_days": daily[daily.date.isin(REPRESENTATIVE_DATES)].to_dict("records"),
        "runtime_seconds": time.perf_counter() - started,
        "hash_checks": "omitted; absolute path, size and mtime are recorded separately",
    }
    (output / f"{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def path_stress_table(data: m.Inputs, forecast: dict, history_days: int = 30) -> pd.DataFrame:
    """Replay complete residual paths; CVaR is an evaluation statistic only."""
    residual = forecast["net_residual_kw"]
    rows = []
    for day in range(31, len(data.dates)):
        history = residual[max(0, day-history_days):day]
        predicted_net = forecast["net_kw"][day]
        costs = []
        for path in history:
            scenario_net = predicted_net + path
            deficit = np.maximum(scenario_net, 0.0) * m.DT_HOURS
            costs.append(float(np.dot(data.typical_price, deficit)))
        values = np.asarray(costs)
        threshold = float(np.quantile(values, .9, method="inverted_cdf"))
        rows.append({
            "date": data.dates[day].date().isoformat(), "path_count": len(values),
            "mean_proxy_cost_yuan": float(values.mean()), "var90_proxy_cost_yuan": threshold,
            "cvar90_proxy_cost_yuan": float(values[values >= threshold].mean()),
            "role": "complete historical path pressure replay; not an optimization objective",
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default="results")
    parser.add_argument("--case", default="F1_W30_A090_H36_M000")
    parser.add_argument("--forecast-model", choices=("F0", "F1", "F2"), default="F1")
    parser.add_argument("--history-days", type=int, choices=(21, 30, 45, 60), default=30)
    parser.add_argument("--alpha", type=float, choices=(.8, .9, .95), default=.9)
    parser.add_argument("--horizon", type=int, choices=(36, 72), default=36)
    parser.add_argument("--tracking-weight", type=float, choices=(0., .01, .05), default=0.)
    parser.add_argument("--controller", choices=("mpc", "delayed_greedy", "greedy"), default="mpc")
    parser.add_argument("--end-day", type=int, default=365)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = m.load_inputs(root)
    archive = m.build_forecast_archive(data)
    prediction_table(data, archive).to_csv(output / "forecast_comparison.csv", index=False, encoding="utf-8-sig")
    price_prediction_table(data).to_csv(output / "price_forecast_comparison.csv", index=False, encoding="utf-8-sig")
    path_stress_table(data, archive[args.forecast_model], args.history_days).to_csv(output / "path_stress_cvar.csv", index=False, encoding="utf-8-sig")
    report = simulate_q2_case(
        data, archive[args.forecast_model], name=args.case, history_days=args.history_days,
        alpha=args.alpha, horizon=args.horizon, tracking_weight=args.tracking_weight,
        controller=args.controller, output=output, end_day=args.end_day,
    )
    files = []
    for path in sorted(output.glob("*")):
        stat = path.stat()
        files.append({"path": str(path.resolve()), "size_bytes": stat.st_size, "modified_time_ns": stat.st_mtime_ns})
    (output / "manifest_nohash.json").write_text(json.dumps({"files": files, "hashes": "not calculated"}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
