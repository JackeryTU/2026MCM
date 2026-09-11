"""National-award-oriented ablations and diagnostic evidence.

Run only after the M1 and P1 gates pass.  January is a causal warm-up and
February--December is the fixed 334-day evaluation period.  The script does
not calculate or compare cryptographic hashes.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
import math
import platform
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn

import microgrid_core as m
from evaluate_results import validate_day


EVALUATION_START = pd.Timestamp("2025-02-01")
ALPHA = 0.80
HISTORY_DAYS = 60
MAX_SCENARIOS = 9
RELEASE_HOURS = (6, 12, 18)
RELEASE_SETS = tuple(
    subset
    for size in range(4)
    for subset in combinations(RELEASE_HOURS, size)
)
NUMERIC_TOLERANCES = {
    "coverage": 1e-12,
    "pinball_kw": 1e-9,
    "cash_cost_yuan": 1e-4,
    "emergency_kwh": 1e-6,
    "day_cost_yuan": 1e-6,
}


def file_metadata(path: Path) -> dict[str, Any]:
    """Identify a file without calculating or comparing a hash."""
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "modified_time_ns": int(stat.st_mtime_ns),
    }


def release_name(hours: tuple[int, ...]) -> str:
    return "S_none" if not hours else "S_" + "_".join(map(str, hours))


def pinball_components(actual_kw: np.ndarray, safe_kw: np.ndarray) -> tuple[int, float]:
    error = np.asarray(actual_kw, dtype=float) - np.asarray(safe_kw, dtype=float)
    loss = np.maximum(ALPHA * error, -(1.0 - ALPHA) * error)
    return int(np.count_nonzero(error <= 1e-10)), float(np.sum(loss))


def day_evidence(
    data: m.Inputs,
    day: int,
    result: dict[str, Any],
    *,
    variant: str,
    adjusted: bool,
    variable_price: bool,
    risk_mode: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    actual = result["actual"]
    initial = result["initial_plan_kwh"] if adjusted else result["plan"]["grid_kwh"]
    contract = result["final_contract_kwh"] if adjusted else initial
    safe_net = result["safe_net_kw"] if adjusted else result["safe_load_kw"] - result["safe_pv_kw"]
    actual_net = data.actual_load_kw[day] - data.actual_pv_kw[day]
    covered_slots, pinball_sum = pinball_components(actual_net, safe_net)
    effective_grid = contract - actual["unused_contract_kwh"] + actual["emergency_kwh"]
    summary = m.summarize_day(result, adjusted=adjusted)
    emergency_mask = actual["emergency_kwh"] > 1e-8
    near_min = actual["soc_kwh"][1:] <= m.DEFAULT_BATTERY.minimum + 1e-7
    source_hours = (
        ",".join(str(item["source_issue_hour"]) for item in result["update_meta"])
        if adjusted
        else ""
    )
    row = {
        "variant": variant,
        "date": data.dates[day].date().isoformat(),
        "month": int(data.dates[day].month),
        "adjusted": adjusted,
        "variable_price": variable_price,
        "risk_mode": risk_mode,
        **summary,
        "contract_cost_yuan": float(np.sum(actual["settlement_cost"])),
        "emergency_cost_yuan": float(np.sum(actual["emergency_cost"])),
        "actual_load_kwh": float(np.sum(data.actual_load_kw[day]) * m.DT_HOURS),
        "actual_pv_kwh": float(np.sum(data.actual_pv_kw[day]) * m.DT_HOURS),
        "pv_used_kwh": float(np.sum(actual["pv_used_kwh"])),
        "actual_effective_grid_kwh": float(np.sum(effective_grid)),
        "emergency_slots": int(np.count_nonzero(emergency_mask)),
        "emergency_day": int(np.any(emergency_mask)),
        "covered_slots": covered_slots,
        "evaluated_slots": m.N_SLOTS,
        "safe_net_coverage": covered_slots / m.N_SLOTS,
        "pinball_sum_kw": pinball_sum,
        "pinball_mean_kw": pinball_sum / m.N_SLOTS,
        "source_issue_hours": source_hours,
    }

    target_soc = result["target_soc_kwh"] if adjusted else result["plan"]["soc_kwh"]
    soc_error = np.asarray(actual["soc_kwh"]) - np.asarray(target_soc)
    initial_error = effective_grid - initial
    final_error = effective_grid - contract
    plan_solutions = result["update_solutions"] if adjusted else [result["plan"]]
    planned_overlap_slots = [int(solution["simultaneous_flow_slots"]) for solution in plan_solutions]
    planned_overlap_max = [float(solution["simultaneous_flow_max"]) for solution in plan_solutions]
    execution_overlap = np.minimum(actual["charge_kwh"], actual["discharge_kwh"])
    diagnosis = {
        "variant": variant,
        "date": row["date"],
        "month": row["month"],
        "risk_mode": risk_mode,
        "lp_solution_count": len(plan_solutions),
        "planning_simultaneous_flow_slots_total": int(sum(planned_overlap_slots)),
        "planning_simultaneous_flow_slots_max_per_lp": int(max(planned_overlap_slots)),
        "planning_simultaneous_flow_max_kwh": float(max(planned_overlap_max)),
        "execution_simultaneous_flow_slots": int(np.count_nonzero(execution_overlap > 1e-5)),
        "execution_simultaneous_flow_max_kwh": float(np.max(execution_overlap)),
        "target_soc_mae_kwh": float(np.mean(np.abs(soc_error))),
        "target_soc_rmse_kwh": float(np.sqrt(np.mean(soc_error**2))),
        "target_soc_max_abs_kwh": float(np.max(np.abs(soc_error))),
        "initial_plan_to_effective_grid_mae_kwh_per_slot": float(np.mean(np.abs(initial_error))),
        "initial_plan_to_effective_grid_rmse_kwh_per_slot": float(np.sqrt(np.mean(initial_error**2))),
        "initial_plan_to_effective_grid_sum_kwh": float(np.sum(effective_grid - initial)),
        "final_contract_to_effective_grid_mae_kwh_per_slot": float(np.mean(np.abs(final_error))),
        "final_contract_to_effective_grid_rmse_kwh_per_slot": float(np.sqrt(np.mean(final_error**2))),
        "final_contract_to_effective_grid_sum_kwh": float(np.sum(effective_grid - contract)),
        "emergency_slots": int(np.count_nonzero(emergency_mask)),
        "emergency_soc_end_at_min_rate": (
            float(np.mean(near_min[emergency_mask])) if np.any(emergency_mask) else np.nan
        ),
        "max_balance_residual_kwh": float(actual["max_balance_residual"]),
    }
    events = result["events"] if adjusted else (0,)
    lp_rows = []
    for event, solution in zip(events, plan_solutions):
        lp_rows.append(
            {
                "variant": variant,
                "date": row["date"],
                "decision_hour": int(event),
                "risk_mode": risk_mode,
                "horizon_slots": int(len(solution["grid_kwh"])),
                "simultaneous_flow_slots": int(solution["simultaneous_flow_slots"]),
                "simultaneous_flow_max_kwh": float(solution["simultaneous_flow_max"]),
                "max_eq_residual_kwh": float(solution["max_eq_residual"]),
                "solver_status": solution["status"],
            }
        )
    return row, diagnosis, lp_rows


def simulate(
    data: m.Inputs,
    forecasts: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    variant: str,
    adjusted: bool,
    variable_price: bool,
    risk_mode: str,
    trajectories: list[Any],
    allowed_issue_hours: tuple[int, ...] = RELEASE_HOURS,
    capture_diagnostics: bool = True,
) -> dict[str, Any]:
    load_point, load_residual, pv_point, pv_residual, issued_residual = forecasts
    soc = m.SOC_INITIAL
    warmup_end_soc = np.nan
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    lp_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for day, date in enumerate(data.dates):
        if adjusted:
            result = m.run_q3_day(
                data,
                day,
                soc,
                load_point,
                load_residual,
                issued_residual,
                variable_price=variable_price,
                alpha=ALPHA,
                safe_updates=trajectories[day],
                allowed_issue_hours=allowed_issue_hours,
                risk_mode=risk_mode,
            )
        else:
            result = m.run_q2_day(
                data,
                day,
                soc,
                load_point,
                pv_point,
                load_residual,
                pv_residual,
                variable_price=variable_price,
                alpha=ALPHA,
                safe_pair=trajectories[day],
                risk_mode=risk_mode,
            )
        validate_day(result, adjusted, soc)
        soc = float(result["actual"]["soc_kwh"][-1])
        if day == 30:
            warmup_end_soc = soc
        if date >= EVALUATION_START:
            row, diagnosis, day_lp_rows = day_evidence(
                data,
                day,
                result,
                variant=variant,
                adjusted=adjusted,
                variable_price=variable_price,
                risk_mode=risk_mode,
            )
            rows.append(row)
            if capture_diagnostics:
                diagnostics.append(diagnosis)
                lp_rows.extend(day_lp_rows)
        if day % 45 == 0 or day == len(data.dates) - 1:
            print(f"[{variant}] {day + 1}/{len(data.dates)}, SOC={soc:.2f}", flush=True)
    frame = pd.DataFrame(rows)
    if len(frame) != 334:
        raise AssertionError(f"{variant} formal evaluation must contain 334 days, got {len(frame)}")
    return {
        "daily": frame,
        "diagnostics": pd.DataFrame(diagnostics),
        "lp_diagnostics": pd.DataFrame(lp_rows),
        "warmup_end_soc_kwh": float(warmup_end_soc),
        "final_soc_kwh": float(soc),
        "runtime_seconds": time.perf_counter() - started,
    }


def precompute_trajectories(
    data: m.Inputs,
    forecasts: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    load_point, load_residual, pv_point, pv_residual, issued_residual = forecasts
    q2_cache: dict[str, list[Any]] = {mode: [] for mode in ("clustered", "unclustered")}
    q3_cache: dict[str, list[Any]] = {mode: [] for mode in ("clustered", "unclustered")}
    for mode in q2_cache:
        for day in range(len(data.dates)):
            q2_cache[mode].append(
                m.safe_analog_trajectory(
                    day,
                    load_point[day],
                    pv_point[day],
                    load_residual,
                    pv_residual,
                    alpha=ALPHA,
                    history_days=HISTORY_DAYS,
                    max_scenarios=MAX_SCENARIOS,
                    risk_mode=mode,
                )
            )
            q3_cache[mode].append(
                [
                    m.safe_update_trajectory(
                        data,
                        day,
                        issue_index,
                        load_point[day],
                        load_residual,
                        issued_residual,
                        alpha=ALPHA,
                        history_days=HISTORY_DAYS,
                        max_scenarios=MAX_SCENARIOS,
                        risk_mode=mode,
                    )
                    for issue_index in range(len(m.ISSUES))
                ]
            )
            if day % 60 == 0 or day == len(data.dates) - 1:
                print(f"[safe-cache:{mode}] {day + 1}/{len(data.dates)}", flush=True)
    return q2_cache, q3_cache


def summarize_frame(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "days": int(len(frame)),
        "cash_cost_yuan": float(frame["cash_cost_yuan"].sum()),
        "contract_cost_yuan": float(frame["contract_cost_yuan"].sum()),
        "emergency_cost_yuan": float(frame["emergency_cost_yuan"].sum()),
        "plan_kwh": float(frame["plan_kwh"].sum()),
        "final_contract_kwh": float(frame["final_contract_kwh"].sum()),
        "actual_effective_grid_kwh": float(frame["actual_effective_grid_kwh"].sum()),
        "emergency_kwh": float(frame["emergency_kwh"].sum()),
        "emergency_days": int(frame["emergency_day"].sum()),
        "emergency_slots": int(frame["emergency_slots"].sum()),
        "unused_contract_kwh": float(frame["unused_contract_kwh"].sum()),
        "pv_curtailed_kwh": float(frame["pv_curtailed_kwh"].sum()),
        "charge_kwh": float(frame["charge_kwh"].sum()),
        "discharge_kwh": float(frame["discharge_kwh"].sum()),
        "safe_net_coverage": float(frame["covered_slots"].sum() / frame["evaluated_slots"].sum()),
        "pinball_mean_kw": float(frame["pinball_sum_kw"].sum() / frame["evaluated_slots"].sum()),
        "soc_end_kwh": float(frame.iloc[-1]["soc_end_kwh"]),
        "max_balance_residual_kwh": float(frame["max_balance_residual_kwh"].max()),
    }


def risk_decision(summary: pd.DataFrame) -> dict[str, Any]:
    tests = []
    for strategy in ("q2_fixed", "q3_fixed"):
        rows = summary[summary["strategy"] == strategy].set_index("risk_mode")
        clustered = rows.loc["clustered"]
        raw = rows.loc["unclustered"]
        checks = {
            "coverage_not_lower": bool(
                raw.safe_net_coverage + NUMERIC_TOLERANCES["coverage"] >= clustered.safe_net_coverage
            ),
            "pinball_not_higher": bool(
                raw.pinball_mean_kw <= clustered.pinball_mean_kw + NUMERIC_TOLERANCES["pinball_kw"]
            ),
            "cost_not_higher": bool(
                raw.cash_cost_yuan <= clustered.cash_cost_yuan + NUMERIC_TOLERANCES["cash_cost_yuan"]
            ),
            "emergency_not_higher": bool(
                raw.emergency_kwh <= clustered.emergency_kwh + NUMERIC_TOLERANCES["emergency_kwh"]
            ),
        }
        tests.append({"strategy": strategy, **checks, "passed": all(checks.values())})
    selected = "unclustered" if all(test["passed"] for test in tests) else "clustered"
    return {
        "selected_risk_mode": selected,
        "rule": (
            "Select unclustered only if, in both Q2 and Q3 fixed-price branches, "
            "coverage is not lower, pinball loss is not higher, cash cost is not "
            "higher, and emergency energy is not higher; otherwise retain clustered."
        ),
        "tests": tests,
        "tolerances": NUMERIC_TOLERANCES,
        "alpha_was_retuned": False,
    }


def monthly_operations(variants: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    sum_columns = (
        "actual_load_kwh",
        "actual_pv_kwh",
        "plan_kwh",
        "final_contract_kwh",
        "actual_effective_grid_kwh",
        "emergency_kwh",
        "unused_contract_kwh",
        "pv_curtailed_kwh",
        "pv_used_kwh",
        "charge_kwh",
        "discharge_kwh",
        "contract_cost_yuan",
        "emergency_cost_yuan",
        "cash_cost_yuan",
    )
    for variant, frame in variants.items():
        for month, group in frame.groupby("month", sort=True):
            row = {"variant": variant, "month": int(month), "days": int(len(group))}
            row.update({column: float(group[column].sum()) for column in sum_columns})
            row.update(
                emergency_days=int(group["emergency_day"].sum()),
                emergency_slots=int(group["emergency_slots"].sum()),
                safe_net_coverage=float(group["covered_slots"].sum() / group["evaluated_slots"].sum()),
                pinball_mean_kw=float(group["pinball_sum_kw"].sum() / group["evaluated_slots"].sum()),
                soc_end_kwh=float(group.iloc[-1]["soc_end_kwh"]),
            )
            rows.append(row)
    return pd.DataFrame(rows)


def low_pv_stratification(variants: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    metrics = (
        "actual_pv_kwh",
        "cash_cost_yuan",
        "contract_cost_yuan",
        "emergency_cost_yuan",
        "emergency_kwh",
        "unused_contract_kwh",
        "pv_curtailed_kwh",
        "actual_effective_grid_kwh",
        "pinball_mean_kw",
    )
    for variant, original in variants.items():
        frame = original.copy()
        for month, month_frame in frame.groupby("month", sort=True):
            threshold = float(np.quantile(month_frame["actual_pv_kwh"], 0.20, method="linear"))
            is_low = month_frame["actual_pv_kwh"] <= threshold
            for label, group in (("low_pv_q20", month_frame[is_low]), ("other", month_frame[~is_low])):
                evaluated_slots = float(group["evaluated_slots"].sum())
                row = {
                    "variant": variant,
                    "month": int(month),
                    "group": label,
                    "q20_threshold_actual_pv_kwh": threshold,
                    "days": int(len(group)),
                    "emergency_day_rate": float(group["emergency_day"].mean()) if len(group) else np.nan,
                    "safe_net_coverage": float(group["covered_slots"].sum() / evaluated_slots) if evaluated_slots else np.nan,
                }
                row.update({f"mean_{metric}": float(group[metric].mean()) for metric in metrics})
                rows.append(row)
    return pd.DataFrame(rows)


def information_outputs(info_frames: dict[tuple[int, ...], pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    baseline = info_frames[()].loc[:, ["date", "cash_cost_yuan", "emergency_kwh"]].rename(
        columns={
            "cash_cost_yuan": "S_none_cash_cost_yuan",
            "emergency_kwh": "S_none_emergency_kwh",
        }
    )
    daily_parts = []
    summary_rows = []
    monthly_rows = []
    totals: dict[tuple[int, ...], float] = {}
    for coalition in RELEASE_SETS:
        frame = info_frames[coalition].copy().merge(baseline, on="date", how="left", validate="one_to_one")
        frame["release_set"] = release_name(coalition)
        frame["allowed_issue_hours"] = ",".join(map(str, coalition)) if coalition else "none"
        frame["saving_vs_S_none_yuan"] = frame["S_none_cash_cost_yuan"] - frame["cash_cost_yuan"]
        frame["emergency_reduction_vs_S_none_kwh"] = frame["S_none_emergency_kwh"] - frame["emergency_kwh"]
        daily_parts.append(frame)
        total = summarize_frame(frame)
        totals[coalition] = total["cash_cost_yuan"]
        difference = frame["saving_vs_S_none_yuan"].to_numpy()
        summary_rows.append(
            {
                "release_set": release_name(coalition),
                "allowed_issue_hours": frame.iloc[0]["allowed_issue_hours"],
                **total,
                "saving_vs_S_none_yuan": float(np.sum(difference)),
                "saving_vs_S_none_percent": float(100 * np.sum(difference) / totals[()] if totals[()] else np.nan),
                "improved_days_vs_S_none": int(np.count_nonzero(difference > NUMERIC_TOLERANCES["day_cost_yuan"])),
                "worsened_days_vs_S_none": int(np.count_nonzero(difference < -NUMERIC_TOLERANCES["day_cost_yuan"])),
                "equal_days_vs_S_none": int(np.count_nonzero(np.abs(difference) <= NUMERIC_TOLERANCES["day_cost_yuan"])),
            }
        )
        for month, group in frame.groupby("month", sort=True):
            month_total = summarize_frame(group)
            monthly_rows.append(
                {
                    "release_set": release_name(coalition),
                    "allowed_issue_hours": frame.iloc[0]["allowed_issue_hours"],
                    "month": int(month),
                    **month_total,
                    "saving_vs_S_none_yuan": float(group["saving_vs_S_none_yuan"].sum()),
                    "emergency_reduction_vs_S_none_kwh": float(group["emergency_reduction_vs_S_none_kwh"].sum()),
                }
            )

    factorial = math.factorial
    marginal_rows = []
    shapley_rows = []
    empty_cost = totals[()]
    values = {coalition: empty_cost - cost for coalition, cost in totals.items()}
    for hour in RELEASE_HOURS:
        shapley = 0.0
        others = tuple(item for item in RELEASE_HOURS if item != hour)
        for size in range(len(others) + 1):
            for base in combinations(others, size):
                base = tuple(sorted(base))
                expanded = tuple(sorted(base + (hour,)))
                marginal = values[expanded] - values[base]
                weight = factorial(len(base)) * factorial(2 - len(base)) / factorial(3)
                shapley += weight * marginal
                marginal_rows.append(
                    {
                        "release_hour": hour,
                        "base_release_set": release_name(base),
                        "expanded_release_set": release_name(expanded),
                        "conditional_marginal_saving_yuan": float(marginal),
                        "shapley_weight": float(weight),
                    }
                )
        shapley_rows.append({"release_hour": hour, "shapley_saving_yuan": float(shapley)})
    return (
        pd.concat(daily_parts, ignore_index=True),
        pd.DataFrame(summary_rows),
        pd.DataFrame(monthly_rows),
        marginal_rows + [{**row, "record_type": "shapley"} for row in shapley_rows],
    )


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="results/national_award")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    output = (root / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    np.random.seed(m.SEED)

    data = m.load_inputs(root)
    load_point, load_residual = m.build_analog_forecasts(
        data.actual_load_kw, data.typical_load_kw, weekday_weight=True
    )
    pv_point, pv_residual = m.build_analog_forecasts(
        data.actual_pv_kw, data.typical_pv_kw, weekday_weight=False
    )
    issued_residual = m.official_pv_residuals(data)
    forecasts = (load_point, load_residual, pv_point, pv_residual, issued_residual)
    q2_cache, q3_cache = precompute_trajectories(data, forecasts)

    risk_runs: dict[tuple[str, str], dict[str, Any]] = {}
    for mode in ("clustered", "unclustered"):
        risk_runs[("q2_fixed", mode)] = simulate(
            data,
            forecasts,
            variant=f"q2_fixed_{mode}",
            adjusted=False,
            variable_price=False,
            risk_mode=mode,
            trajectories=q2_cache[mode],
        )
        risk_runs[("q3_fixed", mode)] = simulate(
            data,
            forecasts,
            variant=f"q3_fixed_{mode}",
            adjusted=True,
            variable_price=False,
            risk_mode=mode,
            trajectories=q3_cache[mode],
        )

    risk_daily_parts = []
    risk_summary_rows = []
    for (strategy, mode), run in risk_runs.items():
        frame = run["daily"].copy()
        frame["strategy"] = strategy
        risk_daily_parts.append(frame)
        risk_summary_rows.append(
            {
                "strategy": strategy,
                "risk_mode": mode,
                **summarize_frame(frame),
                "warmup_end_soc_kwh": run["warmup_end_soc_kwh"],
                "runtime_seconds": run["runtime_seconds"],
            }
        )
    risk_daily = pd.concat(risk_daily_parts, ignore_index=True)
    risk_summary = pd.DataFrame(risk_summary_rows)
    decision = risk_decision(risk_summary)
    selected_mode = decision["selected_risk_mode"]

    info_runs: dict[tuple[int, ...], dict[str, Any]] = {}
    for coalition in RELEASE_SETS:
        if coalition == RELEASE_HOURS:
            info_runs[coalition] = risk_runs[("q3_fixed", selected_mode)]
        else:
            info_runs[coalition] = simulate(
                data,
                forecasts,
                variant=f"q3_info_{release_name(coalition)}",
                adjusted=True,
                variable_price=False,
                risk_mode=selected_mode,
                trajectories=q3_cache[selected_mode],
                allowed_issue_hours=coalition,
                capture_diagnostics=False,
            )
    info_frames = {coalition: run["daily"] for coalition, run in info_runs.items()}
    info_daily, info_summary, info_monthly, marginal_records = information_outputs(info_frames)
    marginal_frame = pd.DataFrame(marginal_records)
    if "record_type" not in marginal_frame:
        marginal_frame["record_type"] = "conditional_marginal"
    else:
        marginal_frame["record_type"] = marginal_frame["record_type"].fillna("conditional_marginal")

    q4_2 = simulate(
        data,
        forecasts,
        variant=f"q4_2_variable_{selected_mode}",
        adjusted=False,
        variable_price=True,
        risk_mode=selected_mode,
        trajectories=q2_cache[selected_mode],
    )
    q4_3 = simulate(
        data,
        forecasts,
        variant=f"q4_3_variable_{selected_mode}",
        adjusted=True,
        variable_price=True,
        risk_mode=selected_mode,
        trajectories=q3_cache[selected_mode],
    )
    main_runs = {
        "q2_fixed": risk_runs[("q2_fixed", selected_mode)],
        "q3_fixed": risk_runs[("q3_fixed", selected_mode)],
        "q4_2_variable": q4_2,
        "q4_3_variable": q4_3,
    }
    main_frames = {name: run["daily"] for name, run in main_runs.items()}
    monthly = monthly_operations(main_frames)
    low_pv = low_pv_stratification(main_frames)
    plan_execution = pd.concat(
        [run["diagnostics"] for run in list(risk_runs.values()) + [q4_2, q4_3]],
        ignore_index=True,
    )
    lp_diagnostics = pd.concat(
        [run["lp_diagnostics"] for run in list(risk_runs.values()) + [q4_2, q4_3]],
        ignore_index=True,
    )

    write_csv(risk_daily, output / "risk_ablation_daily.csv")
    write_csv(risk_summary, output / "risk_ablation_summary.csv")
    write_csv(info_daily, output / "information_value_daily.csv")
    write_csv(info_summary, output / "information_value_summary.csv")
    write_csv(info_monthly, output / "information_value_monthly.csv")
    write_csv(marginal_frame, output / "information_value_marginal.csv")
    write_csv(monthly, output / "monthly_operations.csv")
    write_csv(low_pv, output / "low_pv_stratification.csv")
    write_csv(plan_execution, output / "plan_execution_diagnostics.csv")
    write_csv(lp_diagnostics, output / "lp_solution_diagnostics.csv")

    main_totals = {name: summarize_frame(frame) for name, frame in main_frames.items()}
    full_info = info_summary.loc[info_summary.release_set == "S_6_12_18"].iloc[0]
    low_counts = (
        low_pv[low_pv.group == "low_pv_q20"]
        .groupby("variant", as_index=False)["days"]
        .sum()
        .set_index("variant")["days"]
        .astype(int)
        .to_dict()
    )
    shapley = (
        marginal_frame[marginal_frame.record_type == "shapley"]
        .loc[:, ["release_hour", "shapley_saving_yuan"]]
        .to_dict("records")
    )
    metrics = {
        "status": "full_diagnostics_completed_pending_P2",
        "evaluation": {"warmup": "2025-01-01..2025-01-31", "formal": "2025-02-01..2025-12-31", "days": 334},
        "parameters": {
            "alpha": ALPHA,
            "history_days": HISTORY_DAYS,
            "max_scenarios": MAX_SCENARIOS,
            "seed": m.SEED,
            "alpha_was_retuned": False,
        },
        "risk_decision": decision,
        "risk_ablation": risk_summary.to_dict("records"),
        "main_totals": main_totals,
        "information_value": {
            "full_vs_frozen_saving_yuan": float(full_info["saving_vs_S_none_yuan"]),
            "full_vs_frozen_saving_percent": float(full_info["saving_vs_S_none_percent"]),
            "full_vs_frozen_emergency_reduction_kwh": float(
                info_summary.loc[info_summary.release_set == "S_none", "emergency_kwh"].iloc[0]
                - full_info["emergency_kwh"]
            ),
            "shapley_saving_yuan": shapley,
            "interpretation_limit": (
                "This isolates access to official PV releases while retaining the same 0/6/12/18 "
                "re-optimization opportunities; it is not the total Q2-to-Q3 benefit."
            ),
        },
        "planning_execution": {
            "max_planning_simultaneous_flow_kwh": float(plan_execution["planning_simultaneous_flow_max_kwh"].max()),
            "max_execution_simultaneous_flow_kwh": float(plan_execution["execution_simultaneous_flow_max_kwh"].max()),
            "max_balance_residual_kwh": float(plan_execution["max_balance_residual_kwh"].max()),
            "main_variant_target_soc_mae_kwh": {
                name: float(run["diagnostics"]["target_soc_mae_kwh"].mean())
                for name, run in main_runs.items()
            },
        },
        "low_pv_q20_days": low_counts,
        "hash_checks": "omitted per explicit user request; no hashes calculated or compared; no cryptographic integrity guarantee",
    }
    metrics_path = output / "national_award_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    inputs = [root / "data" / "C题.pdf"] + [root / "data" / "附件" / f"附件{i}.xlsx" for i in range(1, 5)]
    output_files = sorted(path for path in output.iterdir() if path.is_file())
    run_record = {
        "command": "python run_national_award_diagnostics.py --project-root .",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "parameters": metrics["parameters"],
        "input_files": [file_metadata(path) for path in inputs],
        "source_files": [
            file_metadata(root / "microgrid_core.py"),
            file_metadata(root / "run_national_award_diagnostics.py"),
            file_metadata(root / "题目分析报告.md"),
            file_metadata(root / "术语表格.md"),
        ],
        "outputs_before_run_record": [file_metadata(path) for path in output_files],
        "runtime_seconds": time.perf_counter() - started,
        "hash_checks": "omitted per explicit user request; no hashes calculated or compared; no cryptographic integrity guarantee",
    }
    (output / "run_record_nohash.json").write_text(
        json.dumps(run_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
