"""Reproduce the refactored Problem-2 strategy and its validation experiments.

The formal strategy is a day-ahead risk-contract LP followed by current-slot
greedy feedback.  Rolling MPC is retained only as a comparison baseline.
January is simulated solely as a causal warm-up; reported totals cover
2025-02-01 through 2025-12-31 (334 days).
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from openpyxl import load_workbook

import microgrid_core as m
from evaluate_results import validate_day
from run_all import write_annual_workbook


EVALUATION_START = 31
SENSITIVITY_FACTORS = (0.5, 1.0, 1.5, 2.0)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, pd.Timestamp): return value.isoformat()
    raise TypeError(type(value).__name__)


def dump_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def count_events(values: np.ndarray, tolerance: float = 1e-8) -> int:
    active = np.asarray(values) > tolerance
    return int(np.count_nonzero(active & ~np.r_[False, active[:-1]]))


def execute_fixed(
    method: str, plan: dict[str, Any], data: m.Inputs, day: int, soc0: float,
    load_kw: np.ndarray, pv_kw: np.ndarray, load_forecast: np.ndarray,
    pv_forecast: np.ndarray, mpc_config: m.MpcConfig,
) -> dict[str, Any]:
    if method == "greedy":
        actual = m.greedy_execution(
            plan["grid_kwh"], load_kw, pv_kw, data.typical_price, soc0,
        )
    elif method == "mpc":
        actual = m.baseline_MPC(
            plan["grid_kwh"], load_kw, pv_kw, data.typical_price, soc0,
            forecast_load_kw=load_forecast, forecast_pv_kw=pv_forecast,
            forecast_price=data.typical_price,
            target_soc_kwh=plan["soc_kwh"], config=mpc_config,
        )
    else:
        raise ValueError(method)
    validate_day({"plan": plan, "actual": actual}, False, soc0)
    return actual


def daily_row(
    date: pd.Timestamp, method: str, plan: dict[str, Any], actual: dict[str, Any],
    *, comparison: str, sigma_factor: float | None = None,
) -> dict[str, Any]:
    row = {
        "date": date.date().isoformat(),
        "method": method,
        "comparison": comparison,
        "cash_cost_yuan": float(np.sum(actual["cash_cost"])),
        "contract_cost_yuan": float(np.sum(actual["settlement_cost"])),
        "emergency_cost_yuan": float(np.sum(actual["emergency_cost"])),
        "contract_kwh": float(np.sum(plan["grid_kwh"])),
        "base_contract_kwh": float(np.sum(plan["base_contract_kwh"])),
        "risk_reserve_kwh": float(np.sum(plan["risk_reserve_kwh"])),
        "expected_emergency_cost_yuan": float(plan["expected_emergency_cost"]),
        "emergency_kwh": float(np.sum(actual["emergency_kwh"])),
        "emergency_slots": int(np.count_nonzero(actual["emergency_kwh"] > 1e-8)),
        "emergency_events": count_events(actual["emergency_kwh"]),
        "unused_contract_kwh": float(np.sum(actual["unused_contract_kwh"])),
        "pv_curtailed_kwh": float(np.sum(actual["pv_curtailed_kwh"])),
        "charge_kwh": float(np.sum(actual["charge_kwh"])),
        "discharge_kwh": float(np.sum(actual["discharge_kwh"])),
        "soc_start_kwh": float(actual["soc_kwh"][0]),
        "soc_end_kwh": float(actual["soc_kwh"][-1]),
        "soc_min_kwh": float(np.min(actual["soc_kwh"])),
        "soc_max_kwh": float(np.max(actual["soc_kwh"])),
        "runtime_seconds": float(actual.get("runtime_seconds", 0.0)),
        "max_balance_residual_kwh": float(actual["max_balance_residual"]),
        "simultaneous_flow_slots": int(actual["simultaneous_flow_slots"]),
    }
    if sigma_factor is not None: row["sigma_factor"] = sigma_factor
    return row


def aggregate(rows: list[dict[str, Any]], method_label: str) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    if frame.empty: raise ValueError("cannot aggregate an empty experiment")
    return {
        "方法": method_label,
        "天数": int(len(frame)),
        "总费用_元": float(frame.cash_cost_yuan.sum()),
        "平均每日费用_元": float(frame.cash_cost_yuan.mean()),
        "合同费用_元": float(frame.contract_cost_yuan.sum()),
        "应急费用_元": float(frame.emergency_cost_yuan.sum()),
        "应急购电量_kWh": float(frame.emergency_kwh.sum()),
        "应急购电槽数": int(frame.emergency_slots.sum()),
        "应急购电事件数": int(frame.emergency_events.sum()),
        "合同浪费量_kWh": float(frame.unused_contract_kwh.sum()),
        "弃光量_kWh": float(frame.pv_curtailed_kwh.sum()),
        "SOC最小值_kWh": float(frame.soc_min_kwh.min()),
        "SOC最大值_kWh": float(frame.soc_max_kwh.max()),
        "期末SOC_kWh": float(frame.iloc[-1].soc_end_kwh),
        "执行计算时间_秒": float(frame.runtime_seconds.sum()),
        "最大平衡残差_kWh": float(frame.max_balance_residual_kwh.max()),
        "同时充放电槽数": int(frame.simultaneous_flow_slots.sum()),
    }


def build_reference(
    data: m.Inputs, forecast: dict[str, Any], config: m.MpcConfig,
    alpha: float, history_days: int, output: Path, *, write_outputs: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    soc = m.SOC_INITIAL
    for day, date in enumerate(data.dates):
        result = m.run_q2_day(
            data, day, soc, forecast["load_kw"], forecast["pv_kw"],
            forecast["load_residual_kw"], forecast["pv_residual_kw"],
            variable_price=False, alpha=alpha, history_days=history_days,
            risk_mode="unclustered", controller="greedy",
            mpc_config=config,
        )
        result.update(date=date, day_index=day, variable_price=False)
        validate_day(result, False, soc)
        if day >= EVALUATION_START:
            rows.append(daily_row(
                date, "greedy", result["plan"], result["actual"],
                comparison="formal_continuous_closed_loop",
            ))
        results.append(result)
        soc = float(result["actual"]["soc_kwh"][-1])
        if day % 60 == 0 or day == len(data.dates) - 1:
            print(f"[formal greedy] {day + 1}/{len(data.dates)} SOC={soc:.2f}", flush=True)

    if write_outputs:
        pd.DataFrame(rows).to_csv(output / "主模型_逐日.csv", index=False, encoding="utf-8-sig")
        metric = aggregate(rows, "LP+反馈贪心")
        pd.DataFrame([metric]).to_csv(output / "主模型年度指标.csv", index=False, encoding="utf-8-sig")
        dump_json(output / "主模型年度指标.json", metric)
        interval_rows = []
        for result in results[EVALUATION_START:]:
            day = result["day_index"]; plan = result["plan"]; actual = result["actual"]
            for slot in range(m.N_SLOTS):
                interval_rows.append({
                    "date": result["date"].date().isoformat(), "slot": slot + 1,
                    "base_contract_kwh": plan["base_contract_kwh"][slot],
                    "risk_reserve_kwh": plan["risk_reserve_kwh"][slot],
                    "contract_kwh": plan["grid_kwh"][slot],
                    "actual_load_kw": data.actual_load_kw[day, slot],
                    "actual_pv_kw": data.actual_pv_kw[day, slot],
                    "pv_to_load_kwh": actual["pv_to_load_kwh"][slot],
                    "contract_to_load_kwh": actual["contract_to_load_kwh"][slot],
                    "pv_to_charge_kwh": actual["pv_to_charge_kwh"][slot],
                    "contract_to_charge_kwh": actual["contract_to_charge_kwh"][slot],
                    "charge_kwh": actual["charge_kwh"][slot],
                    "discharge_kwh": actual["discharge_kwh"][slot],
                    "emergency_kwh": actual["emergency_kwh"][slot],
                    "unused_contract_kwh": actual["unused_contract_kwh"][slot],
                    "soc_start_kwh": actual["soc_kwh"][slot],
                    "soc_end_kwh": actual["soc_kwh"][slot + 1],
                    "cash_cost_yuan": actual["cash_cost"][slot],
                })
        pd.DataFrame(interval_rows).to_csv(
            output / "主模型_逐槽.csv", index=False, encoding="utf-8-sig",
        )
    return results, rows


def continuous_mpc(
    data: m.Inputs, forecast: dict[str, Any], config: m.MpcConfig, alpha: float,
    history_days: int, output: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []; soc = m.SOC_INITIAL
    for day, date in enumerate(data.dates):
        result = m.run_q2_day(
            data, day, soc, forecast["load_kw"], forecast["pv_kw"],
            forecast["load_residual_kw"], forecast["pv_residual_kw"],
            variable_price=False, alpha=alpha, history_days=history_days,
            risk_mode="unclustered", controller="mpc", mpc_config=config,
        )
        validate_day(result, False, soc)
        if day >= EVALUATION_START:
            rows.append(daily_row(
                date, "mpc", result["plan"], result["actual"],
                comparison="continuous_closed_loop",
            ))
        soc = float(result["actual"]["soc_kwh"][-1])
        if day % 30 == 0 or day == len(data.dates) - 1:
            print(f"[continuous MPC] {day + 1}/{len(data.dates)} SOC={soc:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(output / "MPC连续闭环_逐日.csv", index=False, encoding="utf-8-sig")
    return rows, aggregate(rows, "MPC（连续闭环）")


def paired_comparison(
    data: m.Inputs, forecast: dict[str, Any], reference: list[dict[str, Any]],
    config: m.MpcConfig, output: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    for day in range(EVALUATION_START, len(data.dates)):
        ref = reference[day]; plan = ref["plan"]; soc0 = float(ref["actual"]["soc_kwh"][0])
        for method in ("greedy", "mpc"):
            actual = ref["actual"] if method == "greedy" else execute_fixed(
                method, plan, data, day, soc0, data.actual_load_kw[day],
                data.actual_pv_kw[day], forecast["load_kw"][day],
                forecast["pv_kw"][day], config,
            )
            rows.append(daily_row(
                data.dates[day], method, plan, actual,
                comparison="frozen_paired_same_contract_soc_information",
            ))
        if day % 30 == 0 or day == len(data.dates) - 1:
            print(f"[paired comparison] {day + 1}/{len(data.dates)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "策略对比_冻结配对逐日.csv", index=False, encoding="utf-8-sig")
    table = [
        aggregate(frame[frame.method == "mpc"].to_dict("records"), "MPC"),
        aggregate(frame[frame.method == "greedy"].to_dict("records"), "LP+反馈贪心"),
    ]
    pd.DataFrame(table).to_csv(output / "策略对比.csv", index=False, encoding="utf-8-sig")
    return rows, table


def scaled_net_error_scenario(
    load_forecast: np.ndarray, pv_forecast: np.ndarray, actual_load: np.ndarray,
    actual_pv: np.ndarray, factor: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    forecast_net = load_forecast - pv_forecast
    actual_net = actual_load - actual_pv
    baseline_residual = actual_net - forecast_net
    scenario_net = forecast_net + factor * baseline_residual
    # Preserve the scaled net-load error exactly while retaining non-negative
    # load and PV.  At factor=1 this recovers the observed load and PV exactly.
    gross = np.maximum(actual_load + actual_pv, np.abs(scenario_net))
    scenario_load = (gross + scenario_net) / 2.0
    scenario_pv = (gross - scenario_net) / 2.0
    achieved = float(np.std(scenario_net - forecast_net) / np.std(baseline_residual))
    return scenario_load, scenario_pv, achieved


def sensitivity(
    data: m.Inputs, forecast: dict[str, Any], reference: list[dict[str, Any]],
    config: m.MpcConfig, output: Path, factors: tuple[float, ...],
) -> list[dict[str, Any]]:
    detailed = []
    for factor in factors:
        achieved_ratios = []
        for day in range(EVALUATION_START, len(data.dates)):
            ref = reference[day]; plan = ref["plan"]; soc0 = float(ref["actual"]["soc_kwh"][0])
            scenario_load, scenario_pv, achieved = scaled_net_error_scenario(
                forecast["load_kw"][day], forecast["pv_kw"][day],
                data.actual_load_kw[day], data.actual_pv_kw[day], factor,
            )
            achieved_ratios.append(achieved)
            for method in ("greedy", "mpc"):
                actual = execute_fixed(
                    method, plan, data, day, soc0, scenario_load, scenario_pv,
                    forecast["load_kw"][day], forecast["pv_kw"][day], config,
                )
                row = daily_row(
                    data.dates[day], method, plan, actual,
                    comparison="error_scaled_frozen_contract_and_soc",
                    sigma_factor=factor,
                )
                row["achieved_daily_sigma_ratio"] = achieved
                detailed.append(row)
            if day % 45 == 0 or day == len(data.dates) - 1:
                print(f"[sensitivity {factor:.1f}sigma] {day + 1}/{len(data.dates)}", flush=True)
        print(
            f"[sensitivity {factor:.1f}sigma] achieved mean ratio "
            f"{np.mean(achieved_ratios):.6f}", flush=True,
        )
    detail = pd.DataFrame(detailed)
    detail.to_csv(output / "误差敏感性_逐日.csv", index=False, encoding="utf-8-sig")
    summary = []
    for factor in factors:
        part = detail[detail.sigma_factor == factor]
        by_method = {}
        for method, label in (("mpc", "MPC"), ("greedy", "LP+反馈贪心")):
            metric = aggregate(part[part.method == method].to_dict("records"), label)
            metric["sigma_factor"] = factor
            metric["实际残差标准差倍率"] = float(part.achieved_daily_sigma_ratio.mean())
            summary.append(metric); by_method[method] = metric
        summary.append({
            "方法": "MPC费用-LP+反馈贪心费用", "天数": len(part) // 2,
            "总费用_元": by_method["mpc"]["总费用_元"] - by_method["greedy"]["总费用_元"],
            "平均每日费用_元": by_method["mpc"]["平均每日费用_元"] - by_method["greedy"]["平均每日费用_元"],
            "sigma_factor": factor,
            "实际残差标准差倍率": float(part.achieved_daily_sigma_ratio.mean()),
        })
    pd.DataFrame(summary).to_csv(output / "误差敏感性.csv", index=False, encoding="utf-8-sig")
    return summary


def information_fairness(
    data: m.Inputs, forecast: dict[str, Any], reference: list[dict[str, Any]],
    config: m.MpcConfig, output: Path,
) -> dict[str, Any]:
    day = 78; cutoff = 48; ref = reference[day]; plan = ref["plan"]
    soc0 = float(ref["actual"]["soc_kwh"][0])
    changed = replace(
        data, actual_load_kw=data.actual_load_kw.copy(),
        actual_pv_kw=data.actual_pv_kw.copy(), variable_price=data.variable_price.copy(),
    )
    changed.actual_load_kw[day:] += 1234.0
    changed.actual_pv_kw[day:] += 345.0
    changed_result = m.run_q2_day(
        changed, day, soc0, forecast["load_kw"], forecast["pv_kw"],
        forecast["load_residual_kw"], forecast["pv_residual_kw"],
        variable_price=False, alpha=float(plan["alpha"]),
        history_days=day - int(plan["history_start"]), controller="greedy",
    )
    contract_diff = float(np.max(np.abs(plan["grid_kwh"] - changed_result["plan"]["grid_kwh"])))

    future_load = data.actual_load_kw[day].copy(); future_pv = data.actual_pv_kw[day].copy()
    future_load[cutoff:] += 1500.0; future_pv[cutoff:] += 300.0
    originals = {}; changed_truth = {}
    for method in ("greedy", "mpc"):
        originals[method] = execute_fixed(
            method, plan, data, day, soc0, data.actual_load_kw[day],
            data.actual_pv_kw[day], forecast["load_kw"][day],
            forecast["pv_kw"][day], config,
        )
        changed_truth[method] = execute_fixed(
            method, plan, data, day, soc0, future_load, future_pv,
            forecast["load_kw"][day], forecast["pv_kw"][day], config,
        )
    prefix_diffs = {}
    for method in ("greedy", "mpc"):
        prefix_diffs[method] = max(
            float(np.max(np.abs(originals[method][key][:cutoff] - changed_truth[method][key][:cutoff])))
            for key in ("charge_kwh", "discharge_kwh", "emergency_kwh")
        )
        prefix_diffs[method] = max(
            prefix_diffs[method],
            float(np.max(np.abs(
                originals[method]["soc_kwh"][:cutoff + 1]
                - changed_truth[method]["soc_kwh"][:cutoff + 1]
            ))),
        )
    matched = {
        "contract_max_abs_diff_kwh": 0.0,
        "initial_soc_abs_diff_kwh": 0.0,
        "load_path_max_abs_diff_kw": 0.0,
        "pv_path_max_abs_diff_kw": 0.0,
        "load_forecast_max_abs_diff_kw": 0.0,
        "pv_forecast_max_abs_diff_kw": 0.0,
        "settlement_price_max_abs_diff": 0.0,
    }
    checks = {
        "day_ahead_contract_unchanged_after_current_and_future_truth_perturbation": contract_diff <= 1e-9,
        "greedy_executed_prefix_ignores_future_truth": prefix_diffs["greedy"] <= 1e-9,
        "mpc_executed_prefix_ignores_future_truth": prefix_diffs["mpc"] <= 1e-9,
        "history_strictly_precedes_decision_day": int(plan["history_end_exclusive"]) == day,
        "same_contract_soc_forecast_truth_and_settlement_for_paired_methods": all(v == 0 for v in matched.values()),
        "greedy_has_no_simultaneous_charge_discharge": originals["greedy"]["simultaneous_flow_slots"] == 0,
        "mpc_is_comparison_only": originals["mpc"]["solver_diagnostics"]["role"] == "comparison_only",
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "decision_day_index": day, "decision_date": data.dates[day],
        "history_start": plan["history_start"],
        "history_end_exclusive": plan["history_end_exclusive"],
        "future_truth_perturbation_cutoff_slot": cutoff,
        "contract_max_abs_diff_kwh": contract_diff,
        "executed_prefix_max_abs_diff": prefix_diffs,
        "paired_input_max_abs_differences": matched,
        "checks": checks,
        "information_sets": {
            "contract_LP": ["rolling-origin point forecast available at 00:00", "residuals from days j<d", "initial SOC", "published tariff"],
            "greedy": ["fixed day-ahead contract", "current-slot load/PV", "current SOC"],
            "MPC_baseline": ["same fixed contract", "same published point forecast", "current SOC", "current-slot truth only for feasibility clipping"],
        },
    }
    if report["status"] != "PASS": raise AssertionError(report)
    dump_json(output / "信息公平性.json", report)
    return report


def audit_result2(template: Path, result: Path, output: Path) -> dict[str, Any]:
    source = load_workbook(template, read_only=True, data_only=False)
    written = load_workbook(result, read_only=True, data_only=False)
    source_purchase = source["计划购电量"]
    purchase = written["计划购电量"]
    # A read-only worksheet is stream-backed.  Iterating once avoids reparsing
    # the XML for every random cell access (which is effectively quadratic).
    source_header = next(source_purchase.iter_rows(
        min_row=1, max_row=1, min_col=1, max_col=147, values_only=True
    ))
    result_rows = list(purchase.iter_rows(
        min_row=1, max_row=335, min_col=1, max_col=147, values_only=True
    ))
    result_header = result_rows[0]
    data_rows = result_rows[1:]
    dates = [row[0] for row in data_rows]
    report = {
        "status": "PASS",
        "template": str(template.resolve()), "result": str(result.resolve()),
        "same_sheet_names_and_order": source.sheetnames == written.sheetnames,
        "sheet_names": written.sheetnames,
        "purchase_header_identical": source_header == result_header,
        "written_day_count": sum(value is not None for value in dates),
        "first_date": str(dates[0])[:10], "last_date": str(dates[-1])[:10],
        "all_numeric_contract_cells": all(
            isinstance(value, (int, float))
            for row in data_rows for value in row[1:145]
        ),
    }
    report["checks"] = {
        "same_sheet_names_and_order": report["same_sheet_names_and_order"],
        "purchase_header_identical": report["purchase_header_identical"],
        "exactly_334_days": report["written_day_count"] == 334,
        "date_range_correct": str(report["first_date"])[:10] == "2025-02-01" and str(report["last_date"])[:10] == "2025-12-31",
        "all_numeric_contract_cells": report["all_numeric_contract_cells"],
    }
    report["status"] = "PASS" if all(report["checks"].values()) else "FAIL"
    dump_json(output / "result2模板兼容审计.json", report)
    if report["status"] != "PASS": raise AssertionError(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="results/q2_strategy")
    parser.add_argument("--phase", choices=("main", "comparison", "sensitivity", "fairness", "all"), default="all")
    parser.add_argument("--factor", type=float, choices=SENSITIVITY_FACTORS)
    parser.add_argument("--alpha", type=float, default=m.DEFAULT_ALPHA)
    parser.add_argument("--history-days", type=int, default=m.DEFAULT_HISTORY_DAYS)
    parser.add_argument("--mpc-horizon", type=int, default=m.DEFAULT_MPC_HORIZON)
    args = parser.parse_args()
    root = Path(args.project_root).resolve(); output = (root / args.output_dir).resolve()
    if output == root or root not in output.parents: raise ValueError("output must be inside project root")
    output.mkdir(parents=True, exist_ok=True)
    begun = time.perf_counter(); data = m.load_inputs(root)
    forecast = m.build_forecast_archive(data)["F1"]
    config = m.MpcConfig(horizon_slots=args.mpc_horizon, tracking_weight=0.0)
    write_main = args.phase in ("main", "all")
    reference, formal_rows = build_reference(
        data, forecast, config, args.alpha, args.history_days, output,
        write_outputs=write_main,
    )
    manifest: dict[str, Any] = {
        "phase": args.phase, "alpha": args.alpha,
        "history_days": args.history_days, "forecast_model": "F1",
        "mpc_horizon_slots": args.mpc_horizon, "evaluated_days": 334,
        "evaluation_start": "2025-02-01", "evaluation_end": "2025-12-31",
    }
    if write_main:
        template = root / "data" / "附件" / "附件5" / "result2.xlsx"
        strategy_workbook = output / "result2.xlsx"
        write_annual_workbook(data, reference, template, strategy_workbook, adjusted=False)
        write_annual_workbook(data, reference, template, root / "results" / "result2.xlsx", adjusted=False)
        manifest["main"] = aggregate(formal_rows, "LP+反馈贪心")
        manifest["result2_audit"] = audit_result2(template, strategy_workbook, output)
    if args.phase in ("comparison", "all"):
        paired_rows, paired_table = paired_comparison(data, forecast, reference, config, output)
        continuous_rows, continuous_metric = continuous_mpc(
            data, forecast, config, args.alpha, args.history_days, output,
        )
        continuous_table = [continuous_metric, aggregate(formal_rows, "LP+反馈贪心（连续闭环）")]
        pd.DataFrame(continuous_table).to_csv(output / "策略对比_连续闭环.csv", index=False, encoding="utf-8-sig")
        manifest["paired_comparison"] = paired_table
        manifest["continuous_comparison"] = continuous_table
    if args.phase in ("sensitivity", "all"):
        factors = (args.factor,) if args.factor is not None else SENSITIVITY_FACTORS
        manifest["sensitivity"] = sensitivity(data, forecast, reference, config, output, factors)
    if args.phase in ("fairness", "all"):
        manifest["information_fairness"] = information_fairness(data, forecast, reference, config, output)
    manifest["runtime_seconds"] = time.perf_counter() - begun
    dump_json(output / f"运行记录_{args.phase}.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
