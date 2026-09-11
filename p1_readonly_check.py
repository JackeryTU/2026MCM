"""Short, read-only P1 gate for the revised microgrid model.

The script intentionally writes no files.  It prints one JSON object containing
the measured numerical evidence required by the programmer-stage P1 gate.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
from numpy.testing import assert_allclose

import microgrid_core as m
from test_core import physical


def check_forecasts(data: m.Inputs) -> dict[str, float | int]:
    archive = m.build_forecast_archive(data)
    assert set(archive) == {"F0", "F1", "F2"}
    assert_allclose(archive["F1"]["load_kw"][7], data.actual_load_kw[0])
    assert_allclose(
        archive["F2"]["load_kw"][21],
        np.median(data.actual_load_kw[[14, 7, 0]], axis=0),
    )
    for name in ("F1", "F2"):
        expected = (
            data.actual_load_kw
            - data.actual_pv_kw
            - archive[name]["net_kw"]
        )
        assert_allclose(archive[name]["net_residual_kw"], expected)

    day = 78
    changed = replace(
        data,
        actual_load_kw=data.actual_load_kw.copy(),
        actual_pv_kw=data.actual_pv_kw.copy(),
        variable_price=data.variable_price.copy(),
    )
    changed.actual_load_kw[day:] += 1234.0
    changed.actual_pv_kw[day:] += 345.0
    changed.variable_price[day:] += 2.0
    changed_archive = m.build_forecast_archive(changed)
    maximum_change = 0.0
    for name in ("F1", "F2"):
        maximum_change = max(
            maximum_change,
            float(np.max(np.abs(
                archive[name]["net_kw"][: day + 1]
                - changed_archive[name]["net_kw"][: day + 1]
            ))),
        )
    assert maximum_change == 0.0

    q50 = float(m.empirical_quantile_columns(np.array([[0.0], [10.0]]), 0.5)[0])
    q_above = float(
        m.empirical_quantile_columns(np.array([[0.0], [10.0]]), 0.500001)[0]
    )
    assert q50 == 0.0 and q_above == 10.0
    return {
        "causal_prefix_max_change_kw": maximum_change,
        "residual_bindings_checked": 2,
        "two_point_q_at_alpha_0_5": q50,
        "two_point_q_above_alpha_0_5": q_above,
    }


def check_mpc_physics() -> dict[str, float | int]:
    n = 36
    solution = m.solve_mpc_horizon(
        np.full(n, 500.0),
        np.full(n, 3000.0),
        np.zeros(n),
        np.r_[0.0076, np.full(n - 1, 0.8)],
        m.SOC_INITIAL,
        config=m.MpcConfig(horizon_slots=n, tracking_weight=0.0),
    )
    assert_allclose(solution["charge_kwh"][0], 0.0, atol=1e-8)
    assert_allclose(solution["emergency_kwh"][0], 0.0, atol=1e-8)

    clipped = m.clip_planned_action(
        500.0, 3600.0, 0.0, m.SOC_INITIAL, 100.0, 0.0
    )
    assert_allclose(clipped["charge_kwh"], 0.0, atol=1e-8)
    assert_allclose(clipped["emergency_kwh"], 100.0, atol=1e-8)
    assert abs(clipped["balance_residual_kwh"]) < 1e-6
    emergency_charge_overlap = min(
        clipped["charge_kwh"], clipped["emergency_kwh"]
    )
    assert emergency_charge_overlap <= 1e-7
    return {
        "mpc_first_slot_charge_kwh": float(solution["charge_kwh"][0]),
        "mpc_first_slot_emergency_kwh": float(solution["emergency_kwh"][0]),
        "directed_actual_emergency_kwh": float(clipped["emergency_kwh"]),
        "emergency_charge_overlap_kwh": float(emergency_charge_overlap),
        "mpc_max_equality_residual_kwh": float(solution["max_eq_residual"]),
        "mpc_simultaneous_flow_max_kwh": float(solution["simultaneous_flow_max"]),
    }


def check_q3_information(data: m.Inputs) -> dict[str, object]:
    archive = m.build_forecast_archive(data)["F1"]
    load_point = archive["load_kw"]
    load_residual = archive["load_residual_kw"]
    issue_residual = m.official_pv_residuals(data)
    day = 78
    updates = [
        m.safe_update_trajectory(
            data, day, i, load_point[day], load_residual, issue_residual
        )
        for i in range(4)
    ]
    frozen = m.run_q3_day(
        data, day, m.SOC_INITIAL, load_point, load_residual, issue_residual,
        variable_price=False, safe_updates=updates, allowed_issue_hours=(),
        controller="greedy",
    )
    full = m.run_q3_day(
        data, day, m.SOC_INITIAL, load_point, load_residual, issue_residual,
        variable_price=False, safe_updates=updates,
        allowed_issue_hours=(6, 12, 18), controller="greedy",
    )

    changed = replace(data, pv_forecast_hourly=data.pv_forecast_hourly.copy())
    changed.pv_forecast_hourly[:, 1:] += 5000.0
    changed_issue_residual = m.official_pv_residuals(changed)
    changed_updates = [
        m.safe_update_trajectory(
            changed, day, i, load_point[day], load_residual,
            changed_issue_residual,
        )
        for i in range(4)
    ]
    frozen_changed = m.run_q3_day(
        changed, day, m.SOC_INITIAL, load_point, load_residual,
        changed_issue_residual, variable_price=False,
        safe_updates=changed_updates, allowed_issue_hours=(),
        controller="greedy",
    )
    contract_change = float(np.max(np.abs(
        frozen["final_contract_kwh"] - frozen_changed["final_contract_kwh"]
    )))
    assert contract_change == 0.0
    frozen_sources = [int(x["source_issue_hour"]) for x in frozen["update_meta"]]
    full_sources = [int(x["source_issue_hour"]) for x in full["update_meta"]]
    assert frozen_sources == [0, 0, 0, 0]
    assert full_sources == [0, 6, 12, 18]
    physical(frozen, True)
    physical(full, True)
    return {
        "hidden_release_contract_max_change_kwh": contract_change,
        "frozen_source_issue_hours": frozen_sources,
        "full_source_issue_hours": full_sources,
        "frozen_max_balance_residual_kwh": float(
            frozen["actual"]["max_balance_residual"]
        ),
    }


def check_full_year(data: m.Inputs) -> dict[str, float | int]:
    forecast = m.build_forecast_archive(data)["F1"]
    soc = m.SOC_INITIAL
    max_boundary_gap = 0.0
    max_balance = 0.0
    max_overlap = 0.0
    max_emergency_charge_overlap = 0.0
    soc_min = soc
    soc_max = soc
    total_slots = 0
    for day in range(len(data.dates)):
        result = m.run_q2_day(
            data, day, soc,
            forecast["load_kw"], forecast["pv_kw"],
            forecast["load_residual_kw"], forecast["pv_residual_kw"],
            variable_price=False, controller="greedy",
        )
        actual = result["actual"]
        physical(result, False)
        gap = abs(float(actual["soc_kwh"][0]) - soc)
        max_boundary_gap = max(max_boundary_gap, gap)
        max_balance = max(max_balance, float(actual["max_balance_residual"]))
        max_overlap = max(max_overlap, float(actual["simultaneous_flow_max"]))
        max_emergency_charge_overlap = max(
            max_emergency_charge_overlap,
            float(np.max(np.minimum(
                actual["charge_kwh"], actual["emergency_kwh"]
            ))),
        )
        soc_min = min(soc_min, float(np.min(actual["soc_kwh"])))
        soc_max = max(soc_max, float(np.max(actual["soc_kwh"])))
        total_slots += len(actual["charge_kwh"])
        soc = float(actual["soc_kwh"][-1])

    assert total_slots == 365 * m.N_SLOTS
    assert max_boundary_gap <= 1e-8
    assert max_balance < 1e-6
    assert m.SOC_MIN - 1e-7 <= soc_min <= soc_max <= m.SOC_MAX + 1e-7
    assert max_overlap < 1e-5
    assert max_emergency_charge_overlap <= 1e-7
    return {
        "days": len(data.dates),
        "slots": total_slots,
        "cross_day_soc_max_gap_kwh": max_boundary_gap,
        "soc_min_kwh": soc_min,
        "soc_max_kwh": soc_max,
        "max_balance_residual_kwh": max_balance,
        "simultaneous_flow_max_kwh": max_overlap,
        "emergency_charge_overlap_max_kwh": max_emergency_charge_overlap,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--section",
        choices=("all", "forecasts", "mpc", "q3", "full-year"),
        default="all",
    )
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    begun = time.perf_counter()
    data = m.load_inputs(root)
    runners = {
        "forecasts": lambda: check_forecasts(data),
        "mpc": check_mpc_physics,
        "q3": lambda: check_q3_information(data),
        "full-year": lambda: check_full_year(data),
    }
    selected = runners if args.section == "all" else {args.section: runners[args.section]}
    evidence = {name: runner() for name, runner in selected.items()}
    report = {
        "status": "PASS",
        "read_only": True,
        "hashes": "not calculated",
        "section": args.section,
        "evidence": evidence,
        "runtime_seconds": time.perf_counter() - begun,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
