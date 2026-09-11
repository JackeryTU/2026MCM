"""P1 regression tests: physical feasibility, cold start and non-anticipativity."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
from numpy.testing import assert_allclose
import microgrid_core as m


def forecasts(data):
    lp, lr = m.build_analog_forecasts(data.actual_load_kw, data.typical_load_kw, weekday_weight=True)
    vp, vr = m.build_analog_forecasts(data.actual_pv_kw, data.typical_pv_kw, weekday_weight=False)
    return lp, lr, vp, vr, m.official_pv_residuals(data)


def physical(result, adjusted, battery=m.DEFAULT_BATTERY):
    a = result["actual"]
    assert a["max_balance_residual"] < 1e-6
    assert a["soc_kwh"].min() >= battery.minimum - 1e-7
    assert a["soc_kwh"].max() <= battery.maximum + 1e-7
    for key in ("charge_kwh", "discharge_kwh"):
        assert a[key].min() >= -1e-7 and a[key].max() <= battery.slot_limit + 1e-7
    assert np.max(np.minimum(a["charge_kwh"], a["discharge_kwh"])) < 1e-5
    assert np.max(np.minimum(a["charge_kwh"], a["emergency_kwh"])) <= 1e-7
    assert_allclose(np.diff(a["soc_kwh"]), battery.eta_c*a["charge_kwh"]-a["discharge_kwh"]/battery.eta_d, atol=1e-7)
    plans = result["update_solutions"] if adjusted else [result["plan"]]
    for p in plans:
        assert p["status"] == "optimal" and p["max_eq_residual"] < 1e-6
        assert p["simultaneous_flow_max"] < 1e-5
        assert_allclose(p["solver_objective"], p["cash_contract_cost"]-p["terminal_value_credit"], atol=1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default="results/p1_regression.json")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    begun = time.perf_counter()
    data = m.load_inputs(root)
    lp, lr, vp, vr, ir = forecasts(data)
    checks = []
    archive = m.build_forecast_archive(data)
    assert set(archive) == {"F0", "F1", "F2"}
    assert_allclose(archive["F1"]["load_kw"][7], data.actual_load_kw[0])
    assert_allclose(archive["F1"]["pv_kw"][7], data.actual_pv_kw[:7].mean(axis=0))
    assert_allclose(archive["F2"]["load_kw"][21], np.median(data.actual_load_kw[[14, 7, 0]], axis=0))
    assert_allclose(archive["F1"]["net_residual_kw"], data.actual_load_kw-data.actual_pv_kw-archive["F1"]["net_kw"])
    assert_allclose(m.empirical_quantile_columns(np.array([[0.], [10.]]), .5), [0.])
    assert_allclose(m.empirical_quantile_columns(np.empty((0, 3)), .9), np.zeros(3))
    checks.append("F0_F1_F2_versioned_causal_forecasts_and_empirical_quantile")
    assert_allclose(m.interpolate_hourly_points(0, np.array([60, 120]), 12), np.arange(10, 121, 10))
    assert_allclose(ir[0, 0, 0], data.actual_pv_kw[0, 5]-data.pv_forecast_hourly[0, 0, 0])
    assert_allclose(ir[0, 3, 23], data.actual_pv_kw[1, 107]-data.pv_forecast_hourly[0, 3, 23])
    assert np.isnan(ir[-1, 3, 23])
    checks.append("hourly_point_endpoint_mapping")
    for n in (0, 1, 4):
        z = np.zeros((n, 144))
        cl, cv, w = m._cluster_residuals(z, z)
        assert cl.shape == (1, 144) and cv.shape == cl.shape
        assert_allclose(cl, 0); assert_allclose(w, [1])
    for n in (0, 1):
        z = np.zeros((n, 144))
        ul, uv, uw = m._residual_scenarios(z, z, risk_mode="unclustered", max_scenarios=9)
        assert ul.shape == (1, 144) and uv.shape == ul.shape
        assert_allclose(ul, 0); assert_allclose(uw, [1])
    residual_pair = np.vstack([np.zeros(144), np.full(144, 10.0)])
    ul, uv, uw = m._residual_scenarios(
        residual_pair, np.zeros_like(residual_pair), risk_mode="unclustered", max_scenarios=1
    )
    assert ul.shape == (2, 144) and uv.shape == ul.shape
    assert_allclose(uw, [.5, .5])
    unclustered = m.safe_analog_trajectory(
        2, np.zeros(144), np.zeros(144), residual_pair, np.zeros_like(residual_pair),
        alpha=.8, risk_mode="unclustered",
    )
    assert_allclose(unclustered[0] - unclustered[1], 10)
    assert unclustered[2]["risk_mode"] == "unclustered"
    assert unclustered[2]["scenario_count"] == 2
    try:
        m._residual_scenarios(residual_pair, residual_pair, risk_mode="unknown", max_scenarios=9)
        raise AssertionError("invalid risk_mode accepted")
    except ValueError:
        pass
    assert_allclose(m.weighted_quantile_columns(np.array([[0.], [10.]]), np.array([.5, .5]), .8), [10])
    assert_allclose(m.weighted_quantile_columns(np.array([[0.], [10.]]), np.array([.5, .5]), .5), [0])
    assert_allclose(lp[0], data.typical_load_kw)
    assert_allclose(lp[1], data.actual_load_kw[0]/21+data.typical_load_kw*20/21)
    l0, v0, meta = m.safe_analog_trajectory(0, lp[0], vp[0], lr, vr)
    assert_allclose(l0, lp[0]); assert_allclose(v0, vp[0])
    for i in (0, 1):
        l, v, meta = m.safe_update_trajectory(data, 0, i, lp[0], lr, ir)
        assert meta["release_preserved_without_recentering"]
        assert meta["scenario_count"] == 1 and np.isfinite(l).all() and np.isfinite(v).all()
        assert meta["anchor_kw"] == (0 if i == 0 else data.actual_pv_kw[0, 35])
    checks.append("cold_start_single_duplicate_and_zero_variance")
    p = np.array([10., 10., 10.]); q = np.array([5., 10., 15.])
    a = m.causal_dispatch(q, np.zeros(3), np.zeros(3), np.ones(3), m.SOC_MAX, base_plan_kwh=p)
    assert_allclose(a["settlement_cost"], [7.5, 10, 17.5])
    assert_allclose(a["alternative_settlement_cost"], [12.5, 10, 17.5])
    a = m.causal_dispatch(np.zeros(1), np.array([600.]), np.zeros(1), np.array([2.]), m.SOC_MIN)
    assert_allclose(a["emergency_kwh"], [100]); assert_allclose(a["cash_cost"], [1000])
    assert_allclose(a["planned_charge_kwh"], a["charge_kwh"])
    assert_allclose(a["planned_discharge_kwh"], a["discharge_kwh"])
    checks.append("piecewise_settlement_and_emergency")
    aligned = m.delayed_greedy_dispatch(
        np.zeros(2), np.array([600., 600.]), np.zeros(2),
        np.ones(2), m.SOC_INITIAL, forecast_load_kw=np.array([600., 0.]),
        forecast_pv_kw=np.zeros(2),
    )
    # Slot 2 must use the completed slot-1 load, not its deliberately wrong
    # slot-2 point forecast; both slots remain physically feasible.
    assert aligned["planned_discharge_kwh"][1] > 0
    assert aligned["max_balance_residual"] < 1e-8
    assert aligned["simultaneous_flow_max"] < 1e-8
    checks.append("timing_aligned_delayed_greedy")
    # M1 directed regression: emergency energy may cover only forecast load
    # deficit, never charge the battery for cheap terminal-value arbitrage.
    n = 36
    mpc = m.solve_mpc_horizon(
        np.full(n, 500.0), np.full(n, 3000.0), np.zeros(n),
        np.r_[0.0076, np.full(n-1, 0.8)], 6000.0,
        config=m.MpcConfig(horizon_slots=36, tracking_weight=0.0),
    )
    assert_allclose(mpc["charge_kwh"][0], 0.0, atol=1e-8)
    assert_allclose(mpc["emergency_kwh"][0], 0.0, atol=1e-8)
    clipped = m.clip_planned_action(500.0, 3600.0, 0.0, 6000.0, 100.0, 0.0)
    assert_allclose(clipped["charge_kwh"], 0.0, atol=1e-8)
    assert_allclose(clipped["emergency_kwh"], 100.0, atol=1e-8)
    assert min(clipped["charge_kwh"], clipped["emergency_kwh"]) <= 1e-7
    assert abs(clipped["balance_residual_kwh"]) <= 1e-7
    tracked = m.solve_mpc_horizon(
        np.full(n, 500.0), np.full(n, 3000.0), np.zeros(n), np.full(n, .8),
        6000.0, target_soc_kwh=np.full(n, 6000.0),
        config=m.MpcConfig(horizon_slots=36, tracking_weight=.01),
    )
    assert tracked["simultaneous_flow_max"] <= 1e-7
    checks.append("mpc_emergency_cannot_charge_and_truth_only_shrinks_action")
    rng = np.random.default_rng(m.SEED)
    sampled = np.sort(rng.choice(np.arange(1, 364), size=20, replace=False))
    for d in sampled:
        # Corrupt every future actual datum, but preserve already-published forecasts.
        changed = replace(data, actual_load_kw=data.actual_load_kw.copy(), actual_pv_kw=data.actual_pv_kw.copy(), variable_price=data.variable_price.copy())
        changed.actual_load_kw[d:] += 1234
        changed.actual_pv_kw[d:] += 345
        changed.variable_price[d:] += 2.0
        flp, flr, fvp, fvr, fir = forecasts(changed)
        assert_allclose(lp[:d+1], flp[:d+1], atol=0, rtol=0)
        assert_allclose(vp[:d+1], fvp[:d+1], atol=0, rtol=0)
        assert_allclose(m.predict_price(data, d), m.predict_price(changed, d), atol=0, rtol=0)
        original = m.safe_analog_trajectory(d, lp[d], vp[d], lr, vr)
        altered = m.safe_analog_trajectory(d, flp[d], fvp[d], flr, fvr)
        assert_allclose(original[0], altered[0], atol=0, rtol=0)
        assert_allclose(original[1], altered[1], atol=0, rtol=0)
        original = m.safe_update_trajectory(data, d, 0, lp[d], lr, ir)
        altered = m.safe_update_trajectory(changed, d, 0, flp[d], flr, fir)
        assert_allclose(original[0], altered[0], atol=0, rtol=0)
        assert_allclose(original[1], altered[1], atol=0, rtol=0)
    checks.append("20_random_days_future_actual_perturbation")
    d = 78
    for event in (0, 6, 8, 12, 18):
        # At event, prefix is unchanged; corrupt future observations and future releases.
        changed = replace(data, actual_pv_kw=data.actual_pv_kw.copy(), actual_load_kw=data.actual_load_kw.copy(), variable_price=data.variable_price.copy(), pv_forecast_hourly=data.pv_forecast_hourly.copy())
        changed.actual_pv_kw[d, event*6:] += 1000
        changed.actual_load_kw[d, event*6:] += 1000
        changed.variable_price[d, event*6:] += 10
        for j, hour in enumerate(m.ISSUES):
            if hour > event: changed.pv_forecast_hourly[d, j] += 2000
        fir = m.official_pv_residuals(changed)
        latest = max(i for i, h in enumerate(m.ISSUES) if h <= event)
        source = m.safe_update_trajectory(data, d, latest, lp[d], lr, ir)
        modified = m.safe_update_trajectory(changed, d, latest, lp[d], lr, fir)
        orig = m.nowcast_trajectory(data, d, event, source)
        alt = m.nowcast_trajectory(changed, d, event, modified)
        assert_allclose(orig[0], alt[0], atol=0, rtol=0); assert_allclose(orig[1], alt[1], atol=0, rtol=0)
        price = m.predict_price(data, d, event)
        assert_allclose(price, m.predict_price(changed, d, event), atol=0, rtol=0)
        base = m.solve_schedule(orig[0], orig[1], price, 6000)["grid_kwh"]
        test = m.solve_schedule(alt[0], alt[1], price, 6000)["grid_kwh"]
        assert_allclose(base, test, atol=0, rtol=0)
    checks.append("intraday_nowcast_price_and_decision_nonanticipativity")
    # Information-release ablation: hidden official versions and their
    # calibration residuals must not affect the frozen-00 strategy.
    d = 78
    updates = [m.safe_update_trajectory(data, d, i, lp[d], lr, ir) for i in range(4)]
    frozen = m.run_q3_day(
        data, d, 6000, lp, lr, ir, variable_price=False,
        safe_updates=updates, allowed_issue_hours=(),
    )
    explicit_full = m.run_q3_day(
        data, d, 6000, lp, lr, ir, variable_price=False,
        safe_updates=updates, allowed_issue_hours=(6, 12, 18),
    )
    default_full = m.run_q3_day(
        data, d, 6000, lp, lr, ir, variable_price=False, safe_updates=updates,
    )
    assert_allclose(explicit_full["final_contract_kwh"], default_full["final_contract_kwh"], atol=0, rtol=0)
    changed = replace(
        data,
        pv_forecast_hourly=data.pv_forecast_hourly.copy(),
    )
    changed.pv_forecast_hourly[:, 1:] += 5000
    changed_ir = m.official_pv_residuals(changed)
    changed_updates = [m.safe_update_trajectory(changed, d, i, lp[d], lr, changed_ir) for i in range(4)]
    frozen_changed = m.run_q3_day(
        changed, d, 6000, lp, lr, changed_ir, variable_price=False,
        safe_updates=changed_updates, allowed_issue_hours=(),
    )
    assert_allclose(frozen["initial_plan_kwh"], frozen_changed["initial_plan_kwh"], atol=0, rtol=0)
    assert_allclose(frozen["final_contract_kwh"], frozen_changed["final_contract_kwh"], atol=0, rtol=0)
    assert [x["source_issue_hour"] for x in frozen["update_meta"]] == [0, 0, 0, 0]
    assert [x["source_issue_hour"] for x in explicit_full["update_meta"]] == [0, 6, 12, 18]
    assert np.isfinite(frozen["target_soc_kwh"]).all()
    assert np.isfinite(frozen["safe_net_kw"]).all()
    physical(frozen, True)
    checks.append("unclustered_risk_and_eight_release_set_semantics")
    continuity = {}
    # January is a required initial-state slice, not a parameter scan or annual backtest.
    for adjusted, variable in ((False, False), (True, False), (False, True), (True, True)):
        soc = m.SOC_INITIAL
        jan31 = None
        for d in range(32):
            if adjusted:
                result = m.run_q3_day(data, d, soc, lp, lr, ir, variable_price=variable)
            else:
                result = m.run_q2_day(data, d, soc, lp, vp, lr, vr, variable_price=variable)
            physical(result, adjusted)
            assert result["actual"]["soc_kwh"][0] == soc
            if d == 30: jan31 = float(result["actual"]["soc_kwh"][-1])
            if d == 31: assert result["actual"]["soc_kwh"][0] == jan31
            soc = float(result["actual"]["soc_kwh"][-1])
        continuity[f"adjusted={adjusted},variable={variable}"] = jan31
    checks.append("jan1_through_feb1_all_four_variants_continuous_soc")
    for events in ((0,), (0, 12), tuple(range(0, 24, 2))):
        result = m.run_q3_day(data, 78, 6000, lp, lr, ir, variable_price=True, events=events)
        physical(result, True)
    no_storage = m.Battery(capacity_kwh=0, power_kw=0)
    r = m.run_q2_day(data, 78, 0, lp, vp, lr, vr, variable_price=False, battery=no_storage)
    physical(r, False, no_storage)
    checks.append("candidate_event_slices_and_no_storage")
    files = ["microgrid_core.py", "test_core.py", "smoke_test.py", "题目分析报告.md", "术语表格.md"]
    snapshot = [{"path": str(root/f), "size_bytes": (root/f).stat().st_size,
                 "modified_time_ns": (root/f).stat().st_mtime_ns} for f in files]
    report = {"status": "author_tests_passed_not_independent_gate", "checks": checks,
              "sampled_days": sampled.tolist(), "jan31_soc_kwh": continuity,
              "file_snapshot": snapshot,
              "hash_checks": "omitted per explicit user request; no hashes calculated or compared",
              "runtime_seconds": time.perf_counter()-begun}
    out = root/args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
