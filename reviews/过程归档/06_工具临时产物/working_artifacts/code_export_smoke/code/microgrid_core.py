"""2026 CUMCM C problem: shared data, forecasting, LP, and causal simulation core."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, lil_matrix
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

from sklearn.cluster import KMeans


DT_HOURS = 1.0 / 6.0
N_SLOTS = 144
ETA_C = 0.90
ETA_D = 0.90
SOC_MIN = 1_200.0
SOC_MAX = 10_800.0
SOC_INITIAL = 6_000.0
POWER_LIMIT_KW = 5_000.0
ENERGY_LIMIT = POWER_LIMIT_KW * DT_HOURS
SEED = 2026
DEFAULT_ALPHA = 0.80
ISSUES = (0, 6, 12, 18)


@dataclass(frozen=True)
class Battery:
    capacity_kwh: float = 12000.0
    power_kw: float = POWER_LIMIT_KW
    eta_c: float = ETA_C
    eta_d: float = ETA_D

    @property
    def minimum(self) -> float:
        return 0.1 * self.capacity_kwh

    @property
    def maximum(self) -> float:
        return 0.9 * self.capacity_kwh

    @property
    def initial(self) -> float:
        return 0.5 * self.capacity_kwh

    @property
    def slot_limit(self) -> float:
        return self.power_kw * DT_HOURS


DEFAULT_BATTERY = Battery()


@dataclass(frozen=True)
class Inputs:
    dates: pd.DatetimeIndex
    typical_time: list[Any]
    typical_price: np.ndarray
    typical_load_kw: np.ndarray
    typical_pv_kw: np.ndarray
    actual_load_kw: np.ndarray
    actual_pv_kw: np.ndarray
    variable_price: np.ndarray
    pv_forecast_hourly: np.ndarray


def load_inputs(project_root: str | Path) -> Inputs:
    root = Path(project_root)
    attachment = root / "data" / "附件"

    a1 = pd.read_excel(attachment / "附件1.xlsx", sheet_name=0, header=0)
    if a1.shape != (N_SLOTS, 4):
        raise ValueError(f"附件1维度异常: {a1.shape}")

    load_df = pd.read_excel(attachment / "附件2.xlsx", sheet_name="小区负载", index_col=0, header=0)
    pv_df = pd.read_excel(attachment / "附件2.xlsx", sheet_name="光伏发电实际功率", index_col=0, header=0)
    price_df = pd.read_excel(attachment / "附件4.xlsx", sheet_name=0, index_col=0, header=0)
    def clock_minute(value):
        if hasattr(value, "hour"):
            minute = value.hour * 60 + value.minute
        else:
            clock = str(value).split("+")[0].split(":")
            minute = int(clock[0]) * 60 + int(clock[1])
        return 1440 if minute == 0 else minute
    expected_slots = list(range(10, 1441, 10))
    for labels in (a1.iloc[:, 0], load_df.columns, pv_df.columns, price_df.columns):
        if [clock_minute(v) for v in labels] != expected_slots:
            raise ValueError("附件时间列不是有序的00:10至24:00端点")
    dates = pd.DatetimeIndex(pd.to_datetime(load_df.index))
    if load_df.shape != (365, N_SLOTS) or pv_df.shape != load_df.shape:
        raise ValueError("附件2应为365×144的负荷和光伏矩阵")
    if price_df.shape != load_df.shape:
        raise ValueError("附件4应与附件2严格同形")
    if not dates.equals(pd.DatetimeIndex(pd.to_datetime(pv_df.index))):
        raise ValueError("附件2两个工作表日期不一致")
    if not dates.equals(pd.DatetimeIndex(pd.to_datetime(price_df.index))):
        raise ValueError("附件4日期与附件2不一致")

    a3 = pd.read_excel(attachment / "附件3.xlsx", sheet_name=0, header=0)
    if a3.shape != (1460, 26) or list(a3.columns[2:]) != [f"预报{h}小时" for h in range(1, 25)]:
        raise ValueError("附件3维度或提前期表头异常")
    forecast_dates = pd.to_datetime(a3.iloc[:, 0].replace("", np.nan).ffill())
    forecast = np.full((len(dates), len(ISSUES), 24), np.nan, dtype=float)
    date_pos = {d.normalize(): i for i, d in enumerate(dates)}
    issue_pos = {h: i for i, h in enumerate(ISSUES)}
    for row_index, row in enumerate(a3.itertuples(index=False, name=None)):
        d = pd.Timestamp(forecast_dates.iloc[row_index]).normalize()
        issue_value = row[1]
        issue_hour = int(issue_value.hour) if hasattr(issue_value, "hour") else int(str(issue_value).split(":")[0])
        if d not in date_pos or issue_hour not in issue_pos:
            raise ValueError("附件3含未定义日期或发布时刻")
        target = forecast[date_pos[d], issue_pos[issue_hour]]
        if np.isfinite(target).any():
            raise ValueError("附件3日期与发布时刻重复")
        target[:] = np.asarray(row[2:26], dtype=float)
    if np.isnan(forecast).any():
        raise ValueError("附件3存在未能映射的日期、发布时刻或预测值")

    result = Inputs(
        dates=dates,
        typical_time=a1.iloc[:, 0].tolist(),
        typical_price=a1.iloc[:, 1].to_numpy(dtype=float),
        typical_load_kw=a1.iloc[:, 2].to_numpy(dtype=float),
        typical_pv_kw=a1.iloc[:, 3].to_numpy(dtype=float),
        actual_load_kw=load_df.to_numpy(dtype=float),
        actual_pv_kw=pv_df.to_numpy(dtype=float),
        variable_price=price_df.to_numpy(dtype=float),
        pv_forecast_hourly=forecast,
    )
    _validate_inputs(result)
    return result


def _validate_inputs(data: Inputs) -> None:
    arrays = (
        data.typical_price,
        data.typical_load_kw,
        data.typical_pv_kw,
        data.actual_load_kw,
        data.actual_pv_kw,
        data.variable_price,
        data.pv_forecast_hourly,
    )
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("输入包含NaN或无穷值")
    if any((a < 0).any() for a in arrays):
        raise ValueError("输入包含负的电价、负荷或光伏值")
    expected = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    if not data.dates.equals(expected):
        raise ValueError("附件日期并非2025完整连续自然年")


def build_analog_forecasts(
    actual_kw: np.ndarray,
    typical_kw: np.ndarray,
    *,
    weekday_weight: bool,
    max_history: int = 90,
    decay_days: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Strict rolling-origin analog forecast and its realized residuals."""
    n_days, n_slots = actual_kw.shape
    pred = np.empty_like(actual_kw, dtype=float)
    for d in range(n_days):
        start = max(0, d - max_history)
        idx = np.arange(start, d)
        if idx.size == 0:
            pred[d] = typical_kw
            continue
        ages = d - idx
        weights = np.exp(-ages / decay_days)
        if weekday_weight:
            weights *= np.where(idx % 7 == d % 7, 2.0, 1.0)
        hist_mean = np.average(actual_kw[idx], axis=0, weights=weights)
        shrink = min(1.0, idx.size / 21.0)
        pred[d] = shrink * hist_mean + (1.0 - shrink) * typical_kw
    pred = np.maximum(pred, 0.0)
    return pred, actual_kw - pred


def _cluster_residuals(
    load_residual_kw: np.ndarray,
    pv_residual_kw: np.ndarray,
    *,
    max_scenarios: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = load_residual_kw.shape[0]
    if n == 0:
        zeros = np.zeros((1, load_residual_kw.shape[1]), dtype=float)
        return zeros, zeros.copy(), np.ones(1)
    raw = np.concatenate([load_residual_kw, pv_residual_kw], axis=1)
    scale = np.std(raw, axis=0)
    scale[scale < 1e-9] = 1.0
    k = min(max_scenarios, n, np.unique(raw / scale, axis=0).shape[0])
    if k == 1:
        labels = np.zeros(n, dtype=int)
    else:
        labels = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(raw / scale)
    nonempty = np.unique(labels)
    centers_l = np.stack([load_residual_kw[labels == i].mean(axis=0) for i in nonempty])
    centers_v = np.stack([pv_residual_kw[labels == i].mean(axis=0) for i in nonempty])
    weights = np.asarray([(labels == i).mean() for i in nonempty], dtype=float)
    return centers_l, centers_v, weights


def weighted_quantile_columns(values: np.ndarray, weights: np.ndarray, quantile: float) -> np.ndarray:
    if values.ndim != 2 or values.shape[0] != weights.size:
        raise ValueError("分位数输入维度不一致")
    if not 0 <= quantile <= 1 or not np.isfinite(values).all() or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("分位数或概率非法")
    result = np.empty(values.shape[1], dtype=float)
    for t in range(values.shape[1]):
        order = np.argsort(values[:, t], kind="mergesort")
        v = values[order, t]
        w = weights[order]
        cdf = np.cumsum(w) / np.sum(w)
        result[t] = v[min(np.searchsorted(cdf, quantile, side="left"), len(v) - 1)]
    return result


def safe_analog_trajectory(
    day: int,
    load_point_kw: np.ndarray,
    pv_point_kw: np.ndarray,
    load_residual_kw: np.ndarray,
    pv_residual_kw: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
    history_days: int = 60,
    max_scenarios: int = 9,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    start = max(0, day - history_days)
    c_l, c_v, weights = _cluster_residuals(
        load_residual_kw[start:day], pv_residual_kw[start:day], max_scenarios=max_scenarios
    )
    l_scen = np.maximum(0.0, load_point_kw[None, :] + c_l)
    v_scen = np.maximum(0.0, pv_point_kw[None, :] + c_v)
    net_q = weighted_quantile_columns(l_scen - v_scen, weights, alpha)
    pv_low = weighted_quantile_columns(v_scen, weights, 1.0 - alpha)
    safe_pv = np.maximum.reduce([pv_low, -net_q, np.zeros_like(net_q)])
    safe_load = safe_pv + net_q
    return safe_load, safe_pv, {
        "alpha": alpha,
        "history_start": start,
        "history_end_exclusive": day,
        "scenario_count": int(weights.size),
        "scenario_weights": weights.tolist(),
    }


def _block_slices(n: int, adjusted: bool) -> dict[str, slice]:
    pos = 0
    blocks: dict[str, slice] = {}
    for name, size in (("grid", n), ("charge", n), ("discharge", n), ("soc", n + 1), ("pv_used", n)):
        blocks[name] = slice(pos, pos + size)
        pos += size
    if adjusted:
        blocks["up"] = slice(pos, pos + n)
        pos += n
        blocks["down"] = slice(pos, pos + n)
        pos += n
    blocks["all"] = slice(0, pos)
    return blocks


def solve_schedule(
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    price: np.ndarray,
    soc0: float,
    *,
    terminal_cycle: bool = False,
    terminal_value: float = 0.0,
    base_plan_kwh: np.ndarray | None = None,
    battery: Battery = DEFAULT_BATTERY,
) -> dict[str, Any]:
    """Solve a deterministic LP for an initial or adjusted contract trajectory."""
    load_kw = np.asarray(load_kw, dtype=float)
    pv_kw = np.asarray(pv_kw, dtype=float)
    price = np.asarray(price, dtype=float)
    n = load_kw.size
    if pv_kw.shape != (n,) or price.shape != (n,):
        raise ValueError("负荷、光伏、电价长度不一致")
    if not (battery.minimum - 1e-9 <= soc0 <= battery.maximum + 1e-9):
        raise ValueError(f"初始SOC越界: {soc0}")
    adjusted = base_plan_kwh is not None
    if adjusted:
        base_plan_kwh = np.asarray(base_plan_kwh, dtype=float)
        if base_plan_kwh.shape != (n,):
            raise ValueError("原计划与调整时域长度不一致")
    b = _block_slices(n, adjusted)
    m = b["all"].stop
    c = np.zeros(m, dtype=float)
    if adjusted:
        c[b["up"]] = 1.5 * price
        c[b["down"]] = -0.5 * price
        constant_cost = float(np.dot(price, base_plan_kwh))
    else:
        c[b["grid"]] = price
        constant_cost = 0.0
    c[b["soc"].stop - 1] = -float(terminal_value)

    rows = 2 * n + (n if adjusted else 0)
    aeq = lil_matrix((rows, m), dtype=float)
    beq = np.zeros(rows, dtype=float)
    load_energy = load_kw * DT_HOURS
    pv_energy = pv_kw * DT_HOURS
    for t in range(n):
        aeq[t, b["grid"].start + t] = 1.0
        aeq[t, b["charge"].start + t] = -1.0
        aeq[t, b["discharge"].start + t] = 1.0
        aeq[t, b["pv_used"].start + t] = 1.0
        beq[t] = load_energy[t]

        row = n + t
        aeq[row, b["soc"].start + t] = -1.0
        aeq[row, b["soc"].start + t + 1] = 1.0
        aeq[row, b["charge"].start + t] = -battery.eta_c
        aeq[row, b["discharge"].start + t] = 1.0 / battery.eta_d
        if adjusted:
            row = 2 * n + t
            aeq[row, b["grid"].start + t] = 1.0
            aeq[row, b["up"].start + t] = -1.0
            aeq[row, b["down"].start + t] = 1.0
            beq[row] = base_plan_kwh[t]

    bounds: list[tuple[float | None, float | None]] = []
    bounds.extend([(0.0, None)] * n)
    bounds.extend([(0.0, battery.slot_limit)] * n)
    bounds.extend([(0.0, battery.slot_limit)] * n)
    soc_bounds = [(battery.minimum, battery.maximum)] * (n + 1)
    soc_bounds[0] = (soc0, soc0)
    if terminal_cycle:
        soc_bounds[-1] = (soc0, soc0)
    bounds.extend(soc_bounds)
    bounds.extend([(0.0, float(v)) for v in pv_energy])
    if adjusted:
        bounds.extend([(0.0, None)] * n)
        bounds.extend([(0.0, float(v)) for v in base_plan_kwh])

    res = linprog(c, A_eq=aeq.tocsr(), b_eq=beq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"HiGHS求解失败: status={res.status}, {res.message}")

    economic_optimum = float(res.fun)
    tie = np.zeros(m, dtype=float)
    tie[b["charge"]] = 1.0
    tie[b["discharge"]] = 1.0
    aub = csr_matrix(c.reshape(1, -1))
    tol = 1e-8 * max(1.0, abs(economic_optimum))
    res2 = linprog(
        tie,
        A_ub=aub,
        b_ub=np.asarray([economic_optimum + tol]),
        A_eq=aeq.tocsr(),
        b_eq=beq,
        bounds=bounds,
        method="highs",
    )
    if not res2.success:
        raise RuntimeError(f"吞吐次级LP失败: {res2.status}, {res2.message}")
    x = res2.x
    balance_residual = aeq.tocsr() @ x - beq
    result = {
        "grid_kwh": x[b["grid"]].copy(),
        "charge_kwh": x[b["charge"]].copy(),
        "discharge_kwh": x[b["discharge"]].copy(),
        "soc_kwh": x[b["soc"]].copy(),
        "pv_used_kwh": x[b["pv_used"]].copy(),
        "solver_objective": float(np.dot(c, x) + constant_cost + terminal_value * battery.minimum),
        "cash_contract_cost": float(
            np.dot(price, x[b["grid"]])
            if not adjusted
            else constant_cost
            + np.dot(1.5 * price, x[b["up"]])
            - np.dot(0.5 * price, x[b["down"]])
        ),
        "terminal_value_credit": float(terminal_value * (x[b["soc"].stop - 1] - battery.minimum)),
        "max_eq_residual": float(np.max(np.abs(balance_residual))),
        "simultaneous_flow_max": float(np.max(np.minimum(x[b["charge"]], x[b["discharge"]]))),
        "status": "optimal",
    }
    if adjusted:
        result["up_kwh"] = x[b["up"]].copy()
        result["down_kwh"] = x[b["down"]].copy()
    return result


def causal_dispatch(
    contract_kwh: np.ndarray,
    actual_load_kw: np.ndarray,
    actual_pv_kw: np.ndarray,
    price: np.ndarray,
    soc0: float,
    *,
    base_plan_kwh: np.ndarray | None = None,
    battery: Battery = DEFAULT_BATTERY,
    emergency_multiplier: float = 5.0,
) -> dict[str, Any]:
    """Greedy feasible controller using current observation only."""
    q = np.asarray(contract_kwh, dtype=float)
    load_e = np.asarray(actual_load_kw, dtype=float) * DT_HOURS
    pv_e = np.asarray(actual_pv_kw, dtype=float) * DT_HOURS
    price = np.asarray(price, dtype=float)
    n = q.size
    if load_e.shape != (n,) or pv_e.shape != (n,) or price.shape != (n,):
        raise ValueError("因果控制输入长度不一致")
    charge = np.zeros(n)
    discharge = np.zeros(n)
    emergency = np.zeros(n)
    unused = np.zeros(n)
    curtailed = np.zeros(n)
    pv_used = np.zeros(n)
    soc = np.empty(n + 1)
    soc[0] = soc0
    for t in range(n):
        available = q[t] + pv_e[t] - load_e[t]
        if available >= 0.0:
            charge[t] = min(available, battery.slot_limit, max(0.0, (battery.maximum - soc[t]) / battery.eta_c))
            surplus = max(0.0, available - charge[t])
            unused[t] = min(q[t], surplus)
            curtailed[t] = max(0.0, surplus - unused[t])
            pv_used[t] = pv_e[t] - curtailed[t]
            soc[t + 1] = soc[t] + battery.eta_c * charge[t]
        else:
            need = -available
            discharge[t] = min(need, battery.slot_limit, max(0.0, (soc[t] - battery.minimum) * battery.eta_d))
            emergency[t] = max(0.0, need - discharge[t])
            pv_used[t] = pv_e[t]
            soc[t + 1] = soc[t] - discharge[t] / battery.eta_d
    residual = q - unused + pv_used + discharge + emergency - load_e - charge
    if base_plan_kwh is None:
        settlement = price * q
        alt_settlement = settlement.copy()
    else:
        p = np.asarray(base_plan_kwh, dtype=float)
        if p.shape != (n,):
            raise ValueError("原计划与最终合同长度不一致")
        settlement = price * np.minimum(p, q) + 1.5 * price * np.maximum(q - p, 0.0) + 0.5 * price * np.maximum(p - q, 0.0)
        alt_settlement = price * p + 1.5 * price * np.maximum(q - p, 0.0) + 0.5 * price * np.maximum(p - q, 0.0)
    emergency_cost = emergency_multiplier * price * emergency
    return {
        "charge_kwh": charge,
        "discharge_kwh": discharge,
        "soc_kwh": soc,
        "emergency_kwh": emergency,
        "unused_contract_kwh": unused,
        "pv_curtailed_kwh": curtailed,
        "pv_used_kwh": pv_used,
        "settlement_cost": settlement,
        "alternative_settlement_cost": alt_settlement,
        "emergency_cost": emergency_cost,
        "cash_cost": settlement + emergency_cost,
        "alternative_cash_cost": alt_settlement + emergency_cost,
        "max_balance_residual": float(np.max(np.abs(residual))),
    }


def terminal_value(price: np.ndarray, battery: Battery = DEFAULT_BATTERY) -> float:
    return float(battery.eta_d * np.median(np.asarray(price, dtype=float)))


def predict_price(data: Inputs, day: int, issue_hour: int = 0) -> np.ndarray:
    """Only historical days and the already-realized prefix are readable."""
    base = np.median(data.variable_price[max(0, day - 30):day], axis=0) if day else data.typical_price.copy()
    start = issue_hour * 6
    if start:
        observed = slice(max(0, start - 18), start)
        bias = float(np.mean(data.variable_price[day, observed] - base[observed]))
        tau = np.arange(1, N_SLOTS - start + 1) / 6.0
        return np.maximum(0.0001, base[start:] + bias * np.exp(-tau / 3.0))
    return np.maximum(0.0001, base)


def run_q1(data: Inputs, battery: Battery = DEFAULT_BATTERY) -> dict[str, Any]:
    return solve_schedule(
        data.typical_load_kw,
        data.typical_pv_kw,
        data.typical_price,
        battery.initial,
        terminal_cycle=True,
        battery=battery,
    )


def run_q2_day(
    data: Inputs,
    day: int,
    soc0: float,
    load_point: np.ndarray,
    pv_point: np.ndarray,
    load_residual: np.ndarray,
    pv_residual: np.ndarray,
    *,
    variable_price: bool,
    alpha: float = DEFAULT_ALPHA,
    safe_pair: tuple[np.ndarray, np.ndarray, dict[str, Any]] | None = None,
    battery: Battery = DEFAULT_BATTERY,
    emergency_multiplier: float = 5.0,
    history_days: int = 60,
    max_scenarios: int = 9,
) -> dict[str, Any]:
    price = predict_price(data, day) if variable_price else data.typical_price
    settlement_price = data.variable_price[day] if variable_price else data.typical_price
    if safe_pair is None:
        safe_l, safe_v, meta = safe_analog_trajectory(
            day, load_point[day], pv_point[day], load_residual, pv_residual, alpha=alpha, history_days=history_days, max_scenarios=max_scenarios
        )
    else:
        safe_l, safe_v, meta = safe_pair
    plan = solve_schedule(
        safe_l,
        safe_v,
        price,
        soc0,
        terminal_value=terminal_value(price, battery),
        battery=battery,
    )
    actual = causal_dispatch(
        plan["grid_kwh"], data.actual_load_kw[day], data.actual_pv_kw[day], settlement_price, soc0, battery=battery, emergency_multiplier=emergency_multiplier
    )
    return {"plan": plan, "actual": actual, "forecast_meta": meta, "safe_load_kw": safe_l, "safe_pv_kw": safe_v, "planning_price": price}


def official_pv_residuals(data: Inputs) -> np.ndarray:
    """Actual power at each forecasted clock hour minus that point forecast."""
    n_days = len(data.dates)
    flat = data.actual_pv_kw.reshape(-1)
    residual = np.full_like(data.pv_forecast_hourly, np.nan, dtype=float)
    for d in range(n_days):
        for i, issue in enumerate(ISSUES):
            start = d * N_SLOTS + issue * 6
            for h in range(24):
                # Column h is the point forecast at issue+(h+1) hours.  The
                # 10-minute source rows are interval endpoints, hence -1.
                endpoint = start + (h + 1) * 6 - 1
                if endpoint < flat.size:
                    residual[d, i, h] = flat[endpoint] - data.pv_forecast_hourly[d, i, h]
    return residual


def interpolate_hourly_points(anchor: float, hourly_points: np.ndarray, n_slots: int) -> np.ndarray:
    """Linearly interpolate clock-hour point values onto 10-minute endpoints."""
    values = np.concatenate(([float(anchor)], np.asarray(hourly_points, dtype=float)))
    x_hour = np.arange(values.size, dtype=float)
    x_slot = np.arange(1, n_slots + 1, dtype=float) / 6.0
    return np.interp(x_slot, x_hour, values)


def safe_update_trajectory(
    data: Inputs,
    day: int,
    issue_index: int,
    load_point_day: np.ndarray,
    load_residual: np.ndarray,
    pv_issue_residual: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
    history_days: int = 60,
    max_scenarios: int = 9,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    issue = ISSUES[issue_index]
    start_slot = issue * 6
    n = N_SLOTS - start_slot
    hours = int(np.ceil(n / 6))
    raw = data.pv_forecast_hourly[day, issue_index, :hours]
    hist_start = max(0, day - history_days)
    hist = pv_issue_residual[hist_start:day, issue_index, :hours]
    valid_rows = np.all(np.isfinite(hist), axis=1) if hist.size else np.zeros(0, dtype=bool)
    hist = hist[valid_rows]
    bias = np.nanmean(hist, axis=0) if hist.shape[0] else np.zeros(hours)
    if issue > 0:
        anchor = float(data.actual_pv_kw[day, start_slot - 1])
    elif day > 0:
        anchor = float(data.actual_pv_kw[day - 1, -1])
    else:
        anchor = 0.0
    pv_point = interpolate_hourly_points(anchor, np.maximum(0.0, raw + bias), n)
    l_point = load_point_day[start_slot:]

    hist_ids = np.arange(hist_start, day)[valid_rows]
    if hist_ids.size:
        r_l = load_residual[hist_ids, start_slot:]
        centered_hourly = hist - bias[None, :]
        r_v = np.vstack(
            [interpolate_hourly_points(0.0, row, n) for row in centered_hourly]
        )
    else:
        r_l = np.empty((0, n))
        r_v = np.empty((0, n))
    c_l, c_v, weights = _cluster_residuals(r_l, r_v, max_scenarios=max_scenarios)
    l_scen = np.maximum(0.0, l_point[None, :] + c_l)
    v_scen = np.maximum(0.0, pv_point[None, :] + c_v)
    net_q = weighted_quantile_columns(l_scen - v_scen, weights, alpha)
    pv_low = weighted_quantile_columns(v_scen, weights, 1.0 - alpha)
    safe_v = np.maximum.reduce([pv_low, -net_q, np.zeros_like(net_q)])
    safe_l = safe_v + net_q
    return safe_l, safe_v, {
        "decision_day": int(day),
        "issue_hour": int(issue),
        "history_start": int(hist_start),
        "history_end_exclusive": int(day),
        "scenario_count": int(weights.size),
        "alpha": float(alpha),
        "scenario_weights": weights.tolist(),
        "pv_point_kw": pv_point.tolist(),
        "anchor_kw": anchor,
        "bias_kw": bias.tolist(),
    }


def nowcast_trajectory(
    data: Inputs, day: int, event: int,
    source: tuple[np.ndarray, np.ndarray, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Synthetic nowcast based on the latest available official issue only."""
    safe_l, safe_v, meta = source
    issue = int(meta["issue_hour"])
    offset = (event - issue) * 6
    if offset < 0:
        raise ValueError("不能使用未来发布的预报")
    if not offset:
        return source
    point = np.asarray(meta["pv_point_kw"])
    first = max(0, offset - 18)
    realized = data.actual_pv_kw[day, issue * 6 + first:event * 6]
    bias = float(np.mean(realized - point[first:offset])) if realized.size else 0.0
    tau = np.arange(1, N_SLOTS - event * 6 + 1) / 6.0
    revised = np.maximum(0.0, safe_v[offset:] + bias * np.exp(-tau / 3.0))
    return safe_l[offset:], revised, {
        **{k: v for k, v in meta.items() if k not in ("pv_point_kw", "bias_kw")},
        "decision_hour": event, "synthetic_nowcast": True,
        "observed_end_exclusive": event * 6, "nowcast_bias_kw": bias,
    }


def run_q3_day(
    data: Inputs,
    day: int,
    soc0: float,
    load_point: np.ndarray,
    load_residual: np.ndarray,
    pv_issue_residual: np.ndarray,
    *,
    variable_price: bool,
    alpha: float = DEFAULT_ALPHA,
    safe_updates: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] | None = None,
    events: tuple[int, ...] = ISSUES,
    battery: Battery = DEFAULT_BATTERY,
    emergency_multiplier: float = 5.0,
    history_days: int = 60,
    max_scenarios: int = 9,
) -> dict[str, Any]:
    if not events or events[0] != 0 or tuple(sorted(set(events))) != events or any(h < 0 or h >= 24 or int(h) != h for h in events):
        raise ValueError("更新时刻必须从0开始，严格递增且在[0,24)内")
    price = predict_price(data, day) if variable_price else data.typical_price
    settlement_price = data.variable_price[day] if variable_price else data.typical_price
    if safe_updates is None:
        safe_l0, safe_v0, meta0 = safe_update_trajectory(
            data, day, 0, load_point[day], load_residual, pv_issue_residual, alpha=alpha, history_days=history_days, max_scenarios=max_scenarios
        )
    else:
        safe_l0, safe_v0, meta0 = safe_updates[0]
    initial = solve_schedule(
        safe_l0,
        safe_v0,
        price,
        soc0,
        terminal_value=terminal_value(price, battery),
        battery=battery,
    )
    p0 = initial["grid_kwh"]
    final_contract = p0.copy()
    actual_parts: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "charge_kwh",
            "discharge_kwh",
            "emergency_kwh",
            "unused_contract_kwh",
            "pv_curtailed_kwh",
            "pv_used_kwh",
            "settlement_cost",
            "alternative_settlement_cost",
            "emergency_cost",
            "cash_cost",
            "alternative_cash_cost",
        )
    }
    soc_trace = [float(soc0)]
    update_meta = [meta0]
    update_solutions: list[dict[str, Any]] = []
    current_soc = float(soc0)

    for i, issue in enumerate(events):
        start = issue * 6
        end = events[i + 1] * 6 if i + 1 < len(events) else N_SLOTS
        if i == 0:
            contract_remaining = p0[start:]
            update_solution = initial
        else:
            latest_index = max(j for j, h in enumerate(ISSUES) if h <= issue)
            if safe_updates is None:
                source = safe_update_trajectory(
                    data, day, latest_index, load_point[day], load_residual, pv_issue_residual, alpha=alpha, history_days=history_days, max_scenarios=max_scenarios
                )
            else:
                source = safe_updates[latest_index]
            safe_l, safe_v, meta = nowcast_trajectory(data, day, issue, source)
            update_meta.append(meta)
            remaining_price = predict_price(data, day, issue) if variable_price else price[start:]
            update_solution = solve_schedule(
                safe_l,
                safe_v,
                remaining_price,
                current_soc,
                terminal_value=terminal_value(remaining_price, battery),
                base_plan_kwh=p0[start:],
                battery=battery,
            )
            update_solution["planning_price"] = remaining_price
            contract_remaining = update_solution["grid_kwh"]
        update_solutions.append(update_solution)
        segment_contract = contract_remaining[: end - start]
        final_contract[start:end] = segment_contract
        segment = causal_dispatch(
            segment_contract,
            data.actual_load_kw[day, start:end],
            data.actual_pv_kw[day, start:end],
            settlement_price[start:end],
            current_soc,
            base_plan_kwh=p0[start:end],
            battery=battery,
            emergency_multiplier=emergency_multiplier,
        )
        for key in actual_parts:
            actual_parts[key].append(segment[key])
        soc_trace.extend(segment["soc_kwh"][1:].tolist())
        current_soc = float(segment["soc_kwh"][-1])

    actual = {key: np.concatenate(parts) for key, parts in actual_parts.items()}
    actual["soc_kwh"] = np.asarray(soc_trace)
    balance = final_contract - actual["unused_contract_kwh"] + actual["pv_used_kwh"] + actual["discharge_kwh"] + actual["emergency_kwh"] - data.actual_load_kw[day] * DT_HOURS - actual["charge_kwh"]
    actual["max_balance_residual"] = float(np.max(np.abs(balance)))
    return {
        "initial_plan": initial,
        "initial_plan_kwh": p0,
        "final_contract_kwh": final_contract,
        "actual": actual,
        "update_meta": update_meta,
        "update_solutions": update_solutions,
        "planning_price": price,
        "events": events,
    }


def summarize_day(result: dict[str, Any], *, adjusted: bool) -> dict[str, float]:
    actual = result["actual"]
    plan = result["initial_plan_kwh"] if adjusted else result["plan"]["grid_kwh"]
    contract = result["final_contract_kwh"] if adjusted else plan
    return {
        "plan_kwh": float(np.sum(plan)),
        "final_contract_kwh": float(np.sum(contract)),
        "cash_cost_yuan": float(np.sum(actual["cash_cost"])),
        "alternative_cash_cost_yuan": float(np.sum(actual.get("alternative_cash_cost", actual["cash_cost"]))),
        "emergency_kwh": float(np.sum(actual["emergency_kwh"])),
        "unused_contract_kwh": float(np.sum(actual["unused_contract_kwh"])),
        "pv_curtailed_kwh": float(np.sum(actual["pv_curtailed_kwh"])),
        "charge_kwh": float(np.sum(actual["charge_kwh"])),
        "discharge_kwh": float(np.sum(actual["discharge_kwh"])),
        "soc_start_kwh": float(actual["soc_kwh"][0]),
        "soc_end_kwh": float(actual["soc_kwh"][-1]),
        "soc_min_kwh": float(np.min(actual["soc_kwh"])),
        "soc_max_kwh": float(np.max(actual["soc_kwh"])),
        "max_balance_residual_kwh": float(actual["max_balance_residual"]),
    }
