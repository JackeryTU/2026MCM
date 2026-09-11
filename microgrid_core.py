"""2026 CUMCM C problem: shared data, forecasting, LP, and causal simulation core."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
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
DEFAULT_RISK_MODE = "unclustered"
ISSUES = (0, 6, 12, 18)
DEFAULT_HISTORY_DAYS = 30
DEFAULT_MPC_HORIZON = 36
DEFAULT_TRACKING_WEIGHT = 0.0


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
class MpcConfig:
    """Short-horizon execution settings; horizon is measured in 10-minute slots."""

    horizon_slots: int = DEFAULT_MPC_HORIZON
    tracking_weight: float = DEFAULT_TRACKING_WEIGHT
    solve_budget_seconds: float = 5.0
    residual_tolerance: float = 1e-7
    overlap_tolerance: float = 1e-7

    def __post_init__(self) -> None:
        if self.horizon_slots <= 0:
            raise ValueError("MPC视野必须为正整数")
        if self.tracking_weight < 0 or self.solve_budget_seconds <= 0:
            raise ValueError("MPC跟踪权重和求解预算非法")


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


def build_periodic_forecasts(
    actual_load_kw: np.ndarray,
    actual_pv_kw: np.ndarray,
    typical_load_kw: np.ndarray,
    typical_pv_kw: np.ndarray,
    *,
    load_model: str = "F1",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the F1/F2 causal centre forecasts and their own residual archives.

    F1 copies the same slot from d-7.  F2 takes the median of available
    d-7/d-14/d-21 values.  PV is the same-slot mean of at most seven fully
    completed days.  Cold starts fall back to available history and finally
    to Attachment 1's typical profile.
    """
    load = np.asarray(actual_load_kw, dtype=float)
    pv = np.asarray(actual_pv_kw, dtype=float)
    prior_l = np.asarray(typical_load_kw, dtype=float)
    prior_v = np.asarray(typical_pv_kw, dtype=float)
    if load.ndim != 2 or pv.shape != load.shape or prior_l.shape != (load.shape[1],) or prior_v.shape != prior_l.shape:
        raise ValueError("周期预测输入维度不一致")
    if load_model not in ("F1", "F2"):
        raise ValueError("load_model必须是F1或F2")
    n_days, _ = load.shape
    pred_l = np.empty_like(load)
    pred_v = np.empty_like(pv)
    for d in range(n_days):
        weekly = [d - lag for lag in ((7,) if load_model == "F1" else (7, 14, 21)) if d >= lag]
        if weekly:
            pred_l[d] = load[weekly[0]] if load_model == "F1" else np.median(load[weekly], axis=0)
        elif d:
            pred_l[d] = np.mean(load[:d], axis=0)
        else:
            pred_l[d] = prior_l
        pv_start = max(0, d - 7)
        pred_v[d] = np.mean(pv[pv_start:d], axis=0) if d else prior_v
    pred_l = np.maximum(pred_l, 0.0)
    pred_v = np.maximum(pred_v, 0.0)
    return pred_l, load - pred_l, pred_v, pv - pred_v


def build_forecast_archive(data: Inputs) -> dict[str, dict[str, Any]]:
    """Return independently versioned F0/F1/F2 forecasts and residuals."""
    f0_l, f0_lr = build_analog_forecasts(
        data.actual_load_kw, data.typical_load_kw, weekday_weight=True
    )
    f0_v, f0_vr = build_analog_forecasts(
        data.actual_pv_kw, data.typical_pv_kw, weekday_weight=False
    )
    archive: dict[str, dict[str, Any]] = {
        "F0": {
            "load_kw": f0_l, "pv_kw": f0_v,
            "load_residual_kw": f0_lr, "pv_residual_kw": f0_vr,
            "description": "旧版90日指数衰减相似日基线",
        }
    }
    for name in ("F1", "F2"):
        lp, lr, vp, vr = build_periodic_forecasts(
            data.actual_load_kw, data.actual_pv_kw, data.typical_load_kw,
            data.typical_pv_kw, load_model=name,
        )
        archive[name] = {
            "load_kw": lp, "pv_kw": vp,
            "load_residual_kw": lr, "pv_residual_kw": vr,
            "description": (
                "负荷d-7同槽、光伏前7个完成日同槽均值" if name == "F1"
                else "负荷d-7/d-14/d-21同槽中位数、光伏前7个完成日同槽均值"
            ),
        }
    for item in archive.values():
        item["net_kw"] = item["load_kw"] - item["pv_kw"]
        item["net_residual_kw"] = (
            item["load_residual_kw"] - item["pv_residual_kw"]
        )
    return archive


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


def _residual_scenarios(
    load_residual_kw: np.ndarray,
    pv_residual_kw: np.ndarray,
    *,
    risk_mode: str,
    max_scenarios: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return either K-means representatives or all historical residual paths."""
    load_residual_kw = np.asarray(load_residual_kw, dtype=float)
    pv_residual_kw = np.asarray(pv_residual_kw, dtype=float)
    if load_residual_kw.ndim != 2 or pv_residual_kw.shape != load_residual_kw.shape:
        raise ValueError("负荷与光伏残差必须是同形二维矩阵")
    if risk_mode == "clustered":
        return _cluster_residuals(
            load_residual_kw, pv_residual_kw, max_scenarios=max_scenarios
        )
    if risk_mode != "unclustered":
        raise ValueError("risk_mode必须是'clustered'或'unclustered'")
    n = load_residual_kw.shape[0]
    if n == 0:
        zeros = np.zeros((1, load_residual_kw.shape[1]), dtype=float)
        return zeros, zeros.copy(), np.ones(1)
    return load_residual_kw.copy(), pv_residual_kw.copy(), np.full(n, 1.0 / n)


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


def empirical_quantile_columns(values: np.ndarray, alpha: float) -> np.ndarray:
    """Columnwise ceil(alpha*n) order statistic, without interpolation."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or not 0 <= alpha <= 1 or not np.isfinite(x).all():
        raise ValueError("经验分位数输入非法")
    if x.shape[0] == 0:
        return np.zeros(x.shape[1], dtype=float)
    rank = max(1, int(np.ceil(alpha * x.shape[0]))) - 1
    return np.sort(x, axis=0, kind="stable")[rank]


def safe_analog_trajectory(
    day: int,
    load_point_kw: np.ndarray,
    pv_point_kw: np.ndarray,
    load_residual_kw: np.ndarray,
    pv_residual_kw: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
    history_days: int = DEFAULT_HISTORY_DAYS,
    max_scenarios: int = 9,
    risk_mode: str = DEFAULT_RISK_MODE,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    start = max(0, day - history_days)
    residual_l = np.asarray(load_residual_kw[start:day], dtype=float)
    residual_v = np.asarray(pv_residual_kw[start:day], dtype=float)
    net_residual = residual_l - residual_v
    if risk_mode == "unclustered":
        correction = empirical_quantile_columns(net_residual, alpha)
        scenario_count = net_residual.shape[0] if net_residual.shape[0] else 1
        weights = np.full(scenario_count, 1.0 / scenario_count)
    else:
        c_l, c_v, weights = _residual_scenarios(
            residual_l, residual_v, risk_mode=risk_mode, max_scenarios=max_scenarios,
        )
        correction = weighted_quantile_columns(c_l - c_v, weights, alpha)
        scenario_count = weights.size
    point_net = np.asarray(load_point_kw, dtype=float) - np.asarray(pv_point_kw, dtype=float)
    safe_net = point_net + correction
    safe_pv = np.maximum.reduce([np.asarray(pv_point_kw, dtype=float), -safe_net, np.zeros_like(safe_net)])
    safe_load = safe_pv + safe_net
    return safe_load, safe_pv, {
        "alpha": alpha,
        "history_start": start,
        "history_end_exclusive": day,
        "risk_mode": risk_mode,
        "residual_count": int(residual_l.shape[0]),
        "scenario_count": int(scenario_count),
        "scenario_weights": weights.tolist(),
        "net_residual_quantile_kw": correction.tolist(),
        "load_point_kw": np.asarray(load_point_kw, dtype=float).tolist(),
        "pv_point_kw": np.asarray(pv_point_kw, dtype=float).tolist(),
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
        "simultaneous_flow_slots": int(np.count_nonzero(np.minimum(x[b["charge"]], x[b["discharge"]]) > 1e-5)),
        "simultaneous_flow_max": float(np.max(np.minimum(x[b["charge"]], x[b["discharge"]]))),
        "status": "optimal",
    }
    if adjusted:
        result["up_kwh"] = x[b["up"]].copy()
        result["down_kwh"] = x[b["down"]].copy()
    return result


def solve_risk_contract_lp(
    load_point_kw: np.ndarray,
    pv_point_kw: np.ndarray,
    price: np.ndarray,
    soc0: float,
    historical_net_residual_kw: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
    scenario_weights: np.ndarray | None = None,
    terminal_value: float = 0.0,
    battery: Battery = DEFAULT_BATTERY,
    emergency_multiplier: float = 5.0,
    history_start: int | None = None,
    history_end_exclusive: int | None = None,
    base_plan_kwh: np.ndarray | None = None,
) -> dict[str, Any]:
    """Solve a causal initial or adjusted risk-contract LP.

    ``base_contract_kwh`` balances the centre forecast together with the
    planned battery trajectory.  ``risk_reserve_kwh`` is an already-paid
    reserve contract calibrated only from completed historical residuals.
    Historical shortfall variables price the remaining tail at the fivefold
    emergency tariff; they are planning proxies and never enter cash
    settlement.  When ``base_plan_kwh`` is supplied, the total contract is
    settled against that original plan with the Problem 3 upward/downward
    adjustment tariff; the same residual reserve and shortfall construction
    is retained.
    """
    load = np.asarray(load_point_kw, dtype=float)
    pv = np.asarray(pv_point_kw, dtype=float)
    price = np.asarray(price, dtype=float)
    residual = np.asarray(historical_net_residual_kw, dtype=float)
    n_slots = load.size
    if pv.shape != (n_slots,) or price.shape != (n_slots,):
        raise ValueError("风险合同LP的负荷、光伏和电价长度不一致")
    if residual.ndim != 2 or residual.shape[1] != n_slots:
        raise ValueError("历史净负荷残差必须是样本数×时段数二维矩阵")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("风险分位水平必须位于[0,1]")
    if any(not np.isfinite(v).all() for v in (load, pv, price, residual)):
        raise ValueError("风险合同LP输入包含NaN或无穷值")
    if np.any(load < 0) or np.any(pv < 0) or np.any(price < 0):
        raise ValueError("风险合同LP的预测量和电价不得为负")
    if not battery.minimum - 1e-9 <= soc0 <= battery.maximum + 1e-9:
        raise ValueError(f"初始SOC越界: {soc0}")
    adjusted = base_plan_kwh is not None
    if adjusted:
        base_plan_kwh = np.asarray(base_plan_kwh, dtype=float)
        if base_plan_kwh.shape != (n_slots,) or not np.isfinite(base_plan_kwh).all():
            raise ValueError("原合同必须是与调整时域等长的有限向量")
        if np.any(base_plan_kwh < -1e-7):
            raise ValueError("原合同不得为负")
        base_plan_kwh = np.maximum(base_plan_kwh, 0.0)

    sample_count = residual.shape[0]
    if scenario_weights is None:
        weights = (
            np.full(sample_count, 1.0 / sample_count)
            if sample_count
            else np.empty(0, dtype=float)
        )
    else:
        weights = np.asarray(scenario_weights, dtype=float)
        if weights.shape != (sample_count,) or np.any(weights < 0) or (sample_count and weights.sum() <= 0):
            raise ValueError("历史残差场景权重非法")
        weights = weights / weights.sum() if sample_count else weights

    # Variable blocks: base, reserve, charge, discharge, SOC, PV used, z,
    # and (for an adjusted contract) upward/downward changes from the original.
    pos = 0
    blocks: dict[str, slice] = {}
    for name, size in (
        ("base", n_slots), ("reserve", n_slots),
        ("charge", n_slots), ("discharge", n_slots),
        ("soc", n_slots + 1), ("pv_used", n_slots),
        ("shortfall", sample_count * n_slots),
    ):
        blocks[name] = slice(pos, pos + size)
        pos += size
    if adjusted:
        for name in ("up", "down"):
            blocks[name] = slice(pos, pos + n_slots)
            pos += n_slots

    c = np.zeros(pos, dtype=float)
    if adjusted:
        c[blocks["up"]] = 1.5 * price
        c[blocks["down"]] = -0.5 * price
        constant_cost = float(np.dot(price, base_plan_kwh))
    else:
        c[blocks["base"]] = price
        c[blocks["reserve"]] = price
        constant_cost = 0.0
    if sample_count:
        c[blocks["shortfall"]] = (
            emergency_multiplier * weights[:, None] * price[None, :]
        ).ravel()
    c[blocks["soc"].stop - 1] = -float(terminal_value)

    aeq = lil_matrix((2 * n_slots + (n_slots if adjusted else 0), pos), dtype=float)
    beq = np.zeros(aeq.shape[0], dtype=float)
    load_e = load * DT_HOURS
    pv_e = pv * DT_HOURS
    for t in range(n_slots):
        aeq[t, blocks["base"].start + t] = 1.0
        aeq[t, blocks["charge"].start + t] = -1.0
        aeq[t, blocks["discharge"].start + t] = 1.0
        aeq[t, blocks["pv_used"].start + t] = 1.0
        beq[t] = load_e[t]

        row = n_slots + t
        aeq[row, blocks["soc"].start + t] = -1.0
        aeq[row, blocks["soc"].start + t + 1] = 1.0
        aeq[row, blocks["charge"].start + t] = -battery.eta_c
        aeq[row, blocks["discharge"].start + t] = 1.0 / battery.eta_d

        if adjusted:
            row = 2 * n_slots + t
            # base + reserve = original + upward - downward
            aeq[row, blocks["base"].start + t] = 1.0
            aeq[row, blocks["reserve"].start + t] = 1.0
            aeq[row, blocks["up"].start + t] = -1.0
            aeq[row, blocks["down"].start + t] = 1.0
            beq[row] = base_plan_kwh[t]

    if sample_count:
        aub = lil_matrix((sample_count * n_slots, pos), dtype=float)
        bub = np.empty(sample_count * n_slots, dtype=float)
        residual_e = residual * DT_HOURS
        for j in range(sample_count):
            for t in range(n_slots):
                row = j * n_slots + t
                # z[j,t] >= residual[j,t] - reserve[t]
                aub[row, blocks["reserve"].start + t] = -1.0
                aub[row, blocks["shortfall"].start + row] = -1.0
                bub[row] = -residual_e[j, t]
        aub_csr = aub.tocsr()
    else:
        residual_e = np.empty((0, n_slots), dtype=float)
        aub_csr = None
        bub = None

    if sample_count:
        risk_floor = np.maximum(
            0.0, weighted_quantile_columns(residual, weights, alpha) * DT_HOURS
        )
    else:
        risk_floor = np.zeros(n_slots, dtype=float)
    bounds: list[tuple[float | None, float | None]] = []
    bounds.extend([(0.0, None)] * n_slots)
    bounds.extend([(float(v), None) for v in risk_floor])
    bounds.extend([(0.0, battery.slot_limit)] * n_slots)
    bounds.extend([(0.0, battery.slot_limit)] * n_slots)
    soc_bounds = [(battery.minimum, battery.maximum)] * (n_slots + 1)
    soc_bounds[0] = (soc0, soc0)
    bounds.extend(soc_bounds)
    bounds.extend([(0.0, float(v)) for v in pv_e])
    bounds.extend([(0.0, None)] * (sample_count * n_slots))
    if adjusted:
        bounds.extend([(0.0, None)] * n_slots)
        bounds.extend([(0.0, float(v)) for v in base_plan_kwh])

    solved = linprog(
        c, A_ub=aub_csr, b_ub=bub, A_eq=aeq.tocsr(), b_eq=beq,
        bounds=bounds, method="highs",
    )
    if not solved.success:
        raise RuntimeError(f"风险合同LP求解失败: status={solved.status}, {solved.message}")

    economic_optimum = float(solved.fun)
    tie = np.zeros(pos, dtype=float)
    tie[blocks["charge"]] = 1.0
    tie[blocks["discharge"]] = 1.0
    objective_row = csr_matrix(c.reshape(1, -1))
    tolerance = 1e-8 * max(1.0, abs(economic_optimum))
    if aub_csr is None:
        tie_aub = objective_row
        tie_bub = np.asarray([economic_optimum + tolerance])
    else:
        from scipy.sparse import vstack
        tie_aub = vstack([aub_csr, objective_row], format="csr")
        tie_bub = np.concatenate([bub, [economic_optimum + tolerance]])
    refined = linprog(
        tie, A_ub=tie_aub, b_ub=tie_bub, A_eq=aeq.tocsr(), b_eq=beq,
        bounds=bounds, method="highs",
    )
    if not refined.success:
        raise RuntimeError(f"风险合同LP吞吐次级优化失败: {refined.status}, {refined.message}")
    x = refined.x

    base = x[blocks["base"]].copy()
    reserve = x[blocks["reserve"]].copy()
    shortfall = x[blocks["shortfall"]].reshape(sample_count, n_slots).copy()
    soc = x[blocks["soc"]].copy()
    total_contract = base + reserve
    if adjusted:
        up = x[blocks["up"]].copy()
        down = x[blocks["down"]].copy()
        contract_cost = float(
            np.dot(price, base_plan_kwh)
            + np.dot(1.5 * price, up)
            - np.dot(0.5 * price, down)
        )
        base_only_cost = float(np.sum(
            price * np.minimum(base_plan_kwh, base)
            + 1.5 * price * np.maximum(base - base_plan_kwh, 0.0)
            + 0.5 * price * np.maximum(base_plan_kwh - base, 0.0)
        ))
        base_cost = base_only_cost
        reserve_cost = contract_cost - base_only_cost
    else:
        up = down = None
        base_cost = float(np.dot(price, base))
        reserve_cost = float(np.dot(price, reserve))
        contract_cost = base_cost + reserve_cost
    expected_emergency_cost = float(
        emergency_multiplier * np.sum(weights[:, None] * price[None, :] * shortfall)
    ) if sample_count else 0.0
    terminal_credit = float(terminal_value * (soc[-1] - battery.minimum))
    objective = contract_cost + expected_emergency_cost - terminal_credit
    normalized_highs_objective = float(
        np.dot(c, x) + constant_cost + terminal_value * battery.minimum
    )
    normalized_economic_optimum = float(
        economic_optimum + constant_cost + terminal_value * battery.minimum
    )
    eq_residual = aeq.tocsr() @ x - beq
    overlap = np.minimum(x[blocks["charge"]], x[blocks["discharge"]])
    if sample_count:
        shortfall_violation = np.maximum(
            residual_e - reserve[None, :] - shortfall, 0.0
        )
        max_shortfall_violation = float(np.max(shortfall_violation))
    else:
        max_shortfall_violation = 0.0
    result = {
        "grid_kwh": total_contract,
        "base_contract_kwh": base,
        "risk_reserve_kwh": reserve,
        "risk_reserve_lower_bound_kwh": risk_floor,
        "charge_kwh": x[blocks["charge"]].copy(),
        "discharge_kwh": x[blocks["discharge"]].copy(),
        "soc_kwh": soc,
        "pv_used_kwh": x[blocks["pv_used"]].copy(),
        "historical_shortfall_kwh": shortfall,
        "base_contract_cost": base_cost,
        "cash_contract_cost": contract_cost,
        "risk_reserve_cost": reserve_cost,
        "expected_emergency_cost": expected_emergency_cost,
        "risk_cost": reserve_cost + expected_emergency_cost,
        "terminal_value_credit": terminal_credit,
        "solver_objective": objective,
        "highs_objective": normalized_highs_objective,
        "economic_optimum": normalized_economic_optimum,
        "secondary_tie_break_gap": float(
            normalized_highs_objective - normalized_economic_optimum
        ),
        "objective_solver_residual": float(
            objective - normalized_highs_objective
        ),
        "objective_decomposition_residual": float(
            objective - (base_cost + reserve_cost + expected_emergency_cost - terminal_credit)
        ),
        "max_eq_residual": float(np.max(np.abs(eq_residual))),
        "max_shortfall_violation": max_shortfall_violation,
        "simultaneous_flow_slots": int(np.count_nonzero(overlap > 1e-5)),
        "simultaneous_flow_max": float(np.max(overlap)),
        "historical_sample_count": int(sample_count),
        "history_start": history_start,
        "history_end_exclusive": history_end_exclusive,
        "scenario_weights": weights.tolist(),
        "alpha": float(alpha),
        "status": "optimal",
        "model": "risk_contract_lp_adjusted" if adjusted else "risk_contract_lp",
        "adjusted": adjusted,
    }
    if adjusted:
        result["up_kwh"] = up
        result["down_kwh"] = down
        result["original_contract_kwh"] = base_plan_kwh.copy()
    return result


def greedy_execution(
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
    """Execute the fixed contract with current-slot state feedback only.

    Source priority is explicit: PV serves load, then the paid contract; PV
    surplus charges before contract surplus; the battery discharges only for a
    remaining load deficit; emergency power covers the final deficit.
    """
    begun = time.perf_counter()
    q = np.asarray(contract_kwh, dtype=float)
    load_e = np.asarray(actual_load_kw, dtype=float) * DT_HOURS
    pv_e = np.asarray(actual_pv_kw, dtype=float) * DT_HOURS
    price = np.asarray(price, dtype=float)
    n = q.size
    if load_e.shape != (n,) or pv_e.shape != (n,) or price.shape != (n,):
        raise ValueError("因果控制输入长度不一致")
    if any(not np.isfinite(v).all() for v in (q, load_e, pv_e, price)) or np.any(q < -1e-7) or np.any(load_e < 0) or np.any(pv_e < 0) or np.any(price < 0):
        raise ValueError("反馈执行输入必须有限且非负")
    # LP solvers can return a nominally non-negative contract at roughly
    # -1e-10 because of feasibility tolerances.  Treat only that numerical
    # dust as zero; materially negative contracts remain invalid above.
    q = np.maximum(q, 0.0)
    if not battery.minimum - 1e-9 <= soc0 <= battery.maximum + 1e-9:
        raise ValueError("反馈执行初始SOC越界")
    charge = np.zeros(n)
    discharge = np.zeros(n)
    emergency = np.zeros(n)
    unused = np.zeros(n)
    curtailed = np.zeros(n)
    pv_used = np.zeros(n)
    pv_to_load = np.zeros(n)
    contract_to_load = np.zeros(n)
    pv_to_charge = np.zeros(n)
    contract_to_charge = np.zeros(n)
    soc = np.empty(n + 1)
    soc[0] = soc0
    for t in range(n):
        pv_to_load[t] = min(pv_e[t], load_e[t])
        load_after_pv = load_e[t] - pv_to_load[t]
        contract_to_load[t] = min(q[t], load_after_pv)
        deficit = load_after_pv - contract_to_load[t]
        remaining_pv = pv_e[t] - pv_to_load[t]
        remaining_contract = q[t] - contract_to_load[t]
        if deficit <= 1e-12:
            room = min(
                battery.slot_limit,
                max(0.0, (battery.maximum - soc[t]) / battery.eta_c),
            )
            pv_to_charge[t] = min(remaining_pv, room)
            contract_to_charge[t] = min(remaining_contract, room - pv_to_charge[t])
            charge[t] = pv_to_charge[t] + contract_to_charge[t]
            curtailed[t] = remaining_pv - pv_to_charge[t]
            unused[t] = remaining_contract - contract_to_charge[t]
            pv_used[t] = pv_to_load[t] + pv_to_charge[t]
            soc[t + 1] = soc[t] + battery.eta_c * charge[t]
        else:
            discharge[t] = min(
                deficit, battery.slot_limit,
                max(0.0, (soc[t] - battery.minimum) * battery.eta_d),
            )
            emergency[t] = max(0.0, deficit - discharge[t])
            pv_used[t] = pv_to_load[t]
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
        # Keep the execution-result schema aligned with rolling MPC.  The
        # greedy baseline chooses from the current observation and executes
        # that feasible action immediately, so its planned and executed
        # actions are identical by construction.
        "planned_charge_kwh": charge.copy(),
        "planned_discharge_kwh": discharge.copy(),
        "soc_kwh": soc,
        "emergency_kwh": emergency,
        "unused_contract_kwh": unused,
        "pv_curtailed_kwh": curtailed,
        "pv_used_kwh": pv_used,
        "pv_to_load_kwh": pv_to_load,
        "contract_to_load_kwh": contract_to_load,
        "pv_to_charge_kwh": pv_to_charge,
        "contract_to_charge_kwh": contract_to_charge,
        "settlement_cost": settlement,
        "alternative_settlement_cost": alt_settlement,
        "emergency_cost": emergency_cost,
        "cash_cost": settlement + emergency_cost,
        "alternative_cash_cost": alt_settlement + emergency_cost,
        "max_balance_residual": float(np.max(np.abs(residual))),
        "simultaneous_flow_slots": int(np.count_nonzero(np.minimum(charge, discharge) > 1e-5)),
        "simultaneous_flow_max": float(np.max(np.minimum(charge, discharge))),
        "controller": "greedy_feedback",
        "runtime_seconds": float(time.perf_counter() - begun),
        "controller_diagnostics": {
            "solve_count": 0, "fallback_count": 0, "clip_count": 0,
            "total_seconds": float(time.perf_counter() - begun),
        },
    }


def causal_dispatch(
    contract_kwh: np.ndarray, actual_load_kw: np.ndarray,
    actual_pv_kw: np.ndarray, price: np.ndarray, soc0: float, *,
    base_plan_kwh: np.ndarray | None = None,
    battery: Battery = DEFAULT_BATTERY, emergency_multiplier: float = 5.0,
) -> dict[str, Any]:
    """Compatibility alias for the formal feedback-greedy executor."""
    return greedy_execution(
        contract_kwh, actual_load_kw, actual_pv_kw, price, soc0,
        base_plan_kwh=base_plan_kwh, battery=battery,
        emergency_multiplier=emergency_multiplier,
    )


def _mpc_blocks(n: int, tracking: bool) -> dict[str, slice]:
    pos = 0
    blocks: dict[str, slice] = {}
    for name, size in (
        ("charge", n), ("discharge", n), ("emergency", n),
        ("unused", n), ("pv_used", n), ("soc", n + 1),
    ):
        blocks[name] = slice(pos, pos + size)
        pos += size
    if tracking:
        blocks["tracking"] = slice(pos, pos + n)
        pos += n
    blocks["all"] = slice(0, pos)
    return blocks


def solve_mpc_horizon(
    contract_kwh: np.ndarray,
    forecast_load_kw: np.ndarray,
    forecast_pv_kw: np.ndarray,
    forecast_price: np.ndarray,
    soc0: float,
    *,
    target_soc_kwh: np.ndarray | None = None,
    emergency_multiplier: float = 5.0,
    battery: Battery = DEFAULT_BATTERY,
    config: MpcConfig = MpcConfig(),
) -> dict[str, Any]:
    """Solve one short-horizon execution LP with a fixed signed contract.

    Emergency energy is capped by the forecast load deficit.  Consequently it
    cannot be used to charge the battery or manufacture terminal-value credit.
    """
    q = np.asarray(contract_kwh, dtype=float)
    load_e = np.asarray(forecast_load_kw, dtype=float) * DT_HOURS
    pv_e = np.asarray(forecast_pv_kw, dtype=float) * DT_HOURS
    price = np.asarray(forecast_price, dtype=float)
    n = q.size
    if n == 0 or load_e.shape != (n,) or pv_e.shape != (n,) or price.shape != (n,):
        raise ValueError("MPC输入长度不一致或时域为空")
    if np.any(q < 0) or np.any(load_e < 0) or np.any(pv_e < 0) or np.any(price < 0):
        raise ValueError("MPC输入不得为负")
    if not battery.minimum - 1e-9 <= soc0 <= battery.maximum + 1e-9:
        raise ValueError("MPC初始SOC越界")
    tracking = target_soc_kwh is not None and config.tracking_weight > 0
    if tracking:
        target = np.asarray(target_soc_kwh, dtype=float)
        if target.shape != (n,):
            raise ValueError("目标SOC应给出每个预测槽末的状态")
    else:
        target = np.empty(0)

    b = _mpc_blocks(n, tracking)
    c = np.zeros(b["all"].stop, dtype=float)
    c[b["emergency"]] = emergency_multiplier * price
    lam = terminal_value(price, battery)
    c[b["soc"].stop - 1] = -lam
    if tracking:
        c[b["tracking"]] = config.tracking_weight

    aeq = lil_matrix((2 * n, b["all"].stop), dtype=float)
    beq = np.zeros(2 * n, dtype=float)
    for t in range(n):
        # q-w+v+y+u-L-x=0
        aeq[t, b["charge"].start + t] = -1.0
        aeq[t, b["discharge"].start + t] = 1.0
        aeq[t, b["emergency"].start + t] = 1.0
        aeq[t, b["unused"].start + t] = -1.0
        aeq[t, b["pv_used"].start + t] = 1.0
        beq[t] = load_e[t] - q[t]
        row = n + t
        aeq[row, b["charge"].start + t] = -battery.eta_c
        aeq[row, b["discharge"].start + t] = 1.0 / battery.eta_d
        aeq[row, b["soc"].start + t] = -1.0
        aeq[row, b["soc"].start + t + 1] = 1.0

    aub_rows = 2 * n if tracking else 0
    aub = lil_matrix((aub_rows, b["all"].stop), dtype=float) if tracking else None
    bub = np.zeros(aub_rows, dtype=float) if tracking else None
    if tracking:
        for t in range(n):
            # |S_{t+1}-target_t| <= a_t
            aub[2 * t, b["soc"].start + t + 1] = 1.0
            aub[2 * t, b["tracking"].start + t] = -1.0
            bub[2 * t] = target[t]
            aub[2 * t + 1, b["soc"].start + t + 1] = -1.0
            aub[2 * t + 1, b["tracking"].start + t] = -1.0
            bub[2 * t + 1] = -target[t]

    r_plus = np.maximum(q + pv_e - load_e, 0.0)
    r_minus = np.maximum(load_e - q - pv_e, 0.0)
    bounds: list[tuple[float | None, float | None]] = []
    bounds.extend((0.0, min(battery.slot_limit, float(v))) for v in r_plus)
    bounds.extend([(0.0, battery.slot_limit)] * n)
    bounds.extend((0.0, float(v)) for v in r_minus)
    bounds.extend((0.0, float(v)) for v in q)
    bounds.extend((0.0, float(v)) for v in pv_e)
    soc_bounds = [(battery.minimum, battery.maximum)] * (n + 1)
    soc_bounds[0] = (soc0, soc0)
    bounds.extend(soc_bounds)
    if tracking:
        bounds.extend([(0.0, None)] * n)

    options = {"time_limit": float(config.solve_budget_seconds)}
    started = time.perf_counter()
    res = linprog(
        c, A_ub=None if aub is None else aub.tocsr(), b_ub=bub,
        A_eq=aeq.tocsr(), b_eq=beq, bounds=bounds, method="highs", options=options,
    )
    if not res.success:
        raise RuntimeError(f"MPC主LP失败: status={res.status}, {res.message}")
    economic_optimum = float(res.fun)
    elapsed = time.perf_counter() - started
    remaining = max(1e-3, config.solve_budget_seconds - elapsed)
    tie = np.zeros_like(c)
    tie[b["charge"]] = 1.0
    tie[b["discharge"]] = 1.0
    economic_row = csr_matrix(c.reshape(1, -1))
    if aub is None:
        aub2 = economic_row
        bub2 = np.asarray([economic_optimum + 1e-8 * max(1.0, abs(economic_optimum))])
    else:
        from scipy.sparse import vstack
        aub2 = vstack([aub.tocsr(), economic_row], format="csr")
        bub2 = np.concatenate([bub, [economic_optimum + 1e-8 * max(1.0, abs(economic_optimum))]])
    res2 = linprog(
        tie, A_ub=aub2, b_ub=bub2, A_eq=aeq.tocsr(), b_eq=beq,
        bounds=bounds, method="highs", options={"time_limit": remaining},
    )
    if not res2.success:
        raise RuntimeError(f"MPC吞吐次级LP失败: status={res2.status}, {res2.message}")
    elapsed = time.perf_counter() - started
    x = res2.x
    residual = aeq.tocsr() @ x - beq
    overlap = np.minimum(x[b["charge"]], x[b["discharge"]])
    result = {
        "charge_kwh": x[b["charge"]].copy(),
        "discharge_kwh": x[b["discharge"]].copy(),
        "emergency_kwh": x[b["emergency"]].copy(),
        "unused_contract_kwh": x[b["unused"]].copy(),
        "pv_used_kwh": x[b["pv_used"]].copy(),
        "soc_kwh": x[b["soc"]].copy(),
        "terminal_value_credit": float(lam * (x[b["soc"].stop - 1] - battery.minimum)),
        "max_eq_residual": float(np.max(np.abs(residual))),
        "simultaneous_flow_slots": int(np.count_nonzero(overlap > config.overlap_tolerance)),
        "simultaneous_flow_max": float(np.max(overlap)),
        "solve_seconds": float(elapsed),
        "status": "optimal",
    }
    if elapsed > config.solve_budget_seconds + 0.05:
        raise RuntimeError(f"MPC求解超出预算: {elapsed:.3f}s")
    if result["max_eq_residual"] > config.residual_tolerance:
        raise RuntimeError(f"MPC等式残差超限: {result['max_eq_residual']:.3e}")
    if result["simultaneous_flow_max"] > config.overlap_tolerance:
        raise RuntimeError(f"MPC同时充放电超限: {result['simultaneous_flow_max']:.3e}")
    return result


def _delayed_greedy_action(
    contract_kwh: float, forecast_load_kw: float, forecast_pv_kw: float,
    soc: float, battery: Battery,
) -> tuple[float, float]:
    available = contract_kwh + forecast_pv_kw * DT_HOURS - forecast_load_kw * DT_HOURS
    if available >= 0:
        charge = min(available, battery.slot_limit, max(0.0, (battery.maximum - soc) / battery.eta_c))
        return float(charge), 0.0
    discharge = min(-available, battery.slot_limit, max(0.0, (soc - battery.minimum) * battery.eta_d))
    return 0.0, float(discharge)


def clip_planned_action(
    contract_kwh: float, actual_load_kw: float, actual_pv_kw: float, soc: float,
    planned_charge_kwh: float, planned_discharge_kwh: float,
    *, battery: Battery = DEFAULT_BATTERY,
) -> dict[str, float]:
    """Shrink a planned action against the current truth without re-optimising it."""
    load_e = float(actual_load_kw) * DT_HOURS
    pv_e = float(actual_pv_kw) * DT_HOURS
    charge = min(
        max(0.0, float(planned_charge_kwh)),
        max(float(contract_kwh) + pv_e - load_e, 0.0),
        battery.slot_limit,
        max(0.0, (battery.maximum - soc) / battery.eta_c),
    )
    discharge = min(
        max(0.0, float(planned_discharge_kwh)),
        load_e + charge,
        battery.slot_limit,
        max(0.0, (soc - battery.minimum) * battery.eta_d),
    )
    if min(charge, discharge) > 1e-7:
        raise RuntimeError("计划裁剪后仍同时充放电")
    balance_before_settlement = load_e + charge - discharge - float(contract_kwh) - pv_e
    if balance_before_settlement >= 0:
        emergency = balance_before_settlement
        unused = 0.0
        pv_used = pv_e
    else:
        emergency = 0.0
        surplus = -balance_before_settlement
        unused = min(float(contract_kwh), surplus)
        pv_used = pv_e - (surplus - unused)
    pv_used = min(pv_e, max(0.0, pv_used))
    next_soc = soc + battery.eta_c * charge - discharge / battery.eta_d
    residual = float(contract_kwh) - unused + pv_used + discharge + emergency - load_e - charge
    return {
        "charge_kwh": charge, "discharge_kwh": discharge,
        "emergency_kwh": emergency, "unused_contract_kwh": unused,
        "pv_used_kwh": pv_used, "pv_curtailed_kwh": pv_e - pv_used,
        "soc_next_kwh": next_soc, "balance_residual_kwh": residual,
    }


def rolling_mpc_dispatch(
    contract_kwh: np.ndarray, actual_load_kw: np.ndarray, actual_pv_kw: np.ndarray,
    settlement_price: np.ndarray, soc0: float, *,
    forecast_load_kw: np.ndarray, forecast_pv_kw: np.ndarray,
    forecast_price: np.ndarray | None = None, base_plan_kwh: np.ndarray | None = None,
    target_soc_kwh: np.ndarray | None = None, battery: Battery = DEFAULT_BATTERY,
    emergency_multiplier: float = 5.0, config: MpcConfig = MpcConfig(),
) -> dict[str, Any]:
    """Re-solve a causal LP every ten minutes and execute only its first action."""
    q = np.asarray(contract_kwh, dtype=float)
    load = np.asarray(actual_load_kw, dtype=float)
    pv = np.asarray(actual_pv_kw, dtype=float)
    price = np.asarray(settlement_price, dtype=float)
    fl = np.asarray(forecast_load_kw, dtype=float)
    fv = np.asarray(forecast_pv_kw, dtype=float)
    fp = price if forecast_price is None else np.asarray(forecast_price, dtype=float)
    n = q.size
    execute_n = load.size
    if execute_n == 0 or execute_n > n or pv.shape != (execute_n,) or price.shape != (execute_n,) or any(a.shape != (n,) for a in (fl, fv, fp)):
        raise ValueError("滚动MPC输入长度不一致")
    if base_plan_kwh is not None and np.asarray(base_plan_kwh).shape != (n,):
        raise ValueError("原合同长度不一致")
    if target_soc_kwh is not None and np.asarray(target_soc_kwh).shape not in ((n,), (n + 1,)):
        raise ValueError("目标SOC长度不一致")

    names = ("charge_kwh", "discharge_kwh", "emergency_kwh",
             "unused_contract_kwh", "pv_curtailed_kwh", "pv_used_kwh")
    out = {name: np.zeros(execute_n) for name in names}
    planned_charge = np.zeros(execute_n)
    planned_discharge = np.zeros(execute_n)
    soc = np.empty(execute_n + 1); soc[0] = soc0
    solve_times: list[float] = []
    failures: dict[str, int] = {}
    clip_count = 0
    for t in range(execute_n):
        h = min(config.horizon_slots, n - t)
        if target_soc_kwh is None:
            target = None
        else:
            raw_target = np.asarray(target_soc_kwh, dtype=float)
            target = raw_target[t + 1:t + h + 1] if raw_target.size == n + 1 else raw_target[t:t + h]
        try:
            solution = solve_mpc_horizon(
                q[t:t + h], fl[t:t + h], fv[t:t + h], fp[t:t + h], soc[t],
                target_soc_kwh=target, emergency_multiplier=emergency_multiplier,
                battery=battery, config=config,
            )
            x_m = float(solution["charge_kwh"][0])
            y_m = float(solution["discharge_kwh"][0])
            solve_times.append(float(solution["solve_seconds"]))
        except (RuntimeError, ValueError) as exc:
            reason = str(exc).split(":", 1)[0]
            failures[reason] = failures.get(reason, 0) + 1
            x_m, y_m = _delayed_greedy_action(q[t], fl[t], fv[t], soc[t], battery)
        planned_charge[t], planned_discharge[t] = x_m, y_m
        executed = clip_planned_action(q[t], load[t], pv[t], soc[t], x_m, y_m, battery=battery)
        if abs(executed["charge_kwh"] - x_m) > 1e-8 or abs(executed["discharge_kwh"] - y_m) > 1e-8:
            clip_count += 1
        for name in names:
            out[name][t] = executed[name]
        soc[t + 1] = executed["soc_next_kwh"]

    if base_plan_kwh is None:
        settlement = price * q[:execute_n]
        alt_settlement = settlement.copy()
    else:
        p = np.asarray(base_plan_kwh, dtype=float)[:execute_n]
        q_exec = q[:execute_n]
        settlement = price * np.minimum(p, q_exec) + 1.5 * price * np.maximum(q_exec - p, 0.0) + 0.5 * price * np.maximum(p - q_exec, 0.0)
        alt_settlement = price * p + 1.5 * price * np.maximum(q_exec - p, 0.0) + 0.5 * price * np.maximum(p - q_exec, 0.0)
    emergency_cost = emergency_multiplier * price * out["emergency_kwh"]
    residual = q[:execute_n] - out["unused_contract_kwh"] + out["pv_used_kwh"] + out["discharge_kwh"] + out["emergency_kwh"] - load * DT_HOURS - out["charge_kwh"]
    overlap = np.minimum(out["charge_kwh"], out["discharge_kwh"])
    out.update({
        "soc_kwh": soc, "planned_charge_kwh": planned_charge,
        "planned_discharge_kwh": planned_discharge, "settlement_cost": settlement,
        "alternative_settlement_cost": alt_settlement, "emergency_cost": emergency_cost,
        "cash_cost": settlement + emergency_cost,
        "alternative_cash_cost": alt_settlement + emergency_cost,
        "max_balance_residual": float(np.max(np.abs(residual))),
        "simultaneous_flow_slots": int(np.count_nonzero(overlap > config.overlap_tolerance)),
        "simultaneous_flow_max": float(np.max(overlap)),
        "controller": "rolling_mpc",
        "solver_diagnostics": {
            "solve_count": len(solve_times), "fallback_count": int(sum(failures.values())),
            "fallback_reasons": failures, "clip_count": clip_count,
            "median_seconds": float(np.median(solve_times)) if solve_times else None,
            "p95_seconds": float(np.quantile(solve_times, .95)) if solve_times else None,
            "max_seconds": float(np.max(solve_times)) if solve_times else None,
            "total_seconds": float(np.sum(solve_times)),
            "horizon_slots": config.horizon_slots,
            "tracking_weight": config.tracking_weight,
        },
    })
    return out


def baseline_MPC(
    contract_kwh: np.ndarray, actual_load_kw: np.ndarray,
    actual_pv_kw: np.ndarray, settlement_price: np.ndarray, soc0: float, *,
    forecast_load_kw: np.ndarray, forecast_pv_kw: np.ndarray,
    forecast_price: np.ndarray | None = None,
    base_plan_kwh: np.ndarray | None = None,
    target_soc_kwh: np.ndarray | None = None,
    battery: Battery = DEFAULT_BATTERY, emergency_multiplier: float = 5.0,
    config: MpcConfig = MpcConfig(),
) -> dict[str, Any]:
    """Comparison-only MPC baseline under the same fixed contract."""
    begun = time.perf_counter()
    result = rolling_mpc_dispatch(
        contract_kwh, actual_load_kw, actual_pv_kw, settlement_price, soc0,
        forecast_load_kw=forecast_load_kw, forecast_pv_kw=forecast_pv_kw,
        forecast_price=forecast_price, base_plan_kwh=base_plan_kwh,
        target_soc_kwh=target_soc_kwh, battery=battery,
        emergency_multiplier=emergency_multiplier, config=config,
    )
    result["controller"] = "baseline_MPC"
    result["runtime_seconds"] = float(time.perf_counter() - begun)
    result["solver_diagnostics"]["role"] = "comparison_only"
    return result


def delayed_greedy_dispatch(
    contract_kwh: np.ndarray, actual_load_kw: np.ndarray, actual_pv_kw: np.ndarray,
    settlement_price: np.ndarray, soc0: float, *,
    forecast_load_kw: np.ndarray, forecast_pv_kw: np.ndarray,
    base_plan_kwh: np.ndarray | None = None, battery: Battery = DEFAULT_BATTERY,
    emergency_multiplier: float = 5.0,
) -> dict[str, Any]:
    """Timing-aligned greedy baseline that never sees the current slot in advance.

    The first action uses the causal point forecast available at the decision
    time.  Later actions copy the net load of the latest completed slot.  The
    planned action is then passed through the same one-way clipping and bus
    balancing rule as rolling MPC, so the controller comparison differs only
    in the action policy rather than in information access or feasibility.
    """
    q = np.asarray(contract_kwh, dtype=float)
    load = np.asarray(actual_load_kw, dtype=float)
    pv = np.asarray(actual_pv_kw, dtype=float)
    price = np.asarray(settlement_price, dtype=float)
    fl = np.asarray(forecast_load_kw, dtype=float)
    fv = np.asarray(forecast_pv_kw, dtype=float)
    n = q.size
    if n == 0 or any(a.shape != (n,) for a in (load, pv, price, fl, fv)):
        raise ValueError("延迟贪心输入长度不一致")
    if base_plan_kwh is not None and np.asarray(base_plan_kwh).shape != (n,):
        raise ValueError("原合同长度不一致")

    names = ("charge_kwh", "discharge_kwh", "emergency_kwh",
             "unused_contract_kwh", "pv_curtailed_kwh", "pv_used_kwh")
    out = {name: np.zeros(n) for name in names}
    planned_charge = np.zeros(n)
    planned_discharge = np.zeros(n)
    soc = np.empty(n + 1); soc[0] = soc0
    clip_count = 0
    for t in range(n):
        if t == 0:
            proxy_load, proxy_pv = float(fl[t]), float(fv[t])
        else:
            lagged_net = float(load[t - 1] - pv[t - 1])
            proxy_load, proxy_pv = max(lagged_net, 0.0), max(-lagged_net, 0.0)
        x_m, y_m = _delayed_greedy_action(
            q[t], proxy_load, proxy_pv, soc[t], battery,
        )
        planned_charge[t], planned_discharge[t] = x_m, y_m
        executed = clip_planned_action(
            q[t], load[t], pv[t], soc[t], x_m, y_m, battery=battery,
        )
        if abs(executed["charge_kwh"] - x_m) > 1e-8 or abs(executed["discharge_kwh"] - y_m) > 1e-8:
            clip_count += 1
        for name in names:
            out[name][t] = executed[name]
        soc[t + 1] = executed["soc_next_kwh"]

    if base_plan_kwh is None:
        settlement = price * q
        alt_settlement = settlement.copy()
    else:
        p = np.asarray(base_plan_kwh, dtype=float)
        settlement = price * np.minimum(p, q) + 1.5 * price * np.maximum(q - p, 0.0) + 0.5 * price * np.maximum(p - q, 0.0)
        alt_settlement = price * p + 1.5 * price * np.maximum(q - p, 0.0) + 0.5 * price * np.maximum(p - q, 0.0)
    emergency_cost = emergency_multiplier * price * out["emergency_kwh"]
    residual = q - out["unused_contract_kwh"] + out["pv_used_kwh"] + out["discharge_kwh"] + out["emergency_kwh"] - load * DT_HOURS - out["charge_kwh"]
    overlap = np.minimum(out["charge_kwh"], out["discharge_kwh"])
    out.update({
        "soc_kwh": soc, "planned_charge_kwh": planned_charge,
        "planned_discharge_kwh": planned_discharge,
        "settlement_cost": settlement,
        "alternative_settlement_cost": alt_settlement,
        "emergency_cost": emergency_cost,
        "cash_cost": settlement + emergency_cost,
        "alternative_cash_cost": alt_settlement + emergency_cost,
        "max_balance_residual": float(np.max(np.abs(residual))),
        "simultaneous_flow_slots": int(np.count_nonzero(overlap > 1e-7)),
        "simultaneous_flow_max": float(np.max(overlap)),
        "controller": "delayed_greedy_baseline",
        "controller_diagnostics": {
            "solve_count": 0, "fallback_count": 0,
            "fallback_reasons": {}, "clip_count": clip_count,
            "median_seconds": None, "p95_seconds": None,
            "max_seconds": None, "total_seconds": 0.0,
            "horizon_slots": 1, "tracking_weight": 0.0,
        },
    })
    return out


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


def price_forecast_candidates(data: Inputs, day: int) -> dict[str, np.ndarray]:
    """Causal same-slot price baselines requested by the model review."""
    if day < 0 or day >= len(data.dates):
        raise ValueError("价格预测日期越界")
    prior = data.typical_price.copy()
    if day == 0:
        return {name: prior.copy() for name in ("yesterday", "mean_7", "mean_30", "median_30")}
    history = data.variable_price[:day]
    return {
        "yesterday": history[-1].copy(),
        "mean_7": history[max(0, day - 7):].mean(axis=0),
        "mean_30": history[max(0, day - 30):].mean(axis=0),
        "median_30": np.median(history[max(0, day - 30):], axis=0),
    }


def select_price_forecast(
    data: Inputs, day: int, *, model: str = "mean_7", issue_hour: int = 0,
) -> np.ndarray:
    """Select a frozen causal price baseline and update it from observed prefix bias."""
    candidates = price_forecast_candidates(data, day)
    if model not in candidates:
        raise ValueError(f"未知价格预测模型: {model}")
    base = candidates[model]
    start = issue_hour * 6
    if not start:
        return np.maximum(0.0001, base)
    observed = slice(max(0, start - 18), start)
    bias = float(np.mean(data.variable_price[day, observed] - base[observed]))
    tau = np.arange(1, N_SLOTS - start + 1) / 6.0
    return np.maximum(0.0001, base[start:] + bias * np.exp(-tau / 3.0))


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
    history_days: int = DEFAULT_HISTORY_DAYS,
    max_scenarios: int = 9,
    risk_mode: str = DEFAULT_RISK_MODE,
    controller: str = "greedy",
    mpc_config: MpcConfig = MpcConfig(),
    price_model: str = "mean_7",
) -> dict[str, Any]:
    price = select_price_forecast(data, day, model=price_model) if variable_price else data.typical_price
    settlement_price = data.variable_price[day] if variable_price else data.typical_price
    if safe_pair is None:
        safe_l, safe_v, meta = safe_analog_trajectory(
            day, load_point[day], pv_point[day], load_residual, pv_residual,
            alpha=alpha, history_days=history_days,
            max_scenarios=max_scenarios, risk_mode=risk_mode,
        )
    else:
        safe_l, safe_v, meta = safe_pair
    history_start = max(0, day - history_days)
    residual_l = np.asarray(load_residual[history_start:day], dtype=float)
    residual_v = np.asarray(pv_residual[history_start:day], dtype=float)
    if risk_mode == "unclustered":
        historical_net = residual_l - residual_v
        scenario_weights = None
    else:
        scenario_l, scenario_v, scenario_weights = _residual_scenarios(
            residual_l, residual_v, risk_mode=risk_mode,
            max_scenarios=max_scenarios,
        )
        historical_net = scenario_l - scenario_v
    plan = solve_risk_contract_lp(
        load_point[day], pv_point[day], price, soc0, historical_net,
        alpha=alpha, scenario_weights=scenario_weights,
        terminal_value=terminal_value(price, battery), battery=battery,
        emergency_multiplier=emergency_multiplier,
        history_start=history_start, history_end_exclusive=day,
    )
    if controller == "mpc":
        actual = baseline_MPC(
            plan["grid_kwh"], data.actual_load_kw[day], data.actual_pv_kw[day],
            settlement_price, soc0, forecast_load_kw=load_point[day],
            forecast_pv_kw=pv_point[day], forecast_price=price,
            target_soc_kwh=plan["soc_kwh"], battery=battery,
            emergency_multiplier=emergency_multiplier, config=mpc_config,
        )
    elif controller == "delayed_greedy":
        actual = delayed_greedy_dispatch(
            plan["grid_kwh"], data.actual_load_kw[day], data.actual_pv_kw[day],
            settlement_price, soc0, forecast_load_kw=load_point[day],
            forecast_pv_kw=pv_point[day], battery=battery,
            emergency_multiplier=emergency_multiplier,
        )
    elif controller == "greedy":
        actual = greedy_execution(
            plan["grid_kwh"], data.actual_load_kw[day], data.actual_pv_kw[day],
            settlement_price, soc0, battery=battery,
            emergency_multiplier=emergency_multiplier,
        )
    else:
        raise ValueError("controller必须是mpc、delayed_greedy或greedy")
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
    history_days: int = DEFAULT_HISTORY_DAYS,
    max_scenarios: int = 9,
    risk_mode: str = DEFAULT_RISK_MODE,
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
    if issue > 0:
        anchor = float(data.actual_pv_kw[day, start_slot - 1])
    elif day > 0:
        anchor = float(data.actual_pv_kw[day - 1, -1])
    else:
        anchor = 0.0
    # The official release is archived exactly as issued.  Historical errors
    # calibrate risk but never re-centre a past or current release.
    pv_point = interpolate_hourly_points(anchor, np.maximum(0.0, raw), n)
    l_point = load_point_day[start_slot:]

    hist_ids = np.arange(hist_start, day)[valid_rows]
    if hist_ids.size:
        r_l = load_residual[hist_ids, start_slot:]
        r_v = np.vstack(
            [interpolate_hourly_points(0.0, row, n) for row in hist]
        )
    else:
        r_l = np.empty((0, n))
        r_v = np.empty((0, n))
    if risk_mode == "unclustered":
        historical_net = r_l - r_v
        correction = empirical_quantile_columns(r_l - r_v, alpha)
        scenario_count = hist_ids.size if hist_ids.size else 1
        weights = np.full(scenario_count, 1.0 / scenario_count)
    else:
        c_l, c_v, weights = _residual_scenarios(
            r_l, r_v, risk_mode=risk_mode, max_scenarios=max_scenarios
        )
        historical_net = c_l - c_v
        correction = weighted_quantile_columns(historical_net, weights, alpha)
        scenario_count = weights.size
    safe_net = l_point - pv_point + correction
    safe_v = np.maximum.reduce([pv_point, -safe_net, np.zeros_like(safe_net)])
    safe_l = safe_v + safe_net
    return safe_l, safe_v, {
        "decision_day": int(day),
        "issue_hour": int(issue),
        "history_start": int(hist_start),
        "history_end_exclusive": int(day),
        "risk_mode": risk_mode,
        "residual_count": int(hist_ids.size),
        "scenario_count": int(scenario_count),
        "alpha": float(alpha),
        "scenario_weights": weights.tolist(),
        "pv_point_kw": pv_point.tolist(),
        "load_point_kw": np.asarray(l_point, dtype=float).tolist(),
        "net_residual_quantile_kw": correction.tolist(),
        # Kept as a compact ndarray for the downstream risk-contract LP.  It
        # contains only completed-day residual paths available at this issue.
        "historical_net_residual_kw": historical_net,
        "history_day_indices": hist_ids.tolist(),
        "anchor_kw": anchor,
        "release_preserved_without_recentering": True,
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
    load_point = np.asarray(meta["load_point_kw"])
    correction = np.asarray(meta["net_residual_quantile_kw"])
    historical_net = np.asarray(meta["historical_net_residual_kw"], dtype=float)
    first = max(0, offset - 18)
    realized = data.actual_pv_kw[day, issue * 6 + first:event * 6]
    bias = float(np.mean(realized - point[first:offset])) if realized.size else 0.0
    tau = np.arange(1, N_SLOTS - event * 6 + 1) / 6.0
    revised_point = np.maximum(0.0, point[offset:] + bias * np.exp(-tau / 3.0))
    revised_net = load_point[offset:] - revised_point + correction[offset:]
    revised_pv = np.maximum.reduce([revised_point, -revised_net, np.zeros_like(revised_net)])
    revised_load = revised_pv + revised_net
    return revised_load, revised_pv, {
        **{k: v for k, v in meta.items() if k not in ("pv_point_kw", "load_point_kw", "net_residual_quantile_kw", "historical_net_residual_kw")},
        "decision_hour": event, "synthetic_nowcast": True,
        "observed_end_exclusive": event * 6, "nowcast_bias_kw": bias,
        "pv_point_kw": revised_point.tolist(),
        "load_point_kw": load_point[offset:].tolist(),
        "net_residual_quantile_kw": correction[offset:].tolist(),
        "historical_net_residual_kw": historical_net[:, offset:],
    }


def _solve_update_risk_contract(
    meta: dict[str, Any],
    price: np.ndarray,
    soc0: float,
    *,
    alpha: float,
    battery: Battery,
    emergency_multiplier: float,
    base_plan_kwh: np.ndarray | None = None,
) -> dict[str, Any]:
    """Map one causally available Q3/Q4 forecast release to the common LP."""
    historical_net = np.asarray(meta["historical_net_residual_kw"], dtype=float)
    raw_weights = np.asarray(meta.get("scenario_weights", []), dtype=float)
    weights = raw_weights if historical_net.shape[0] and raw_weights.shape == (historical_net.shape[0],) else None
    return solve_risk_contract_lp(
        np.asarray(meta["load_point_kw"], dtype=float),
        np.asarray(meta["pv_point_kw"], dtype=float),
        np.asarray(price, dtype=float),
        soc0,
        historical_net,
        alpha=alpha, scenario_weights=weights,
        terminal_value=terminal_value(np.asarray(price, dtype=float), battery),
        battery=battery, emergency_multiplier=emergency_multiplier,
        history_start=int(meta["history_start"]),
        history_end_exclusive=int(meta["history_end_exclusive"]),
        base_plan_kwh=base_plan_kwh,
    )


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
    allowed_issue_hours: tuple[int, ...] = ISSUES[1:],
    battery: Battery = DEFAULT_BATTERY,
    emergency_multiplier: float = 5.0,
    history_days: int = DEFAULT_HISTORY_DAYS,
    max_scenarios: int = 9,
    risk_mode: str = DEFAULT_RISK_MODE,
    controller: str = "greedy",
    mpc_config: MpcConfig = MpcConfig(),
    price_model: str = "mean_7",
) -> dict[str, Any]:
    if not events or events[0] != 0 or tuple(sorted(set(events))) != events or any(h < 0 or h >= 24 or int(h) != h for h in events):
        raise ValueError("更新时刻必须从0开始，严格递增且在[0,24)内")
    allowed_issue_hours = tuple(allowed_issue_hours)
    if tuple(sorted(set(allowed_issue_hours))) != allowed_issue_hours or any(h not in ISSUES[1:] for h in allowed_issue_hours):
        raise ValueError("获准新发布时间必须是{6,12,18}的严格递增子集")
    readable_issues = (0,) + allowed_issue_hours
    price = select_price_forecast(data, day, model=price_model) if variable_price else data.typical_price
    settlement_price = data.variable_price[day] if variable_price else data.typical_price
    if safe_updates is None:
        safe_l0, safe_v0, meta0 = safe_update_trajectory(
            data, day, 0, load_point[day], load_residual, pv_issue_residual,
            alpha=alpha, history_days=history_days,
            max_scenarios=max_scenarios, risk_mode=risk_mode,
        )
    else:
        safe_l0, safe_v0, meta0 = safe_updates[0]
    initial = _solve_update_risk_contract(
        meta0, price, soc0, alpha=alpha, battery=battery,
        emergency_multiplier=emergency_multiplier,
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
    update_meta = [{
        **meta0,
        "decision_hour": 0,
        "source_issue_hour": 0,
        "is_new_official_release": True,
        "allowed_issue_hours": list(allowed_issue_hours),
    }]
    update_solutions: list[dict[str, Any]] = []
    controller_diagnostics: list[dict[str, Any]] = []
    target_soc = np.full(N_SLOTS + 1, np.nan, dtype=float)
    safe_net_kw = np.full(N_SLOTS, np.nan, dtype=float)
    current_soc = float(soc0)

    for i, issue in enumerate(events):
        start = issue * 6
        end = events[i + 1] * 6 if i + 1 < len(events) else N_SLOTS
        if i == 0:
            safe_l, safe_v = safe_l0, safe_v0
            execution_meta = meta0
            contract_remaining = p0[start:]
            update_solution = initial
        else:
            source_issue = max(h for h in readable_issues if h <= issue)
            latest_index = ISSUES.index(source_issue)
            if safe_updates is None:
                source = safe_update_trajectory(
                    data, day, latest_index, load_point[day], load_residual,
                    pv_issue_residual, alpha=alpha, history_days=history_days,
                    max_scenarios=max_scenarios, risk_mode=risk_mode,
                )
            else:
                source = safe_updates[latest_index]
            safe_l, safe_v, meta = nowcast_trajectory(data, day, issue, source)
            meta = {
                **meta,
                "decision_hour": int(issue),
                "source_issue_hour": int(source_issue),
                "is_new_official_release": bool(issue == source_issue),
                "allowed_issue_hours": list(allowed_issue_hours),
            }
            update_meta.append(meta)
            execution_meta = meta
            remaining_price = select_price_forecast(data, day, model=price_model, issue_hour=issue) if variable_price else price[start:]
            update_solution = _solve_update_risk_contract(
                meta, remaining_price, current_soc, alpha=alpha,
                battery=battery, emergency_multiplier=emergency_multiplier,
                base_plan_kwh=p0[start:],
            )
            update_solution["planning_price"] = remaining_price
            contract_remaining = update_solution["grid_kwh"]
        update_solutions.append(update_solution)
        target_soc[start:end + 1] = update_solution["soc_kwh"][: end - start + 1]
        safe_net_kw[start:end] = (safe_l - safe_v)[: end - start]
        segment_contract = contract_remaining[: end - start]
        final_contract[start:end] = segment_contract
        if controller == "mpc":
            remaining_price = select_price_forecast(data, day, model=price_model, issue_hour=issue) if variable_price else price[start:]
            segment = baseline_MPC(
                segment_contract, data.actual_load_kw[day, start:end],
                data.actual_pv_kw[day, start:end], settlement_price[start:end],
                current_soc,
                forecast_load_kw=np.asarray(execution_meta["load_point_kw"])[:end-start],
                forecast_pv_kw=np.asarray(execution_meta["pv_point_kw"])[:end-start],
                forecast_price=remaining_price[:end-start], base_plan_kwh=p0[start:end],
                target_soc_kwh=update_solution["soc_kwh"][:end-start+1],
                battery=battery, emergency_multiplier=emergency_multiplier,
                config=mpc_config,
            )
            controller_diagnostics.append(segment["solver_diagnostics"])
        elif controller == "delayed_greedy":
            segment = delayed_greedy_dispatch(
                segment_contract, data.actual_load_kw[day, start:end],
                data.actual_pv_kw[day, start:end], settlement_price[start:end],
                current_soc,
                forecast_load_kw=np.asarray(execution_meta["load_point_kw"])[:end-start],
                forecast_pv_kw=np.asarray(execution_meta["pv_point_kw"])[:end-start],
                base_plan_kwh=p0[start:end], battery=battery,
                emergency_multiplier=emergency_multiplier,
            )
        elif controller == "greedy":
            segment = greedy_execution(
                segment_contract, data.actual_load_kw[day, start:end],
                data.actual_pv_kw[day, start:end], settlement_price[start:end],
                current_soc, base_plan_kwh=p0[start:end], battery=battery,
                emergency_multiplier=emergency_multiplier,
            )
        else:
            raise ValueError("controller必须是mpc、delayed_greedy或greedy")
        for key in actual_parts:
            actual_parts[key].append(segment[key])
        soc_trace.extend(segment["soc_kwh"][1:].tolist())
        current_soc = float(segment["soc_kwh"][-1])

    actual = {key: np.concatenate(parts) for key, parts in actual_parts.items()}
    actual["soc_kwh"] = np.asarray(soc_trace)
    balance = final_contract - actual["unused_contract_kwh"] + actual["pv_used_kwh"] + actual["discharge_kwh"] + actual["emergency_kwh"] - data.actual_load_kw[day] * DT_HOURS - actual["charge_kwh"]
    actual["max_balance_residual"] = float(np.max(np.abs(balance)))
    overlap = np.minimum(actual["charge_kwh"], actual["discharge_kwh"])
    actual["simultaneous_flow_slots"] = int(np.count_nonzero(overlap > mpc_config.overlap_tolerance))
    actual["simultaneous_flow_max"] = float(np.max(overlap))
    actual["controller"] = {
        "mpc": "baseline_MPC",
        "delayed_greedy": "delayed_greedy_baseline",
        "greedy": "greedy_feedback",
    }[controller]
    if controller_diagnostics:
        times = [v for d in controller_diagnostics for v in ([d["median_seconds"]] if d["median_seconds"] is not None else [])]
        actual["solver_diagnostics"] = {
            "solve_count": int(sum(d["solve_count"] for d in controller_diagnostics)),
            "fallback_count": int(sum(d["fallback_count"] for d in controller_diagnostics)),
            "fallback_reasons": {
                reason: int(sum(d["fallback_reasons"].get(reason, 0) for d in controller_diagnostics))
                for reason in sorted({r for d in controller_diagnostics for r in d["fallback_reasons"]})
            },
            "clip_count": int(sum(d["clip_count"] for d in controller_diagnostics)),
            "median_segment_seconds": float(np.median(times)) if times else None,
            "total_seconds": float(sum(d["total_seconds"] for d in controller_diagnostics)),
            "horizon_slots": mpc_config.horizon_slots,
            "tracking_weight": mpc_config.tracking_weight,
        }
    return {
        "initial_plan": initial,
        "initial_plan_kwh": p0,
        "final_contract_kwh": final_contract,
        "actual": actual,
        "update_meta": update_meta,
        "update_solutions": update_solutions,
        "target_soc_kwh": target_soc,
        "safe_net_kw": safe_net_kw,
        "planning_price": price,
        "events": events,
        "allowed_issue_hours": allowed_issue_hours,
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
