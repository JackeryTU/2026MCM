# -*- coding: utf-8 -*-
"""2026 全国大学生数学建模竞赛 C 题：微网与外部电网电力调控策略。

核心模型库（编程手主线实现）。

模块职责
--------
* 数据层：读取附件 1--4，统一为 (天数, 144) 的 10 min 功率矩阵；
* 优化层：确定性线性规划（日前合同、滚动执行）与结算；
* 执行层：问题二因果贪心、问题三/四滚动 LP + 槽内裁剪；
* 预测层：F1 周周期点预测、净负荷残差分位安全轨迹、官方预报偏差校正、
  价格预测与日内修正；
* 输出层：模板工作簿、CSV、图件。

所有能量单位为 kWh，功率单位为 kW，电价单位为 元/kWh，时间步长 1/6 h。
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

# ---------------------------------------------------------------------------
# 全局常数
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "附件"
RESULT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"

T = 144                      # 每日时段数
DT = 1.0 / 6.0               # 每时段小时数
ETA_C = 0.9                  # 充电效率
ETA_D = 0.9                  # 放电效率
E_MAX = 12000.0              # 额定容量 kWh
S_MIN = 1200.0               # SOC 下限 kWh
S_MAX = 10800.0              # SOC 上限 kWh
P_MAX = 5000.0               # 充/放电功率上限 kW
B_MAX = P_MAX * DT           # 每时段充/放电能量上限 kWh
S0 = 6000.0                  # 2025-01-01 0:00 初始储电量 kWh
EMERGENCY_FACTOR = 5.0       # 应急购电价格倍数
OVER_FACTOR = 1.5            # 高于计划购电的结算倍数
UNDER_FACTOR = 0.5           # 低于计划购电的违约结算比例

DAY0 = _dt.date(2025, 1, 1)
START_DATE = _dt.date(2025, 2, 1)     # 正式结果起始日
END_DATE = _dt.date(2025, 12, 31)     # 正式结果终止日
N_OFFICIAL = (END_DATE - START_DATE).days + 1   # 334

EPS_THROUGHPUT = 1e-4        # 次级目标（充放吞吐）平局打破权重；1e-4 可消解
                             # LP 退化引起的同槽充放重叠，成本影响可忽略


# ---------------------------------------------------------------------------
# 数据层
# ---------------------------------------------------------------------------
_TIME_COLS = None


def _read_power_matrix(path: Path, sheet) -> np.ndarray:
    """读取 (365, 145) 的 [日期 | 144 个 10 min 值] 工作表，返回 (365, 144) 功率。"""
    df = pd.read_excel(path, sheet_name=sheet, header=0)
    if df.shape[1] != 145:
        raise ValueError(f"{path.name}/{sheet} 列数异常：{df.shape[1]}")
    vals = df.iloc[:, 1:].to_numpy(dtype=float)
    if vals.shape != (365, T):
        raise ValueError(f"{path.name}/{sheet} 形状异常：{vals.shape}")
    if not np.isfinite(vals).all():
        raise ValueError(f"{path.name}/{sheet} 含缺失值")
    return vals


def load_inputs() -> dict:
    """返回全部输入数据。功率单位 kW，价格为 元/kWh。"""
    global _TIME_COLS
    a1 = pd.read_excel(DATA_DIR / "附件1.xlsx", sheet_name=0, header=0)
    a1.columns = [str(c).strip() for c in a1.columns]
    price1 = a1.iloc[:, 1].to_numpy(dtype=float)
    load1 = a1.iloc[:, 2].to_numpy(dtype=float)
    pv1 = a1.iloc[:, 3].to_numpy(dtype=float)
    if not (len(price1) == len(load1) == len(pv1) == T):
        raise ValueError("附件1 长度异常")

    load_act = _read_power_matrix(DATA_DIR / "附件2.xlsx", "小区负载")
    pv_act = _read_power_matrix(DATA_DIR / "附件2.xlsx", "光伏发电实际功率")
    price_act = _read_power_matrix(DATA_DIR / "附件4.xlsx", 0)

    a3 = pd.read_excel(DATA_DIR / "附件3.xlsx", sheet_name=0, header=0)
    if a3.shape != (1460, 26):
        raise ValueError(f"附件3 形状异常：{a3.shape}")
    raw_date = a3.iloc[:, 0].ffill()
    dates3 = pd.to_datetime(raw_date, format="%Y-%m-%d", errors="coerce")
    if dates3.isna().any():
        # 回退：按字符串解析
        parsed = []
        cur = None
        for v in a3.iloc[:, 0]:
            if isinstance(v, str) and v.strip():
                cur = pd.Timestamp(_dt.datetime.strptime(v.strip(), "%Y-%m-%d").date())
            parsed.append(cur)
        dates3 = pd.Series(parsed)
    issue = a3.iloc[:, 1].astype(str).str.strip().to_numpy()
    fc_raw = a3.iloc[:, 2:26].to_numpy(dtype=float)   # (1460, 24) 提前期 1..24 h

    issue_hours = np.array([int(s.split(":")[0]) for s in issue])

    # 组装 (365, 4, 24) 的原始预报张量
    day_idx = np.array([(d.date() - DAY0).days for d in dates3])
    slot_idx = issue_hours // 6
    fc = np.full((365, 4, 24), np.nan)
    for k in range(len(day_idx)):
        fc[day_idx[k], slot_idx[k]] = fc_raw[k]
    if not np.isfinite(fc).all():
        raise ValueError("附件3 预报张量存在空洞")

    _TIME_COLS = [str(c) for c in pd.read_excel(DATA_DIR / "附件2.xlsx", sheet_name="小区负载",
                                                header=0).columns[1:]]
    return {
        "price1": price1,
        "load1": load1,
        "pv1": pv1,
        "load_act": load_act,
        "pv_act": pv_act,
        "price_act": price_act,
        "fc_raw": fc,               # (365, 4, 24) kW，提前期 1..24 h
    }


def date_of(day_index: int) -> _dt.date:
    return DAY0 + _dt.timedelta(days=int(day_index))


# ---------------------------------------------------------------------------
# 确定性线性规划：日前合同 / 问题一
# ---------------------------------------------------------------------------
_DAY_LP_CACHE: dict = {}


def _day_lp_structure(n: int):
    """构造并缓存 n 个时段 LP 的等式矩阵结构。

    变量顺序：[x(0..n-1), y, g, v, S(0..n-1)]，S[t] 表示第 t 槽末（= S_{t+1}）。
    """
    if n in _DAY_LP_CACHE:
        return _DAY_LP_CACHE[n]
    N = 5 * n
    A_eq = np.zeros((2 * n, N))
    b_eq = np.zeros(2 * n)
    for t in range(n):
        # 母线平衡  g + v + y - x = L
        A_eq[t, 2 * n + t] = 1.0    # g
        A_eq[t, 3 * n + t] = 1.0    # v（实际利用光伏）
        A_eq[t, n + t] = 1.0        # y
        A_eq[t, t] = -1.0           # x
        # SOC 递推 S[t] - S[t-1] - eta_c x + y/eta_d = 0
        r = n + t
        A_eq[r, 4 * n + t] = 1.0
        if t > 0:
            A_eq[r, 4 * n + t - 1] = -1.0
        A_eq[r, t] = -ETA_C
        A_eq[r, n + t] = 1.0 / ETA_D
    _DAY_LP_CACHE[n] = (A_eq, b_eq)
    return A_eq, b_eq


def solve_day_lp(c, load_kwh, pv_kwh, s_init, s_end=None, lam=None,
                 charge_cap=None, extra_throughput_penalty=EPS_THROUGHPUT):
    """日前确定性 LP。

    参数
    ----
    c : (n,) 电价 元/kWh
    load_kwh, pv_kwh : (n,) 负荷/可用光伏能量 kWh
    s_init : 起始储电量 kWh
    s_end : 若不为 None，强制末槽储电量等于该值（问题一取 6000）
    lam : 若不为 None，末槽储电量相对 S_MIN 的线性价值（元/kWh）
    charge_cap : 可选 (n,) 每槽充电能量上限（防止应急电充电）

    返回 dict：x, y, g, v, S（S 长度 n，S[t] 为第 t 槽末储电量）
    """
    c = np.asarray(c, dtype=float)
    n = len(c)
    A_eq, _ = _day_lp_structure(n)
    N = 5 * n

    b_eq = np.empty(2 * n)
    b_eq[:n] = load_kwh
    b_eq[n] = s_init
    b_eq[n + 1:] = 0.0

    obj = np.zeros(N)
    obj[2 * n:3 * n] = c                   # 购电成本
    obj[:n] += extra_throughput_penalty    # 充电吞吐
    obj[n:2 * n] += extra_throughput_penalty  # 放电吞吐
    if lam is not None:
        obj[4 * n + n - 1] = -lam

    lb = np.zeros(N)
    ub = np.full(N, np.inf)
    ub[:n] = B_MAX                       # x
    ub[n:2 * n] = B_MAX                  # y
    ub[2 * n:3 * n] = np.inf             # g
    ub[3 * n:4 * n] = pv_kwh             # v
    lb[4 * n:5 * n] = S_MIN
    ub[4 * n:5 * n] = S_MAX
    if charge_cap is not None:
        ub[:n] = np.minimum(ub[:n], charge_cap)
    if s_end is not None:
        lb[4 * n + n - 1] = s_end
        ub[4 * n + n - 1] = s_end

    bounds = list(zip(lb, ub))
    res = linprog(obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"日前 LP 求解失败：{res.message}")
    z = res.x
    return {
        "x": z[:n], "y": z[n:2 * n], "g": z[2 * n:3 * n],
        "v": z[3 * n:4 * n], "S": z[4 * n:5 * n],
        "curtail": pv_kwh - z[3 * n:4 * n],
        "objective": float(res.fun), "status": res.status,
    }


# ---------------------------------------------------------------------------
# 问题二执行层：因果贪心
# ---------------------------------------------------------------------------
def causal_dispatch(p, load_kwh, pv_kwh, s_init):
    """按题目分析报告 §6.6 的因果贪心逐槽执行。

    p : (n,) 已生效合同购电量 kWh
    返回 x, y, u, w, v, S（S[t] 为第 t 槽末储电量）。
    """
    n = len(p)
    x = np.zeros(n); y = np.zeros(n); u = np.zeros(n)
    w = np.zeros(n); v = np.zeros(n); S = np.zeros(n)
    s = float(s_init)
    for t in range(n):
        A = p[t] + pv_kwh[t] - load_kwh[t]
        if A >= 0.0:
            xe = min(A, B_MAX, (S_MAX - s) / ETA_C)
            xe = max(xe, 0.0)
            ye = 0.0
            ue = 0.0
            r = A - xe
            we = min(p[t], r)
            ve = pv_kwh[t] - max(0.0, r - we)
        else:
            xe = 0.0
            ye = min(-A, B_MAX, (s - S_MIN) * ETA_D)
            ue = -A - ye
            we = 0.0
            ve = pv_kwh[t]
        x[t], y[t], u[t], w[t], v[t] = xe, ye, ue, we, ve
        s = s + ETA_C * xe - ye / ETA_D
        S[t] = s
    return {"x": x, "y": y, "u": u, "w": w, "v": v, "S": S}


# ---------------------------------------------------------------------------
# 问题三/四执行层：滚动 LP + 槽内裁剪
# ---------------------------------------------------------------------------
_ROLL_LP_CACHE: dict = {}


def _rolling_lp_structure(n: int):
    """缓存 n 槽滚动 LP 的等式矩阵结构（只依赖 n，可用于任何数据）。"""
    cached = _ROLL_LP_CACHE.get(n)
    if cached is not None:
        return cached
    # 变量：x,y,v,u,w (各 n)，S(0..n)，S[0] 固定为 s_init
    N = 5 * n + (n + 1)
    o = {k: i * n for i, k in enumerate(("x", "y", "v", "u", "w"))}
    s0 = 5 * n
    A_eq = np.zeros((2 * n, N))
    for t in range(n):
        A_eq[t, o["x"] + t] = -1.0
        A_eq[t, o["y"] + t] = 1.0
        A_eq[t, o["v"] + t] = 1.0
        A_eq[t, o["u"] + t] = 1.0
        A_eq[t, o["w"] + t] = -1.0
        A_eq[n + t, o["x"] + t] = -ETA_C
        A_eq[n + t, o["y"] + t] = 1.0 / ETA_D
        A_eq[n + t, s0 + t] = -1.0
        A_eq[n + t, s0 + t + 1] = 1.0
    result = (A_eq, o, s0, N)
    _ROLL_LP_CACHE[n] = result
    return result


def rolling_lp_action(s_init, g_seg, load_seg, pv_seg, c_seg, lam,
                      time_limit=None):
    """§7.2 滚动 LP，返回首槽动作 (x_M, y_M)。

    time_limit 为单次求解秒数预算；超时/非最优返回 None，由调用方回退。
    """
    n = len(g_seg)
    A_eq, o, s0, N = _rolling_lp_structure(n)
    b_eq = np.concatenate([load_seg - g_seg, np.zeros(n)])

    obj = np.zeros(N)
    obj[o["u"]:o["u"] + n] = EMERGENCY_FACTOR * c_seg
    obj[o["x"]:o["x"] + n] += EPS_THROUGHPUT
    obj[o["y"]:o["y"] + n] += EPS_THROUGHPUT
    obj[s0 + n] = -lam

    surplus = np.maximum(g_seg + pv_seg - load_seg, 0.0)
    deficit = np.maximum(load_seg - g_seg - pv_seg, 0.0)

    lb = np.zeros(N); ub = np.full(N, np.inf)
    ub[o["x"]:o["x"] + n] = np.minimum(B_MAX, surplus)
    ub[o["y"]:o["y"] + n] = B_MAX
    ub[o["v"]:o["v"] + n] = pv_seg
    ub[o["u"]:o["u"] + n] = deficit
    ub[o["w"]:o["w"] + n] = g_seg
    lb[s0] = ub[s0] = s_init
    lb[s0 + 1:s0 + n + 1] = S_MIN
    ub[s0 + 1:s0 + n + 1] = S_MAX

    kwargs = {} if time_limit is None else {"options": {"time_limit": float(time_limit)}}
    res = linprog(obj, A_eq=A_eq, b_eq=b_eq, bounds=list(zip(lb, ub)),
                  method="highs", **kwargs)
    if not res.success or res.x is None:
        return None
    return float(res.x[o["x"]]), float(res.x[o["y"]])


def clip_slot(xm, ym, g, load_kwh, pv_kwh):
    """§7.3 槽内供需可行性裁剪，返回 (x_E, y_E, u, w, v, balance)。"""
    xe = min(xm, max(g + pv_kwh - load_kwh, 0.0))
    xe = max(xe, 0.0)
    ye = min(ym, load_kwh + xe)
    ye = max(ye, 0.0)
    b = load_kwh + xe - ye - g - pv_kwh
    if b >= 0.0:
        u, w, v = b, 0.0, pv_kwh
    else:
        u = 0.0
        w = min(g, -b)
        v = pv_kwh - (-b - w)
        v = max(v, 0.0)
    return xe, ye, u, w, v, b


def rolling_execute(g_contract, load_kwh, pv_kwh, price, s_init, horizon=36,
                    lam_price=None):
    """对给定合同逐槽滚动 LP + 裁剪执行，返回真实轨迹。"""
    n = len(g_contract)
    x = np.zeros(n); y = np.zeros(n); u = np.zeros(n)
    w = np.zeros(n); v = np.zeros(n); S = np.zeros(n)
    s = float(s_init)
    for t in range(n):
        K = min(t + horizon - 1, n - 1)
        seg = slice(t, K + 1)
        c_seg = price[seg]
        if lam_price is None:
            lam = ETA_D * float(np.median(c_seg))
        else:
            lam = lam_price[t]
        act = rolling_lp_action(s, g_contract[seg], load_kwh[seg], pv_kwh[seg],
                                c_seg, lam)
        if act is None:
            xm = ym = 0.0
        else:
            xm, ym = act
        xe, ye, ue, we, ve, _ = clip_slot(xm, ym, g_contract[t], load_kwh[t], pv_kwh[t])
        x[t], y[t], u[t], w[t], v[t] = xe, ye, ue, we, ve
        s = s + ETA_C * xe - ye / ETA_D
        if s < S_MIN - 1e-6 or s > S_MAX + 1e-6:
            raise RuntimeError(f"滚动执行 SOC 越界：t={t}, S={s}")
        s = min(max(s, S_MIN), S_MAX)
        S[t] = s
    return {"x": x, "y": y, "u": u, "w": w, "v": v, "S": S}


# ---------------------------------------------------------------------------
# 合同调整 LP（问题三 §8.2）
# ---------------------------------------------------------------------------
_CONTRACT_UPDATE_CACHE: dict = {}


def _contract_update_structure(n: int):
    """缓存 n 槽合同调整 LP 的等式结构：变量 [x, y, v, δ+, δ-, u, S] 各 n 个。"""
    cached = _CONTRACT_UPDATE_CACHE.get(n)
    if cached is not None:
        return cached
    N = 7 * n
    o = {"x": 0, "y": n, "v": 2 * n, "d": 3 * n, "e": 4 * n,
         "u": 5 * n, "S": 6 * n}
    A_eq = np.zeros((2 * n, N))
    for t in range(n):
        A_eq[t, o["x"] + t] = -1.0
        A_eq[t, o["y"] + t] = 1.0
        A_eq[t, o["v"] + t] = 1.0
        A_eq[t, o["d"] + t] = 1.0        # δ+
        A_eq[t, o["e"] + t] = -1.0       # δ-
        A_eq[t, o["u"] + t] = 1.0        # u
        r = n + t
        A_eq[r, o["S"] + t] = 1.0
        if t > 0:
            A_eq[r, o["S"] + t - 1] = -1.0
        A_eq[r, o["x"] + t] = -ETA_C
        A_eq[r, o["y"] + t] = 1.0 / ETA_D
    result = (A_eq, o, N)
    _CONTRACT_UPDATE_CACHE[n] = result
    return result


def contract_update_lp(p_rem, c_rem, load_kwh, pv_kwh, s_init, s_end=None, lam=None):
    """给定原始合同 p，优化剩余时段最终合同 q 与运行方式。

    C_t = c min(p,q) + 1.5c (q-p)^+ + 0.5c (p-q)^+ ，逐槽只结算一次。
    返回 q, x, y, v, u, S。
    """
    n = len(p_rem)
    A_eq, o, N = _contract_update_structure(n)
    bx, by, bv, bd, be, bu, bS = (o[k] for k in ("x", "y", "v", "d", "e", "u", "S"))
    b_eq = np.zeros(2 * n)
    for t in range(n):
        b_eq[t] = load_kwh[t] - p_rem[t]
    b_eq[n] = s_init

    obj = np.zeros(N)
    obj[bd:bd + n] = OVER_FACTOR * c_rem
    obj[be:be + n] = -UNDER_FACTOR * c_rem
    obj[bu:bu + n] = EMERGENCY_FACTOR * c_rem
    obj[bx:bx + n] += EPS_THROUGHPUT
    obj[by:by + n] += EPS_THROUGHPUT
    if lam is not None:
        obj[bS + n - 1] = -lam

    lb = np.zeros(N); ub = np.full(N, np.inf)
    # 禁止用应急电主动充电：充电上限 = 原合同与预测形成的盈余
    ub[bx:bx + n] = np.minimum(B_MAX, np.maximum(p_rem + pv_kwh - load_kwh, 0.0))
    ub[by:by + n] = B_MAX
    ub[bv:bv + n] = pv_kwh
    ub[bu:bu + n] = np.maximum(load_kwh - pv_kwh, 0.0)
    ub[be:be + n] = p_rem          # δ- <= p，保证 q >= 0
    lb[bS:bS + n] = S_MIN
    ub[bS:bS + n] = S_MAX
    if s_end is not None:
        lb[bS + n - 1] = ub[bS + n - 1] = s_end

    res = linprog(obj, A_eq=A_eq, b_eq=b_eq, bounds=list(zip(lb, ub)), method="highs")
    if not res.success:
        return None
    z = res.x
    q = p_rem + z[bd:bd + n] - z[be:be + n]
    return {
        "q": q, "x": z[bx:bx + n], "y": z[by:by + n],
        "v": z[bv:bv + n], "u": z[bu:bu + n],
        "S": z[bS:bS + n], "objective": float(res.fun),
    }


# ---------------------------------------------------------------------------
# 结算
# ---------------------------------------------------------------------------
def settle_q12(p, u, price):
    """问题一/二现金结算：Σ c p + 5 Σ c u。"""
    return float(np.sum(price * p) + EMERGENCY_FACTOR * np.sum(price * u))


def settle_q3(p, q, u, price):
    """问题三逐槽结算：c min(p,q) + 1.5c(q-p)^+ + 0.5c(p-q)^+ + 5c u。"""
    over = np.maximum(q - p, 0.0)
    under = np.maximum(p - q, 0.0)
    return float(np.sum(price * np.minimum(p, q) + OVER_FACTOR * price * over
                        + UNDER_FACTOR * price * under
                        + EMERGENCY_FACTOR * price * u))


# ---------------------------------------------------------------------------
# 预测与风险
# ---------------------------------------------------------------------------
def forecast_f1(day_index, load_act, pv_act, load1, pv1):
    """F1 中心预测：负荷取上周同槽，光伏取前 7 日均值（kW）。"""
    if day_index >= 7:
        lhat = load_act[day_index - 7].copy()
        vhat = pv_act[day_index - 7:day_index].mean(axis=0)
    elif day_index >= 1:
        lhat = load_act[:day_index].mean(axis=0)
        vhat = pv_act[:day_index].mean(axis=0)
    else:
        lhat = load1.copy()
        vhat = pv1.copy()
    return lhat, vhat


def empirical_quantile(samples, alpha):
    """有限样本等权左逆分位：排序后第 ceil(alpha*n) 个值，不插值。"""
    s = np.sort(np.asarray(samples, dtype=float))
    if s.size == 0:
        return 0.0
    k = int(np.ceil(alpha * s.size))
    k = min(max(k, 1), s.size)
    return float(s[k - 1])


def residual_quantile_safe(day_index, nhat_kw, vhat_kw, resid_hist, alpha, W):
    """净负荷残差分位安全轨迹（kW）。

    resid_hist : dict[(day, t)] -> 净负荷残差 kW，按日归档
    """
    lo = max(1, day_index - W)
    n = len(nhat_kw)
    q = np.zeros(n)
    for t in range(n):
        samples = [resid_hist[(j, t)] for j in range(lo, day_index)
                   if (j, t) in resid_hist]
        q[t] = empirical_quantile(samples, alpha) if samples else 0.0
    ntilde = nhat_kw + q
    vtilde = np.maximum(np.maximum(vhat_kw, -ntilde), 0.0)
    ltilde = vtilde + ntilde
    return ltilde, vtilde, ntilde


def bias_corrected_official_pv(day_index, issue_slot, pv_act, fc_raw, err_hist, W,
                               anchor_prev):
    """附件3 官方预报的 (发布时刻, 提前期) 历史偏差校正。

    返回该次发布对未来整点（提前期 1..24 h）的校正后功率 kW。
    """
    raw = fc_raw[day_index, issue_slot, :].copy()
    corr = np.zeros(24)
    for h in range(1, 25):
        key_h = h - 1
        samples = [err_hist[(j, issue_slot, key_h)] for j in range(
            max(0, day_index - W), day_index) if (j, issue_slot, key_h) in err_hist]
        corr[key_h] = float(np.mean(samples)) if samples else 0.0
    return np.maximum(raw + corr, 0.0)


def interp_official_profile(corrected_hourly, issue_hour, anchor_kw):
    """把整点校正预报 + 发布时刻锚点线性插值到 10 min 网格。

    返回长度 144 的数组，索引 p 对应对 p*10 min 的功率（p=0 为 0:00）。
    仅发布时刻之后的部分有效。
    """
    pos0 = issue_hour * 6
    knots_x = [pos0]
    knots_y = [anchor_kw]
    for h in range(1, 25):
        pos = pos0 + 6 * h
        if pos > 144:
            break
        knots_x.append(pos)
        knots_y.append(corrected_hourly[h - 1])
    grid = np.arange(knots_x[0], knots_x[-1] + 1)
    prof = np.interp(grid, knots_x, knots_y)
    out = np.zeros(T + 1)
    out[knots_x[0]:knots_x[-1] + 1] = prof
    if knots_x[0] > 0:
        out[:knots_x[0]] = knots_y[0]
    return out


def price_forecast_0h(day_index, price_act, price1, base="median30"):
    """0:00 价格基准（元/kWh），只用已完成历史。"""
    if day_index <= 0:
        return price1.copy()
    lo = max(0, day_index - 30)
    hist = price_act[lo:day_index]
    if hist.shape[0] == 0:
        return price1.copy()
    if base == "median30":
        return np.median(hist, axis=0)
    if base == "mean30":
        return hist.mean(axis=0)
    if base == "mean7":
        return price_act[max(0, day_index - 7):day_index].mean(axis=0)
    if base == "prev_day":
        return price_act[day_index - 1].copy()
    raise ValueError(base)


def price_intraday_correct(base_pred, price_act_day, t, n_recent=18, decay=3.0, floor=1e-4):
    """日内价格修正：最近已完成槽 (真实-基准) 均值的指数衰减外推（kW 同量纲）。"""
    lo = max(0, t - n_recent)
    if t <= 0 or lo >= t:
        m = 0.0
    else:
        m = float(np.mean(price_act_day[lo:t] - base_pred[lo:t]))
    lead = (np.arange(len(base_pred)) - t) * DT
    corr = np.where(lead > 0, m * np.exp(-lead / decay), 0.0)
    return np.maximum(base_pred + corr, floor)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def four_hour_blocks(arr):
    """把 144 槽序列按 0-4/4-8/.../20-24 聚合成 6 段。"""
    a = np.asarray(arr, dtype=float).reshape(6, 24)
    return a.sum(axis=1)


def merge_intervals(mask, threshold=1e-8):
    """把同一日内连续正值的槽合并成最大连续区间。

    mask : 长度 144 的布尔数组
    返回 [(起槽, 止槽), ...]，闭区间。
    """
    spans = []
    t = 0
    n = len(mask)
    while t < n:
        if mask[t]:
            start = t
            while t + 1 < n and mask[t + 1]:
                t += 1
            spans.append((start, t))
        t += 1
    return spans


def slot_label_start(i):
    """第 i 槽（0 基）的起始时刻字符串。"""
    m = i * 10
    return f"{m // 60}:{m % 60:02d}"


def slot_label_end(i):
    m = (i + 1) * 10
    if m == 1440:
        return "0:00+1"
    return f"{m // 60}:{m % 60:02d}"


def interval_label(i0, i1):
    return f"{slot_label_start(i0)}-{slot_label_end(i1)}"
