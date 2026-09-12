# -*- coding: utf-8 -*-
"""生成问题一至问题四的出版级图表（扁平命名，论文与质检共用）。

产物
----
figures/raw_qN_*.svg|.png、figures/process_qN_*.svg|.png、
figures/result_qN_*.svg|.png，以及 figures/图表面板.html。
全部图形同时导出 300 dpi PNG 与可编辑文本 SVG，导出前通过版面与设计预检。

用法
----
    python make_figures.py
"""
from __future__ import annotations

import datetime as _dt
import html
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import microgrid_core as mc
import solution_lib as sl
from utils import plot_style as ps

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'
CSV = sl.CSV
DPI = 300
X = np.arange(mc.T) * mc.DT + mc.DT / 2.0
PANELS: list = []

HOURS_TICKS = np.arange(0.0, 24.01, 4.0)
HOURS_LABELS = ['%d:00' % int(h) for h in HOURS_TICKS]
MONTH_LABELS = ['%02d月' % m for m in range(1, 13)]
Q3_MAIN = 'main'


def _save(fig, stem, title, question, category, note):
    outputs = ps.export_figure(fig, FIG / stem, dpi=DPI)
    PANELS.append({'stem': stem, 'title': title, 'question': question,
                   'category': category, 'note': note,
                   'png': Path(outputs['png']).name,
                   'svg': Path(outputs['svg']).name})
    plt.close(fig)
    print('[fig] ' + stem, flush=True)


def _read(name):
    return pd.read_csv(CSV / name, encoding='utf-8-sig')


def _hours(ax):
    ax.set_xticks(HOURS_TICKS)
    ax.set_xticklabels(HOURS_LABELS)
    ax.set_xlim(0.0, 24.0)


def _plabel(ax, text):
    ax.text(0.02, 0.96, text, transform=ax.transAxes, ha='left', va='top',
            fontsize=8, fontweight='bold')


def _annotate_bars(ax, bars, values, fmt):
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, fmt % value,
                ha='center', va='bottom', fontsize=6.5)


def _zero_top(values, slack=1.15):
    return 0.0, float(np.max(values)) * slack


# ---------------------------------------------------------------------------
# 问题一
# ---------------------------------------------------------------------------
def fig_q1_raw(data):
    fig, ax = ps.publication_subplots(width='report', aspect=0.5)
    ax.plot(X, data['load1'], color=ps.PALETTE['primary'], label='负荷')
    ax.plot(X, data['pv1'], color=ps.PALETTE['secondary'], label='光伏可发')
    ax.set_xlabel('时刻')
    ax.set_ylabel('功率 / kW')
    ax.set_title('典型日负荷与光伏出力')
    ax.legend(loc='upper left', ncols=2)
    _hours(ax)
    _save(fig, 'raw_q1_典型日负荷光伏', '典型日负荷与光伏出力', 'q1', 'raw',
          '附件 1 给出的典型日负荷与光伏可发功率，用于问题一确定性优化。')


def fig_q1_process(data):
    fig, axes = ps.publication_subplots(2, 1, width='report', aspect=0.72)
    a, b = axes
    a.step(X, data['price1'], where='post', color=ps.PALETTE['contrast'])
    a.set_ylabel('电价 / (元/kWh)')
    a.set_title('分时电价')
    _hours(a)
    b.plot(X, data['load1'] - data['pv1'], color=ps.PALETTE['primary'])
    b.axhline(0.0, color=ps.PALETTE['neutral'], lw=0.7)
    b.set_xlabel('时刻')
    b.set_ylabel('净负荷 / kW')
    b.set_title('净负荷曲线')
    _hours(b)
    _plabel(a, 'a')
    _plabel(b, 'b')
    _save(fig, 'process_q1_分时电价与净负荷', '分时电价与净负荷', 'q1', 'process',
          '问题一的价格信号与净负荷形态，决定储能低储高发。')


def fig_q1_result_battery():
    row = _read('问题1_汇总.csv').iloc[0]
    values = [float(row['全天计划购电量_kWh']), float(row['无储能购电量_kWh'])]
    fig, ax = ps.publication_subplots(width='report', aspect=0.55)
    bars = ax.bar([0.0, 1.0], values, width=0.45,
                  color=[ps.PALETTE['positive'], ps.PALETTE['neutral']])
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(['配置储能', '不配置储能'])
    ax.set_ylabel('全天购电量 / kWh')
    ax.set_title('储能对全天购电量的影响')
    ax.set_ylim(*_zero_top(values))
    _annotate_bars(ax, bars, values, '%.0f')
    _save(fig, 'result_q1_储能效益对比', '储能效益对比', 'q1', 'result',
          '配置储能后全天购电量由约 61790 kWh 降至约 59483 kWh。')


def fig_q1_result_plan():
    df = _read('问题1_逐槽计划.csv')
    fig, axes = ps.publication_subplots(2, 1, width='report', aspect=0.72)
    a, b = axes
    a.step(X, df['充电量_kWh'].to_numpy() / mc.DT, where='post',
           color=ps.PALETTE['primary'], label='充电')
    a.step(X, df['放电量_kWh'].to_numpy() / mc.DT, where='post',
           color=ps.PALETTE['contrast'], label='放电')
    a.set_ylabel('功率 / kW')
    a.set_title('最优充放电功率')
    a.legend(loc='upper left', ncols=2)
    _hours(a)
    b.plot(X, df['槽末储电量_kWh'].to_numpy(), color=ps.PALETTE['positive'])
    b.axhline(mc.S_MIN, color=ps.PALETTE['neutral'], lw=0.7)
    b.set_xlabel('时刻')
    b.set_ylabel('储电量 / kWh')
    b.set_title('储电量轨迹')
    _hours(b)
    _plabel(a, 'a')
    _plabel(b, 'b')
    _save(fig, 'result_q1_最优充放电与储电量', '最优充放电与储电量', 'q1', 'result',
          '典型日最优充放电功率与储电量轨迹，储电量始终位于上下限之间。')


# ---------------------------------------------------------------------------
# 问题二
# ---------------------------------------------------------------------------
def fig_q2_raw(data):
    load, pv = data['load_act'], data['pv_act']
    n = load.shape[0]
    months = np.array([(mc.DAY0 + _dt.timedelta(days=i)).month for i in range(n)])
    mean_load = np.array([load[months == m].mean() for m in range(1, 13)])
    mean_pv = np.array([pv[months == m].mean() for m in range(1, 13)])
    fig, ax = ps.publication_subplots(width='report', aspect=0.5)
    index = np.arange(12)
    ax.bar(index - 0.2, mean_load, width=0.4,
           color=ps.PALETTE['primary'], label='负荷')
    ax.bar(index + 0.2, mean_pv, width=0.4,
           color=ps.PALETTE['secondary'], label='光伏')
    ax.set_xticks(index)
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_ylabel('月均功率 / kW')
    ax.set_title('全年逐月平均负荷与光伏')
    ax.set_ylim(*_zero_top(np.concatenate([mean_load, mean_pv])))
    ax.legend(loc='upper left', ncols=2)
    _save(fig, 'raw_q2_年度负荷与光伏', '全年逐月平均负荷与光伏', 'q2', 'raw',
          '附件 2 全年实测负荷与光伏的逐月均值，全年呈夏高冬低特征。')


def fig_q2_process_error():
    df = _read('问题2_预测误差对比.csv')
    names = list(df['预测器'])
    series = [('load_MAE', '负荷', 'primary'),
              ('pv_MAE', '光伏', 'secondary'),
              ('net_MAE', '净负荷', 'contrast')]
    fig, ax = ps.publication_subplots(width='report', aspect=0.5)
    width = 0.8 / len(series)
    for k, (col, label, color_key) in enumerate(series):
        values = df[col].to_numpy(dtype=float)
        ax.bar(np.arange(len(names)) + (k - 1) * width, values, width=width,
               color=ps.PALETTE[color_key], label=label)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel('MAE / kW')
    ax.set_title('不同预测器的净负荷误差')
    top = df[[col for col, _, _ in series]].to_numpy(dtype=float).max()
    ax.set_ylim(0.0, float(top) * 1.15)
    ax.legend(loc='upper left', ncols=3)
    _save(fig, 'process_q2_预测误差对比', '不同预测器的净负荷误差', 'q2',
          'process', 'F0 为直接外推，F1 为同槽多日中心，F2 为平滑中心。')


def fig_q2_process_safe(data):
    prof = sl.build_pv_archive(data, 30)
    resid, _ = sl.q3_net_residual(data, prof, (0, 1, 2, 3), 'F1', 'archive')
    day = sl.DAY_INDEX[sl.REP_DATES[1]]
    lhat, vhat = sl.predict_center('F1', day, data)
    reserve_kw, _ = sl.q3_reserve(day, resid, 0.9, 30, 'stratified')
    fig, ax = ps.publication_subplots(width='report', aspect=0.5)
    ax.plot(X, data['load_act'][day] - data['pv_act'][day],
            color=ps.PALETTE['dark'], label='实测净负荷')
    ax.plot(X, lhat - vhat, color=ps.PALETTE['primary'], label='中心预测')
    ax.plot(X, lhat - vhat + reserve_kw, color=ps.PALETTE['contrast'],
            label='安全轨迹')
    ax.set_xlabel('时刻')
    ax.set_ylabel('功率 / kW')
    ax.set_title('安全轨迹与实测净负荷')
    ax.legend(loc='upper left', ncols=1)
    _hours(ax)
    _save(fig, 'process_q2_安全轨迹覆盖示意', '安全轨迹与实测净负荷', 'q2',
          'process', '以 6 月 21 日为例，安全轨迹在中心预测上叠加分层风险储备，' 
          '使实测净负荷基本不越界。')


def fig_q2_result_sens():
    df = _read('问题2_配置敏感性.csv')
    labels = ['%s/W%d/α%.2f' % (p, int(w), float(a))
              for p, w, a in zip(df['预测器'], df['W'], df['alpha'])]
    values = df['cash_total'].to_numpy(dtype=float) / 1e4
    fig, ax = ps.publication_subplots(width='report', aspect=0.62)
    ypos = np.arange(len(values))
    bars = ax.barh(ypos, values, height=0.62, color=ps.PALETTE['primary'])
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('年度现金费 / 万元')
    ax.set_title('配置敏感性：年度现金费')
    ax.set_xlim(0.0, float(values.max()) * 1.16)
    ax.invert_yaxis()
    for bar, value in zip(bars, values):
        ax.text(value, bar.get_y() + bar.get_height() / 2.0, ' %.1f' % value,
                ha='left', va='center', fontsize=6.5)
    _save(fig, 'result_q2_配置敏感性', '配置敏感性：年度现金费', 'q2', 'result',
          '预测器 F1 配合储备窗口 30 天与风险水平 0.9 的年度现金费最低。')


def fig_q2_result_monthly():
    df = _read('问题2_逐日汇总.csv')
    grouped = df.groupby('月份', sort=True)
    cash = grouped['现金费_元'].sum().to_numpy() / 1e4
    emergency = grouped['应急购电量_kWh'].sum().to_numpy()
    labels = ['%02d月' % int(str(m)[-2:]) for m in grouped.groups]
    index = np.arange(len(cash))
    fig, axes = ps.publication_subplots(2, 1, width='report', aspect=0.72)
    a, b = axes
    a.bar(index, cash, width=0.6, color=ps.PALETTE['primary'])
    a.set_xticks(index)
    a.set_xticklabels(labels)
    a.set_ylabel('费用 / 万元')
    a.set_title('逐月现金费')
    a.set_ylim(*_zero_top(cash))
    b.bar(index, emergency, width=0.6, color=ps.PALETTE['contrast'])
    b.set_xticks(index)
    b.set_xticklabels(labels)
    b.set_xlabel('月份')
    b.set_ylabel('电量 / kWh')
    b.set_title('逐月应急购电量')
    b.set_ylim(*_zero_top(emergency))
    _plabel(a, 'a')
    _plabel(b, 'b')
    _save(fig, 'result_q2_月度费用与应急', '逐月现金费与应急购电量', 'q2', 'result',
          '应急购电量集中在冬季与晚高峰，全年约 4.7 万 kWh。')


# ---------------------------------------------------------------------------
# 问题三
# ---------------------------------------------------------------------------
def fig_q3_raw(data):
    prof = sl.build_pv_archive(data, 30)
    resid, _ = sl.q3_net_residual(data, prof, (0, 1, 2, 3), 'F1', 'archive')
    values = resid[31:].ravel()
    fig, ax = ps.publication_subplots(width='report', aspect=0.5)
    ax.hist(values, bins=60, color=ps.PALETTE['primary'])
    ax.axvline(float(np.quantile(values, 0.10)),
               color=ps.PALETTE['contrast'], lw=0.9, label='10 分位')
    ax.axvline(float(np.quantile(values, 0.90)),
               color=ps.PALETTE['secondary'], lw=0.9, label='90 分位')
    ax.set_xlabel('净负荷预测残差 / kW')
    ax.set_ylabel('频数')
    ax.set_title('净负荷预测残差分布')
    ax.legend(loc='upper right', ncols=1)
    _save(fig, 'raw_q3_净负荷残差分布', '净负荷预测残差分布', 'q3', 'raw',
          '正式窗口内 F1 预测的净负荷残差分布，尾部不厚但存在正偏。')


def fig_q3_process_leadtime():
    df = _read('问题3_预见期尺度.csv')
    # 有效提前期仅 1..19 h：构造上 20 h 以后只有 0 时机在范围内，
    # 而 20:00 以后实际光伏恒为 0，误差恒等于 0，属于构造性 0，
    # 画出来会误导，故截断到 19 h 并在注释中说明。
    df = df[df['提前期_h'] <= 19]
    series = [('MAE_kW', 'MAE', 'primary'),
              ('RMSE_kW', 'RMSE', 'secondary'),
              ('q90_kW', '90 分位', 'accent')]
    fig, ax = ps.publication_subplots(width='report', aspect=0.5)
    for col, label, color_key in series:
        ax.plot(df['提前期_h'], df[col], color=ps.PALETTE[color_key], label=label)
    ax.set_xlabel('提前期 / h')
    ax.set_ylabel('误差 / kW')
    ax.set_title('净负荷误差随提前期变化')
    ax.legend(loc='upper left', ncols=3)
    _save(fig, 'process_q3_预见期误差', '净负荷误差随提前期变化', 'q3', 'process',
          '误差随提前期增长，13 h 达峰（MAE 约 324 kW，此时 00/06 两个发布时刻在范围内）；'
          '20 h 起只剩 00 时机、且目标落在夜间光伏恒零段，误差为构造性零，故截断至 19 h；'
          '13 h 以后的下降含“在范围内发布时刻减少”的成分效应，不宜单独解读。')


def fig_q3_process_info():
    df = _read('问题3_信息消融.csv')
    labels = []
    for item in df['获准集合S']:
        text = str(item)
        labels.append('空集' if text.startswith('空集') else text)
    values = df['kappa_star_元每次'].to_numpy(dtype=float)
    return labels, values


def _fig_q3_process_info(labels, values):
    fig, ax = ps.publication_subplots(width='report', aspect=0.55)
    index = np.arange(len(values))
    ax.bar(index, values, width=0.6, color=ps.PALETTE['primary'])
    ax.axhline(0.0, color=ps.PALETTE['neutral'], lw=0.7)
    ax.set_xticks(index)
    ax.set_xticklabels(labels)
    ax.set_xlabel('获准发布时刻集合')
    ax.set_ylabel('边际收益 / 元每次')
    ax.set_title('信息消融的边际收益')
    span = float(np.nanmax(np.abs(values))) * 1.25
    ax.set_ylim(-span, span)
    _save(fig, 'process_q3_信息消融边际收益', '信息消融的边际收益', 'q3',
          'process', '新增 6 时点边际收益最高，18 时点单独放开为负，说明信息需组合使用。')


def fig_q3_result_compare():
    df = _read('问题3_方案对比.csv')
    tags = ['main', 'uniform', 'h72']
    names = {'main': '分层储备', 'uniform': '统一分位', 'h72': 'H=72'}
    sub = df.set_index('tag').loc[tags]
    cash = sub['cash_total'].to_numpy(dtype=float) / 1e4
    emergency = sub['emergency_kwh'].to_numpy(dtype=float)
    index = np.arange(len(tags))
    labels = [names[t] for t in tags]
    fig, axes = ps.publication_subplots(2, 1, width='report', aspect=0.72)
    a, b = axes
    bar_a = a.bar(index, cash, width=0.55, color=ps.PALETTE['primary'])
    a.set_xticks(index)
    a.set_xticklabels(labels)
    a.set_ylabel('费用 / 万元')
    a.set_title('年度现金费对比')
    a.set_ylim(*_zero_top(cash))
    _annotate_bars(a, bar_a, cash, '%.1f')
    bar_b = b.bar(index, emergency, width=0.55, color=ps.PALETTE['contrast'])
    b.set_xticks(index)
    b.set_xticklabels(labels)
    b.set_xlabel('方案')
    b.set_ylabel('电量 / kWh')
    b.set_title('应急购电量对比')
    b.set_ylim(*_zero_top(emergency))
    _annotate_bars(b, bar_b, emergency, '%.0f')
    _plabel(a, 'a')
    _plabel(b, 'b')
    _save(fig, 'result_q3_方案对比', '方案对比：费用与应急购电', 'q3', 'result',
          '分层储备优于统一分位，H=72 因迟滞最优而费用与应急量同时上升。')


def fig_q3_result_day():
    bundle = np.load(CSV / 'q3_bundle.npz')
    day = sl.DAY_INDEX[sl.REP_DATES[1]]
    fig, axes = ps.publication_subplots(2, 1, width='report', aspect=0.72)
    a, b = axes
    a.step(X, bundle[Q3_MAIN + '_X'][day] / mc.DT, where='post',
           color=ps.PALETTE['primary'], label='充电')
    a.step(X, bundle[Q3_MAIN + '_Y'][day] / mc.DT, where='post',
           color=ps.PALETTE['contrast'], label='放电')
    a.set_ylabel('功率 / kW')
    a.set_title('重点日充放电功率')
    a.legend(loc='upper left', ncols=2)
    _hours(a)
    b.plot(X, bundle[Q3_MAIN + '_S'][day], color=ps.PALETTE['positive'])
    b.axhline(mc.S_MIN, color=ps.PALETTE['neutral'], lw=0.7)
    b.axhline(mc.S_MAX, color=ps.PALETTE['neutral'], lw=0.7)
    b.set_xlabel('时刻')
    b.set_ylabel('储电量 / kWh')
    b.set_title('重点日储电量轨迹')
    _hours(b)
    _plabel(a, 'a')
    _plabel(b, 'b')
    _save(fig, 'result_q3_重点日充放电与储能', '重点日充放电与储电量', 'q3',
          'result', '以 6 月 21 日为例，滚动执行层在电价高峰放电、低谷充电。')


# ---------------------------------------------------------------------------
# 问题四
# ---------------------------------------------------------------------------
def fig_q4_raw():
    df = _read('问题4_价格基线误差.csv')
    labels = list(df['基线说明'])
    mae0 = df['MAE0_元每kWh'].to_numpy(dtype=float)
    mae12 = df['MAE12_元每kWh'].to_numpy(dtype=float)
    index = np.arange(len(labels))
    width = 0.38
    fig, ax = ps.publication_subplots(width='report', aspect=0.5)
    ax.bar(index - width / 2, mae0, width=width,
           color=ps.PALETTE['primary'], label='0 时基准')
    ax.bar(index + width / 2, mae12, width=width,
           color=ps.PALETTE['secondary'], label='12 时修正')
    ax.set_xticks(index)
    ax.set_xticklabels(labels)
    ax.set_ylabel('MAE / (元/kWh)')
    ax.set_title('价格基线预测误差')
    ax.set_ylim(0.0, float(np.max([mae0.max(), mae12.max()])) * 1.18)
    ax.legend(loc='upper left', ncols=2)
    _save(fig, 'raw_q4_价格基线误差', '价格基线预测误差', 'q4', 'raw',
          '四条价格基线在 0 时与 12 时修正后的绝对误差；日内修正显著降低误差。')


def fig_q4_process(data):
    day = sl.DAY_INDEX[sl.REP_DATES[1]]
    actual = data['price_act']
    base = mc.price_forecast_0h(day, actual, data['price1'], 'median30')
    corrected = mc.price_intraday_correct(base, actual[day], 72)
    fig, ax = ps.publication_subplots(width='report', aspect=0.5)
    ax.plot(X, actual[day], color=ps.PALETTE['dark'], label='实际电价')
    ax.plot(X, base, color=ps.PALETTE['primary'], label='0 时基准')
    ax.plot(X, corrected, color=ps.PALETTE['contrast'], label='12 时修正')
    ax.set_xlabel('时刻')
    ax.set_ylabel('电价 / (元/kWh)')
    ax.set_title('价格预测的日内修正')
    ax.legend(loc='upper left', ncols=1)
    _hours(ax)
    _save(fig, 'process_q4_日内价格修正', '价格预测的日内修正', 'q4', 'process',
          '以 6 月 21 日为例，利用已完成时段误差外推可显著逼近真实价格。')


def fig_q4_result():
    df = _read('问题4_方案对比.csv')
    tags = ['Q4-2_median30', 'Q4-3_median30', 'Q4-2_oracle', 'Q4-3_oracle']
    names = ['模式一', '模式二', '模式一I', '模式二I']
    sub = df.set_index('tag').loc[tags]
    cash = sub['cash_total'].to_numpy(dtype=float) / 1e4
    emergency = sub['emergency_kwh'].to_numpy(dtype=float)
    index = np.arange(len(tags))
    fig, axes = ps.publication_subplots(2, 1, width='report', aspect=0.72)
    a, b = axes
    bar_a = a.bar(index, cash, width=0.55, color=ps.PALETTE['primary'])
    a.set_xticks(index)
    a.set_xticklabels(names)
    a.set_ylabel('费用 / 万元')
    a.set_title('两模式年度现金费')
    a.set_ylim(*_zero_top(cash))
    _annotate_bars(a, bar_a, cash, '%.1f')
    bar_b = b.bar(index, emergency, width=0.55, color=ps.PALETTE['contrast'])
    b.set_xticks(index)
    b.set_xticklabels(names)
    b.set_xlabel('方案（I 表示 0 时已知真价的理想信息）')
    b.set_ylabel('电量 / kWh')
    b.set_title('两模式应急购电量')
    b.set_ylim(*_zero_top(emergency))
    _annotate_bars(b, bar_b, emergency, '%.0f')
    _plabel(a, 'a')
    _plabel(b, 'b')
    _save(fig, 'result_q4_两模式费用对比', '两模式费用与应急购电对比', 'q4',
          'result', '模式二可调合同较模式一降低费用；理想信息下两者进一步下降。')


def write_panel():
    order = {'q1': 0, 'q2': 1, 'q3': 2, 'q4': 3}
    cat_order = {'raw': 0, 'process': 1, 'result': 2}
    cat_name = {'raw': '原始数据图', 'process': '过程机理图', 'result': '结果图'}
    items = sorted(PANELS, key=lambda r: (order.get(r['question'], 9),
                                          cat_order.get(r['category'], 9),
                                          r['stem']))
    parts = ['<!DOCTYPE html>', '<html lang="zh-CN">', '<head>',
             '<meta charset="utf-8">',
             '<title>图表面板</title>',
             '<style>body{font-family:"Microsoft YaHei",Arial,sans-serif;'
             'margin:24px;color:#222}h1{font-size:20px}'
             'h2{font-size:15px;margin-top:22px}figure{display:inline-block;'
             'margin:8px;width:320px;vertical-align:top}'
             'img{width:100%;border:1px solid #ddd}figcaption{font-size:12px;'
             'color:#444;margin-top:4px}</style>', '</head>', '<body>',
             '<h1>图表面板索引</h1>']
    for question in ('q1', 'q2', 'q3', 'q4'):
        for category in ('raw', 'process', 'result'):
            group = [r for r in items if r['question'] == question
                     and r['category'] == category]
            if not group:
                continue
            parts.append('<h2>问题 %s · %s</h2>'
                         % (question[1:], cat_name[category]))
            for record in group:
                parts.append('<figure><img src="%s" alt="%s">'
                             '<figcaption><strong>%s</strong><br>%s</figcaption>'
                             '</figure>'
                             % (html.escape(record['png']),
                                html.escape(record['title']),
                                html.escape(record['title']),
                                html.escape(record['note'])))
    parts.extend(['</body>', '</html>'])
    body = chr(10).join(parts)
    (FIG / '图表面板.html').write_text(body, encoding='utf-8')
    print('[html] 图表面板.html panels=%d' % len(items), flush=True)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    ps.apply_publication_style('zh', 'report')
    data = mc.load_inputs()

    fig_q1_raw(data)
    fig_q1_process(data)
    fig_q1_result_battery()
    fig_q1_result_plan()

    fig_q2_raw(data)
    fig_q2_process_error()
    fig_q2_process_safe(data)
    fig_q2_result_sens()
    fig_q2_result_monthly()

    fig_q3_raw(data)
    fig_q3_process_leadtime()
    labels, values = fig_q3_process_info()
    _fig_q3_process_info(labels, values)
    fig_q3_result_compare()
    fig_q3_result_day()

    fig_q4_raw()
    fig_q4_process(data)
    fig_q4_result()

    write_panel()
    print('[done] figures=%d' % len(PANELS))


if __name__ == '__main__':
    main()
