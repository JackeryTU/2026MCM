"""Causal forecast diagnostics, full-run checks and required paper table extracts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from microgrid_core import DT_HOURS, ISSUES, N_SLOTS
from test_core import physical


def forecast_diagnostics(data, lp, vp, lr, vr, ir):
    rows = []
    def record(name, truth, pred, threshold, **extra):
        valid = np.isfinite(truth) & np.isfinite(pred)
        truth, pred = truth[valid], pred[valid]
        err = truth-pred
        eligible = truth > threshold
        row = dict(model=name, count=int(len(err)), mae_kw=float(np.mean(abs(err))),
                   rmse_kw=float(np.sqrt(np.mean(err**2))), bias_kw=float(np.mean(err)),
                   mape_percent=float(np.mean(abs(err[eligible])/truth[eligible])*100) if eligible.any() else np.nan,
                   mape_count=int(eligible.sum()), **extra)
        rows.append(row)
        return row
    for name, truth, pred, residual, threshold in (("analog_load", data.actual_load_kw, lp, lr, 1), ("analog_pv", data.actual_pv_kw, vp, vr, 100)):
        row = record(name, truth[31:], pred[31:], threshold, evaluation="Feb-Dec, ten-minute endpoint")
        covered, width = [], []
        for d in range(31, 365):
            lo, hi = np.quantile(residual[max(0,d-60):d], [.05,.95], axis=0, method="inverted_cdf")
            lo, hi = np.maximum(0,pred[d]+lo), np.maximum(0,pred[d]+hi)
            covered.extend(((truth[d]>=lo)&(truth[d]<=hi)).tolist())
            width.extend((hi-lo).tolist())
        row.update(empirical_90_coverage=float(np.mean(covered)), interval_mean_width_kw=float(np.mean(width)), interval_source="past <=60 raw rolling-origin residual days; no guarantee")
    truth_h = data.pv_forecast_hourly + ir
    # Raw long-lead diagnostics may cross midnight; unavailable next-year truth excluded.
    for i, hour in enumerate(ISSUES):
        record("official_pv_raw", truth_h[31:,i,:], data.pv_forecast_hourly[31:,i,:], 100, issue_hour=hour, evaluation="Feb-Dec issues, all available 1-24h targets")
        h = 24-hour
        raw = data.pv_forecast_hourly[31:,i,:h]
        corrected, cover, widths = [], [], []
        for d in range(31,365):
            past = ir[max(0,d-60):d,i,:h]
            bias = np.mean(past,axis=0)
            corrected.append(np.maximum(0, data.pv_forecast_hourly[d,i,:h]+bias))
            lo,hi = np.quantile(past,[.05,.95],axis=0,method="inverted_cdf")
            lo,hi = np.maximum(0,data.pv_forecast_hourly[d,i,:h]+lo),np.maximum(0,data.pv_forecast_hourly[d,i,:h]+hi)
            cover.extend(((truth_h[d,i,:h]>=lo)&(truth_h[d,i,:h]<=hi)).tolist())
            widths.extend((hi-lo).tolist())
        row = record("official_pv_bias_corrected", truth_h[31:,i,:h],np.asarray(corrected),100,issue_hour=hour,evaluation="Feb-Dec, same-day hourly targets only")
        row.update(empirical_90_coverage=float(np.mean(cover)),interval_mean_width_kw=float(np.mean(widths)),interval_source="past <=60 raw hourly residual days; no guarantee")
        record("official_pv_raw_same_day",truth_h[31:,i,:h],raw,100,issue_hour=hour,evaluation="Feb-Dec, same-day hourly targets only")
    for lead in range(1,25):
        record("official_pv_raw_lead",truth_h[31:,:,lead-1],data.pv_forecast_hourly[31:,:,lead-1],100,lead_hour=lead,evaluation="Feb-Dec issues, available targets")
    return pd.DataFrame(rows)


def validate_day(result, adjusted, previous_soc, battery=None):
    if battery is None: physical(result, adjusted)
    else: physical(result, adjusted,battery)
    assert_allclose(result["actual"]["soc_kwh"][0],previous_soc,atol=1e-8,rtol=0)
    a=result["actual"]
    for key,value in a.items():
        if isinstance(value,np.ndarray): assert np.isfinite(value).all(),key
    assert_allclose(a["cash_cost"],a["settlement_cost"]+a["emergency_cost"],atol=1e-7)


def write_representative_tables(root, variants, dates, output_dir=None):
    output_dir = Path(output_dir) if output_dir is not None else Path(root) / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    purchase,storage,emergency,updates=[],[],[],[]
    for name,(results,adjusted) in variants.items():
        for r in results:
            if r["date"] not in dates: continue
            date=r["date"].date().isoformat(); a=r["actual"]
            p=r["initial_plan_kwh"] if adjusted else r["plan"]["grid_kwh"]
            q=r["final_contract_kwh"] if adjusted else p
            for mode,values in (("initial_plan",p),("final_contract",q)):
                row=dict(variant=name,date=date,quantity=mode,**{f"{h:02d}:00-{h:02d}:10_kwh":float(values[h*6]) for h in (10,12,14,16,18,20)},day_purchase_kwh=float(values.sum()),actual_cash_cost_yuan=float(a["cash_cost"].sum()))
                purchase.append(row)
            for g in range(6):
                sl=slice(g*24,(g+1)*24)
                storage.append(dict(variant=name,date=date,period=f"{g*4}:00-{(g+1)*4}:00",charge_kwh=float(a["charge_kwh"][sl].sum()),discharge_kwh=float(a["discharge_kwh"][sl].sum()),soc_0_kwh=float(a["soc_kwh"][0]),soc_24_kwh=float(a["soc_kwh"][-1])))
            slots=np.flatnonzero(a["emergency_kwh"]>1e-8)
            for t in slots:
                m=int(t)*10
                emergency.append(dict(variant=name,date=date,interval=f"{m//60:02d}:{m%60:02d}-{(m+10)//60:02d}:{(m+10)%60:02d}",emergency_kwh=float(a["emergency_kwh"][t])))
            if not len(slots): emergency.append(dict(variant=name,date=date,interval="无",emergency_kwh=0.))
            if adjusted:
                for event,pred in zip(r["events"],r["update_solutions"]):
                    for j,value in enumerate(pred["grid_kwh"]):
                        updates.append(dict(variant=name,date=date,issue_hour=event,slot=event*6+j+1,purchase_kwh=float(value)))
    for filename,rows in (("表1_代表日购电.csv",purchase),("表2_代表日储能.csv",storage),("表3_代表日应急.csv",emergency),("代表日历次更新.csv",updates)):
        pd.DataFrame(rows).to_csv(output_dir/filename,index=False,encoding="utf-8-sig")


def raw_data_profile(data):
    rows=[]
    for name,values,unit in (("actual_load",data.actual_load_kw,"kW"),("actual_pv",data.actual_pv_kw,"kW"),("actual_price",data.variable_price,"yuan/kWh")):
        for month in range(1,13):
            x=values[data.dates.month==month].ravel()
            q1,q3=np.quantile(x,[.25,.75]);iqr=q3-q1
            rows.append(dict(variable=name,month=month,unit=unit,count=x.size,missing=int(np.isnan(x).sum()),minimum=x.min(),q25=q1,median=np.median(x),q75=q3,maximum=x.max(),mean=x.mean(),std=x.std(),tukey_outlier_count=int(((x<q1-1.5*iqr)|(x>q3+1.5*iqr)).sum()),outlier_action="retain physical nonnegative observations"))
    return pd.DataFrame(rows)
