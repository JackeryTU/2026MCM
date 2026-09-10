"""Causal annual baselines, update ablations and one-at-a-time sensitivity.

Run only after independent P1 PASS. All cases start January 1 and carry SOC;
the reported evaluation period is February--December. No file hashes are used.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import microgrid_core as m
from evaluate_results import validate_day
from run_all import REPRESENTATIVE_DATES, file_metadata


def cases():
    specs = []
    def add(name, **kwargs):
        specs.append(dict(name=name, adjusted=False, variable=False, alpha=.8,
                          history_days=60, max_scenarios=9, baseline="safe",
                          capacity=12000., power=5000., eta=.9, emergency=5.,
                          events=m.ISSUES, **{} ) | kwargs)
    for variable, prefix in ((False, "q2"), (True, "q42")):
        for baseline in ("safe", "point", "typical", "no_storage"):
            add(f"{prefix}_{baseline}", variable=variable, baseline=baseline)
    for variable, prefix in ((False, "q3"), (True, "q43")):
        for label, events in (("0h", (0,)), ("12h", (0,12)),
                              ("6h", m.ISSUES), ("2h", tuple(range(0,24,2)))):
            add(f"{prefix}_{label}", adjusted=True, variable=variable, events=events)
    # Both a day-ahead and an intraday strategy: changes are not pooled.
    for adjusted, variable, prefix in ((False, False, "q2"), (True, True, "q43")):
        common = dict(adjusted=adjusted, variable=variable)
        for a in (.70,.75,.85,.90,.95):
            add(f"{prefix}_alpha_{a:.2f}", **common, alpha=a)
        for k in (5,12,16):
            add(f"{prefix}_clusters_{k}", **common, max_scenarios=k)
        for h in (30,90):
            add(f"{prefix}_history_{h}", **common, history_days=h)
        for factor in (.8,1.2):
            add(f"{prefix}_capacity_{factor}", **common, capacity=12000*factor)
            add(f"{prefix}_power_{factor}", **common, power=5000*factor)
        add(f"{prefix}_roundtrip_0.9", **common, eta=float(np.sqrt(.9)))
        # Holding alpha=.8 isolates tariff exposure; not optimal-policy tuning.
        for penalty in (3.,4.,6.,8.):
            add(f"{prefix}_emergency_{penalty:g}", **common, emergency=penalty)
    return specs


def prepare_case(spec, data, forecasts, cache):
    lp,lr,vp,vr,ir = forecasts
    key = (spec["adjusted"],spec["alpha"],spec["history_days"],spec["max_scenarios"])
    if spec["baseline"] in ("point","typical"):
        if spec["baseline"] == "typical":
            return [(data.typical_load_kw,data.typical_pv_kw,{"baseline":"typical"}) for _ in data.dates]
        return [(lp[d],vp[d],{"baseline":"uncorrected_point"}) for d in range(len(data.dates))]
    if key not in cache:
        kwargs = dict(alpha=spec["alpha"], history_days=spec["history_days"], max_scenarios=spec["max_scenarios"])
        if spec["adjusted"]:
            value = [[m.safe_update_trajectory(data,d,i,lp[d],lr,ir,**kwargs) for i in range(4)] for d in range(len(data.dates))]
        else:
            value = [m.safe_analog_trajectory(d,lp[d],vp[d],lr,vr,**kwargs) for d in range(len(data.dates))]
        cache[key] = value
    return cache[key]


def simulate(spec, data, forecasts, safe_cache, output):
    started=time.perf_counter()
    lp,lr,vp,vr,ir=forecasts
    no_storage=spec["baseline"]=="no_storage"
    battery=m.Battery(capacity_kwh=0. if no_storage else spec["capacity"],
                      power_kw=0. if no_storage else spec["power"],
                      eta_c=spec["eta"],eta_d=spec["eta"])
    trajectories=prepare_case(spec,data,forecasts,safe_cache)
    soc=battery.initial
    rows=[]
    for d,date in enumerate(data.dates):
        common=dict(variable_price=spec["variable"], battery=battery,
                    alpha=spec["alpha"], emergency_multiplier=spec["emergency"])
        if spec["adjusted"]:
            r=m.run_q3_day(data,d,soc,lp,lr,ir,safe_updates=trajectories[d],
                          events=tuple(spec["events"]),**common)
        else:
            r=m.run_q2_day(data,d,soc,lp,vp,lr,vr,safe_pair=trajectories[d],**common)
        validate_day(r,spec["adjusted"],soc,battery)
        soc=float(r["actual"]["soc_kwh"][-1])
        a=r["actual"]
        row=dict(date=date.date().isoformat(),**m.summarize_day(r,adjusted=spec["adjusted"]))
        row.update(emergency_slots=int(np.count_nonzero(a["emergency_kwh"]>1e-8)),
                   settlement_yuan=float(a["settlement_cost"].sum()),
                   emergency_yuan=float(a["emergency_cost"].sum()),
                   soc_bound_slots=int(np.count_nonzero((a["soc_kwh"][:-1]<=battery.minimum+1e-7)|(a["soc_kwh"][:-1]>=battery.maximum-1e-7))))
        rows.append(row)
        if d%90==0 or d==len(data.dates)-1:
            print(f"[{spec['name']}] {d+1}/{len(data.dates)}",flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(output/f"{spec['name']}_daily.csv",index=False,encoding="utf-8-sig")
    evaluated=frame[frame.date>="2025-02-01"]
    assert len(evaluated)==334
    totals={k:float(evaluated[k].sum()) for k in ("cash_cost_yuan","alternative_cash_cost_yuan","plan_kwh","final_contract_kwh","emergency_kwh","unused_contract_kwh","pv_curtailed_kwh","charge_kwh","discharge_kwh","emergency_slots","settlement_yuan","emergency_yuan")}
    total_pv=float(data.actual_pv_kw[31:].sum()*m.DT_HOURS)
    totals.update(days=334,soc_end_kwh=soc,
                  soc_bound_rate=float(evaluated.soc_bound_slots.sum()/(334*144)),
                  equivalent_cycles=(totals["charge_kwh"]+totals["discharge_kwh"])/(2*battery.capacity_kwh) if battery.capacity_kwh else None,
                  pv_curtailment_rate=totals["pv_curtailed_kwh"]/total_pv,
                  max_balance_residual_kwh=float(evaluated.max_balance_residual_kwh.max()))
    report=dict(spec=spec,battery=asdict(battery),initial_soc_kwh=battery.initial,
                totals=totals,representative=frame[frame.date.isin([d.date().isoformat() for d in REPRESENTATIVE_DATES])].to_dict("records"),
                source_files=[file_metadata(Path(__file__)),file_metadata(Path(m.__file__))],
                hash_checks="omitted per user request",runtime_seconds=time.perf_counter()-started)
    (output/f"{spec['name']}.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report


def aggregate(output):
    reports={}
    for spec in cases():
        path=output/f"{spec['name']}.json"
        if path.exists():
            reports[spec["name"]]=json.loads(path.read_text(encoding="utf-8"))
    rows=[r["spec"]|r["totals"] for r in reports.values()]
    pd.DataFrame(rows).to_csv(output/"实验总表.csv",index=False,encoding="utf-8-sig")
    updates=[]
    for prefix in ("q3","q43"):
        base=reports.get(f"{prefix}_6h")
        if base is None: continue
        for suffix in ("0h","12h","6h","2h"):
            candidate=reports.get(f"{prefix}_{suffix}")
            if candidate is None: continue
            benefit=base["totals"]["cash_cost_yuan"]-candidate["totals"]["cash_cost_yuan"]
            extra=(len(candidate["spec"]["events"])-4)*334
            rep_base={r["date"]:r["cash_cost_yuan"] for r in base["representative"]}
            savings=[rep_base[r["date"]]-r["cash_cost_yuan"] for r in candidate["representative"]]
            updates.append(dict(variant=prefix,candidate=suffix,saving_yuan=benefit,
                saving_percent=100*benefit/base["totals"]["cash_cost_yuan"],
                extra_events=extra,breakeven_yuan_per_extra_event=benefit/extra if extra>0 else None,
                all_four_days_nonworse=all(s>=-1e-5 for s in savings),
                recommended_extra_updates=extra>0 and benefit>.001*base["totals"]["cash_cost_yuan"] and all(s>=-1e-5 for s in savings),
                alternative_saving_yuan=base["totals"]["alternative_cash_cost_yuan"]-candidate["totals"]["alternative_cash_cost_yuan"]))
    pd.DataFrame(updates).to_csv(output/"更新频率比较.csv",index=False,encoding="utf-8-sig")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--project-root",default=".")
    parser.add_argument("--only",nargs="*",help="exact case names; omitted runs every case")
    parser.add_argument("--list",action="store_true")
    args=parser.parse_args()
    specs=cases()
    if args.list:
        print(json.dumps(specs,ensure_ascii=False,indent=2)); return
    if args.only is not None:
        unknown=set(args.only)-{s["name"] for s in specs}
        if unknown: parser.error(f"unknown cases: {sorted(unknown)}")
        specs=[s for s in specs if s["name"] in args.only]
    root=Path(args.project_root).resolve()
    output=root/"results"/"experiments"
    output.mkdir(parents=True,exist_ok=True)
    data=m.load_inputs(root)
    lp,lr=m.build_analog_forecasts(data.actual_load_kw,data.typical_load_kw,weekday_weight=True)
    vp,vr=m.build_analog_forecasts(data.actual_pv_kw,data.typical_pv_kw,weekday_weight=False)
    ir=m.official_pv_residuals(data)
    cache={}
    for spec in specs:
        simulate(spec,data,(lp,lr,vp,vr,ir),cache,output)
        aggregate(output)
    # Typical-day alternative endpoint interpretation: a simultaneous +10min roll.
    shifted=replace(data,typical_load_kw=np.roll(data.typical_load_kw,1),
                    typical_pv_kw=np.roll(data.typical_pv_kw,1),typical_price=np.roll(data.typical_price,1))
    time_rows=[]
    for name,inputs in (("natural_endpoints",data),("template_start_shift_10min",shifted)):
        r=m.run_q1(inputs)
        assert r["max_eq_residual"]<1e-6 and r["simultaneous_flow_max"]<1e-5
        time_rows.append(dict(mapping=name,cost_yuan=r["cash_contract_cost"],purchase_kwh=float(r["grid_kwh"].sum()),soc_start_kwh=r["soc_kwh"][0],soc_end_kwh=r["soc_kwh"][-1]))
    pd.DataFrame(time_rows).to_csv(output/"时间口径敏感性.csv",index=False,encoding="utf-8-sig")
    print(f"completed {len(specs)} requested experiments; hashes omitted")


if __name__=="__main__":
    main()
