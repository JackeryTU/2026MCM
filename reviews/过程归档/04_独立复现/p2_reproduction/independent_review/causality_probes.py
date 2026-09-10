"""Additional reviewer-only nonanticipativity probes; no annual simulation."""
from dataclasses import replace
from pathlib import Path
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
import microgrid_core as m

data=m.load_inputs(ROOT)
lp,lr=m.build_analog_forecasts(data.actual_load_kw,data.typical_load_kw,weekday_weight=True)
ir=m.official_pv_residuals(data)
records=[]
for day in (0,1,364):
    for event in range(0,24,2):
        changed=replace(data,actual_load_kw=data.actual_load_kw.copy(),actual_pv_kw=data.actual_pv_kw.copy(),variable_price=data.variable_price.copy(),pv_forecast_hourly=data.pv_forecast_hourly.copy())
        for arr,delta in ((changed.actual_load_kw,1732),(changed.actual_pv_kw,2537),(changed.variable_price,11)):
            arr[day,event*6:]+=delta
            arr[day+1:]+=delta
        changed.pv_forecast_hourly[day+1:]+=4317
        for j,h in enumerate((0,6,12,18)):
            if h>event:changed.pv_forecast_hourly[day,j]+=4317
        lp2,lr2=m.build_analog_forecasts(changed.actual_load_kw,changed.typical_load_kw,weekday_weight=True)
        ir2=m.official_pv_residuals(changed)
        issue=event//6
        one=m.safe_update_trajectory(data,day,issue,lp[day],lr,ir)
        two=m.safe_update_trajectory(changed,day,issue,lp2[day],lr2,ir2)
        one=m.nowcast_trajectory(data,day,event,one)
        two=m.nowcast_trajectory(changed,day,event,two)
        price=m.predict_price(data,day,event)
        price2=m.predict_price(changed,day,event)
        differences=[float(np.max(abs(a-b),initial=0)) for a,b in zip((lp[day],one[0],one[1],price),(lp2[day],two[0],two[1],price2))]
        first=m.solve_schedule(one[0],one[1],price,6000)['grid_kwh']
        second=m.solve_schedule(two[0],two[1],price2,6000)['grid_kwh']
        differences.append(float(np.max(abs(first-second),initial=0)))
        assert max(differences)==0,(day,event,differences)
        assert (event in (0,6,12,18)) or one[2]['synthetic_nowcast'] is True
        records.append(dict(day=day,event=event,maximum_difference=max(differences)))
# Physical controller decisions do not read later actuals, even if later
# settlement prices change: only the bill suffix is allowed to change.
q=np.full(144,650.);day=78;cut=73
original=m.causal_dispatch(q,data.actual_load_kw[day],data.actual_pv_kw[day],data.variable_price[day],6000)
load=data.actual_load_kw[day].copy();pv=data.actual_pv_kw[day].copy();price=data.variable_price[day].copy()
load[cut:]+=1800;pv[cut:]+=2750;price[cut:]+=9
altered=m.causal_dispatch(q,load,pv,price,6000)
for key in ('charge_kwh','discharge_kwh','emergency_kwh','cash_cost'):
    assert np.array_equal(original[key][:cut],altered[key][:cut]),key
assert np.array_equal(original['soc_kwh'][:cut+1],altered['soc_kwh'][:cut+1])
report=dict(status='PASS',scope='36 boundary/event forecast-price-LP probes and observed-prefix physical controller; not annual re-run',checks=records,controller_prefix_slots=cut,maximum_difference=0,method='numerical arrays only; no file digests')
(HERE/'causality_probes.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status='PASS',event_probes=len(records),maximum_difference=0,controller_prefix_slots=cut),ensure_ascii=False))
