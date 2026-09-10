"""Fast local unit tests for diagnostics and table extraction, not annual gates."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
import microgrid_core as m
from evaluate_results import forecast_diagnostics, raw_data_profile, validate_day, write_representative_tables
from test_core import forecasts


def main():
    data=m.load_inputs('.')
    # Constant synthetic truth has exact forecasts, zero errors, 100% coverage.
    constant=replace(data, actual_load_kw=np.full((365,144),600.), actual_pv_kw=np.full((365,144),200.), pv_forecast_hourly=np.full((365,4,24),200.))
    lp=constant.actual_load_kw.copy();vp=constant.actual_pv_kw.copy()
    metrics=forecast_diagnostics(constant,lp,vp,np.zeros_like(lp),np.zeros_like(vp),m.official_pv_residuals(constant))
    assert np.all(metrics['mae_kw']==0)
    assert np.all(metrics['rmse_kw']==0)
    assert np.all(metrics['mape_percent']==0)
    assert np.all(metrics['empirical_90_coverage'].dropna()==1)
    analog=metrics[metrics.model=='analog_load'].iloc[0]
    assert analog['count']==334*144
    assert int(metrics[(metrics.model=='official_pv_raw_lead')&(metrics.lead_hour==24)].iloc[0]['count'])==334*4-3
    print('synthetic forecast metric and coverage checks PASS')
    profile=raw_data_profile(data)
    assert len(profile)==36 and profile['missing'].sum()==0
    assert profile.groupby('variable')['count'].sum().eq(365*144).all()
    lp,lr,vp,vr,ir=forecasts(data)
    d=78
    r=m.run_q3_day(data,d,6000,lp,lr,ir,variable_price=True)
    validate_day(r,True,6000)
    r['date']=data.dates[d]
    # Temporary outputs stay in reviews and are removed by the context manager.
    with TemporaryDirectory(prefix='evaluation-unit-',dir='reviews') as tmp:
        root=Path(tmp);(root/'results').mkdir()
        write_representative_tables(root,{'q4_3':([r],True)},[data.dates[d]])
        p=pd.read_csv(root/'results'/'表1_代表日购电.csv')
        assert len(p)==2
        np.testing.assert_allclose(p['10:00-10:10_kwh'],[r['initial_plan_kwh'][60],r['final_contract_kwh'][60]])
        s=pd.read_csv(root/'results'/'表2_代表日储能.csv')
        np.testing.assert_allclose(s.charge_kwh.sum(),r['actual']['charge_kwh'].sum())
        np.testing.assert_allclose(s.discharge_kwh.sum(),r['actual']['discharge_kwh'].sum())
        e=pd.read_csv(root/'results'/'表3_代表日应急.csv')
        np.testing.assert_allclose(e.emergency_kwh.sum(),r['actual']['emergency_kwh'].sum())
        u=pd.read_csv(root/'results'/'代表日历次更新.csv')
        assert len(u)==144+108+72+36
    print('one-day physical and required-table extraction checks PASS; not an annual result')


if __name__=='__main__': main()
