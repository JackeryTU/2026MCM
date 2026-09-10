"""Validate every annual experiment and its aggregation, without file hashes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from run_experiments import cases


def audit(root):
    output=root/'results'/'experiments'
    aggregate=pd.read_csv(output/'实验总表.csv').set_index('name')
    expected=cases()
    assert len(aggregate)==len(expected)==54
    assert set(aggregate.index)=={s['name'] for s in expected}
    reports=[]
    for spec in expected:
        name=spec['name']
        report=json.loads((output/f'{name}.json').read_text(encoding='utf-8'))
        # Compare declared parameters, not file fingerprints.
        assert json.loads(json.dumps(spec))==report['spec']
        frame=pd.read_csv(output/f'{name}_daily.csv',parse_dates=['date'])
        assert pd.DatetimeIndex(frame.date).equals(pd.date_range('2025-01-01','2025-12-31'))
        assert np.isfinite(frame.select_dtypes('number')).all().all()
        battery=report['battery']; capacity=battery['capacity_kwh']
        effective_capacity=0 if spec['baseline']=='no_storage' else spec['capacity']
        assert capacity==effective_capacity
        assert report['initial_soc_kwh']==.5*capacity
        assert_allclose(frame.soc_start_kwh.iloc[0],.5*capacity,atol=1e-8,rtol=0)
        assert_allclose(frame.soc_start_kwh.iloc[1:],frame.soc_end_kwh.iloc[:-1],atol=1e-7,rtol=0)
        assert frame.soc_min_kwh.min()>=.1*capacity-1e-6
        assert frame.soc_max_kwh.max()<=.9*capacity+1e-6
        assert frame.max_balance_residual_kwh.max()<1e-6
        assert_allclose(frame.cash_cost_yuan,frame.settlement_yuan+frame.emergency_yuan,atol=1e-6,rtol=1e-12)
        evaluated=frame[frame.date>='2025-02-01']
        assert len(evaluated)==334
        for column in ('cash_cost_yuan','alternative_cash_cost_yuan','plan_kwh','final_contract_kwh','emergency_kwh','unused_contract_kwh','pv_curtailed_kwh','charge_kwh','discharge_kwh','emergency_slots','settlement_yuan','emergency_yuan'):
            assert_allclose(evaluated[column].sum(),report['totals'][column],atol=1e-5,rtol=1e-12)
            assert_allclose(aggregate.loc[name,column],report['totals'][column],atol=1e-5,rtol=1e-12)
        reports.append(dict(name=name,days=365,evaluation_days=334,
            effective_capacity_kwh=capacity,effective_power_kw=battery['power_kw'],
            initial_soc_kwh=report['initial_soc_kwh'],
            max_balance_residual_kwh=float(frame.max_balance_residual_kwh.max())))
    metrics=json.loads((root/'results'/'核心指标.json').read_text(encoding='utf-8'))
    for main,experiment in (('q2','q2_safe'),('q3','q3_6h'),('q4_2','q42_safe'),('q4_3','q43_6h')):
        for col in ('cash_cost_yuan','alternative_cash_cost_yuan','emergency_kwh','unused_contract_kwh','pv_curtailed_kwh','soc_end_kwh'):
            assert_allclose(aggregate.loc[experiment,col],metrics['annual_feb_dec'][main][col],atol=1e-5,rtol=1e-12)
    result=dict(status='PASS',case_count=len(reports),
        hash_checks='omitted per explicit user request',
        no_storage_note='spec capacity/power are nominal configuration; battery fields and effective fields are zero',
        cases=reports)
    (root/'results'/'实验审计.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-root',default='.')
    audit(Path(parser.parse_args().project_root).resolve())
