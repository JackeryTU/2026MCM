"""Single entry point: tests -> all results -> 54 experiments -> figures -> audits.

No file hashes are computed or compared. --verify-only audits already computed
outputs; it is an incremental check, NOT the full reproduction command.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from utils.repro_manifest_nohash import build_manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-root',default='.')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    root=Path(args.project_root).resolve()
    skill=Path(os.environ.get('MATH_MODELING_SKILL_ROOT',r'C:/Users/lenovo/.codex/skills/math-modeling-skill-main'))
    logs=root/'results'/'reproduction_logs'; logs.mkdir(parents=True,exist_ok=True)
    (root/'reviews').mkdir(exist_ok=True)
    records=[]; start=time.perf_counter()

    def run(name,script,*arguments):
        command=[sys.executable,'-X','utf8',str(script),*map(str,arguments)]
        print(f'[{name}] START',flush=True)
        begun=time.perf_counter()
        log=logs/f'{name}.log'
        with log.open('w',encoding='utf-8') as stream:
            completed=subprocess.run(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT)
        record=dict(step=name,command=command,exit_code=completed.returncode,
                    runtime_seconds=time.perf_counter()-begun,log=str(log.relative_to(root)))
        records.append(record)
        (logs/'steps.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'[{name}] exit={completed.returncode}, seconds={record["runtime_seconds"]:.1f}',flush=True)
        if completed.returncode:
            print(log.read_text(encoding='utf-8')[-9000:])
            raise RuntimeError(f'{name} failed; see {log}')

    run('environment',skill/'references/roles/编程手/scripts/check_env.py','--features','data','visualization','optimization','excel','machine-learning')
    run('core_tests',root/'test_core.py','--project-root',root)
    run('evaluation_tests',root/'test_evaluation.py')
    if not args.verify_only:
        run('main',root/'run_all.py','--project-root',root,'--alpha','0.8')
        run('experiments',root/'run_experiments.py','--project-root',root)
    run('workbook_audit',root/'audit_results.py','--project-root',root)
    run('experiment_audit',root/'audit_experiments.py','--project-root',root)
    for name in ('问题1_逐时结果','q2_每日汇总','q3_每日汇总','q4_3_每日汇总','原始数据剖析'):
        run('profile_'+name,skill/'tools/figure/scripts/profile_data.py',root/'results'/f'{name}.csv')
    run('profile_experiments',skill/'tools/figure/scripts/profile_data.py',root/'results/experiments/实验总表.csv')
    run('figures',root/'plot_all.py','--project-root',root)
    files=sorted(p for p in (root/'figures').glob('*') if p.suffix in ('.png','.svg'))
    # check_figure accepts file arguments, not a directory (directory only warns).
    run('figure_format',skill/'tools/figure/scripts/check_figure.py',*files,'--strict')
    format_log=(logs/'figure_format.log').read_text(encoding='utf-8')
    if '[WARN]' in format_log or '[FAIL]' in format_log:
        raise RuntimeError('Unresolved figure-format warning: see figure_format.log')
    run('figure_coverage',skill/'references/roles/编程手/scripts/figure_audit.py',root/'figures','--questions','q1','q2','q3','q4','--strict')
    inputs=[root/'data/C题.pdf',*[root/'data/附件'/f'附件{i}.xlsx' for i in range(1,5)],*sorted((root/'data/附件/附件5').glob('*.xlsx'))]
    manifest=build_manifest(inputs,2026,dict(alpha=.8,delta_hours=1/6,capacity_kwh=12000,
        soc_fractions=[.1,.9],initial_soc_fraction=.5,power_kw=5000,eta_c=.9,eta_d=.9,
        analog_history_days=90,residual_history_days=60,clusters=9,kmeans_n_init=10,
        price_history_days=30,official_updates=[0,6,12,18],simulation_days=365,evaluation_days=334,
        evaluation_start='2025-02-01',experiment_cases=54,
        model='causal marginal-quantile trajectory LP; not full stochastic MPC',
        settlement='executed-slot final contract relative to 0h plan; alternative is same-policy recost'),
        'python -X utf8 reproduce.py --project-root .',['numpy','pandas','scipy','scikit-learn','openpyxl','matplotlib','Pillow'])
    source_files=sorted(root.glob('*.py'))+sorted((root/'utils').glob('*.py'))+[root/'题目分析报告.md',root/'术语表格.md',root/'figures/图表契约-重算版.md']
    manifest['source_files']=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in source_files]
    manifest['last_invocation_mode']='verify_existing_outputs' if args.verify_only else 'full_reproduction'
    manifest['steps']=records
    manifest['runtime_seconds']=time.perf_counter()-start
    manifest['completed_at_utc']=datetime.now(timezone.utc).isoformat()
    manifest['skill_root']=str(skill)
    manifest['independent_gate']='not inferred from this author/runner audit; consult reviews/P2独立验收回执.md'
    (root/'results/复现清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Reproduction pipeline passed. Hash checks omitted. Independent visual review is separate.',flush=True)


if __name__=='__main__':
    main()
