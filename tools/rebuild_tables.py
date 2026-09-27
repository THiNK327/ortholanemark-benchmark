"""Reconstruct saved score summaries without model inference or selection."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def summarize(report, wheel, seed=None):
    rows = report['per_image']
    counts = dict(tp=0, fp=0, fn=0, tn=0)
    joint=[]
    cats = {k: [] for k in ['both', 'left', 'right', 'none']}
    for r in rows:
        for side in ['L', 'R']:
            pred, gt = r['pred_exists_'+side], r['gt_exists_'+side]
            counts['tp' if pred and gt else 'fp' if pred else 'fn' if gt else 'tn'] += 1
        correct = r['pred_exists_L']==r['gt_exists_L'] and r['pred_exists_R']==r['gt_exists_R']
        joint.append(correct)
        cat = 'both' if r['gt_exists_L'] and r['gt_exists_R'] else 'left' if r['gt_exists_L'] else 'right' if r['gt_exists_R'] else 'none'
        cats[cat].append(correct)
    precision=counts['tp']/(counts['tp']+counts['fp'])
    recall=counts['tp']/(counts['tp']+counts['fn'])
    run = dict(seed=seed, tau=report.get('tuned_tau'),
               gra=float(np.mean([r['gated_region_accuracy'] for r in rows])),
               fp=counts['fp'], fn=counts['fn'], precision=precision, recall=recall,
               f1=2*precision*recall/(precision+recall), joint=100*float(np.mean(joint)),
               mae=report['boundary']['boundary_mae_on_gt']*4,
               wheelpath=100*float(np.mean([r['disagreement'] for r in wheel['per_image']])))
    run.update({f'within_{n}':100*report['boundary'][f'within_{n}mm'] for n in [10,25,50]})
    run.update({'category_'+k:100*float(np.mean(v)) for k,v in cats.items()})
    return run

def build(root=ROOT):
    manifest=json.loads((root/'provenance/run_manifest.json').read_text(encoding='utf-8'))
    methods=sorted({r['method'] for r in manifest['runs']})
    results=[]
    for method in methods:
        runs=[]
        for seed in [0,1,2]:
            path=root/'artifacts/learning'/f'{method}_seed{seed}'
            report=json.loads((path/'report_tuned.json').read_text(encoding='utf-8'))
            wheel=json.loads((path/'wheelpath.json').read_text(encoding='utf-8'))
            runs.append(summarize(report,wheel,seed))
        keys=[k for k in runs[0] if k not in ['seed','tau']]
        results.append(dict(method=method,runs=runs,mean={k:float(np.mean([r[k] for r in runs])) for k in keys},std={k:float(np.std([r[k] for r in runs])) for k in keys}))
    expected=json.loads((root/'artifacts/learning_summary.json').read_text(encoding='utf-8'))
    for row in results:
        archived=next(e for e in expected if e['method']==row['method'])
        for aggregation in ['mean','std']:
            for key,value in row[aggregation].items():
                if not np.isclose(value,archived[aggregation][key],rtol=0,atol=1e-12):
                    raise AssertionError((row['method'],aggregation,key,value,archived[aggregation][key]))
    classical=[]
    for method in ['canny_hough','lsd','steger_ridge']:
        path=root/'artifacts/traditional'/method
        r=summarize(json.loads((path/'report_test.json').read_text()),json.loads((path/'wheelpath.json').read_text()))
        r.pop('seed');r.pop('tau')
        classical.append(dict(method=method,mean=r,std=None))
    return results+classical

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'verification/reconstructed_tables.json')
    a=p.parse_args()
    rows=build()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    with a.output.with_suffix('.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['method','aggregation']+list(rows[0]['mean']))
        w.writeheader()
        for row in rows:
            for aggregation in ['mean','std']:
                if row[aggregation] is not None:w.writerow(dict(method=row['method'],aggregation=aggregation,**row[aggregation]))
    print('PASS: six learning-method summaries match saved values (absolute tolerance 1e-12); three traditional summaries reconstructed.')

if __name__=='__main__': main()
