"""Evaluate saved numeric test predictions using frozen original metric helpers."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ortholanemark.scripts.reselect_eval import _gt, _build_report
from ortholanemark.scripts.wheelpath_diagnostic import assess
from prepare_manifest import resolve_manifest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    entries=resolve_manifest(a.dataset_root,a.manifest)['splits']['test']
    with np.load(a.predictions,allow_pickle=False) as z:pred={k:z[k] for k in z.files}
    assert len(entries)==len(pred['stems'])
    records=[]
    for i,e in enumerate(entries):
        assert str(pred['stems'][i])==e['stem'] and str(pred['projects'][i])==e['project'],(i,'ID/order mismatch')
        d=json.loads(Path(e['clean_gt_path']).read_text(encoding='utf-8'))
        gl,gr=_gt(e['clean_gt_path'])
        records.append(dict(xL=pred['xL'][i],xR=pred['xR'][i],cL=float(pred['existsL'][i]),cR=float(pred['existsR'][i]),
                            gL=gl,gR=gr,H=d['image_size']['height'],W=d['image_size']['width'],ms=0,
                            stem=e['stem'],project=e['project']))
    # Flags already contain historical selected-threshold decisions; 0.5 simply
    # transports binary flags through the original shared report helper.
    report=_build_report(records,.5,'test','not_loaded',None)
    report['tuned_tau']=None
    report['decision_source']='Saved existsL/existsR; no threshold selected or applied to continuous scores'
    report['wheelpath']=assess(pred,entries)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('Evaluated',len(entries),'saved predictions; GRA =',report['region']['gated_region_accuracy'])

if __name__=='__main__':main()
