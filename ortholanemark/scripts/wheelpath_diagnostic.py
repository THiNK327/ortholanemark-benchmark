"""Downstream wheelpath geometry from frozen checkpoints/thresholds; no training.

Wheelpath inner edges are 0.375 m from the lane center; each strip is 1 m
wide outward. Disagreement is 1 - IoU of the two-strip unions, ungated.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from ortholanemark.scripts.reselect_eval import _gt, _build_report
from ortholanemark.evaluation.metrics import _operational_xs, _operational_xs_from_gt

ROOT=Path('ortholanemark/runs_wheelpath_20260921')
CLASSICAL=Path('ortholanemark/runs_classical_validation_20260921')
METHODS=['clrnet','polylanenet','scnn','laneatt','ufldv2','unet_seg']
LABELS=['CLRNet','PolyLaneNet','SCNN','LaneATT','UFLDv2','U-Net']
MM_PER_PIXEL=4.0


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def strips(center, width, valid=None):
    offset=375/MM_PER_PIXEL
    band=1000/MM_PER_PIXEL
    left=np.stack([center-offset-band,center+offset],axis=1)
    right=left+band
    # Integer pixel columns, matching the existing region rasterizer.
    lo=np.maximum(np.ceil(left),0)
    hi=np.minimum(np.floor(right),width-1)
    if valid is not None:hi[~valid,:]=-1
    return lo,hi


def disagreement(pl,pr,gl,gr,width):
    a,b=strips((pl+pr)/2,width,pl<=pr)
    c,d=strips((gl+gr)/2,width,gl<=gr)
    pa=np.maximum(b-a+1,0).sum()
    ga=np.maximum(d-c+1,0).sum()
    inter=0.0
    for i in range(2):
        for j in range(2):inter+=np.maximum(np.minimum(b[:,i],d[:,j])-np.maximum(a[:,i],c[:,j])+1,0).sum()
    union=pa+ga-inter
    return float(1-inter/union) if union else 0.0


def assess(pred, manifest):
    records=[]
    for i,e in enumerate(manifest):
        assert pred['stems'][i]==e['stem'] and pred['projects'][i]==e['project']
        gt=json.loads(Path(e['clean_gt_path']).read_text())
        h,w=gt['image_size']['height'],gt['image_size']['width']
        gl,gr=_gt(e['clean_gt_path'])
        xl,xr=_operational_xs(pred['xL'][i],pred['existsL'][i],pred['xR'][i],pred['existsR'][i],h,w)
        yl,yr=_operational_xs_from_gt(gl,gr,h,w)
        records.append(dict(stem=e['stem'],project=e['project'],
                            disagreement=disagreement(xl,xr,yl,yr,w),
                            gt_left=bool(gl.has_marking),gt_right=bool(gr.has_marking),
                            pred_left=bool(pred['existsL'][i]),pred_right=bool(pred['existsR'][i])))
    return dict(n_images=len(records),wheelpath_disagreement_percent=100*np.mean([r['disagreement'] for r in records]),per_image=records)


def infer(name,seed):
    import torch
    from ortholanemark.literature import build_method
    suffix='' if seed==0 else f'_seed{seed}'
    report_path=Path(f'ortholanemark/runs_reselect/{name}{suffix}__gra/report_tuned.json')
    archived=json.loads(report_path.read_text())
    out=ROOT/f'{name}_seed{seed}'
    if (out/'wheelpath.json').exists():
        print('EXISTS',name,seed,flush=True);return
    manifest=json.loads(Path('ortholanemark/clean_gt/v1/manifest_paper.json').read_text())['splits']['test']
    model=build_method(name,ckpt_path=archived['ckpt'])
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    cv2.setNumThreads(1)
    recs=[]
    for i,e in enumerate(manifest):
        image=cv2.imread(str(Path('../dataset')/e['image_path']),0)
        assert image is not None
        with torch.no_grad():p=model.predict_image(image)
        gl,gr=_gt(e['clean_gt_path'])
        recs.append(dict(xL=p['pred_x_L'],xR=p['pred_x_R'],
                         cL=float(p.get('pred_conf_L',p['pred_exists_L'])),
                         cR=float(p.get('pred_conf_R',p['pred_exists_R'])),
                         gL=gl,gR=gr,H=image.shape[0],W=image.shape[1],ms=0,
                         stem=e['stem'],project=e['project']))
        if (i+1)%100==0:print(name,seed,i+1,'/294',flush=True)
    tau=archived['tuned_tau']
    report=_build_report(recs,tau,'test',archived['ckpt'],archived['params'])
    old=archived['per_image'];new=report['per_image']
    flips=sum(a['pred_exists_'+s]!=b['pred_exists_'+s] for a,b in zip(old,new) for s in ['L','R'])
    delta=max(abs(a['gated_region_accuracy']-b['gated_region_accuracy']) for a,b in zip(old,new))
    mean_delta=abs(report['region']['gated_region_accuracy']-archived['region']['gated_region_accuracy'])
    verification=dict(presence_mismatches=flips,max_image_gra_difference=delta,mean_gra_difference=mean_delta,
                      torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
                      checkpoint=archived['ckpt'],threshold=tau,source_report=str(report_path),
                      checkpoint_sha256=hashlib.sha256(Path(archived['ckpt']).read_bytes()).hexdigest())
    write(out/'reproduction_check.json',verification)
    assert flips==0 and mean_delta<1e-5 and delta<1e-3,verification
    pred=dict(stems=np.array([r['stem'] for r in recs]),projects=np.array([r['project'] for r in recs]),
              xL=np.array([r['xL'] for r in recs]),xR=np.array([r['xR'] for r in recs]),
              existsL=np.array([r['cL']>=tau for r in recs]),existsR=np.array([r['cR']>=tau for r in recs]))
    np.savez_compressed(out/'predictions_test.npz',**pred)
    result=assess(pred,manifest);result['reproduction']=verification
    write(out/'wheelpath.json',result)
    print('DONE',name,seed,result['wheelpath_disagreement_percent'],verification,flush=True)
    del model;gc.collect();torch.cuda.empty_cache()


def summarize():
    manifest=json.loads(Path('ortholanemark/clean_gt/v1/manifest_paper.json').read_text())['splits']['test']
    rows=[]
    for name,label in zip(METHODS,LABELS):
        results=[json.loads((ROOT/f'{name}_seed{s}'/'wheelpath.json').read_text()) for s in range(3)]
        vals=[r['wheelpath_disagreement_percent'] for r in results]
        rows.append(dict(method=name,label=label,mean=float(np.mean(vals)),std=float(np.std(vals)),n_runs=3))
    for name,label in [('lsd','LSD'),('canny_hough','Canny-Hough'),('steger_ridge','Steger ridge')]:
        with np.load(CLASSICAL/name/'predictions_test.npz') as p:result=assess(p,manifest)
        write(ROOT/name/'wheelpath.json',result)
        rows.append(dict(method=name,label=label,mean=result['wheelpath_disagreement_percent'],std=None,n_runs=1))
    write(ROOT/'summary.json',rows)
    for row in rows:print(row,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS);p.add_argument('--seed',type=int);p.add_argument('--summarize',action='store_true');p.add_argument('--all',action='store_true');args=p.parse_args()
    if args.summarize:summarize()
    elif args.all:
        for name in METHODS:
            for seed in range(3):infer(name,seed)
        summarize()
    else:infer(args.method,args.seed)


if __name__=='__main__':main()
