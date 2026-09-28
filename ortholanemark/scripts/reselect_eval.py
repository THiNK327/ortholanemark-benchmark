"""Per-checkpoint evaluator for the checkpoint-selection sensitivity analysis.

Loads ONE deep-baseline checkpoint via its benchmark adapter, runs inference over
val / test / held_out_test (``method.predict_image`` per image — the SAME inference
path as ``run_literature_benchmark``), tunes the existence threshold tau on val to
maximize GRA (51-point sweep, existence := ``pred_conf >= tau`` — the SAME rule as
``scripts/tune_tau.py``), and writes ``report_val_tuning.json`` / ``report_tuned.json``
/ ``report_tuned_heldout.json`` into an output dir.

The report schema mirrors ``car/eval_car.py`` so the sensitivity-table assembler reads
baseline and CAR reports identically. Because the tau grid, the existence rule, and the
metric driver (``per_image_boundary_errors_full`` / ``aggregate_per_image``) are all the
same as the canonical baseline pipeline, running this on an existing ``best.pth``
reproduces the published ``runs_literature_deep/<method>/report_tuned*.json`` numbers
(same weights -> same predictions -> same metrics). That equality is the validation gate.

Usage:
    python -m ortholanemark.scripts.reselect_eval \
        --method clrnet \
        --ckpt   ../checkpoints/weights/clrnet_seed0/best_gra.pth \
        --output_dir ortholanemark/runs_reselect/clrnet__strict \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ortholanemark.literature import build_method
from ortholanemark.data.lane_gt import LaneGeometricTruth
from ortholanemark.evaluation.metrics import (
    per_image_boundary_errors_full, aggregate_per_image,
    existence_f1_accumulator, existence_f1_update, existence_f1_finalize,
)

N_TAU = 51


def _gt(path):
    d = json.load(open(path))
    gL = LaneGeometricTruth.from_clean_gt_dict(d['left_lane'])
    gR = LaneGeometricTruth.from_clean_gt_dict(d['right_lane'])
    z = LaneGeometricTruth(False, np.zeros(1), 0, None, [])
    return (gL or z), (gR or z)


def _infer(method, entries):
    """Run the adapter over a split. Returns raw per-image records carrying the
    continuous existence conf so tau can be swept without re-inference."""
    recs = []
    for e in entries:
        img = cv2.imread(e['image_path'], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        p = method.predict_image(img)
        gL, gR = _gt(e['clean_gt_path'])
        recs.append({
            'xL': p['pred_x_L'], 'xR': p['pred_x_R'],
            'cL': float(p.get('pred_conf_L', float(p['pred_exists_L']))),
            'cR': float(p.get('pred_conf_R', float(p['pred_exists_R']))),
            'gL': gL, 'gR': gR,
            'H': img.shape[0], 'W': img.shape[1],
            'ms': float(p.get('inference_ms', 0.0) or 0.0),
            'stem': e.get('stem', ''),
            'project': e.get('project', 'unknown'),
        })
    return recs


def _build_report(recs, tau, split, ckpt_path, n_params):
    """Existence := conf >= tau, then full metric vector. Schema matches
    car/eval_car.py::_build_report so reports are uniform across methods."""
    accum = existence_f1_accumulator()
    per = []
    per_project = defaultdict(list)
    for r in recs:
        eL, eR = r['cL'] >= tau, r['cR'] >= tau
        m = per_image_boundary_errors_full(r['xL'], eL, r['xR'], eR,
                                           r['gL'], r['gR'], r['H'], r['W'])
        m = dict(m)
        m['stem'] = r['stem']
        m['project'] = r['project']
        m['gt_exists_L'] = bool(r['gL'].has_marking)
        m['gt_exists_R'] = bool(r['gR'].has_marking)
        m['pred_exists_L'] = bool(eL)
        m['pred_exists_R'] = bool(eR)
        m['pred_conf_L'] = float(r['cL'])
        m['pred_conf_R'] = float(r['cR'])
        per.append(m)
        per_project[m['project']].append(m)
        existence_f1_update(accum, 'L', eL, r['gL'].has_marking)
        existence_f1_update(accum, 'R', eR, r['gR'].has_marking)
    agg = aggregate_per_image(per)
    existence = existence_f1_finalize(accum)

    proj_summary = {}
    for proj, rows in sorted(per_project.items()):
        a = aggregate_per_image(rows)
        proj_summary[proj] = {
            'n_images': a['n_images'],
            'operational_iou': a.get('operational_iou'),
            'gated_region_accuracy': a.get('gated_region_accuracy'),
            'operational_width_mae': a.get('operational_width_mae'),
            'boundary_mae_on_gt': a.get('boundary_mae_on_gt_macro'),
        }

    return {
        'region': {
            'operational_iou': agg.get('operational_iou'),
            'strict_operational_iou': agg.get('strict_operational_iou'),
            'gated_region_accuracy': agg.get('gated_region_accuracy'),
            'area_error_pct': agg.get('operational_area_error_pct'),
            'operational_width_mae': agg.get('operational_width_mae'),
            'misassigned_area_fraction': agg.get('misassigned_area_fraction'),
        },
        'boundary': {
            'boundary_mae_on_gt': agg.get('boundary_mae_on_gt_macro'),
            'boundary_mae_on_gt_left': agg.get('boundary_mae_on_gt_left'),
            'boundary_mae_on_gt_right': agg.get('boundary_mae_on_gt_right'),
            'within_10mm': agg.get('within_10mm'),
            'within_25mm': agg.get('within_25mm'),
            'within_50mm': agg.get('within_50mm'),
            'within_100mm': agg.get('within_100mm'),
        },
        'existence': {
            'left': existence['L'],
            'right': existence['R'],
            'f1': existence['macro_f1'],
        },
        'per_project': proj_summary,
        'n_images': agg['n_images'],
        'split': split,
        'tuned_tau': tau,
        'latency_ms_mean': float(np.mean([r['ms'] for r in recs]))
        if recs and any(r['ms'] for r in recs) else None,
        'params': int(n_params) if n_params else None,
        'ckpt': str(ckpt_path),
        'per_image': per,
    }


def _score_gra(recs, tau):
    vals = []
    for r in recs:
        eL, eR = r['cL'] >= tau, r['cR'] >= tau
        m = per_image_boundary_errors_full(r['xL'], eL, r['xR'], eR,
                                           r['gL'], r['gR'], r['H'], r['W'])
        vals.append(m['gated_region_accuracy'])
    return float(np.mean(vals)) if vals else 0.0


def _count_params(method):
    for attr in ('model', 'net', 'module'):
        mod = getattr(method, attr, None)
        if mod is not None and hasattr(mod, 'parameters'):
            try:
                return sum(p.numel() for p in mod.parameters())
            except Exception:
                pass
    return None


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True,
                   help='Registry name, e.g. clrnet, scnn, '
                        'ufldv2, ufldv2_tusimple, '
                        'laneatt, polylanenet, unet_seg.')
    p.add_argument('--ckpt', default=None,
                   help='Checkpoint for learned methods; omit (or NONE) for '
                        'checkpoint-free classical detectors.')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--manifest',
                   default='ortholanemark/clean_gt/v1/manifest_paper.json')
    p.add_argument('--device', default='cuda')
    p.add_argument('--n_tau', type=int, default=N_TAU)
    args = p.parse_args(argv)

    man = json.load(open(args.manifest))
    # Adapters auto-select device (cuda if available), matching
    # eval_ablation_ckpt.py — they take ckpt_path only, not a device kwarg.
    # Classical detectors are checkpoint-free: build without ckpt_path.
    if args.ckpt and args.ckpt.upper() != 'NONE':
        method = build_method(args.method, ckpt_path=args.ckpt)
    else:
        method = build_method(args.method)
    n_params = _count_params(method)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"reselect_eval  method={args.method}  ckpt={args.ckpt}")
    if n_params:
        print(f"  params={n_params:,}")

    # Tune tau on val by GRA (existence := conf >= tau).
    val = _infer(method, man['splits'].get('val', []))
    confs = np.concatenate([[r['cL'] for r in val], [r['cR'] for r in val]]) \
        if val else np.array([])
    binary_conf = confs.size > 0 and len(np.unique(confs)) <= 2
    if binary_conf:
        best_t, best_g = 0.5, _score_gra(val, 0.5)
        print(f"  binary existence conf -> default tau=0.50  val_GRA={best_g:.4f}")
    else:
        best_t, best_g = 0.5, -1.0
        for t in np.linspace(0.0, 1.0, args.n_tau):
            g = _score_gra(val, float(t))
            if g > best_g:
                best_g, best_t = g, float(t)
        print(f"  tuned tau*={best_t:.2f}  val_GRA*={best_g:.4f}")
    (out_dir / 'report_val_tuning.json').write_text(
        json.dumps(_build_report(val, best_t, 'val', args.ckpt, n_params),
                   indent=2), encoding='utf-8')

    for split, suffix in (('test', ''), ('held_out_test', '_heldout')):
        recs = _infer(method, man['splits'].get(split, []))
        rep = _build_report(recs, best_t, split, args.ckpt, n_params)
        name = 'report_tuned.json' if split == 'test' \
            else 'report_tuned_heldout.json'
        (out_dir / name).write_text(json.dumps(rep, indent=2), encoding='utf-8')
        reg = rep['region']
        print(f"  {split:<14} GRA={reg['gated_region_accuracy']:.4f}  "
              f"op_iou={reg['operational_iou']:.4f}  "
              f"strict={reg['strict_operational_iou']:.4f}  "
              f"exF1={rep['existence']['f1']:.3f}  (n={rep['n_images']})")
    print(f"  wrote reports to {out_dir}")


if __name__ == '__main__':
    main()
