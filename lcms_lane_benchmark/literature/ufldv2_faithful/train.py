"""Faithful UFLDv2 training entrypoint.

Mirrors upstream `train.py` + `utils/common.py:calc_loss` +
`utils/factory.py:get_loss_dict`:

  - Model: `parsingNet` (vendored verbatim) with `num_lanes=2`.
  - Loss dict for the TuSimple/CULane branch (verbatim from upstream
    factory.py:38-46):
      cls_loss        = SoftmaxFocalLoss(gamma=2, ignore_lb=-1)  (weight 1.0)
      relation_loss   = ParsingRelationLoss()                    (weight sim_loss_w)
      relation_dis    = ParsingRelationDis()                     (weight shp_loss_w)
      cls_loss_col    = SoftmaxFocalLoss(gamma=2, ignore_lb=-1)  (weight 1.0)
      cls_ext         = CrossEntropyLoss                         (weight 1.0)
      cls_ext_col     = CrossEntropyLoss                         (weight 1.0)
      mean_loss_row   = MeanLoss                                 (weight mean_loss_w)
      mean_loss_col   = MeanLoss                                 (weight mean_loss_w)
    Default weights (upstream cfg `tusimple_res18.py`):
      sim_loss_w=0.0, shp_loss_w=0.0, mean_loss_w=0.05
  - Optimizer: SGD, lr=0.05, momentum=0.9, weight_decay=1e-4, no
    nesterov (matches upstream cfg).
  - Scheduler: MultiStepLR at [50, 75] gamma=0.1 with linear warmup
    over 100 iters (matches upstream cfg + utils/factory.py).
  - Epochs: default 100, matching upstream `configs/tusimple_res18.py`.
    Batch size and portrait geometry remain LCMS memory/application
    adaptations.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lcms_lane_benchmark.literature.ufldv2_faithful.model.model_culane import (
    parsingNet,
)
from lcms_lane_benchmark.literature.ufldv2_faithful.utils.loss import (
    SoftmaxFocalLoss, ParsingRelationLoss, ParsingRelationDis, MeanLoss,
)
from lcms_lane_benchmark.literature.ufldv2_faithful.dataset import (
    UFLDv2FaithfulDataset, collate,
)
from lcms_lane_benchmark.literature.ufldv2_faithful.predict import (
    predict_with_model, predict_batch_with_model,
)
from lcms_lane_benchmark.literature._train_helpers import (
    set_global_seed, make_loader_generator, rng_snapshot, rng_restore,
    atomic_save, atomic_json, env_metadata, load_resume_checkpoint,
)
from lcms_lane_benchmark.literature._val_strict_iou import (
    compute_val_strict_iou,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--save_dir', required=True)
    p.add_argument('--epochs', type=int, default=100,
                   help='upstream configs/tusimple_res18.py default')
    p.add_argument('--batch_size', type=int, default=8,
                   help='upstream cfg uses 32 (TuSimple). We use 8 to fit '
                        'in single-GPU memory at portrait 800x320.')
    p.add_argument('--input_h', type=int, default=800)
    p.add_argument('--input_w', type=int, default=320)
    p.add_argument('--lr', type=float, default=0.05,
                   help='upstream cfg default (tusimple_res18.py)')
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--num_lanes', type=int, default=2)
    p.add_argument('--num_cls_row', type=int, default=56)
    p.add_argument('--num_cls_col', type=int, default=41)
    p.add_argument('--num_cell_row', type=int, default=100)
    p.add_argument('--num_cell_col', type=int, default=100)
    p.add_argument('--backbone', default='18')
    p.add_argument('--sim_loss_w', type=float, default=0.0,
                   help='upstream cfg default')
    p.add_argument('--shp_loss_w', type=float, default=0.0)
    p.add_argument('--mean_loss_w', type=float, default=0.05)
    p.add_argument('--milestones', type=int, nargs='+', default=[50, 75],
                   help='upstream cfg steps for the multi scheduler')
    p.add_argument('--gamma', type=float, default=0.1,
                   help='upstream cfg scheduler gamma')
    p.add_argument('--warmup_iters', type=int, default=100,
                   help='upstream cfg linear warmup iterations')
    p.add_argument('--use_aux', action='store_true',
                   help='Enable upstream CULane-style auxiliary seg head. '
                        'Adds a per-pixel CE loss over a multi-class lane '
                        'mask painted from the polynomial GT (mirrors '
                        'SCNN-faithful painter; line_width=8). Needed for '
                        'narrow lane-position prior datasets (e.g. LCMS) '
                        'where the flatten+MLP row-anchor head otherwise '
                        'collapses to the dataset-mean position.')
    p.add_argument('--aux_loss_w', type=float, default=1.0,
                   help='Weight on the auxiliary seg CE loss; upstream '
                        'CULane cfg uses 1.0.')
    p.add_argument('--seg_stride', type=int, default=8,
                   help='SegHead spatial stride relative to input. Output '
                        'shape is (input_h//stride, input_w//stride).')
    p.add_argument('--seg_line_width', type=int, default=8)
    p.add_argument('--seed', type=int, default=0,
                   help='Global seed for torch / numpy / random / cudnn.')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--no_auto_resume', action='store_true')
    return p.parse_args()


def _val_predict_batch_fn(model, image_grays, input_h, input_w, num_lanes,
                            num_cls_row, num_cell_row):
    return predict_batch_with_model(
        model, image_grays, input_h=input_h, input_w=input_w,
        num_lanes=num_lanes, num_cls_row=num_cls_row,
        num_cell_row=num_cell_row)


def _build_loss_dict(sim_w: float, shp_w: float, mean_w: float,
                      aux_w: float = 0.0):
    """Mirror upstream utils/factory.py:38-46 (TuSimple/CULane branch).

    Returns a list of (name, op, weight, data_src) tuples that the train
    loop iterates. Note: ignore_lb=-1 matches the -1 sentinel we put in
    labels_row / labels_col for off-frame anchors.

    When `aux_w > 0`, appends the CULane-style aux seg CE loss
    (`seg_out`, `seg_label`).
    """
    specs = [
        ('cls_loss',      SoftmaxFocalLoss(2, ignore_lb=-1), 1.0,
         ('cls_out', 'cls_label')),
        ('relation_loss', ParsingRelationLoss(),             sim_w,
         ('cls_out',)),
        ('relation_dis',  ParsingRelationDis(),              shp_w,
         ('cls_out',)),
        ('cls_loss_col',  SoftmaxFocalLoss(2, ignore_lb=-1), 1.0,
         ('cls_out_col', 'cls_label_col')),
        ('cls_ext',       torch.nn.CrossEntropyLoss(),       1.0,
         ('cls_out_ext', 'cls_out_ext_label')),
        ('cls_ext_col',   torch.nn.CrossEntropyLoss(),       1.0,
         ('cls_out_col_ext', 'cls_out_col_ext_label')),
        ('mean_loss_row', MeanLoss(),                        mean_w,
         ('cls_out', 'cls_label')),
        ('mean_loss_col', MeanLoss(),                        mean_w,
         ('cls_out_col', 'cls_label_col')),
    ]
    if aux_w > 0:
        specs.append(
            ('aux_seg_loss', torch.nn.CrossEntropyLoss(),    aux_w,
             ('seg_out', 'seg_label'))
        )
    return specs


def _inference(net, batch, device):
    """Mirror upstream `inference_culane_tusimple` from utils/common.py."""
    images = batch['images'].to(device)
    labels_row = batch['labels_row'].to(device)
    labels_col = batch['labels_col'].to(device)
    pred = net(images)
    cls_out_ext_label = (labels_row != -1).long()
    cls_out_col_ext_label = (labels_col != -1).long()
    out = {
        'cls_out': pred['loc_row'],
        'cls_label': labels_row,
        'cls_out_col': pred['loc_col'],
        'cls_label_col': labels_col,
        'cls_out_ext': pred['exist_row'],
        'cls_out_ext_label': cls_out_ext_label,
        'cls_out_col_ext': pred['exist_col'],
        'cls_out_col_ext_label': cls_out_col_ext_label,
        'labels_row_float': batch['labels_row_float'].to(device),
        'labels_col_float': batch['labels_col_float'].to(device),
    }
    if 'seg_out' in pred and 'seg_label' in batch:
        seg_label = batch['seg_label'].to(device)
        seg_out = pred['seg_out']
        # SegHead's dilated convs preserve stride-8 size, but downstream
        # dilations can shift by a few px. Force-match by F.interpolate
        # so the painted label aligns with whatever SegHead emits.
        if seg_label.shape[-2:] != seg_out.shape[-2:]:
            import torch.nn.functional as F
            seg_label = F.interpolate(
                seg_label.unsqueeze(1).float(),
                size=seg_out.shape[-2:], mode='nearest'
            ).long().squeeze(1)
        out['seg_out'] = seg_out
        out['seg_label'] = seg_label
    return out


def _calc_loss(loss_specs, results):
    """Mirror upstream `calc_loss`."""
    total = 0.0
    breakdown = {}
    for name, op, weight, srcs in loss_specs:
        if weight == 0:
            continue
        datas = [results[s] for s in srcs]
        loss_cur = op(*datas)
        total = total + loss_cur * weight
        breakdown[name] = float(loss_cur.detach().cpu())
    return total, breakdown


class _UFLDMultiStepLR:
    """Port of upstream `utils/factory.py:MultiStepLR`."""

    def __init__(self, optimizer, steps, gamma, iters_per_epoch,
                 warmup_iters):
        self.optimizer = optimizer
        self.steps = sorted(int(s) for s in steps)
        self.gamma = float(gamma)
        self.iters_per_epoch = int(iters_per_epoch)
        self.warmup_iters = int(warmup_iters)
        self.base_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self, external_iter: int) -> None:
        iters = int(external_iter)
        if self.warmup_iters > 0 and iters < self.warmup_iters:
            rate = iters / self.warmup_iters
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * rate
            return

        if self.iters_per_epoch <= 0 or iters % self.iters_per_epoch != 0:
            return

        epoch = int(iters / self.iters_per_epoch)
        power = len(self.steps)
        for i, step_epoch in enumerate(self.steps):
            if epoch < step_epoch:
                power = i
                break
        for group, lr in zip(self.optimizer.param_groups, self.base_lr):
            group['lr'] = lr * (self.gamma ** power)


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)
    gen = make_loader_generator(args.seed)

    train_ds = UFLDv2FaithfulDataset(
        args.manifest, 'train', input_h=args.input_h, input_w=args.input_w,
        num_cls_row=args.num_cls_row, num_cls_col=args.num_cls_col,
        num_cell_row=args.num_cell_row, num_cell_col=args.num_cell_col,
        num_lanes=args.num_lanes, rotation_deg=2.0,
        paint_seg=args.use_aux,
        seg_stride=args.seg_stride, seg_line_width=args.seg_line_width)
    val_ds = UFLDv2FaithfulDataset(
        args.manifest, 'val', input_h=args.input_h, input_w=args.input_w,
        num_cls_row=args.num_cls_row, num_cls_col=args.num_cls_col,
        num_cell_row=args.num_cell_row, num_cell_col=args.num_cell_col,
        num_lanes=args.num_lanes, rotation_deg=0.0,
        paint_seg=args.use_aux,
        seg_stride=args.seg_stride, seg_line_width=args.seg_line_width)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate,
                          drop_last=True, generator=gen)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate)

    net = parsingNet(
        pretrained=True, backbone=args.backbone,
        num_grid_row=args.num_cell_row, num_cls_row=args.num_cls_row,
        num_grid_col=args.num_cell_col, num_cls_col=args.num_cls_col,
        num_lane_on_row=args.num_lanes, num_lane_on_col=args.num_lanes,
        use_aux=args.use_aux, input_height=args.input_h, input_width=args.input_w,
        fc_norm=False).to(args.device)

    loss_specs = _build_loss_dict(
        args.sim_loss_w, args.shp_loss_w, args.mean_loss_w,
        aux_w=(args.aux_loss_w if args.use_aux else 0.0))

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay)
    sched = _UFLDMultiStepLR(optimizer, args.milestones, args.gamma,
                             len(train_dl), args.warmup_iters)

    history = []
    best_val_strict_iou = -1.0
    best_val_gra = -1.0
    best_val_loss_for_log = float('inf')
    start_epoch = 1
    t_start = time.time()
    base_elapsed = 0.0

    # Resume.
    resume_ckpt = None
    if args.resume or (not args.no_auto_resume):
        resume_ckpt = load_resume_checkpoint(save_dir)
    if resume_ckpt is not None and not args.no_auto_resume:
        net.load_state_dict(resume_ckpt['model'])
        optimizer.load_state_dict(resume_ckpt['optimizer'])
        # _UFLDMultiStepLR isn't a torch scheduler — state is `base_lr`
        # + the optimizer state. We restore base_lr from saved ckpt.
        if 'lr_scheduler' in resume_ckpt:
            sched.base_lr = list(resume_ckpt['lr_scheduler'].get('base_lr',
                                                                  sched.base_lr))
        history = list(resume_ckpt.get('history', []))
        best_val_strict_iou = float(
            resume_ckpt.get('best_val_strict_iou', -1.0))
        best_val_gra = float(resume_ckpt.get('best_val_gra', -1.0))
        best_val_loss_for_log = float(
            resume_ckpt.get('best_val_loss_for_log', float('inf')))
        start_epoch = int(resume_ckpt.get('epoch', 0)) + 1
        if 'rng_state' in resume_ckpt:
            rng_restore(resume_ckpt['rng_state'])
        if history:
            base_elapsed = float(history[-1].get('elapsed_min', 0.0))
        print(f'RESUMING from epoch {start_epoch - 1} -> {args.epochs} '
              f'(best_val_strict_iou={best_val_strict_iou:.4f})')

    if start_epoch > args.epochs:
        print(f'Already at epoch {start_epoch - 1} >= {args.epochs}; done.')
        return

    print(f'Train: {len(train_ds)}  Val: {len(val_ds)}  '
          f'Batches/epoch: {len(train_dl)}  Params: '
          f'{sum(p.numel() for p in net.parameters()):,}  seed: {args.seed}')

    for ep in range(start_epoch, args.epochs + 1):
        net.train()
        tr_loss = 0.0
        n_tr = 0
        for bi, batch in enumerate(train_dl):
            results = _inference(net, batch, args.device)
            loss, _ = _calc_loss(loss_specs, results)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
            sched.step((ep - 1) * len(train_dl) + bi)
            tr_loss += float(loss.detach().cpu()) * batch['images'].size(0)
            n_tr += batch['images'].size(0)
        tr_loss /= max(n_tr, 1)

        # Diagnostic val loss.
        net.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_dl:
                results = _inference(net, batch, args.device)
                loss, _ = _calc_loss(loss_specs, results)
                val_loss += float(loss.cpu()) * batch['images'].size(0)
                n_val += batch['images'].size(0)
        val_loss /= max(n_val, 1)

        # Selection metric: per-epoch val_strict_iou (batched).
        v = compute_val_strict_iou(
            net, args.manifest, _val_predict_batch_fn,
            predict_kwargs={'input_h': args.input_h,
                            'input_w': args.input_w,
                            'num_lanes': args.num_lanes,
                            'num_cls_row': args.num_cls_row,
                            'num_cell_row': args.num_cell_row},
            batch_size=args.batch_size, presence_rule='native')
        val_strict_iou = v['val_strict_iou']
        val_op_iou = v['val_op_iou']
        val_gra = v['val_gra']

        lr_now = optimizer.param_groups[0]['lr']
        elapsed = base_elapsed + (time.time() - t_start) / 60.0
        rec = {'epoch': ep, 'train_loss': tr_loss, 'val_loss': val_loss,
               'val_strict_iou': val_strict_iou, 'val_op_iou': val_op_iou,
               'val_gra': val_gra,
               'lr': lr_now, 'elapsed_min': elapsed}
        history.append(rec)
        print(f'[ep {ep:3d}] tr={tr_loss:.4f}  val_loss={val_loss:.4f}  '
              f'val_strict_iou={val_strict_iou:.4f}  '
              f'val_op_iou={val_op_iou:.4f}  '
              f'val_gra={val_gra:.4f}  '
              f'lr={lr_now:.5f}  +{elapsed:.1f}min')

        ckpt = {
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': {'base_lr': sched.base_lr},
            'history': history,
            'epoch': ep,
            'best_val_strict_iou': best_val_strict_iou,
            'best_val_gra': best_val_gra,
            'best_val_loss_for_log': best_val_loss_for_log,
            'val_loss': val_loss,
            'val_strict_iou': val_strict_iou,
            'val_gra': val_gra,
            'val_op_iou': val_op_iou,
            'rng_state': rng_snapshot(),
            'seed': args.seed,
            'config': {
                'input_h': args.input_h, 'input_w': args.input_w,
                'num_lanes': args.num_lanes,
                'num_cls_row': args.num_cls_row, 'num_cls_col': args.num_cls_col,
                'num_cell_row': args.num_cell_row, 'num_cell_col': args.num_cell_col,
                'backbone': args.backbone, 'batch_size': args.batch_size,
                'lr': args.lr, 'epochs': args.epochs,
                'rotation_deg': 2.0,
                'use_aux': bool(args.use_aux),
                'aux_loss_w': float(args.aux_loss_w),
                'seg_stride': int(args.seg_stride),
                'seg_line_width': int(args.seg_line_width),
            },
            'env': env_metadata(),
        }
        # Dual-best checkpoint selection (selection-sensitivity analysis):
        # best.pth = strict-IoU-best epoch (UNCHANGED criterion, so the
        # published checkpoint reproduces); best_gra.pth = GRA-best epoch
        # (val GRA at fixed tau=0.5). Both bests come from the SAME run.
        new_best_strict = val_strict_iou > best_val_strict_iou
        new_best_gra = val_gra > best_val_gra
        if val_loss < best_val_loss_for_log:
            best_val_loss_for_log = val_loss
        if new_best_strict:
            best_val_strict_iou = val_strict_iou
        if new_best_gra:
            best_val_gra = val_gra
        ckpt['best_val_strict_iou'] = best_val_strict_iou
        ckpt['best_val_gra'] = best_val_gra
        atomic_save(ckpt, save_dir / 'last.pth')
        atomic_json({'history': history}, save_dir / 'history.json')

        if new_best_strict:
            atomic_save(ckpt, save_dir / 'best.pth')
            # Re-save last.pth so a resume restores the UPDATED best —
            # otherwise a post-resume epoch worse than the true best
            # could overwrite best.pth.
            atomic_save(ckpt, save_dir / 'last.pth')
            print(f'  ** new best_val_strict_iou = {best_val_strict_iou:.4f}, '
                  f'best.pth updated')
        if new_best_gra:
            atomic_save(ckpt, save_dir / 'best_gra.pth')
            print(f'  ** new best_val_gra = {best_val_gra:.4f}, '
                  f'best_gra.pth updated')

    print(f'Training done. best_val_strict_iou={best_val_strict_iou:.4f} '
          f'best_val_gra={best_val_gra:.4f}')


if __name__ == '__main__':
    main()
