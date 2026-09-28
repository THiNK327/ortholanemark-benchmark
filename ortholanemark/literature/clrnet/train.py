"""CLRNet training entrypoint.

Mirrors upstream `main.py` + `engine/runner.py` (Turoad/CLRNet @ 7269e9d)
as closely as practical given our manifest-based dataset:

  - Optimizer: AdamW, lr=3e-4 (upstream's documented rate for batch 8;
    the config's 6e-4 is for batch 24). Scheduler: per-iteration
    CosineAnnealingLR with T_max = total training iterations.
  - Loss: the head-internal 4-term loss (focal cls + smooth-L1 xytl +
    Line-IoU + aux seg), upstream weights (2 / 0.2 / 2 / 1).
  - Augmentation: Rotation(±2°) — the project-uniform policy applied to
    every benchmark baseline in place of each upstream's own pipeline
    (documented gap; see FIDELITY_REVIEW.md).
  - Checkpoint selection: per-epoch val `strict_operational_iou`, the
    same criterion as every other benchmark baseline.
  - Reproducibility + stop-and-go resume: shared `_train_helpers`
    machinery (seeded RNG, atomic per-epoch last.pth, auto-resume).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ortholanemark.literature.clrnet.model.detector import (
    CLRNet, CLRNetConfig,
)
from ortholanemark.literature.clrnet.dataset import (
    CLRNetDataset, collate,
)
from ortholanemark.literature.clrnet.predict import (
    predict_batch_with_model,
)
from ortholanemark.literature._train_helpers import (
    set_global_seed, make_loader_generator, rng_snapshot, rng_restore,
    atomic_save, atomic_json, env_metadata, load_resume_checkpoint,
)
from ortholanemark.literature._val_strict_iou import (
    compute_val_strict_iou,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--save_dir', required=True)
    p.add_argument('--epochs', type=int, default=100,
                   help='Fair-comparison budget: identical 100 epochs '
                        'across all benchmark baselines.')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-4,
                   help="Upstream: '3e-4 for batchsize 8'.")
    p.add_argument('--num_priors', type=int, default=192)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--resume', action='store_true',
                   help='Resume from last.pth if present (auto-detects).')
    p.add_argument('--no_auto_resume', action='store_true',
                   help='Force training from scratch even if last.pth '
                        'exists.')
    p.add_argument('--img_w', type=int, default=800,
                   help='Input width (landscape convention). Resolution '
                        'ablation: scale up from the 800x320 default.')
    p.add_argument('--img_h', type=int, default=320,
                   help='Input height.')
    return p.parse_args()


def _val_predict_batch_fn(model, image_grays):
    return predict_batch_with_model(model, image_grays)


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)
    gen = make_loader_generator(args.seed)

    train_ds = CLRNetDataset(args.manifest, 'train',
                                     rotation_deg=2.0,
                                     img_w=args.img_w, img_h=args.img_h)
    val_ds = CLRNetDataset(args.manifest, 'val', rotation_deg=0.0,
                                   img_w=args.img_w, img_h=args.img_h)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=args.num_workers,
                          collate_fn=collate, drop_last=True,
                          generator=gen)

    cfg = CLRNetConfig(num_priors=args.num_priors,
                       img_w=args.img_w, img_h=args.img_h)
    model = CLRNet(cfg).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    max_iter = args.epochs * len(train_dl)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_iter)

    history = []
    best_val_strict_iou = -1.0
    best_val_gra = -1.0
    start_epoch = 1
    t_start = time.time()
    base_elapsed = 0.0

    resume_ckpt = None
    if args.resume or (not args.no_auto_resume):
        resume_ckpt = load_resume_checkpoint(save_dir)
    if resume_ckpt is not None and not args.no_auto_resume:
        model.load_state_dict(resume_ckpt['model'])
        optimizer.load_state_dict(resume_ckpt['optimizer'])
        lr_sched.load_state_dict(resume_ckpt['lr_scheduler'])
        history = list(resume_ckpt.get('history', []))
        best_val_strict_iou = float(
            resume_ckpt.get('best_val_strict_iou', -1.0))
        best_val_gra = float(resume_ckpt.get('best_val_gra', -1.0))
        start_epoch = int(resume_ckpt.get('epoch', 0)) + 1
        if 'rng_state' in resume_ckpt:
            rng_restore(resume_ckpt['rng_state'])
        if history:
            base_elapsed = float(history[-1].get('elapsed_min', 0.0))
        print(f'RESUMING from epoch {start_epoch - 1} -> {args.epochs} '
              f'(best_val_strict_iou={best_val_strict_iou:.4f})')

    if start_epoch > args.epochs:
        print(f'Already at epoch {start_epoch - 1} >= {args.epochs}; '
              f'nothing to do.')
        return

    print(f'Train: {len(train_ds)}  Val: {len(val_ds)}  '
          f'Batches/epoch: {len(train_dl)}  max_iter: {max_iter}  '
          f'seed: {args.seed}')

    for ep in range(start_epoch, args.epochs + 1):
        model.train()
        tr_loss = tr_cls = tr_iou = tr_seg = 0.0
        n_tr = 0
        for sample in train_dl:
            batch = {
                'img': sample['img'].to(args.device),
                'lane_line': sample['lane_line'].to(args.device),
                'seg': sample['seg'].to(args.device),
            }
            optimizer.zero_grad()
            out = model(batch)
            loss = out['loss']
            loss.backward()
            optimizer.step()
            lr_sched.step()
            bs = batch['img'].size(0)
            stats = out['loss_stats']
            tr_loss += float(loss.item()) * bs
            tr_cls += float(stats['cls_loss'].item()) * bs
            tr_iou += float(stats['iou_loss'].item()) * bs
            tr_seg += float(stats['seg_loss'].item()) * bs
            n_tr += bs
        tr_loss /= max(n_tr, 1)
        tr_cls /= max(n_tr, 1)
        tr_iou /= max(n_tr, 1)
        tr_seg /= max(n_tr, 1)

        # Selection criterion: per-epoch val strict_operational_iou.
        model.eval()
        v = compute_val_strict_iou(
            model, args.manifest, _val_predict_batch_fn,
            batch_size=args.batch_size)
        val_strict_iou = v['val_strict_iou']
        val_op_iou = v['val_op_iou']
        val_gra = v['val_gra']

        lr_now = optimizer.param_groups[0]['lr']
        elapsed = base_elapsed + (time.time() - t_start) / 60.0
        rec = {
            'epoch': ep, 'train_loss': tr_loss, 'train_cls': tr_cls,
            'train_iou': tr_iou, 'train_seg': tr_seg,
            'val_strict_iou': val_strict_iou, 'val_op_iou': val_op_iou,
            'val_gra': val_gra,
            'lr': lr_now, 'elapsed_min': elapsed,
        }
        history.append(rec)
        print(f'[ep {ep:3d}] tr={tr_loss:.4f}  '
              f'val_strict_iou={val_strict_iou:.4f}  '
              f'val_gra={val_gra:.4f}  '
              f'val_op_iou={val_op_iou:.4f}  '
              f'lr={lr_now:.6f}  +{elapsed:.1f}min')

        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_sched.state_dict(),
            'history': history,
            'epoch': ep,
            'best_val_strict_iou': best_val_strict_iou,
            'best_val_gra': best_val_gra,
            'val_strict_iou': val_strict_iou,
            'val_op_iou': val_op_iou,
            'val_gra': val_gra,
            'rng_state': rng_snapshot(),
            'seed': args.seed,
            'config': {
                'batch_size': args.batch_size, 'lr': args.lr,
                'epochs': args.epochs, 'rotation_deg': 2.0,
            },
            # Model-constructor overrides for predict.py lazy load.
            'config_model': {'num_priors': args.num_priors,
                             'img_w': args.img_w, 'img_h': args.img_h},
            'env': env_metadata(),
        }
        # Dual-best checkpoint selection (selection-sensitivity analysis):
        # best.pth = strict-IoU-best epoch (UNCHANGED criterion, so the
        # published checkpoint reproduces); best_gra.pth = GRA-best epoch
        # (val GRA at fixed tau=0.5). Both bests come from the SAME run.
        new_best_strict = val_strict_iou > best_val_strict_iou
        new_best_gra = val_gra > best_val_gra
        if new_best_strict:
            best_val_strict_iou = val_strict_iou
        if new_best_gra:
            best_val_gra = val_gra
        ckpt['best_val_strict_iou'] = best_val_strict_iou
        ckpt['best_val_gra'] = best_val_gra
        # last.pth always carries the up-to-date bests so a resume cannot
        # regress and overwrite either best checkpoint.
        atomic_save(ckpt, save_dir / 'last.pth')
        atomic_json({'history': history}, save_dir / 'history.json')
        if new_best_strict:
            atomic_save(ckpt, save_dir / 'best.pth')
            print(f'  ** new best_val_strict_iou = '
                  f'{best_val_strict_iou:.4f}, best.pth updated')
        if new_best_gra:
            atomic_save(ckpt, save_dir / 'best_gra.pth')
            print(f'  ** new best_val_gra = '
                  f'{best_val_gra:.4f}, best_gra.pth updated')

    print(f'Training done. best_val_strict_iou={best_val_strict_iou:.4f}  '
          f'best_val_gra={best_val_gra:.4f}')


if __name__ == '__main__':
    main()
