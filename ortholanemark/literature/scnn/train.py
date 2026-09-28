"""SCNN training entrypoint.

Mirrors upstream `train.py` (harryhan618/SCNN_Pytorch) as closely as
practical given our manifest-based dataset:

  - Optimizer: SGD + Nesterov + PolyLR. Loss: model-internal CE+BCE.
  - Augmentation: Rotation(±2°) (matches upstream + uniform-aug policy).
  - Checkpoint selection: `best.pth` is the epoch with the HIGHEST val
    `strict_operational_iou` (the benchmark metric), NOT the lowest
    val_loss. This is the convention used in LaneATT (Tabelini CVPR 2021),
    LLAMAS benchmark (Behrendt 2019), and most CV benchmark papers.
  - Reproducibility: explicit `--seed`, `torch.backends.cudnn.deterministic`,
    DataLoader generator seeded, full RNG state persisted in `last.pth`.
  - Resume: `--resume` (or auto-detect last.pth) restores model, optimizer,
    scheduler, history, best_val_strict_iou, and RNG state, then
    continues from `epoch + 1`. Per-epoch atomic save (write-temp +
    rename) means a Ctrl-C at most loses the in-flight epoch.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ortholanemark.literature.scnn.model import SCNN
from ortholanemark.literature.scnn.lr_scheduler import (
    PolyLR,
)
from ortholanemark.literature.scnn.dataset import (
    SCNNDataset, collate,
)
from ortholanemark.literature.scnn.predict import (
    predict_with_model, predict_batch_with_model,
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
                   help='Fair-comparison budget: identical 100 epochs across '
                        'all learning baselines.')
    p.add_argument('--batch_size', type=int, default=4,
                   help='Upstream uses batch=32 on TuSimple at 512x288; '
                        'we use 4 (VGG-16 + 800x320 portrait, GPU bound).')
    p.add_argument('--input_h', type=int, default=800)
    p.add_argument('--input_w', type=int, default=320)
    p.add_argument('--lr', type=float, default=5e-2,
                   help='Linear-scaled from upstream lr=0.15 (batch 32 -> 4).')
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--num_lanes', type=int, default=2)
    p.add_argument('--seed', type=int, default=0,
                   help='Global seed for torch / numpy / random / cudnn.')
    p.add_argument('--resume', action='store_true',
                   help='Resume from last.pth if present (auto-detects).')
    p.add_argument('--no_auto_resume', action='store_true',
                   help='Force training from scratch even if last.pth exists.')
    return p.parse_args()


def _val_predict_batch_fn(model, image_grays, input_h, input_w, num_lanes):
    return predict_batch_with_model(model, image_grays, input_h=input_h,
                                     input_w=input_w, num_lanes=num_lanes)


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)
    gen = make_loader_generator(args.seed)

    # rotation_deg=2 mirrors upstream's `Rotation(2)`.
    train_ds = SCNNDataset(args.manifest, 'train',
                                   input_h=args.input_h,
                                   input_w=args.input_w,
                                   rotation_deg=2.0)
    val_ds = SCNNDataset(args.manifest, 'val',
                                 input_h=args.input_h,
                                 input_w=args.input_w,
                                 rotation_deg=0.0)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate,
                          drop_last=True, generator=gen)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate)

    model = SCNN(input_size=(args.input_w, args.input_h),
                 num_lanes=args.num_lanes, pretrained=True).to(args.device)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    max_iter = args.epochs * len(train_dl)
    lr_sched = PolyLR(optimizer, pow=0.9, max_iter=max_iter,
                      min_lrs=1e-10, warmup=20)

    history = []
    best_val_strict_iou = -1.0
    best_val_gra = -1.0
    best_val_loss_for_log = float('inf')
    start_epoch = 1
    t_start = time.time()
    base_elapsed = 0.0

    # Resume logic.
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
        print(f'Already at epoch {start_epoch - 1} >= {args.epochs}; '
              f'nothing to do.')
        return

    print(f'Train: {len(train_ds)}  Val: {len(val_ds)}  '
          f'Batches/epoch: {len(train_dl)}  max_iter: {max_iter}  '
          f'seed: {args.seed}')

    for ep in range(start_epoch, args.epochs + 1):
        model.train()
        tr_loss = tr_seg = tr_exist = 0.0
        n_tr = 0
        for sample in train_dl:
            img = sample['img'].to(args.device)
            seg = sample['segLabel'].to(args.device)
            exist = sample['exist'].to(args.device)
            optimizer.zero_grad()
            _, _, loss_seg, loss_exist, loss = model(img, seg, exist)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            lr_sched.step()
            tr_loss += float(loss.item()) * img.size(0)
            tr_seg += float(loss_seg.item()) * img.size(0)
            tr_exist += float(loss_exist.item()) * img.size(0)
            n_tr += img.size(0)
        tr_loss /= max(n_tr, 1)
        tr_seg /= max(n_tr, 1)
        tr_exist /= max(n_tr, 1)

        # Diagnostic: method-internal val loss
        model.eval()
        val_loss = val_seg = val_exist = 0.0
        n_val = 0
        with torch.no_grad():
            for sample in val_dl:
                img = sample['img'].to(args.device)
                seg = sample['segLabel'].to(args.device)
                exist = sample['exist'].to(args.device)
                _, _, lseg, lexist, loss = model(img, seg, exist)
                val_loss += float(loss.item()) * img.size(0)
                val_seg += float(lseg.item()) * img.size(0)
                val_exist += float(lexist.item()) * img.size(0)
                n_val += img.size(0)
        val_loss /= max(n_val, 1)
        val_seg /= max(n_val, 1)
        val_exist /= max(n_val, 1)

        # Selection criterion: per-epoch val strict_operational_iou (batched).
        v = compute_val_strict_iou(
            model, args.manifest, _val_predict_batch_fn,
            predict_kwargs={'input_h': args.input_h,
                            'input_w': args.input_w,
                            'num_lanes': args.num_lanes},
            batch_size=args.batch_size, presence_rule='native')
        val_strict_iou = v['val_strict_iou']
        val_op_iou = v['val_op_iou']
        val_gra = v['val_gra']

        lr_now = optimizer.param_groups[0]['lr']
        elapsed = base_elapsed + (time.time() - t_start) / 60.0
        rec = {
            'epoch': ep, 'train_loss': tr_loss, 'train_seg': tr_seg,
            'train_exist': tr_exist,
            'val_loss': val_loss, 'val_seg': val_seg, 'val_exist': val_exist,
            'val_strict_iou': val_strict_iou, 'val_op_iou': val_op_iou,
            'val_gra': val_gra,
            'lr': lr_now, 'elapsed_min': elapsed,
        }
        history.append(rec)
        print(f'[ep {ep:3d}] tr={tr_loss:.4f}  val_loss={val_loss:.4f}  '
              f'val_strict_iou={val_strict_iou:.4f}  '
              f'val_op_iou={val_op_iou:.4f}  '
              f'val_gra={val_gra:.4f}  '
              f'lr={lr_now:.5f}  +{elapsed:.1f}min')

        # Save last.pth (every epoch, atomic).
        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_sched.state_dict(),
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
                'num_lanes': args.num_lanes, 'batch_size': args.batch_size,
                'lr': args.lr, 'epochs': args.epochs,
                'rotation_deg': 2.0,
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
