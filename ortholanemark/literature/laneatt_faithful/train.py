"""Faithful LaneATT training entrypoint.

Architecture / loss / optimizer / scheduler verbatim from upstream
`cfgs/laneatt_tusimple_resnet34.yml` + `lib/runner.py`. Uses the
OrthoLaneMark-calibrated anchor frequency file with topk_anchors=1000 (matches
upstream's anchor-freq + topk policy).

Checkpoint selection: per-epoch val_strict_iou.
Reproducibility: explicit seed + cudnn det + atomic save + RNG state.
Resume: `--resume` (default auto-detect) restores everything.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ortholanemark.literature.laneatt_faithful.model.laneatt import (
    LaneATT,
)
from ortholanemark.literature.laneatt_faithful.dataset import (
    LaneATTFaithfulDataset,
)
from ortholanemark.literature.laneatt_faithful.predict import (
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
                   help='Fair-comparison budget across all faithful baselines.')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--img_h', type=int, default=360)
    p.add_argument('--img_w', type=int, default=640)
    p.add_argument('--S', type=int, default=72)
    p.add_argument('--max_lanes', type=int, default=2)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--backbone', default='resnet34')
    p.add_argument('--cls_loss_weight', type=float, default=10.0)
    p.add_argument('--train_nms_thres', type=float, default=15.0,
                   help='Upstream `train_parameters.nms_thres` = 15.')
    p.add_argument('--train_nms_topk', type=int, default=3000)
    p.add_argument('--anchors_freq_path', default=None,
                   help='OrthoLaneMark-calibrated anchor freq .pt; if set, model '
                        'keeps top-K anchors. Upstream does the same.')
    p.add_argument('--topk_anchors', type=int, default=1000)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--no_auto_resume', action='store_true')
    return p.parse_args()


def _val_predict_batch_fn(model, image_grays, img_h, img_w, S, max_lanes):
    return predict_batch_with_model(model, image_grays, img_h=img_h,
                                     img_w=img_w, S=S, max_lanes=max_lanes)


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)
    gen = make_loader_generator(args.seed)

    train_ds = LaneATTFaithfulDataset(
        args.manifest, 'train', img_h=args.img_h, img_w=args.img_w,
        S=args.S, max_lanes=args.max_lanes, rotation_deg=2.0)
    val_ds = LaneATTFaithfulDataset(
        args.manifest, 'val', img_h=args.img_h, img_w=args.img_w,
        S=args.S, max_lanes=args.max_lanes, rotation_deg=0.0)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=True,
                          generator=gen)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    freq_path = args.anchors_freq_path
    topk = args.topk_anchors if freq_path is not None else None
    net = LaneATT(backbone=args.backbone, pretrained_backbone=True,
                  S=args.S, img_w=args.img_w, img_h=args.img_h,
                  anchors_freq_path=freq_path, topk_anchors=topk,
                  anchor_feat_channels=64).to(args.device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    total_iters = max(args.epochs * len(train_dl), 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max=total_iters)

    history = []
    best_val_strict_iou = -1.0
    best_val_gra = -1.0
    best_val_loss_for_log = float('inf')
    start_epoch = 1
    t_start = time.time()
    base_elapsed = 0.0

    resume_ckpt = None
    if args.resume or (not args.no_auto_resume):
        resume_ckpt = load_resume_checkpoint(save_dir)
    if resume_ckpt is not None and not args.no_auto_resume:
        net.load_state_dict(resume_ckpt['model'])
        optimizer.load_state_dict(resume_ckpt['optimizer'])
        sched.load_state_dict(resume_ckpt['lr_scheduler'])
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
          f'Params: {sum(p.numel() for p in net.parameters()):,}  '
          f'Anchors: {len(net.anchors)}  seed: {args.seed}')

    for ep in range(start_epoch, args.epochs + 1):
        net.train()
        tr_loss = 0.0
        n_tr = 0
        for images, labels, _ in train_dl:
            images = images.to(args.device)
            labels = labels.to(args.device)
            proposals_list = net(images, conf_threshold=None,
                                  nms_thres=args.train_nms_thres,
                                  nms_topk=args.train_nms_topk)
            loss, _ = net.loss(proposals_list, labels,
                                cls_loss_weight=args.cls_loss_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
            sched.step()
            tr_loss += float(loss.detach().cpu()) * images.size(0)
            n_tr += images.size(0)
        tr_loss /= max(n_tr, 1)

        net.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for images, labels, _ in val_dl:
                images = images.to(args.device)
                labels = labels.to(args.device)
                proposals_list = net(images, conf_threshold=None,
                                      nms_thres=args.train_nms_thres,
                                      nms_topk=args.train_nms_topk)
                loss, _ = net.loss(proposals_list, labels,
                                    cls_loss_weight=args.cls_loss_weight)
                val_loss += float(loss.cpu()) * images.size(0)
                n_val += images.size(0)
        val_loss /= max(n_val, 1)

        v = compute_val_strict_iou(
            net, args.manifest, _val_predict_batch_fn,
            predict_kwargs={'img_h': args.img_h, 'img_w': args.img_w,
                            'S': args.S, 'max_lanes': args.max_lanes},
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
              f'lr={lr_now:.6f}  +{elapsed:.1f}min')

        ckpt = {
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': sched.state_dict(),
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
                'img_h': args.img_h, 'img_w': args.img_w, 'S': args.S,
                'max_lanes': args.max_lanes, 'backbone': args.backbone,
                'batch_size': args.batch_size, 'lr': args.lr,
                'epochs': args.epochs, 'rotation_deg': 2.0,
                'anchors_freq_path': args.anchors_freq_path,
                'topk_anchors': args.topk_anchors if args.anchors_freq_path else None,
                'train_nms_thres': args.train_nms_thres,
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
        ckpt['best_val_loss_for_log'] = best_val_loss_for_log
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
