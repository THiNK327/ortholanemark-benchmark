"""OrthoLaneMark anchor-frequency calibration for LaneATT — mirrors upstream's
`utils/gen_anchor_mask.py`.

For each training image, run `match_proposals_with_targets` against the
2784 generated anchors and count how often each anchor matches a
positive target. The resulting frequency vector is saved as a `.pt`
tensor; at training time the model loads it via
`anchors_freq_path` + `topk_anchors=1000` and keeps only the
top-1000 most-frequently-matched anchors.

Algorithm verbatim from upstream gen_anchor_mask.py:
    for sample in train_dataset:
        positives_mask, _, _, target_indices = match_proposals_with_targets(
            model, model.anchors, targets, t_pos=30., t_neg=35.)
        anchors_frequency += positives_mask

Usage:
    python -m ortholanemark.literature.laneatt.gen_anchor_mask \\
        --manifest ortholanemark/clean_gt/v1/manifest_paper.json \\
        --output ortholanemark/literature/laneatt/anchor_frequencies.pt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ortholanemark.literature.laneatt.model.laneatt import (
    LaneATT,
)
from ortholanemark.literature.laneatt.model.matching import (
    match_proposals_with_targets,
)
from ortholanemark.literature.laneatt.dataset import (
    LaneATTDataset,
)


def get_anchors_use_frequency(model: LaneATT,
                                dataset: LaneATTDataset,
                                t_pos: float = 30.0,
                                t_neg: float = 35.0) -> torch.Tensor:
    """Verbatim port of upstream get_anchors_use_frequency."""
    anchors_frequency = torch.zeros(len(model.anchors), dtype=torch.int32)
    nb_unmatched_targets = 0
    n_iter = len(dataset)
    for idx in tqdm(range(n_iter), desc='anchor calibration'):
        _img, targets, _idx = dataset[idx]
        targets = targets[targets[:, 1] == 1]
        n_targets = len(targets)
        if n_targets == 0:
            continue
        # `targets` is already a torch tensor from the dataset
        # (LaneATTDataset returns torch.from_numpy(lanes)).
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets)
        positives_mask, _, _, target_indices = match_proposals_with_targets(
            model, model.anchors, targets, t_pos=t_pos, t_neg=t_neg)
        n_matches = len(set(target_indices.tolist()))
        nb_unmatched_targets += n_targets - n_matches
        assert (n_targets - n_matches) >= 0
        anchors_frequency += positives_mask.cpu().to(torch.int32)
    print(f'unmatched_targets total: {nb_unmatched_targets}')
    return anchors_frequency


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True,
                   help='destination .pt path (consumed by '
                        'LaneATT.__init__ via anchors_freq_path)')
    p.add_argument('--img_h', type=int, default=360)
    p.add_argument('--img_w', type=int, default=640)
    p.add_argument('--S', type=int, default=72)
    p.add_argument('--max_lanes', type=int, default=2)
    p.add_argument('--backbone', default='resnet34')
    p.add_argument('--t_pos', type=float, default=30.0,
                   help='upstream gen_anchor_mask default')
    p.add_argument('--t_neg', type=float, default=35.0,
                   help='upstream gen_anchor_mask default')
    p.add_argument('--device', default='cpu',
                   help='matching is on CPU in upstream; no GPU needed')
    return p.parse_args()


def main():
    args = parse_args()
    # Fix seeds (upstream gen_anchor_mask.py:64-69).
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Build model just to expose `model.anchors` + `model.n_strips` —
    # we don't actually run the forward pass.
    model = LaneATT(backbone=args.backbone, pretrained_backbone=False,
                    S=args.S, img_w=args.img_w, img_h=args.img_h,
                    anchors_freq_path=None, topk_anchors=None,
                    anchor_feat_channels=64)
    model.eval()

    dataset = LaneATTDataset(
        args.manifest, 'train',
        img_h=args.img_h, img_w=args.img_w,
        S=args.S, max_lanes=args.max_lanes)
    print(f'Calibrating on {len(dataset)} train images, '
          f'{len(model.anchors)} anchors, t_pos={args.t_pos}, t_neg={args.t_neg}')

    freq = get_anchors_use_frequency(model, dataset,
                                      t_pos=args.t_pos, t_neg=args.t_neg)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(freq, out)
    n_nonzero = int((freq > 0).sum())
    print(f'Saved {out}  shape={tuple(freq.shape)}  nonzero={n_nonzero} '
          f'(top freq={int(freq.max())}, sum={int(freq.sum())})')


if __name__ == '__main__':
    main()
