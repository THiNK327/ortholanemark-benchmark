# Modified port of the LaneATT lane-NMS implementation.
# Copyright (c) 2018, Grégoire Payen de La Garanderie, Durham University.
# Incorporates upstream notices for Copyright (c) 2015 Microsoft Corporation.
# BSD-3-Clause and MIT terms: see repository
# LICENSES/LaneATT-NMS-BSD-3-Clause-and-MIT.txt and THIRD_PARTY_NOTICES.md.

"""Pure-Python lane-NMS replacement for LaneATT's compiled CUDA op.

Upstream imports `nms` from `lib/nms/`, whose CUDA kernel suppresses
lower-scoring lanes when their mean absolute x-distance over the
overlapping valid strip range is below the `overlap` threshold. This
module mirrors that kernel's proposal layout and suppression predicate
WITHOUT requiring a CUDA toolchain.

Vectorisation strategy
----------------------
Upstream's inner C++ loop is O(N²) over proposals: for each "current
best", check whether each later candidate overlaps. That predicate is
the mean per-strip x-distance over the overlapping valid strip range
[max(start_a, start_b), min(end_a, end_b, n_offsets-1)].

Naive Python translation does ~N² `.item()` calls forcing CPU↔GPU sync
per call — at N≈1000 anchors × batch 2, that's ~10 minutes per
forward pass on this hardware. We instead:

  1. Pre-compute `start_strip[N]` and `end_strip[N]` once.
  2. For each kept anchor `i`, vectorise the check against ALL
     not-yet-suppressed candidates:
        s = max(start[i], start[j])  (vectorised over j)
        e = min(end[i], end[j], n_offsets-1)
        range_valid = e >= s
        per-strip distances:    diffs[j, k] = |x_i[k] - x_j[k]|
        range distance:         sum diffs[j, s_j : e_j + 1]
        suppressed_j = (range_dist < overlap * (e_j - s_j + 1))
                       AND range_valid
     The range-sum is implemented with a cumulative-sum trick so we
     can index `cumsum[..., e+1] - cumsum[..., s]` over the batch.

Suppression predicate is BIT-EXACT to upstream — we only replace the
inner Python loop with a vectorised tensor implementation.

Signature compatibility:
    keep, num_to_keep, parent_object_index = nms(
        proposals, scores, overlap=..., top_k=...)
"""

from __future__ import annotations

import torch


def _strip_indices(proposals: torch.Tensor):
    """Mirror upstream's start/end computation.

    Upstream (nms_kernel.cu):
        start_strip = int(start_y * n_strips + 0.5)
        end_strip   = int(start_strip + length - 1 + 0.5
                          - ((length - 1) < 0))
        end_strip   = min(end_strip, n_offsets - 1)
    """
    n_offsets = proposals.shape[1] - 5
    n_strips = n_offsets - 1
    start_y = proposals[:, 2]
    length = proposals[:, 4]
    start_strip = (start_y * n_strips + 0.5).long().clamp(min=0)
    length_minus_one = length - 1
    adjust = torch.where(length_minus_one < 0,
                         torch.ones_like(length),
                         torch.zeros_like(length))
    end_strip = (start_strip.float() + length_minus_one + 0.5 - adjust).long()
    end_strip = end_strip.clamp(max=n_offsets - 1)
    return start_strip, end_strip, n_offsets


@torch.no_grad()
def nms(proposals: torch.Tensor, scores: torch.Tensor,
        overlap: float = 50.0, top_k: int = 3000):
    """Vectorised pure-Python NMS, suppression-predicate-exact w.r.t. upstream.

    proposals : (N, 5 + S)  [cls0, cls1, start_y, start_x, length, x_offsets...]
    scores    : (N,)        confidence = softmax(cls)[:, 1] typically
    overlap   : float       mean x-distance threshold in proposal x units
    top_k     : int         maximum number of kept proposals
    """
    n = proposals.shape[0]
    dev = proposals.device
    if n == 0:
        keep = torch.zeros(0, dtype=torch.long, device=dev)
        parent = torch.zeros(0, dtype=torch.long, device=dev)
        return keep, 0, parent

    n_offsets = proposals.shape[1] - 5
    start_strip, end_strip, _ = _strip_indices(proposals)
    xs = proposals[:, 5:]                              # (N, n_offsets)

    # Cumulative sum of |x_a - x_b| → range-sum via cum[end+1] − cum[start].
    # We precompute per-anchor cumsum of |xs[i]|² etc lazily inside the
    # loop because the LHS anchor changes each iteration.

    order = torch.argsort(scores, descending=True)
    suppressed = torch.zeros(n, dtype=torch.bool, device=dev)
    keep = torch.zeros(n, dtype=torch.long, device=dev)
    parent = torch.zeros(n, dtype=torch.long, device=dev)
    num_to_keep = 0
    threshold = float(overlap)
    # Precompute a (n_offsets + 1) zero column for cumsum prefix.
    zero_col = torch.zeros((1, 1), device=dev, dtype=xs.dtype)

    # Move the order list to Python ints once — avoids per-iteration
    # .item() calls (which were the main slow-down).
    order_list = order.cpu().tolist()
    start_list = start_strip.cpu().tolist()
    end_list = end_strip.cpu().tolist()

    for ii in range(len(order_list)):
        i = order_list[ii]
        if bool(suppressed[i]):
            continue
        keep[num_to_keep] = i
        parent[i] = num_to_keep + 1

        # Gather not-yet-suppressed candidates after `i` in score order.
        cand_pos = ii + 1
        if cand_pos >= len(order_list):
            num_to_keep += 1
            if num_to_keep == top_k:
                break
            continue

        # Collect remaining indices as a tensor for vectorised ops.
        rest = torch.as_tensor(order_list[cand_pos:], device=dev,
                                dtype=torch.long)
        valid = ~suppressed[rest]
        if not bool(valid.any()):
            num_to_keep += 1
            if num_to_keep == top_k:
                break
            continue
        cand = rest[valid]

        # Per-candidate (s, e) ranges intersected with proposal i.
        s_i = int(start_list[i])
        e_i = int(end_list[i])
        s_j = start_strip[cand]
        e_j = end_strip[cand]
        s = torch.maximum(s_j, torch.full_like(s_j, s_i))
        e = torch.minimum(e_j, torch.full_like(e_j, min(e_i, n_offsets - 1)))
        rng_valid = e >= s

        # |xs[i] − xs[cand]|: shape (k, n_offsets). Then cumsum over strips
        # so the range [s, e] sum is cum[e+1] − cum[s].
        diffs = (xs[i].unsqueeze(0) - xs[cand]).abs()                # (k, n_offsets)
        cum = torch.cat([zero_col.expand(diffs.shape[0], 1), diffs.cumsum(dim=1)],
                         dim=1)                                       # (k, n_offsets+1)
        idx_e = (e + 1).clamp(min=0, max=n_offsets)
        idx_s = s.clamp(min=0, max=n_offsets)
        row = torch.arange(diffs.shape[0], device=dev)
        dist = cum[row, idx_e] - cum[row, idx_s]
        strip_count = (e - s + 1).clamp(min=1).to(dist.dtype)
        suppress_mask = (dist < threshold * strip_count) & rng_valid

        if bool(suppress_mask.any()):
            to_suppress = cand[suppress_mask]
            suppressed[to_suppress] = True
            parent[to_suppress] = num_to_keep + 1

        num_to_keep += 1
        if num_to_keep == top_k:
            break

    return keep, min(int(top_k), num_to_keep), parent
