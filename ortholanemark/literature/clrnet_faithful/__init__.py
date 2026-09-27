"""Paper-faithful CLRNet baseline (Zheng et al., CVPR 2022) — the modern
anchor-based SOTA representative in the roster.

Vendored from Turoad/CLRNet @ 7269e9d1c1c650343b6c7febb8e764be538b1aed
(see `_vendor/clrnet_src/`). The model code under `model/` is
upstream-verbatim except for mechanical changes documented per file:
mmcv's ConvModule replaced by a minimal local equivalent, the compiled
CUDA line-NMS replaced by the suppression-predicate-exact pure-Python
stub shared with laneatt_faithful, and registry plumbing removed.
"""
