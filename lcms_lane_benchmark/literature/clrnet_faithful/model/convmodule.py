"""Minimal drop-in for the subset of `mmcv.cnn.ConvModule` that CLRNet uses.

mmcv semantics reproduced exactly for the call sites in fpn.py and
roi_gather.py:
  - order: conv -> norm -> act
  - bias='auto': False when a norm layer follows, True otherwise
  - act_cfg defaults to ReLU; passing act_cfg=None disables activation
  - norm_cfg=dict(type='BN') -> BatchNorm2d; None -> no norm
"""

from __future__ import annotations

import torch.nn as nn

_DEFAULT_ACT = {'type': 'ReLU'}


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias='auto', conv_cfg=None, norm_cfg=None,
                 act_cfg=_DEFAULT_ACT, inplace=True):
        super().__init__()
        assert conv_cfg is None, 'only plain Conv2d is supported'
        self.with_norm = norm_cfg is not None
        self.with_act = act_cfg is not None
        if bias == 'auto':
            bias = not self.with_norm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.with_norm:
            assert norm_cfg.get('type', 'BN') == 'BN', \
                'only BN norm_cfg is supported'
            self.bn = nn.BatchNorm2d(out_channels)
        if self.with_act:
            assert act_cfg.get('type', 'ReLU') == 'ReLU', \
                'only ReLU act_cfg is supported'
            self.activate = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.with_norm:
            x = self.bn(x)
        if self.with_act:
            x = self.activate(x)
        return x
