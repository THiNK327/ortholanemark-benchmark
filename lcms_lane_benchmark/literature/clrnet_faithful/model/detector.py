"""CLRNet detector assembly (backbone -> FPN -> CLRHead), mirroring
upstream `clrnet/models/nets/detector.py` with the registry replaced by
direct construction from a config object (upstream clr_resnet18_culane
hyperparameters are the defaults)."""

from __future__ import annotations

import torch.nn as nn

from lcms_lane_benchmark.literature.clrnet_faithful.model.resnet import (
    ResNetWrapper,
)
from lcms_lane_benchmark.literature.clrnet_faithful.model.fpn import FPN
from lcms_lane_benchmark.literature.clrnet_faithful.model.clr_head import (
    CLRHead,
)


class CLRNetConfig:
    """Attribute bag mirroring the upstream mmcv-style Config for the
    fields the model actually reads. Defaults = clr_resnet18_culane.py
    with the LCMS task adaptations (max_lanes=2, num_classes=3,
    cut_height=0)."""

    def __init__(self, **overrides):
        self.img_w = 800
        self.img_h = 320
        self.num_points = 72
        self.max_lanes = 2          # LCMS: at most one marking per side
        self.num_classes = 2 + 1    # bg + left + right (aux seg)
        self.ignore_label = 255
        self.bg_weight = 0.4
        self.cls_loss_weight = 2.
        self.xyt_loss_weight = 0.2
        self.iou_loss_weight = 2.
        self.seg_loss_weight = 1.0
        self.num_priors = 192
        self.refine_layers = 3
        self.fc_hidden_dim = 64
        self.sample_points = 36
        self.resnet = 'resnet18'
        self.pretrained = True
        self.in_channels = [128, 256, 512]   # resnet18 layers 2-4
        self.fpn_out_channels = 64
        self.test_parameters = dict(conf_threshold=0.4, nms_thres=50,
                                    nms_topk=2)
        for k, v in overrides.items():
            setattr(self, k, v)

    def haskey(self, k):
        return hasattr(self, k)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class CLRNet(nn.Module):
    def __init__(self, cfg: CLRNetConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = ResNetWrapper(
            resnet=cfg.resnet,
            pretrained=cfg.pretrained,
            replace_stride_with_dilation=[False, False, False],
            out_conv=False,
        )
        self.neck = FPN(in_channels=cfg.in_channels,
                        out_channels=cfg.fpn_out_channels,
                        num_outs=len(cfg.in_channels),
                        attention=False)
        self.heads = CLRHead(num_points=cfg.num_points,
                             prior_feat_channels=cfg.fpn_out_channels,
                             fc_hidden_dim=cfg.fc_hidden_dim,
                             num_priors=cfg.num_priors,
                             num_fc=2,
                             refine_layers=cfg.refine_layers,
                             sample_points=cfg.sample_points,
                             cfg=cfg)

    def forward(self, batch):
        img = batch['img'] if isinstance(batch, dict) else batch
        fea = self.backbone(img)
        fea = self.neck(fea)
        if self.training:
            return self.heads(fea, batch=batch)
        return self.heads(fea)

    def get_lanes(self, output, **kw):
        return self.heads.get_lanes(output, **kw)
