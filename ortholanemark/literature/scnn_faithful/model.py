# Verbatim port of https://github.com/harryhan618/SCNN_Pytorch/blob/master/model.py
# (MIT, see LICENSE_upstream).
#
# Adaptation for our 2-lane (L, R) task vs upstream's 4-lane (CULane/TuSimple):
#   - num_lanes is now a constructor argument (default 4 = upstream behaviour).
#     We pass num_lanes=2 (seg head outputs 3 classes: bg, L, R;
#     existence head outputs 2 sigmoids).
#   - CE class weights mirror upstream: bg = scale_background (0.4),
#     every lane = 1.0. Same shape rule, dynamically sized to num_lanes.
#   - Loss scales (0.4 / 1.0 / 0.1) unchanged.
#   - Backbone (dilated VGG-16 BN, ImageNet-pretrained), spatial message
#     passing layer, all conv shapes, dropout, AvgPool, and the
#     fc_input_feature formula are byte-for-byte from upstream.
#
# Per Pan et al. (AAAI 2018) §3.4 + §4.1, num_classes (here num_lanes + 1)
# is dataset-specific: 5 on CULane (4 lanes + bg), 7 on TuSimple
# (6 lanes + bg). Adapting it to 3 for our 2-lane OrthoLaneMark task is a
# standard task adaptation.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SCNN(nn.Module):
    def __init__(
            self,
            input_size,
            ms_ks=9,
            pretrained=True,
            num_lanes=4,
    ):
        """
        Argument
            ms_ks: kernel size in message passing conv
            num_lanes: number of lane classes (excluding background).
                       Upstream hardcoded 4 for CULane. We expose this so we
                       can train num_lanes=2 (L, R) without touching any
                       other line. Default 4 reproduces upstream behaviour.
        """
        super(SCNN, self).__init__()
        self.pretrained = pretrained
        self.num_lanes = int(num_lanes)
        self.net_init(input_size, ms_ks)
        if not pretrained:
            self.weight_init()

        self.scale_background = 0.4
        self.scale_seg = 1.0
        self.scale_exist = 0.1

        # CE weight vector: [bg_weight, 1, 1, ..., 1] of length (num_lanes + 1).
        # Upstream had a literal [0.4, 1, 1, 1, 1] for 4 lanes — same rule.
        ce_weight = torch.tensor(
            [self.scale_background] + [1.0] * self.num_lanes,
            dtype=torch.float32,
        )
        self.ce_loss = nn.CrossEntropyLoss(weight=ce_weight)
        self.bce_loss = nn.BCELoss()

    def forward(self, img, seg_gt=None, exist_gt=None):
        x = self.backbone(img)
        x = self.layer1(x)
        x = self.message_passing_forward(x)
        x = self.layer2(x)

        seg_pred = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
        x = self.layer3(x)
        x = x.view(-1, self.fc_input_feature)
        exist_pred = self.fc(x)

        if seg_gt is not None and exist_gt is not None:
            loss_seg = self.ce_loss(seg_pred, seg_gt)
            loss_exist = self.bce_loss(exist_pred, exist_gt)
            loss = loss_seg * self.scale_seg + loss_exist * self.scale_exist
        else:
            loss_seg = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss_exist = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss = torch.tensor(0, dtype=img.dtype, device=img.device)

        return seg_pred, exist_pred, loss_seg, loss_exist, loss

    def message_passing_forward(self, x):
        Vertical = [True, True, False, False]
        Reverse = [False, True, False, True]
        for ms_conv, v, r in zip(self.message_passing, Vertical, Reverse):
            x = self.message_passing_once(x, ms_conv, v, r)
        return x

    def message_passing_once(self, x, conv, vertical=True, reverse=False):
        """
        Argument:
        ----------
        x: input tensor
        vertical: vertical message passing or horizontal
        reverse: False for up-down or left-right, True for down-up or right-left
        """
        nB, C, H, W = x.shape
        if vertical:
            slices = [x[:, :, i:(i + 1), :] for i in range(H)]
            dim = 2
        else:
            slices = [x[:, :, :, i:(i + 1)] for i in range(W)]
            dim = 3
        if reverse:
            slices = slices[::-1]

        out = [slices[0]]
        for i in range(1, len(slices)):
            out.append(slices[i] + F.relu(conv(out[i - 1])))
        if reverse:
            out = out[::-1]
        return torch.cat(out, dim=dim)

    def net_init(self, input_size, ms_ks):
        input_w, input_h = input_size
        # Upstream literal: 5 = num_classes (bg + 4 lanes). Same formula,
        # generalised to (num_lanes + 1) classes.
        self.fc_input_feature = (self.num_lanes + 1) * int(input_w / 16) * int(input_h / 16)
        self.backbone = models.vgg16_bn(pretrained=self.pretrained).features

        # ----------------- process backbone -----------------
        for i in [34, 37, 40]:
            conv = self.backbone._modules[str(i)]
            dilated_conv = nn.Conv2d(
                conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride,
                padding=tuple(p * 2 for p in conv.padding), dilation=2, bias=(conv.bias is not None)
            )
            dilated_conv.load_state_dict(conv.state_dict())
            self.backbone._modules[str(i)] = dilated_conv
        self.backbone._modules.pop('33')
        self.backbone._modules.pop('43')

        # ----------------- SCNN part -----------------
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()  # (nB, 128, 36, 100)
        )

        # ----------------- add message passing -----------------
        self.message_passing = nn.ModuleList()
        self.message_passing.add_module('up_down', nn.Conv2d(128, 128, (1, ms_ks), padding=(0, ms_ks // 2), bias=False))
        self.message_passing.add_module('down_up', nn.Conv2d(128, 128, (1, ms_ks), padding=(0, ms_ks // 2), bias=False))
        self.message_passing.add_module('left_right',
                                        nn.Conv2d(128, 128, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))
        self.message_passing.add_module('right_left',
                                        nn.Conv2d(128, 128, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))
        # (nB, 128, 36, 100)

        # ----------------- SCNN part -----------------
        # Upstream: nn.Conv2d(128, 5, 1) for 5 classes (bg + 4 lanes).
        # Generalised to (num_lanes + 1) classes.
        self.layer2 = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(128, self.num_lanes + 1, 1),
        )

        self.layer3 = nn.Sequential(
            nn.Softmax(dim=1),
            nn.AvgPool2d(2, 2),
        )
        # Upstream: final fc emits 4 (one existence sigmoid per lane).
        # Generalised to num_lanes outputs.
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_feature, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_lanes),
            nn.Sigmoid()
        )

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data[:] = 1.
                m.bias.data.zero_()
