# Verbatim port of `initialize_weights` + `real_init_weights` from
# https://github.com/cfzd/Ultra-Fast-Lane-Detection-V2/blob/master/utils/common.py
# (MIT, see ../LICENSE_upstream). Only this helper is needed downstream
# of model_culane.py and seg_model.py; the rest of upstream's
# utils/common.py is config-loading / DALI / distributed helpers we
# don't use.
import torch


def initialize_weights(*models):
    for model in models:
        real_init_weights(model)


def real_init_weights(m):
    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            pass
