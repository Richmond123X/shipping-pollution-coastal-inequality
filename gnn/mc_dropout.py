# -*- coding: utf-8 -*-
"""MC Dropout 不确定性量化（方案 5.6）。

推断时保持 dropout 开启，对每个输入窗口采样 n_samples 次（默认 100），
输出预测均值与方差（标准化空间；反变换由调用方用 ChannelScaler.inverse）。
"""
from __future__ import annotations

import torch
import torch.nn as nn


def enable_mc_dropout(model: nn.Module) -> None:
    """eval 模式下单独把 Dropout 层切回 train（BatchNorm 等保持 eval）。"""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


@torch.no_grad()
def mc_predict(model: nn.Module, x: torch.Tensor, supports,
               n_samples: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """x [B, T_in, N, C] → (均值, 方差) 各 [B, horizon, N, 12]。"""
    enable_mc_dropout(model)
    acc, acc2 = None, None
    for _ in range(n_samples):
        y = model(x, supports)
        acc = y if acc is None else acc + y
        acc2 = y * y if acc2 is None else acc2 + y * y
    mean = acc / n_samples
    var = acc2 / n_samples - mean * mean
    return mean, var.clamp(min=0.0)
