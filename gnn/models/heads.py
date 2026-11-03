# -*- coding: utf-8 -*-
"""多任务输出头与损失（方案 5.3）：共享编码器 + 12 个并行任务头（4 船型 × 3 污染物）。"""
from __future__ import annotations

import torch
import torch.nn as nn

N_TASKS = 12  # 4 船型 × 3 污染物


class MultiTaskHeads(nn.Module):
    """编码器末端隐状态 [B, N, hidden] → [B, horizon, N, n_tasks]。

    每个任务一个独立线性头（hidden → horizon），参数量相对共享主干增加 <10%。
    """

    def __init__(self, hidden_dim: int, horizon: int, n_tasks: int = N_TASKS):
        super().__init__()
        self.horizon = horizon
        self.n_tasks = n_tasks
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, horizon) for _ in range(n_tasks)]
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, N, hidden]
        outs = [head(h) for head in self.heads]          # n_tasks × [B, N, horizon]
        y = torch.stack(outs, dim=-1)                    # [B, N, horizon, n_tasks]
        return y.permute(0, 2, 1, 3).contiguous()        # [B, horizon, N, n_tasks]


def masked_mae(pred: torch.Tensor, true: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """12 任务 Masked MAE 求和（任务初始等权，方案 5.3 训练配置）。

    pred/true/mask: [B, horizon, N, n_tasks]；mask=1 表示该观测有效。
    """
    losses = []
    for t in range(pred.shape[-1]):
        m = mask[..., t]
        denom = m.sum().clamp(min=1.0)
        losses.append(((pred[..., t] - true[..., t]).abs() * m).sum() / denom)
    return torch.stack(losses).sum()
