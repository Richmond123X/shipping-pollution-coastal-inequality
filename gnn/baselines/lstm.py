# -*- coding: utf-8 -*-
"""基准 3：逐城市 LSTM（无网络结构，方案 5.4）。

所有城市共享一个 LSTM（节点作为 batch 维展开），无图信息；
输入输出与 GNN 相同（12 月 → 12 月 × 12 任务），训练走 train.py 同一循环。
"""
from __future__ import annotations

import torch
import torch.nn as nn

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.heads import MultiTaskHeads  # noqa: E402


class PerCityLSTM(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 horizon: int = 12, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.heads = MultiTaskHeads(hidden_dim, horizon)

    def forward(self, x: torch.Tensor, supports=None) -> torch.Tensor:
        # x: [B, T_in, N, C] → 把 N 摊进 batch（城市间无信息交换）
        B, T, N, C = x.shape
        z = x.permute(0, 2, 1, 3).reshape(B * N, T, C)
        out, _ = self.lstm(z)
        h = out[:, -1].reshape(B, N, -1)            # [B, N, hidden]
        return self.heads(self.dropout(h))          # [B, horizon, N, 12]
