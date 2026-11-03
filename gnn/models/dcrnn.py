# -*- coding: utf-8 -*-
"""DCRNN (Li et al. 2018)：双向扩散卷积 + GRU，seq2seq。纯 PyTorch 稠密实现。"""
from __future__ import annotations

import torch
import torch.nn as nn

from .heads import MultiTaskHeads


class DiffusionConv(nn.Module):
    """K 阶双向扩散卷积：x [B, N, C_in] → [B, N, C_out]。supports: 2 个 [N, N]。"""

    def __init__(self, c_in: int, c_out: int, n_supports: int, k: int = 2):
        super().__init__()
        self.k = k
        self.lin = nn.Linear(c_in * (1 + n_supports * k), c_out)

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        out = [x]
        for A in supports:
            xk = x
            for _ in range(self.k):
                xk = torch.einsum("ij,bjc->bic", A, xk)
                out.append(xk)
        return self.lin(torch.cat(out, dim=-1))


class DCGRUCell(nn.Module):
    def __init__(self, c_in: int, hidden: int, n_supports: int, k: int = 2):
        super().__init__()
        self.hidden = hidden
        self.gate = DiffusionConv(c_in + hidden, 2 * hidden, n_supports, k)
        self.cand = DiffusionConv(c_in + hidden, hidden, n_supports, k)

    def forward(self, x, h, supports):
        zr = torch.sigmoid(self.gate(torch.cat([x, h], dim=-1), supports))
        z, r = zr.chunk(2, dim=-1)
        c = torch.tanh(self.cand(torch.cat([x, r * h], dim=-1), supports))
        return z * h + (1 - z) * c


class DCRNN(nn.Module):
    """编码器读入 12 个月 → 末端隐状态 → 多任务头输出 12 个月 × 12 任务。

    与原论文逐步解码不同，这里用「编码末态 + 任务头直推 horizon」的简化解码
    （多任务 12 头直推；scheduled sampling 接口保留 use_curriculum 参数位）。
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 horizon: int = 12, k_hops: int = 2, dropout: float = 0.3,
                 n_supports: int = 2, use_curriculum: bool = False):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                DCGRUCell(in_dim if i == 0 else hidden_dim, hidden_dim, n_supports, k_hops)
            )
        self.dropout = nn.Dropout(dropout)
        self.heads = MultiTaskHeads(hidden_dim, horizon)
        self.use_curriculum = use_curriculum

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        # x: [B, T_in, N, C]
        B, T, N, _ = x.shape
        hs = [x.new_zeros(B, N, layer.hidden) for layer in self.layers]
        for t in range(T):
            inp = x[:, t]
            for li, layer in enumerate(self.layers):
                hs[li] = layer(inp, hs[li], supports)
                inp = self.dropout(hs[li])
        return self.heads(self.dropout(hs[-1]))  # [B, horizon, N, 12]
