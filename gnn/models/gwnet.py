# -*- coding: utf-8 -*-
"""Graph WaveNet (Wu et al. 2019)：自适应邻接 + 时间扩张因果卷积 + skip 连接。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import MultiTaskHeads


class GCNMix(nn.Module):
    """支撑集图卷积（固定支撑 + 自适应邻接），x [B, C, N, T]。"""

    def __init__(self, c_in: int, c_out: int, n_supports: int, order: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.order = order
        c_total = c_in * (1 + n_supports * order)
        self.mlp = nn.Conv2d(c_total, c_out, kernel_size=(1, 1))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        out = [x]
        for A in supports:
            xk = x
            for _ in range(self.order):
                xk = torch.einsum("ncvl,vw->ncwl", xk, A)
                out.append(xk)
        h = self.mlp(torch.cat(out, dim=1))
        return F.dropout(h, self.dropout, training=self.training)


class GraphWaveNet(nn.Module):
    def __init__(self, n_nodes: int, in_dim: int, hidden_dim: int = 64,
                 horizon: int = 12, n_layers: int = 8, kernel: int = 2,
                 emb_dim: int = 10, dropout: float = 0.3, n_fixed_supports: int = 1):
        super().__init__()
        self.start = nn.Conv2d(in_dim, hidden_dim, kernel_size=(1, 1))
        # 自适应邻接的源/汇节点嵌入
        self.e1 = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
        self.e2 = nn.Parameter(torch.randn(emb_dim, n_nodes) * 0.1)

        self.tcn_f = nn.ModuleList()
        self.tcn_g = nn.ModuleList()
        self.gcn = nn.ModuleList()
        self.skip = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.receptive = 1
        d = 1
        for i in range(n_layers):
            self.tcn_f.append(nn.Conv2d(hidden_dim, hidden_dim, (1, kernel), dilation=(1, d)))
            self.tcn_g.append(nn.Conv2d(hidden_dim, hidden_dim, (1, kernel), dilation=(1, d)))
            self.gcn.append(GCNMix(hidden_dim, hidden_dim,
                                   n_supports=n_fixed_supports + 1, dropout=dropout))
            self.skip.append(nn.Conv2d(hidden_dim, hidden_dim * 2, (1, 1)))
            self.bn.append(nn.BatchNorm2d(hidden_dim))
            self.receptive += d * (kernel - 1)
            d = 2 if d == 1 else (1 if d == 4 else 4)  # 1,2,4 循环扩张
        self.end1 = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, (1, 1))
        self.heads = MultiTaskHeads(hidden_dim * 2, horizon)
        self.dropout = nn.Dropout(dropout)

    def adaptive_support(self) -> torch.Tensor:
        return F.softmax(F.relu(self.e1 @ self.e2), dim=1)

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        # x: [B, T_in, N, C] → [B, C, N, T_in]
        x = x.permute(0, 3, 2, 1)
        if x.shape[-1] < self.receptive:
            x = F.pad(x, (self.receptive - x.shape[-1], 0))
        h = self.start(x)
        sup = supports + [self.adaptive_support()]
        skip_sum = 0
        for f, g, gc, sk, bn in zip(self.tcn_f, self.tcn_g, self.gcn, self.skip, self.bn):
            res = h
            ht = torch.tanh(f(h)) * torch.sigmoid(g(h))
            s = sk(ht)
            skip_sum = s[..., -1:] + (skip_sum if isinstance(skip_sum, int) else skip_sum)
            h = gc(ht, sup)
            h = bn(h + res[..., -h.shape[-1]:])
        out = F.relu(self.end1(F.relu(skip_sum)))      # [B, 2*hidden, N, 1]
        out = out.squeeze(-1).permute(0, 2, 1)         # [B, N, 2*hidden]
        return self.heads(self.dropout(out))           # [B, horizon, N, 12]
