# -*- coding: utf-8 -*-
"""AGCRN (Bai et al. 2020)：节点自适应参数学习(NAPL) + 数据自适应图 + GRU。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import MultiTaskHeads


class AVWGCN(nn.Module):
    """自适应图卷积：A_adp = softmax(relu(E Eᵀ))；权重由节点嵌入生成（NAPL）。"""

    def __init__(self, c_in: int, c_out: int, emb_dim: int, cheb_k: int = 2):
        super().__init__()
        self.cheb_k = cheb_k
        self.W_pool = nn.Parameter(torch.randn(emb_dim, cheb_k, c_in, c_out) * 0.1)
        self.b_pool = nn.Parameter(torch.zeros(emb_dim, c_out))

    def forward(self, x: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C_in], E: [N, emb]
        N = E.shape[0]
        A = F.softmax(F.relu(E @ E.T), dim=1)
        support_set = [torch.eye(N, device=x.device), A]
        for _ in range(2, self.cheb_k):
            support_set.append(2 * A @ support_set[-1] - support_set[-2])
        supports = torch.stack(support_set, dim=0)            # [K, N, N]
        W = torch.einsum("ne,ekio->nkio", E, self.W_pool)     # [N, K, C_in, C_out]
        b = E @ self.b_pool                                   # [N, C_out]
        xg = torch.einsum("knm,bmc->bknc", supports, x)       # [B, K, N, C_in]
        xg = xg.permute(0, 2, 1, 3)                           # [B, N, K, C_in]
        return torch.einsum("bnki,nkio->bno", xg, W) + b


class AGCRNCell(nn.Module):
    def __init__(self, c_in: int, hidden: int, emb_dim: int, cheb_k: int = 2):
        super().__init__()
        self.hidden = hidden
        self.gate = AVWGCN(c_in + hidden, 2 * hidden, emb_dim, cheb_k)
        self.cand = AVWGCN(c_in + hidden, hidden, emb_dim, cheb_k)

    def forward(self, x, h, E):
        zr = torch.sigmoid(self.gate(torch.cat([x, h], dim=-1), E))
        z, r = zr.chunk(2, dim=-1)
        c = torch.tanh(self.cand(torch.cat([x, r * h], dim=-1), E))
        return z * h + (1 - z) * c


class AGCRN(nn.Module):
    """不需要预定义邻接（图由节点嵌入自学习）；forward 的 supports 参数忽略。"""

    def __init__(self, n_nodes: int, in_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, horizon: int = 12, emb_dim: int = 10,
                 cheb_k: int = 2, dropout: float = 0.3):
        super().__init__()
        self.E = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
        self.layers = nn.ModuleList(
            [AGCRNCell(in_dim if i == 0 else hidden_dim, hidden_dim, emb_dim, cheb_k)
             for i in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.heads = MultiTaskHeads(hidden_dim, horizon)

    def forward(self, x: torch.Tensor, supports=None) -> torch.Tensor:
        # x: [B, T_in, N, C]
        B, T, N, _ = x.shape
        hs = [x.new_zeros(B, N, layer.hidden) for layer in self.layers]
        for t in range(T):
            inp = x[:, t]
            for li, layer in enumerate(self.layers):
                hs[li] = layer(inp, hs[li], self.E)
                inp = self.dropout(hs[li])
        return self.heads(self.dropout(hs[-1]))  # [B, horizon, N, 12]
