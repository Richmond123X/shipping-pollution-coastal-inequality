# -*- coding: utf-8 -*-
"""邻接矩阵构建：月度边权 → 静态图（训练期均值）。

- 边权 log(1+x) 变换（方案 5.2 数据预处理）
- DCRNN：双向扩散转移矩阵 [D_o^{-1}W, D_i^{-1}W^T]
- Graph WaveNet：对称归一化 + 模型内部另有自适应邻接
- AGCRN：不需要预定义图（节点嵌入自学习），此处仅供可选正则/对照
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dataset import MonthlyData


def static_adjacency(data: MonthlyData, train_hi: str = "202412") -> np.ndarray:
    """训练期（≤ train_hi）月度综合边权均值 → log1p → [N, N]。"""
    n_idx = {nid: i for i, nid in enumerate(data.nodes)}
    N = len(data.nodes)
    acc = np.zeros((N, N), dtype=np.float64)
    n_months = 0
    for ym, e in data.edges.items():
        if ym > train_hi.replace("-", ""):
            continue
        n_months += 1
        src = e["From_City"].map(n_idx)
        dst = e["To_City"].map(n_idx)
        ok = src.notna() & dst.notna()
        np.add.at(acc, (src[ok].astype(int), dst[ok].astype(int)), e.loc[ok, "w"].values)
    if n_months:
        acc /= n_months
    return np.log1p(acc)


def row_normalize(A: np.ndarray) -> np.ndarray:
    d = A.sum(axis=1, keepdims=True)
    d[d == 0] = 1.0
    return A / d


def sym_normalize(A: np.ndarray, add_self_loop: bool = True) -> np.ndarray:
    """D^{-1/2}(A+I)D^{-1/2}（GWNet 固定支撑）。先对称化。"""
    S = np.maximum(A, A.T)
    if add_self_loop:
        S = S + np.eye(len(S))
    d = S.sum(axis=1)
    d_inv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    return (S * d_inv[:, None]) * d_inv[None, :]


def dual_random_walk(A: np.ndarray) -> list[np.ndarray]:
    """DCRNN 双向扩散支撑：[前向随机游走, 反向随机游走]。"""
    return [row_normalize(A), row_normalize(A.T)]
