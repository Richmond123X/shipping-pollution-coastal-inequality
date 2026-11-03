# -*- coding: utf-8 -*-
"""GNN 框架 shape 冒烟检查：随机张量前向（小 N），不训练。"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from models import DCRNN, GraphWaveNet, AGCRN, masked_mae  # noqa: E402
from baselines.lstm import PerCityLSTM  # noqa: E402
from adjacency import dual_random_walk, sym_normalize  # noqa: E402
from mc_dropout import mc_predict  # noqa: E402

B, T_IN, HORIZON, N, C_IN, N_TASKS = 4, 12, 12, 30, 16, 12


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(B, T_IN, N, C_IN)
    y = torch.randn(B, HORIZON, N, N_TASKS)
    m = torch.ones_like(y)
    A = np.abs(np.random.rand(N, N)) * (np.random.rand(N, N) > 0.7)

    cases = [
        ("DCRNN", DCRNN(C_IN, 64, 2, HORIZON),
         [torch.tensor(s, dtype=torch.float32) for s in dual_random_walk(A)]),
        ("GraphWaveNet", GraphWaveNet(N, C_IN, 64, HORIZON),
         [torch.tensor(sym_normalize(A), dtype=torch.float32)]),
        ("AGCRN", AGCRN(N, C_IN, 64, 2, HORIZON), []),
        ("PerCityLSTM", PerCityLSTM(C_IN, 64, 2, HORIZON), []),
    ]
    for name, model, supports in cases:
        out = model(x, supports)
        assert out.shape == (B, HORIZON, N, N_TASKS), f"{name}: {out.shape}"
        loss = masked_mae(out, y, m)
        loss.backward()  # 梯度可回传
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {name:14s} 输出 {tuple(out.shape)}  loss {float(loss):.3f}  "
              f"参数 {n_params:,}")

    # MC Dropout
    model = AGCRN(N, C_IN, 64, 2, HORIZON)
    mean, var = mc_predict(model, x, [], n_samples=8)
    assert mean.shape == (B, HORIZON, N, N_TASKS) and (var >= 0).all()
    print(f"  MC Dropout     均值 {tuple(mean.shape)}  方差非负 OK")
    print("\n✓ GNN shape 冒烟检查全部通过")


if __name__ == "__main__":
    main()
