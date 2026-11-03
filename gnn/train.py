# -*- coding: utf-8 -*-
"""GNN 训练骨架（方案 5.3 训练配置）。本阶段只搭建，不执行训练。

用法（迁到 GPU 服务器、Part B 月度数据就绪后）：
  python train.py --model dcrnn      # dcrnn / gwnet / agcrn / lstm
  python train.py --model gwnet --device cuda

配置（config.yaml: gnn）：输入 12 月 / 视域 12 月 / hidden 64 / 2 层 / dropout 0.3 /
Adam lr 1e-3, wd 1e-5 / ReduceLROnPlateau patience 10 / batch 32 / max 200 epoch /
early stop patience 20（验证集综合 Masked MAE）/ 损失 = 12 任务 Masked MAE 求和。

注：评价指标对比模块（MAE/RMSE/MAPE 表，方案 5.4/Table 2）按计划暂未实现。
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dataset import (ChannelScaler, build_monthly_data, load_config,
                     make_windows, MAIN_POLLUTANTS)
from adjacency import static_adjacency, dual_random_walk, sym_normalize
from models import DCRNN, GraphWaveNet, AGCRN, masked_mae
from baselines.lstm import PerCityLSTM

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "checkpoints")


def build_model(name: str, n_nodes: int, in_dim: int, g: dict):
    h, L, hor, dp = (int(g["hidden_dim"]), int(g["num_layers"]),
                     int(g["horizon"]), float(g["dropout"]))
    if name == "dcrnn":
        return DCRNN(in_dim, h, L, hor, dropout=dp, n_supports=2)
    if name == "gwnet":
        return GraphWaveNet(n_nodes, in_dim, h, hor, dropout=dp, n_fixed_supports=1)
    if name == "agcrn":
        return AGCRN(n_nodes, in_dim, h, L, hor, dropout=dp)
    if name == "lstm":
        return PerCityLSTM(in_dim, h, L, hor, dp)
    raise SystemExit(f"未知模型: {name}")


def build_supports(name: str, A: np.ndarray, device) -> list[torch.Tensor]:
    if name == "dcrnn":
        mats = dual_random_walk(A)
    elif name == "gwnet":
        mats = [sym_normalize(A)]
    else:  # agcrn / lstm 不用预定义图
        mats = []
    return [torch.tensor(m, dtype=torch.float32, device=device) for m in mats]


def to_loader(x, y, m, batch_size, shuffle):
    ds = TensorDataset(*(torch.tensor(a, dtype=torch.float32) for a in (x, y, m)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["dcrnn", "gwnet", "agcrn", "lstm"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config()
    g = cfg["gnn"]
    device = torch.device(args.device)

    # ---- 数据 ----
    print("构建月度数据 ...")
    data = build_monthly_data(cfg)
    n_targets = data.target.shape[-1]
    scaler = ChannelScaler().fit(
        data.target[:len([m for m in data.months if m <= g["train_range"][1].replace('-', '')])],
        data.mask[:len([m for m in data.months if m <= g["train_range"][1].replace('-', '')])],
    )
    data.target = scaler.transform(data.target).astype(np.float32)
    os.makedirs(CKPT_DIR, exist_ok=True)
    scaler.save(os.path.join(CKPT_DIR, "scaler.npz"))

    xtr, ytr, mtr = make_windows(data, cfg, "train")
    xva, yva, mva = make_windows(data, cfg, "val")
    print(f"train {len(xtr)} 窗  val {len(xva)} 窗  N={len(data.nodes)}  "
          f"C_in={xtr.shape[-1]}  任务={n_targets}")
    tr_loader = to_loader(xtr, ytr, mtr, int(g["batch_size"]), True)
    va_loader = to_loader(xva, yva, mva, int(g["batch_size"]), False)

    # ---- 图与模型 ----
    A = static_adjacency(data, g["train_range"][1])
    supports = build_supports(args.model, A, device)
    model = build_model(args.model, len(data.nodes), xtr.shape[-1], g).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(g["lr"]),
                           weight_decay=float(g["weight_decay"]))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=int(g["lr_plateau_patience"]), factor=0.5)

    # ---- 训练循环（早停基于验证集综合 Masked MAE）----
    best_val, best_epoch = float("inf"), -1
    patience = int(g["early_stop_patience"])
    ckpt = os.path.join(CKPT_DIR, f"{args.model}_best.pt")
    for epoch in range(int(g["max_epochs"])):
        model.train()
        tr_loss = 0.0
        for xb, yb, mb in tr_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            # 数据增强：节点特征随机扰动（正则化，方案 5.3）
            xb = xb + torch.randn_like(xb) * 0.01
            opt.zero_grad()
            loss = masked_mae(model(xb, supports), yb, mb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_loss += float(loss)
        model.eval()
        va_loss, nb = 0.0, 0
        with torch.no_grad():
            for xb, yb, mb in va_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                va_loss += float(masked_mae(model(xb, supports), yb, mb))
                nb += 1
        va_loss /= max(nb, 1)
        sched.step(va_loss)
        print(f"epoch {epoch:3d}  train {tr_loss / max(len(tr_loader), 1):.4f}  "
              f"val {va_loss:.4f}")
        if va_loss < best_val:
            best_val, best_epoch = va_loss, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val": va_loss, "nodes": data.nodes,
                        "months": data.months}, ckpt)
        elif epoch - best_epoch >= patience:
            print(f"early stop @ {epoch} (best {best_epoch}: {best_val:.4f})")
            break
    print(f"完成: {ckpt}")


if __name__ == "__main__":
    main()
