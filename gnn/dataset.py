# -*- coding: utf-8 -*-
"""GNN 月度数据管线（研究方案 5.2）。

从 Part B 月度城市排放输出构建张量：
  X_target [T=84, N, 12]   4 船型 × 3 污染物（SO2/PM/NOx）月度排放（主预测目标）
  X_aux    [T, N, 4]       月份 sin/cos + SECA/NECA 虚拟变量（随时间变化）
  E_w      [T, N, N]       月度综合排放边权（5:2:1 加权、跨船型求和）→ adjacency.py 用

划分：训练 2019.1-2024.12 / 验证 2025.1-6 / 测试 2025.7-12（config.yaml: gnn）。
标准化：z-score 仅用训练段统计量；缺月/缺节点用 mask 标记（Masked MAE）。
滑窗：输入 12 个月 → 预测 12 个月。
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MAIN_POLLUTANTS = ["SO2", "PM", "NOx"]


def load_config() -> dict:
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def month_range(a: str, b: str) -> list[str]:
    return pd.period_range(a, b, freq="M").strftime("%Y%m").tolist()


@dataclass
class MonthlyData:
    months: list[str]          # T 个 YYYYMM
    nodes: list[str]           # N 个 node_id
    ship_types: list[str]      # 4 船型（任务通道顺序：ship × pollutant）
    target: np.ndarray         # [T, N, 12]
    aux: np.ndarray            # [T, N, 4]  (sin, cos, seca, neca)
    mask: np.ndarray           # [T, N, 12] 1=观测到（该月该船型文件存在）
    edges: dict[str, pd.DataFrame]  # YYYYMM -> (From_City, To_City, w_综合)


def _safe_folder_name(name: str) -> str:
    import re
    s = re.sub(r'[\\/:*?"<>|]', "_", str(name).strip())
    return re.sub(r"\s+", " ", s).strip()


def build_monthly_data(cfg: dict | None = None) -> MonthlyData:
    cfg = cfg or load_config()
    g = cfg["gnn"]
    em = cfg["emission_network"]
    paths = cfg["paths"]
    ship_types = list(em["ship_types"])
    w = em["pollutant_weights"]

    months = month_range(g["train_range"][0], g["test_range"][1])

    import geopandas as gpd
    cities = gpd.read_file(paths["coastal_cities_gpkg"], ignore_geometry=True) \
        if hasattr(gpd, "read_file") else None
    nodes = sorted(cities["node_id"].tolist())
    n_idx = {nid: i for i, nid in enumerate(nodes)}
    T, N = len(months), len(nodes)
    C = len(ship_types) * len(MAIN_POLLUTANTS)

    target = np.zeros((T, N, C), dtype=np.float32)
    mask = np.zeros((T, N, C), dtype=np.float32)
    edges: dict[str, pd.DataFrame] = {}

    root = paths["results_root"]
    for t, ym in enumerate(months):
        edge_parts = []
        for s, st in enumerate(ship_types):
            sub = os.path.join(root, ym, _safe_folder_name(st))
            f = os.path.join(sub, f"city_emissions_{st}_{ym}.csv")
            if os.path.isfile(f):
                df = pd.read_csv(f)
                for p, pol in enumerate(MAIN_POLLUTANTS):
                    c = s * len(MAIN_POLLUTANTS) + p
                    col = f"{pol}_Total"
                    if col in df.columns and len(df):
                        idx = df["node_id"].map(n_idx).dropna().astype(int)
                        target[t, idx.values, c] = df.loc[idx.index, col].values
                    mask[t, :, c] = 1.0  # 文件存在=该月该船型有观测（含真实 0）
            # 边：综合加权
            for pol in MAIN_POLLUTANTS:
                ef = os.path.join(sub, f"city_edge_list_{pol}_{st}_{ym}.csv")
                if os.path.isfile(ef):
                    e = pd.read_csv(ef)
                    if len(e):
                        e["w"] = e["Total_Emission"] * w[pol]
                        edge_parts.append(e[["From_City", "To_City", "w"]])
        if edge_parts:
            edges[ym] = (pd.concat(edge_parts, ignore_index=True)
                         .groupby(["From_City", "To_City"], as_index=False)["w"].sum())

    # ---- 辅助特征 ----
    aux = np.zeros((T, N, 4), dtype=np.float32)
    mm = np.array([int(ym[4:]) for ym in months], dtype=np.float32)
    aux[:, :, 0] = np.sin(2 * np.pi * mm / 12.0)[:, None]
    aux[:, :, 1] = np.cos(2 * np.pi * mm / 12.0)[:, None]
    sn_path = os.path.join(paths["covariates_dir"], "seca_neca_monthly.csv")
    if os.path.isfile(sn_path):
        sn = pd.read_csv(sn_path)
        sn["ym"] = sn["month"].str.replace("-", "")
        sn = sn[sn.ym.isin(set(months))]
        t_of = {ym: t for t, ym in enumerate(months)}
        rows = sn["ym"].map(t_of).values
        cols = sn["node_id"].map(n_idx).values
        ok = ~(pd.isna(rows) | pd.isna(cols))
        aux[rows[ok].astype(int), cols[ok].astype(int), 2] = sn.loc[ok, "seca"].values
        aux[rows[ok].astype(int), cols[ok].astype(int), 3] = sn.loc[ok, "neca"].values

    return MonthlyData(months, nodes, ship_types, target, aux, mask, edges)


# ---------------------------------------------------------------------------
# 标准化与滑窗
# ---------------------------------------------------------------------------
class ChannelScaler:
    """逐通道 z-score；统计量只来自训练段（防信息泄漏）；持久化供反变换。"""

    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.mean, self.std = mean, std

    def fit(self, x: np.ndarray, mask: np.ndarray) -> "ChannelScaler":
        m = mask.astype(bool)
        C = x.shape[-1]
        self.mean = np.zeros(C, dtype=np.float64)
        self.std = np.ones(C, dtype=np.float64)
        for c in range(C):
            v = x[..., c][m[..., c]]
            if v.size:
                self.mean[c] = v.mean()
                self.std[c] = max(v.std(), 1e-8)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str) -> "ChannelScaler":
        d = np.load(path)
        return cls(d["mean"], d["std"])


def make_windows(data: MonthlyData, cfg: dict, split: str):
    """滑窗样本。split ∈ {train, val, test}。

    返回 (X[B, T_in, N, C_in], Y[B, T_out, N, 12], M[B, T_out, N, 12])，
    其中窗口末月落在该 split 的月份范围内即归入该 split（输入可回看前段历史）。
    """
    g = cfg["gnn"]
    t_in, t_out = int(g["input_len"]), int(g["horizon"])
    lo, hi = {"train": g["train_range"], "val": g["val_range"],
              "test": g["test_range"]}[split]
    lo, hi = lo.replace("-", ""), hi.replace("-", "")

    feats = np.concatenate([data.target, data.aux], axis=-1)  # [T,N,C+4]
    xs, ys, ms = [], [], []
    T = len(data.months)
    for end in range(t_in, T - t_out + 1):
        y_last = data.months[end + t_out - 1]
        if not (lo <= y_last <= hi):
            continue
        xs.append(feats[end - t_in:end])
        ys.append(data.target[end:end + t_out])
        ms.append(data.mask[end:end + t_out])
    if not xs:
        return (np.empty((0, t_in, len(data.nodes), feats.shape[-1])),
                np.empty((0, t_out, len(data.nodes), data.target.shape[-1])),
                np.empty((0, t_out, len(data.nodes), data.target.shape[-1])))
    return np.stack(xs), np.stack(ys), np.stack(ms)
