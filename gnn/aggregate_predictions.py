# -*- coding: utf-8 -*-
"""月度预测 → 年度综合排放（方案 5.5/5.6），供研究 3 的 F 项。

输入：预测张量 npz（pred_mean / pred_var，[T_pred, N, 12]，已反标准化、单位与
      城市排放一致）+ 节点列表 + 预测月份列表
步骤：① 污染物 5:2:1 加权 → ② 4 船型加和 → ③ 月度聚合到年度（2026/2027）
不确定性：月度方差求和聚合（独立近似），给出年度均值 ± 1.96σ。
输出：covariates/outputs/f_input_2026_2027.csv
      (node_id, year, E_pred, E_pred_lo, E_pred_hi)
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from dataset import load_config, MAIN_POLLUTANTS

HERE = os.path.dirname(os.path.abspath(__file__))


def aggregate(pred_mean: np.ndarray, pred_var: np.ndarray,
              months: list[str], nodes: list[str],
              ship_types: list[str], weights: dict) -> pd.DataFrame:
    T, N, C = pred_mean.shape
    assert C == len(ship_types) * len(MAIN_POLLUTANTS)

    w = np.array([weights[p] for p in MAIN_POLLUTANTS], dtype=np.float64)
    w_full = np.tile(w, len(ship_types))                # [12] 通道顺序 ship × pollutant

    # ①② 综合加权 + 船型加和（线性组合 → 方差按权重平方组合，独立近似）
    comp_mean = pred_mean @ w_full                      # [T, N]
    comp_var = pred_var @ (w_full ** 2)                 # [T, N]

    rows = []
    years = sorted({m[:4] for m in months})
    for y in years:
        idx = [t for t, m in enumerate(months) if m[:4] == y]
        mu = comp_mean[idx].sum(axis=0)                 # ③ 月度→年度求和
        sd = np.sqrt(comp_var[idx].sum(axis=0))
        for i, nid in enumerate(nodes):
            rows.append({"node_id": nid, "year": int(y),
                         "E_pred": mu[i],
                         "E_pred_lo": max(mu[i] - 1.96 * sd[i], 0.0),
                         "E_pred_hi": mu[i] + 1.96 * sd[i]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-npz", required=True,
                    help="预测文件: pred_mean/pred_var/months/nodes")
    args = ap.parse_args()
    cfg = load_config()
    d = np.load(args.pred_npz, allow_pickle=True)
    out = aggregate(d["pred_mean"], d["pred_var"],
                    list(d["months"]), list(d["nodes"]),
                    list(cfg["emission_network"]["ship_types"]),
                    cfg["emission_network"]["pollutant_weights"])
    dst = os.path.join(cfg["paths"]["covariates_dir"], "f_input_2026_2027.csv")
    out.to_csv(dst, index=False, encoding="utf-8-sig")
    print(f"已写出 F 项输入: {dst}  ({out.node_id.nunique()} 城市 × {out.year.nunique()} 年)")


if __name__ == "__main__":
    main()
