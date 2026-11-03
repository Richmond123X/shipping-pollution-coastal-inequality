# -*- coding: utf-8 -*-
"""基准 1：历史平均 HA —— 同月份过去几年均值（方案 5.4）。"""
from __future__ import annotations

import numpy as np


def historical_average(target: np.ndarray, months: list[str],
                       predict_months: list[str],
                       train_hi: str = "202412") -> np.ndarray:
    """target [T, N, C]；对每个预测月取训练期内同历法月的均值。

    返回 [len(predict_months), N, C]。
    """
    hi = train_hi.replace("-", "")
    out = np.zeros((len(predict_months),) + target.shape[1:], dtype=np.float64)
    for i, pm in enumerate(predict_months):
        mm = pm[-2:]
        idx = [t for t, ym in enumerate(months) if ym[-2:] == mm and ym <= hi]
        if idx:
            out[i] = target[idx].mean(axis=0)
    return out
