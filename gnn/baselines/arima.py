# -*- coding: utf-8 -*-
"""基准 2：逐城市独立 ARIMA（方案 5.4）。无网络结构，逐 (节点, 任务) 单变量拟合。"""
from __future__ import annotations

import warnings

import numpy as np


def arima_forecast(target: np.ndarray, horizon: int,
                   orders=((1, 1, 1), (0, 1, 1), (1, 0, 0)),
                   seasonal: bool = True) -> np.ndarray:
    """target [T_train, N, C] → 预测 [horizon, N, C]。

    每个 (n, c) 序列独立拟合（AIC 选阶；可选 12 期季节差分）；
    失败退化为末值延展。计算量大，由调用方决定并行/抽样策略。
    """
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    T, N, C = target.shape
    out = np.zeros((horizon, N, C), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in range(N):
            for c in range(C):
                y = target[:, n, c].astype(float)
                if not np.isfinite(y).all() or y.std() < 1e-12:
                    out[:, n, c] = y[-1] if len(y) else 0.0
                    continue
                best_aic, fc = np.inf, None
                for order in orders:
                    try:
                        if seasonal:
                            fit = SARIMAX(y, order=order,
                                          seasonal_order=(0, 1, 1, 12)).fit(disp=False)
                        else:
                            fit = ARIMA(y, order=order).fit()
                        if np.isfinite(fit.aic) and fit.aic < best_aic:
                            best_aic = fit.aic
                            fc = fit.forecast(steps=horizon)
                    except Exception:  # noqa: BLE001
                        continue
                out[:, n, c] = np.clip(fc, 0, None) if fc is not None else y[-1]
    return out
