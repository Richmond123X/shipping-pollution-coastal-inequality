# -*- coding: utf-8 -*-
"""X_eval / SPRI 指标标准化（研究方案 §6.5 + §6.6）。

依据各指标分布的实测偏度，区分两类处理：
  - 「水平量」(金额/计数/密度/排放，右尾重，实测 skew 2.6~9.7)：先 log1p，
    再 1% winsorize，再 min-max → [0,1]。包括 AC1、S1_pop、pop_density_km2、E1、E2。
  - 「有界比例/评分类」(本就有界或近对称，部分含负值)：不 log，仅 1% winsorize +
    min-max。包括 AC2、AC3、AC4、AC5、S2_vuln_share、E3。

理由：log 仅修正"跨数量级的右偏"；对 [0,1] 比例或 ±2.5 评分做 log 无意义且对负值无效。
log1p 后上述水平量 |skew| 全部回落到 <1.2。与方案 §6.4(F 用 log)、§6.6(1% winsorize)、
GNN(边权 log(1+x)) 的处理精神一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 需要先 log1p 的"水平量"列名（金额/计数/密度/排放）
LOG_LEVEL_COLS = frozenset(
    {"AC1", "S1_pop", "pop_density_km2", "E1", "E2"}
)
# 有界比例/评分类：不 log（AC4/AC5 还含负值）。此集合仅供文档/校验参考。
BOUNDED_COLS = frozenset(
    {"AC2", "AC3", "AC4", "AC5", "S2_vuln_share", "E3"}
)


def winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    """双侧 p 分位截尾（默认 1%，与方案 §6.6 CRITIC 一致）。"""
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def normalize_for_xeval(
    df: pd.DataFrame,
    cols,
    log_cols=None,
    winsor_p: float = 0.01,
) -> pd.DataFrame:
    """对 ``cols`` 逐列做 [可选 log1p] → winsorize(p) → min-max[0,1]。

    Parameters
    ----------
    df : 含待标准化列的 DataFrame
    cols : 要标准化的列名可迭代对象
    log_cols : 需要先 log1p 的列集合；None=取 ``LOG_LEVEL_COLS`` 与 ``cols`` 的交集
    winsor_p : winsorize 双侧分位，默认 0.01

    Returns
    -------
    仅含标准化结果的 DataFrame（列名加后缀 ``_norm``，索引与 df 对齐）。
    水平量列若出现负值（理论不应有）先截 0 再 log1p。常数列归一化为 0。
    """
    log_cols = LOG_LEVEL_COLS if log_cols is None else set(log_cols)
    out = pd.DataFrame(index=df.index)
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").astype(float)
        if c in log_cols:
            x = np.log1p(x.clip(lower=0.0))  # 水平量假定非负
        x = winsorize(x, winsor_p)
        mn, mx = x.min(), x.max()
        out[f"{c}_norm"] = 0.0 if mx == mn else (x - mn) / (mx - mn)
    return out
