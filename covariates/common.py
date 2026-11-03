# -*- coding: utf-8 -*-
"""协变量脚本公共工具。"""
from __future__ import annotations

import os

import pandas as pd
import geopandas as gpd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "outputs")

YEARS = list(range(2019, 2029))  # 2019-2028


def load_config() -> dict:
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_cities(cfg: dict | None = None) -> gpd.GeoDataFrame:
    cfg = cfg or load_config()
    return gpd.read_file(cfg["paths"]["coastal_cities_gpkg"])


def city_iso3_set(cfg: dict | None = None) -> set[str]:
    return set(load_cities(cfg)["iso3"].unique())


# 属地/特殊单元 → 主权国（国家级指标 AC2/AC4/AC5 缺失时的代理；写入数据字典与 SI）
TERRITORY_PARENT = {
    # 法国海外
    "GLP": "FRA", "MTQ": "FRA", "GUF": "FRA", "REU": "FRA", "MYT": "FRA",
    "NCL": "FRA", "PYF": "FRA", "BLM": "FRA", "MAF": "FRA", "SPM": "FRA", "WLF": "FRA",
    # 英国海外/皇家属地
    "GIB": "GBR", "IMN": "GBR", "GGY": "GBR", "JEY": "GBR", "FLK": "GBR",
    "TCA": "GBR", "VGB": "GBR", "MSR": "GBR", "SGS": "GBR", "IOT": "GBR",
    "BMU": "GBR", "CYM": "GBR",
    # 美国
    "PRI": "USA", "VIR": "USA", "GUM": "USA", "ASM": "USA", "MNP": "USA", "UMI": "USA",
    # 荷兰加勒比
    "ABW": "NLD", "CUW": "NLD", "SXM": "NLD", "BES": "NLD",
    # 丹麦 / 挪威 / 芬兰
    "FRO": "DNK", "GRL": "DNK", "SJM": "NOR", "ALA": "FIN",
    # 澳大利亚 / 新西兰（含新西兰自由结合区 Cook Islands / Niue）
    "CXR": "AUS", "CCK": "AUS", "NFK": "AUS", "TKL": "NZL",
    "COK": "NZL", "NIU": "NZL",
    # 中国特别行政区（医疗体系独立，作代理时在 SI 注明局限）
    "HKG": "CHN", "MAC": "CHN",
    # 其他
    "ESH": "MAR", "XNC": "CYP",
}


def interp_and_extend(series: pd.Series, years: list[int] = YEARS,
                      arima_min_obs: int = 10, arima_min_lastyear: int = 2015,
                      use_arima: bool = True) -> tuple[pd.Series, str]:
    """逐国年度序列 → 目标年份：内部线性插值 + 尾部门槛式 ARIMA / 末值延展。

    series: index=年份(int), values=观测值（含 NaN 的年份可缺省）
    返回 (目标年份序列, 外推方式标签)。
    """
    s = series.dropna().sort_index()
    if len(s) == 0:
        return pd.Series(index=years, dtype=float), "no_data"
    if len(s) == 1:
        return pd.Series(float(s.iloc[0]), index=years), "constant"

    # 内部缺口线性插值（仅在观测范围内）
    full_idx = range(int(s.index.min()), int(s.index.max()) + 1)
    s_interp = s.reindex(full_idx).interpolate(method="linear")

    last_year = int(s.index.max())
    method = "last_value"
    extended = s_interp.copy()

    need_tail = [y for y in years if y > last_year]
    if need_tail and use_arima and len(s) >= arima_min_obs and last_year >= arima_min_lastyear:
        try:
            from statsmodels.tsa.arima.model import ARIMA
            import numpy as np
            import warnings
            best_aic, best_fit = float("inf"), None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for order in [(1, 1, 0), (0, 1, 1), (1, 1, 1), (0, 1, 0), (1, 0, 0)]:
                    try:
                        fit = ARIMA(s_interp.values.astype(float), order=order).fit()
                        if np.isfinite(fit.aic) and fit.aic < best_aic:
                            best_aic, best_fit = fit.aic, fit
                    except Exception:  # noqa: BLE001
                        continue
            if best_fit is not None:
                fc = best_fit.forecast(steps=len(need_tail))
                fc = pd.Series(fc, index=need_tail).clip(lower=0.0)  # 非负截断
                extended = pd.concat([extended, fc])
                method = "arima"
        except Exception:  # noqa: BLE001
            pass

    out = extended.reindex(years)
    out = out.ffill().bfill()  # 头部早于首个观测的年份用首值；尾部末值延展
    return out, method
