# -*- coding: utf-8 -*-
"""AC1 人均 GDP：Kummu 栅格分区统计 + IMF WEO 不变价缩放（方案 3.9.4）。

2019-2022：Kummu 人均 GDP PPP 栅格（5 弧分，波段=年，1990-2022）城市多边形内均值；
小面/无值多边形退路：30km 集水区内均值。
2023-2027：AC1_t = AC1_2022 × (pc_t / pc_2022)，pc 取 WEO
  "Gross domestic product (GDP), Constant prices, Per capita, PPP ... ICP benchmark 2021"；
缺该序列的国家退用实际 GDP 增长率（Constant prices, Percent change）累乘；
两者皆缺 → 2022 值常数延展。禁止现价 PPPPC（含通胀，扭曲跨国比较）。

输出:
  outputs/ac1_gdp.csv: node_id, iso3, year, AC1, scale_method
  outputs/ac1_weo_boundary.csv: 各国 WEO 实际数据/预测分界年（SI）
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import geopandas as gpd
from rasterstats import zonal_stats

from common import OUT, YEARS, load_config, load_cities

KUMMU_FIRST_YEAR = 1990
RASTER_YEARS = [2019, 2020, 2021, 2022]

PC_NAME = ("gross domestic product (gdp), constant prices, per capita, "
           "purchasing power parity (ppp) international dollar, icp benchmark 2021")
GROWTH_NAME = "gross domestic product (gdp), constant prices, percent change"


def zonal_mean_by_year(cities: gpd.GeoDataFrame, catchments: gpd.GeoDataFrame,
                       tif: str) -> pd.DataFrame:
    rows = []
    for y in RASTER_YEARS:
        band = y - KUMMU_FIRST_YEAR + 1
        print(f"  Kummu 分区统计 {y} (band {band}) ...")
        zs = zonal_stats(cities.geometry, tif, stats="mean", band=band,
                         all_touched=True, nodata=np.nan)
        vals = [z["mean"] for z in zs]
        # 无值多边形退路：集水区
        miss_idx = [i for i, v in enumerate(vals) if v is None or not np.isfinite(v)]
        if miss_idx:
            sub = catchments.set_index("node_id").loc[
                cities["node_id"].iloc[miss_idx]].geometry
            zs2 = zonal_stats(sub, tif, stats="mean", band=band,
                              all_touched=True, nodata=np.nan)
            for k, i in enumerate(miss_idx):
                vals[i] = zs2[k]["mean"]
        for i, v in enumerate(vals):
            rows.append({"node_id": cities["node_id"].iloc[i],
                         "iso3": cities["iso3"].iloc[i], "year": y, "AC1_raster": v})
    return pd.DataFrame(rows)


def load_weo(weo_csv: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weo = pd.read_csv(weo_csv, low_memory=False)
    weo["iso3"] = weo["SERIES_CODE"].astype(str).str.split(".").str[0]
    name_col = "SERIES_NAME" if "SERIES_NAME" in weo.columns else "INDICATOR"
    norm = weo[name_col].astype(str).str.casefold().str.strip()

    def extract(mask: pd.Series) -> pd.DataFrame:
        sub = weo[mask]
        yr_cols = [c for c in weo.columns if str(c).strip().isdigit()]
        long = sub.melt(id_vars=["iso3"], value_vars=yr_cols, var_name="year", value_name="v")
        long["year"] = long["year"].astype(int)
        long["v"] = pd.to_numeric(long["v"], errors="coerce")
        return long.dropna(subset=["v"])

    pc = extract(norm.str.contains("constant prices, per capita, purchasing power parity", regex=False)
                 & norm.str.contains("icp benchmark 2021", regex=False))
    growth = extract(norm == GROWTH_NAME)
    bound = (weo[["iso3", "LATEST_ACTUAL_ANNUAL_DATA"]].dropna()
             .groupby("iso3")["LATEST_ACTUAL_ANNUAL_DATA"].max().reset_index())
    print(f"WEO: 不变价人均PPP {pc.iso3.nunique()} 国; 实际增长率 {growth.iso3.nunique()} 国")
    return pc, growth, bound


def scale_factor(iso3: str, pc: pd.DataFrame, growth: pd.DataFrame) -> tuple[dict[int, float], str]:
    """返回 {year: 相对 2022 的缩放系数} 与方式标签。"""
    tail = [y for y in YEARS if y > 2022]
    s = pc[pc.iso3 == iso3].set_index("year")["v"]
    if 2022 in s.index and all(y in s.index for y in tail):
        return {y: float(s[y] / s[2022]) for y in tail}, "weo_pc_constant_ppp"
    g = growth[growth.iso3 == iso3].set_index("year")["v"]
    if all(y in g.index for y in tail):
        f, out = 1.0, {}
        for y in tail:
            f *= 1.0 + float(g[y]) / 100.0
            out[y] = f
        return out, "weo_real_growth"
    return {y: 1.0 for y in tail}, "constant_2022"


def main() -> None:
    cfg = load_config()
    cities = load_cities(cfg)
    catchments = gpd.read_file(cfg["paths"]["city_catchments_gpkg"])

    raster = zonal_mean_by_year(cities, catchments, cfg["paths"]["kummu_gdp_tif"])
    pc, growth, bound = load_weo(cfg["paths"]["weo_csv"])

    base = raster[raster.year == 2022].set_index("node_id")["AC1_raster"]

    # 国家级不变价人均 PPP 序列（WEO），供栅格取不到值的微型属地作国家级代理基准。
    # 属地按主权国（TERRITORY_PARENT）取值；本国有 WEO 但栅格漏取的（如 Batanes/PHL）取本国。
    from common import TERRITORY_PARENT
    pc_level = pc.pivot_table(index="iso3", columns="year", values="v", aggfunc="first")

    def national_series(iso: str) -> tuple[dict, str] | None:
        for src in (iso, TERRITORY_PARENT.get(iso)):
            if src and src in pc_level.index and 2022 in pc_level.columns \
                    and np.isfinite(pc_level.loc[src, 2022]):
                row = pc_level.loc[src]
                return ({y: (float(row[y]) if y in pc_level.columns
                             and np.isfinite(row[y]) else float(row[2022]))
                         for y in YEARS}, f"national_proxy_{src}")
        return None

    rows = []
    factors_cache: dict[str, tuple[dict, str]] = {}
    n_proxy = 0
    for _, c in cities.iterrows():
        nid, iso = c["node_id"], c["iso3"]
        if iso not in factors_cache:
            factors_cache[iso] = scale_factor(iso, pc, growth)
        fac, method = factors_cache[iso]
        hist = raster[raster.node_id == nid].set_index("year")["AC1_raster"]
        b = base.get(nid, np.nan)
        nat = national_series(iso) if not np.isfinite(b) else None
        if nat is not None:
            n_proxy += 1
        for y in YEARS:
            if nat is not None:
                # 栅格基准缺失 → 整段用国家级不变价人均 GDP（文档化代理，写入 SI 局限）
                v, m = nat[0][y], nat[1]
            elif y <= 2022:
                v, m = hist.get(y, np.nan), "kummu_raster"
            else:
                v = b * fac[y] if np.isfinite(b) else np.nan
                m = method
            rows.append({"node_id": nid, "iso3": iso, "year": y, "AC1": v, "scale_method": m})
    print(f"  国家级代理基准节点: {n_proxy}")

    out = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(os.path.join(OUT, "ac1_gdp.csv"), index=False, encoding="utf-8-sig")
    bound.to_csv(os.path.join(OUT, "ac1_weo_boundary.csv"), index=False)
    n_nan = out["AC1"].isna().sum()
    print(f"AC1 完成: {out.node_id.nunique()} 节点 × {len(YEARS)} 年; NaN {n_nan} "
          f"({n_nan / len(out):.2%})")
    print("缩放方式分布:", out[out.year == 2023]["scale_method"].value_counts().to_dict())


if __name__ == "__main__":
    main()
