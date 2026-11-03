# -*- coding: utf-8 -*-
"""合并全部协变量 → 城市×年度面板 city_covariates_2019_2028.csv + 数据字典。

国家级指标（AC2/AC4/AC5）按 iso3 连接（同国共用）；
节点级指标（AC1/AC3/S1/S2/密度）按 node_id 连接。
QC：逐列非空率、值域检查；缺失清单写 covariate_gaps.csv。
"""
from __future__ import annotations

import os

import pandas as pd

from common import OUT, TERRITORY_PARENT, YEARS, load_config, load_cities


def main() -> None:
    cfg = load_config()
    cities = load_cities(cfg)
    base = cities[["node_id", "city", "country", "iso3", "area_km2", "n_ports",
                   "rep_lon", "rep_lat"]].copy()
    # 国家级指标连接键：属地缺数据时按主权国代理（proxy 列记录）
    base["iso3_join"] = base["iso3"]
    panel = base.merge(pd.DataFrame({"year": YEARS}), how="cross")

    def add(fname: str, on: list[str], cols: list[str]):
        nonlocal panel
        df = pd.read_csv(os.path.join(OUT, fname), encoding="utf-8-sig")
        if "iso3" in on:  # 国家级：先按本 iso3 连接，缺失行回退主权国
            df = df.rename(columns={"iso3": "iso3_join"})
            on = ["iso3_join" if c == "iso3" else c for c in on]
            panel = panel.merge(df[on + cols], on=on, how="left")
            key = cols[0]
            miss = panel[key].isna() & panel["iso3_join"].isin(TERRITORY_PARENT)
            if miss.any():
                fb = panel.loc[miss, ["iso3_join", "year"]].copy()
                fb["iso3_join"] = fb["iso3_join"].map(TERRITORY_PARENT)
                fb = fb.merge(df[on + cols], on=on, how="left")
                for c in cols:
                    panel.loc[miss, c] = fb[c].values
                panel.loc[miss, key + "_proxy"] = "parent_country"
        else:
            panel = panel.merge(df[on + cols], on=on, how="left")

    add("ac1_gdp.csv", ["node_id", "year"], ["AC1"])
    add("ac2_medical.csv", ["iso3", "year"], ["AC2", "doctor_per_10k", "bed_per_10k"])
    add("ac3_regulation.csv", ["node_id", "year"], ["AC3", "seca", "neca"])
    add("ac4_wgi.csv", ["iso3", "year"], ["AC4"])
    add("ac5_pv.csv", ["iso3", "year"], ["AC5"])
    add("pop_s1_s2.csv", ["node_id", "year"],
        ["S1_pop", "S2_vuln_share", "pop_density_km2", "pop_flag"])

    panel = panel.drop(columns=["iso3_join"])
    # 超大集水区标记：area_km2 > 方案 area_max 80,000（多为 GADM 无更细层级的
    # irreducible 节点，会过度归集过境排放）；analysis 阶段可据此一键排除。
    panel["oversized_flag"] = (panel["area_km2"] > 80000).astype(int)
    out_csv = os.path.join(OUT, "city_covariates_2019_2028.csv")
    panel.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # ---- QC ----
    ind_cols = ["AC1", "AC2", "AC3", "AC4", "AC5", "S1_pop", "S2_vuln_share",
                "pop_density_km2"]
    print(f"面板: {panel.node_id.nunique()} 节点 × {len(YEARS)} 年 = {len(panel)} 行")
    print("\n非空率:")
    for c in ind_cols:
        print(f"  {c}: {panel[c].notna().mean():.2%}")
    checks = {
        "AC1": (panel.AC1.dropna() > 0).all(),
        "AC4∈[-3,3]": panel.AC4.dropna().between(-3, 3).all(),
        "AC5∈[-3,3]": panel.AC5.dropna().between(-3, 3).all(),
        "AC3∈{0,1}": panel.AC3.dropna().isin([0, 1]).all(),
        "S2∈[0,1]": panel.S2_vuln_share.dropna().between(0, 1).all(),
        "S1≥0": (panel.S1_pop.dropna() >= 0).all(),
    }
    print("\n值域检查:", {k: bool(v) for k, v in checks.items()})

    gaps = panel[panel[ind_cols].isna().any(axis=1)]
    if len(gaps):
        gap_summary = (gaps.groupby(["node_id", "country"])[ind_cols]
                       .apply(lambda g: ",".join(c for c in ind_cols if g[c].isna().any()))
                       .rename("missing_cols").reset_index())
    else:
        gap_summary = pd.DataFrame(columns=["node_id", "country", "missing_cols"])
    gap_summary.to_csv(os.path.join(OUT, "covariate_gaps.csv"),
                       index=False, encoding="utf-8-sig")
    print(f"\n有缺失的节点: {gaps.node_id.nunique()} / {panel.node_id.nunique()} "
          f"(明细 covariate_gaps.csv)")

    # ---- 数据字典 ----
    dd = """# 沿海城市协变量面板数据字典 (city_covariates_2019_2028.csv)

| 列 | 含义 | 来源/口径 |
|---|---|---|
| node_id | 城市节点标识 {NAME_0}__{City} | Port2City Agent (coastal_cities.gpkg) |
| city / country / iso3 | 城市名 / 国家名(GADM NAME_0) / ISO3(GID_0) | 同上 |
| area_km2 | 城市多边形面积 (EPSG:6933) | 同上 |
| n_ports | 节点内港口数 | 同上 |
| rep_lon / rep_lat | 代表点位（节点内港口坐标算术平均，仅用于可视化与距离） | 同上 |
| year | 2019-2028 | - |
| AC1 | 人均 GDP (2017 international USD PPP, 不变价)。2019-2022 Kummu 栅格城市多边形内均值；2023-2028 按 IMF WEO 不变价人均 GDP(PPP, ICP2021) 比值缩放，缺失国退用实际增长率累乘 | Kummu et al. 2025 + IMF WEO；详见 ac1_gdp.csv 的 scale_method |
| AC2 | 医疗资源指数 = 医生密度、床位密度各 min-max 后等权平均（国家级，同国共用） | WHO GHO HWF_0001 + WHS6_102；台湾为卫福部口径补录；插值/ARIMA 外推方式见 ac2_medical.csv |
| doctor_per_10k / bed_per_10k | 每万人医生数 / 床位数 | 同上 |
| AC3 | 监管强度（年末处于 SECA 或 NECA 内 = 1） | IMO MARPOL Annex VI 区划（30km 集水区相交判定）；月度版见 seca_neca_monthly.csv |
| seca / neca | 分项虚拟变量（年末状态） | 同上 |
| AC4 | 治理质量 = WGI 五维 (VA/GE/RQ/RL/CC) 等权平均，剔除 PV（国家级）；2024+ 末值延展 | World Bank WGI (DS57, 含台湾) |
| AC5 | 政治稳定性 PV.EST，正向（越高越稳定）；2024+ 末值延展 | World Bank WGI (DS57, 含台湾) |
| S1_pop | 城市多边形内总人口 | WorldPop 1km R2025A（注意为 1km 版本而非 100m，写入 SI） |
| S2_vuln_share | 脆弱人群比例 = (0-4 岁 + ≥65 岁)/总人口；t_01 2019 用 2020 回填 | WorldPop 年龄结构 1km |
| pop_density_km2 | 人口密度 = S1/area_km2（E2 人口加权暴露的人口端） | 同上 |

注：E1/E2 排放端/E3 停泊占比来自排放网络月度输出聚合（Part B），不在本面板。
缺失值策略：国家级指标对无数据微型属地留空（见 covariate_gaps.csv），SPRI 计算时另行处理。
"""
    with open(os.path.join(OUT, "DATA_DICTIONARY.md"), "w", encoding="utf-8") as f:
        f.write(dd)
    print(f"\n输出: {out_csv}")


if __name__ == "__main__":
    main()
