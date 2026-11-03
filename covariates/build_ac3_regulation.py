# -*- coding: utf-8 -*-
"""AC3 监管强度：SECA/NECA 判定（年度 AC3 + 城市×月度虚拟变量供 GNN）。

几何：城市 30km 集水区与 ECA 多边形相交即视为受管控（船舶在该城市近岸水域受限）。
区划与生效时间登记表（REGISTRY）基于 MARPOL Annex VI 与各国法规：
  SECA: 波罗的海(2006)/北海(2007)/北美(2012, 含夏威夷)/美加勒比(2014)/
        中国 DECA(2019-01, 0.5%)/韩国(2020-09 主要港区→2022-01 全水域)/地中海(2025-05)
  NECA: 北美(2016)/波罗的海+北海(2021-01)
已批准未来扩张（挪威海/加拿大北极 2026-03、东北大西洋 ~2027）暂无矢量边界，
未纳入（见输出 README 注记；获得边界后在 REGISTRY 续增即可）。

输出:
  outputs/ac3_regulation.csv:        node_id, year, AC3, seca, neca（年末状态）
  outputs/seca_neca_monthly.csv:     node_id, month(YYYY-MM), seca, neca（GNN 特征）
"""
from __future__ import annotations

import os

import geopandas as gpd
import pandas as pd

from common import OUT, YEARS, load_config

# (shapefile, 类型, 生效年月)
REGISTRY = [
    ("Baltic_sea_ECA.shp", "seca", "2006-05"),
    ("North_sea_ECA.shp", "seca", "2007-11"),
    ("North_America_ECA_All.shp", "seca", "2012-08"),
    ("Hawaii_ECA.shp", "seca", "2012-08"),
    ("Caribbean_ECA.shp", "seca", "2014-01"),
    ("China_ECA_All.shp", "seca", "2019-01"),
    ("South_Korea_ECA_All.shp", "seca", "2020-09"),
    ("Mediterranean.shp", "seca", "2025-05"),
    ("North_America_ECA_All.shp", "neca", "2016-01"),
    ("Hawaii_ECA.shp", "neca", "2016-01"),
    ("Baltic_sea_ECA.shp", "neca", "2021-01"),
    ("North_sea_ECA.shp", "neca", "2021-01"),
]


def main() -> None:
    cfg = load_config()
    eca_dir = cfg["paths"]["eca_shp_dir"]
    catch = gpd.read_file(cfg["paths"]["city_catchments_gpkg"])
    sindex = catch.sindex

    # 每个城市的 (类型 → 最早生效年月)
    eff: dict[str, dict[str, str]] = {nid: {} for nid in catch["node_id"]}
    for shp, kind, since in REGISTRY:
        path = os.path.join(eca_dir, shp)
        if not os.path.exists(path):
            print(f"警告: 缺少 {shp}，跳过")
            continue
        eca = gpd.read_file(path).to_crs("EPSG:4326")
        geom = eca.union_all() if hasattr(eca, "union_all") else eca.unary_union
        hit_idx = sindex.query(geom, predicate="intersects")
        n = 0
        for i in hit_idx:
            nid = catch["node_id"].iloc[i]
            cur = eff[nid].get(kind)
            if cur is None or since < cur:
                eff[nid][kind] = since
            n += 1
        print(f"{shp} [{kind} @{since}]: 命中 {n} 城市")

    months = pd.period_range("2019-01", "2028-12", freq="M").strftime("%Y-%m")
    monthly_rows, yearly_rows = [], []
    for nid in catch["node_id"]:
        s_since = eff[nid].get("seca")
        n_since = eff[nid].get("neca")
        for m in months:
            monthly_rows.append({"node_id": nid, "month": m,
                                 "seca": int(s_since is not None and m >= s_since),
                                 "neca": int(n_since is not None and m >= n_since)})
        for y in YEARS:
            ym = f"{y}-12"  # 年末状态
            s = int(s_since is not None and ym >= s_since)
            n = int(n_since is not None and ym >= n_since)
            yearly_rows.append({"node_id": nid, "year": y, "AC3": max(s, n),
                                "seca": s, "neca": n})

    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(monthly_rows).to_csv(os.path.join(OUT, "seca_neca_monthly.csv"),
                                      index=False, encoding="utf-8-sig")
    ydf = pd.DataFrame(yearly_rows)
    ydf.to_csv(os.path.join(OUT, "ac3_regulation.csv"), index=False, encoding="utf-8-sig")
    print("2025 年 SECA 城市数:", ydf[(ydf.year == 2025)]["seca"].sum(),
          "; NECA:", ydf[(ydf.year == 2025)]["neca"].sum(),
          "; 2019:", ydf[(ydf.year == 2019)]["seca"].sum())


if __name__ == "__main__":
    main()
