# -*- coding: utf-8 -*-
"""带宽校准后的定点协变量补算。

只对 band_changed_nodes.json 列出的几何变化节点（合并组合节点 + 下钻新建节点）
重算栅格指标（S1/S2/密度、AC1），并把旧 csv 中已不存在的节点行清掉；
其余 1300+ 节点的数值原样保留，不碰全量栅格。
"""
from __future__ import annotations

import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import geopandas as gpd
from rasterstats import zonal_stats

from common import OUT, YEARS, load_config, load_cities
from build_pop_s1_s2 import AGE_PREFIXES
from build_ac1_gdp import (RASTER_YEARS, KUMMU_FIRST_YEAR, load_weo, scale_factor)

HERE = os.path.dirname(os.path.abspath(__file__))
CHANGED_JSON = os.path.join(os.path.dirname(HERE), "port2city", "outputs",
                            "band_changed_nodes.json")

_w_geoms = None
_w_jobs_meta = None


def _init(geoms_wkb):
    global _w_geoms
    from shapely import wkb
    _w_geoms = [wkb.loads(g) for g in geoms_wkb]


def _zonal_one(job):
    kind, key, tif, band = job
    zs = zonal_stats(_w_geoms, tif, stats="sum" if kind == "sum" else "mean",
                     band=band, all_touched=(kind == "mean"), nodata=np.nan if kind == "mean" else -99999)
    stat = "sum" if kind == "sum" else "mean"
    return key, np.array([z[stat] if z[stat] is not None else np.nan for z in zs])


def main() -> None:
    cfg = load_config()
    cities = load_cities(cfg)
    with open(CHANGED_JSON, encoding="utf-8") as f:
        changed = [n for n in json.load(f) if n in set(cities.node_id)]
    sub = cities[cities.node_id.isin(changed)].reset_index(drop=True)
    print(f"定点补算节点: {len(sub)} / 全集 {len(cities)}")
    valid_ids = set(cities.node_id)

    # ---- 构建 zonal 任务 ----
    jobs = []
    for y in YEARS:
        jobs.append(("sum", f"pop_{y}",
                     os.path.join(cfg["paths"]["pop_dir"],
                                  f"global_pop_{y}_CN_1km_R2025A_UA_v1.tif"), 1))
        for pref in AGE_PREFIXES:
            sy = 2020 if (pref == "t_01" and y == 2019) else y
            key = f"{pref}_{sy}"
            if not any(j[1] == key for j in jobs):
                jobs.append(("sum", key,
                             os.path.join(cfg["paths"]["pop_age_dir"],
                                          f"global_{pref}_{sy}_CN_1km_R2025A_UA_v1.tif"), 1))
    for y in RASTER_YEARS:
        jobs.append(("mean", f"gdp_{y}", cfg["paths"]["kummu_gdp_tif"],
                     y - KUMMU_FIRST_YEAR + 1))

    from shapely import wkb
    geoms_wkb = [wkb.dumps(g) for g in sub.geometry]
    results = {}
    with Pool(8, initializer=_init, initargs=(geoms_wkb,)) as pool:
        for i, (key, vals) in enumerate(pool.imap_unordered(_zonal_one, jobs)):
            results[key] = vals
            print(f"  {i + 1}/{len(jobs)} {key}", flush=True)

    # ---- 补 pop_s1_s2.csv ----
    pop = pd.read_csv(os.path.join(OUT, "pop_s1_s2.csv"), encoding="utf-8-sig")
    pop = pop[pop.node_id.isin(valid_ids) & ~pop.node_id.isin(changed)]
    rows = []
    for y in YEARS:
        s1 = results[f"pop_{y}"]
        vuln = np.zeros(len(sub))
        for pref in AGE_PREFIXES:
            sy = 2020 if (pref == "t_01" and y == 2019) else y
            vuln += np.nan_to_num(results[f"{pref}_{sy}"], nan=0.0)
        share = np.divide(vuln, s1, out=np.full(len(sub), np.nan),
                          where=np.nan_to_num(s1, nan=0.0) > 0)
        for i in range(len(sub)):
            rows.append({"node_id": sub.node_id.iloc[i], "year": y,
                         "S1_pop": s1[i], "S2_vuln_share": share[i],
                         "pop_density_km2": s1[i] / sub.area_km2.iloc[i]})
    pop = pd.concat([pop, pd.DataFrame(rows)], ignore_index=True)
    pop.to_csv(os.path.join(OUT, "pop_s1_s2.csv"), index=False, encoding="utf-8-sig")
    print(f"pop_s1_s2.csv: {pop.node_id.nunique()} 节点")

    # ---- 补 ac1_gdp.csv ----
    pc, growth, _ = load_weo(cfg["paths"]["weo_csv"])
    ac1 = pd.read_csv(os.path.join(OUT, "ac1_gdp.csv"), encoding="utf-8-sig")
    ac1 = ac1[ac1.node_id.isin(valid_ids) & ~ac1.node_id.isin(changed)]
    rows = []
    fac_cache = {}
    for i in range(len(sub)):
        nid, iso = sub.node_id.iloc[i], sub.iso3.iloc[i]
        if iso not in fac_cache:
            fac_cache[iso] = scale_factor(iso, pc, growth)
        fac, method = fac_cache[iso]
        base = results["gdp_2022"][i]
        for y in YEARS:
            if y <= 2022:
                v, m = results[f"gdp_{y}"][i], "kummu_raster"
            else:
                v = base * fac[y] if np.isfinite(base) else np.nan
                m = method
            rows.append({"node_id": nid, "iso3": iso, "year": y, "AC1": v,
                         "scale_method": m})
    ac1 = pd.concat([ac1, pd.DataFrame(rows)], ignore_index=True)
    ac1.to_csv(os.path.join(OUT, "ac1_gdp.csv"), index=False, encoding="utf-8-sig")
    print(f"ac1_gdp.csv: {ac1.node_id.nunique()} 节点")


if __name__ == "__main__":
    main()
