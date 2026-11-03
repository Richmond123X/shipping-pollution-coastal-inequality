# -*- coding: utf-8 -*-
"""排放集水区构建（研究方案 3.6.5）。

集水区 = 城市多边形 ∪ 自多边形边界向外的等距缓冲带（主规格 30 km）。
实现：EPSG:6933 等积投影下 buffer(width_m)，再转回 EPSG:4326；
不做海陆裁剪（落在邻市/邻国陆地上的部分无 AIS 排放记录，实际无效）。

用法：
  python build_catchments.py            # 30 km 主规格
  python build_catchments.py 50 100    # 敏感性规格（SI）
"""
from __future__ import annotations

import os
import sys

import geopandas as gpd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_config() -> dict:
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_catchments(width_km: float) -> str:
    cfg = load_config()
    eq = cfg["port2city"]["equal_area_crs"]
    cities = gpd.read_file(cfg["paths"]["coastal_cities_gpkg"])
    print(f"读取 {len(cities)} 个城市节点；构建 {width_km:.0f} km 集水区 ...")

    # 跨/近 180° 经线节点（斐济/阿留申/楚科奇/图瓦卢等）在全局 EPSG:6933 缓冲后过
    # 日界线会自相交 → 几何无效。这些节点改用「节点中心 AEQD 方位等距投影」缓冲（无日界线缝）。
    from shapely import make_valid
    b = cities.geometry.bounds
    antimeridian = ((b.maxx - b.minx) > 180) | ((b.maxx > 179) & (b.minx < -179)) \
        | (b.maxx > 179.5) | (b.minx < -179.5)
    normal_idx = cities.index[~antimeridian]
    anti_idx = cities.index[antimeridian]
    print(f"  常规缓冲 {len(normal_idx)} 节点；跨日界线 AEQD 缓冲 {len(anti_idx)} 节点")

    geoms = {}
    # --- 常规节点：批量 EPSG:6933 ---
    proj = cities.loc[normal_idx].to_crs(eq)
    simplified = proj.geometry.buffer(0).simplify(1000.0)
    buffered = simplified.buffer(width_km * 1000.0, resolution=4)
    merged = buffered.union(proj.geometry.buffer(0)).to_crs("EPSG:4326")
    for idx, g in zip(normal_idx, merged.values):
        geoms[idx] = make_valid(g)

    # --- 跨日界线节点：逐个节点中心 AEQD ---
    for idx in anti_idx:
        c = cities.loc[idx]
        aeqd = f"+proj=aeqd +lat_0={c.rep_lat} +lon_0={c.rep_lon} +datum=WGS84 +units=m"
        g4326 = gpd.GeoSeries([c.geometry], crs="EPSG:4326")
        gp = g4326.to_crs(aeqd)
        bp = gp.buffer(0).simplify(1000.0).buffer(width_km * 1000.0, resolution=4)
        out_g = bp.union(gp.buffer(0)).to_crs("EPSG:4326").iloc[0]
        geoms[idx] = make_valid(out_g)

    out = cities.copy()
    out["geometry"] = [geoms[i] for i in cities.index]
    out["buffer_km"] = width_km
    n_invalid = int((~out.geometry.is_valid).sum())
    print(f"  无效几何: {n_invalid}")

    if abs(width_km - cfg["port2city"]["catchment_buffer_km"]) < 1e-9:
        path = cfg["paths"]["city_catchments_gpkg"]
    else:
        path = os.path.join(HERE, "outputs", f"city_catchments_{width_km:.0f}km.gpkg")
    out.to_file(path, layer="catchments", driver="GPKG")
    print(f"已写出: {path}")
    return path


if __name__ == "__main__":
    widths = [float(a) for a in sys.argv[1:]] or [load_config()["port2city"]["catchment_buffer_km"]]
    for w in widths:
        build_catchments(w)
