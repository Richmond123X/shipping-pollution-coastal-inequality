# -*- coding: utf-8 -*-
"""研究 4 步骤 1：高风险区域筛选 + 12 海里政策区几何生成（方案 v6 7.1 节）。

流程：
  1. 读研究 3 SPRI 得分表(node_id, SPRI)，取降序前 top_pct(默认 20%)为高风险节点；
  2. 节点经协变量面板(node_id → country/iso3)聚合到国家，按国内高风险节点数
     排序取前 top_n_countries 国为主分析实施单元(政策以国家为颁布主体)；
     config policy.blocs 可额外定义区域经济体联合单元(iso3 列表)；
  3. 每个单元：GADM(World.shp, GID_0=iso3) 陆地面 dissolve → 局地等距方位投影
     (aeqd) 简化+向海缓冲 12 海里(22.224 km) → 扣除本国及邻国陆地 → 纯海域带；
  4. 输出 candidate_policy_zones.gpkg(zone_id, iso3s, n_trigger_nodes, geometry)
     + policy_trigger_nodes.csv(zone_id, node_id, country, iso3, SPRI, rank)。

调试：--countries CHN,VNM 可跳过 SPRI 直接指定单元(研究 3 未产出时用)。

用法：
  python policy/build_policy_zones.py                     # 全部按 config.yaml
  python policy/build_policy_zones.py --countries CHN     # 调试
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import make_valid
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "emission_network"))

NM_TO_M = 1852.0


def _load_cfg():
    # 共享加载器：含跨机器路径自动迁移(用户名/盘符不同的电脑无需改 config.yaml)
    from config_loader import load_config
    return load_config()


def _node_country_map(covariates_csv: str) -> pd.DataFrame:
    """协变量面板 → 每个节点一行 (node_id, country, iso3)。"""
    df = pd.read_csv(covariates_csv, usecols=["node_id", "country", "iso3"])
    return df.drop_duplicates(subset="node_id").reset_index(drop=True)


def screen_high_risk(spri_csv: str, cov_csv: str, top_pct: float,
                     top_n: int, blocs: dict) -> tuple[pd.DataFrame, dict]:
    """SPRI 前 top_pct 节点 → 国家聚合 → 前 top_n 国 + blocs 联合单元。

    返回 (trigger_nodes_df, {zone_id: [iso3, ...]})。
    """
    spri = pd.read_csv(spri_csv)
    need = {"node_id", "SPRI"}
    if not need <= set(spri.columns):
        raise SystemExit(f"SPRI 表缺列 {need - set(spri.columns)}: {spri_csv}")
    spri = spri.dropna(subset=["SPRI"]).sort_values("SPRI", ascending=False)
    n_top = max(1, int(round(len(spri) * top_pct)))
    top = spri.head(n_top).copy()
    top["rank"] = np.arange(1, len(top) + 1)

    nc = _node_country_map(cov_csv)
    top = top.merge(nc, on="node_id", how="left")
    miss = top["iso3"].isna().sum()
    if miss:
        print(f"  ⚠ {miss} 个高风险节点未能映射国家(不参与国家计数): "
              f"{top.loc[top['iso3'].isna(), 'node_id'].head(5).tolist()} ...")
        top = top.dropna(subset=["iso3"])

    by_ctry = (top.groupby(["iso3", "country"], as_index=False)
               .agg(n_nodes=("node_id", "count"), spri_sum=("SPRI", "sum"))
               .sort_values(["n_nodes", "spri_sum"], ascending=False))
    print("  高风险节点国家分布(前 15):")
    print(by_ctry.head(15).to_string(index=False))
    main_ctrys = by_ctry.head(top_n)

    zones: dict[str, list[str]] = {
        str(r.iso3): [str(r.iso3)] for r in main_ctrys.itertuples()
    }
    for bloc_name, iso3s in (blocs or {}).items():
        zones[str(bloc_name)] = [str(x) for x in iso3s]

    top["zone_id"] = top["iso3"].where(top["iso3"].isin(zones), other=pd.NA)
    return top, zones


def _read_land(world_shp: str, iso3s: list[str] = None, bbox=None) -> gpd.GeoDataFrame:
    """按 GID_0 或 bbox 过滤读取 GADM 陆地面(只取几何,避免整世界入内存)。"""
    kw = {"columns": ["GID_0"]}
    if iso3s:
        quoted = ",".join(f"'{c}'" for c in iso3s)
        kw["where"] = f"GID_0 IN ({quoted})"
    if bbox is not None:
        kw["bbox"] = tuple(bbox)
    gdf = gpd.read_file(world_shp, **kw)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


def build_zone_geometry(world_shp: str, iso3s: list[str], nm: float,
                        simplify_m: float):
    """单个实施单元的 12 海里纯海域带(EPSG:4326)。"""
    land = _read_land(world_shp, iso3s=iso3s)
    if len(land) == 0:
        print(f"  ⚠ GADM 无 {iso3s} 的面, 跳过")
        return None
    b = land.total_bounds
    if b[2] - b[0] > 300:
        print(f"  ⚠ {iso3s} 跨反子午线(经度范围 {b[0]:.0f}~{b[2]:.0f}), "
              f"aeqd 缓冲不可靠, 跳过(需手工处理)")
        return None
    lon0, lat0 = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    aeqd = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"

    own = land.to_crs(aeqd).geometry.simplify(simplify_m).buffer(0)
    own_u = unary_union(own.values)
    buffered = own_u.buffer(nm * NM_TO_M)

    # 邻国陆地(bbox 外扩缓冲距离+0.5°)一并扣除 → 纯海域、不越邻国领海基线侧陆地
    pad = nm * NM_TO_M / 111000.0 + 0.5
    nb_bbox = (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)
    nb = _read_land(world_shp, bbox=nb_bbox)
    nb_u = unary_union(nb.to_crs(aeqd).geometry.simplify(simplify_m).buffer(0).values)
    band = buffered.difference(nb_u)

    out = gpd.GeoSeries([band], crs=aeqd).to_crs("EPSG:4326").iloc[0]

    # Simplified coastlines can leave collapsed polygon rings (too few points),
    # which makes the GeoPackage unusable for reliable trajectory intersection.
    # Repair after the final reprojection and retain polygonal components only.
    out = make_valid(out, method="structure", keep_collapsed=False)
    if not isinstance(out, (Polygon, MultiPolygon)):
        polygon_parts = []
        for part in getattr(out, "geoms", []):
            if isinstance(part, Polygon):
                polygon_parts.append(part)
            elif isinstance(part, MultiPolygon):
                polygon_parts.extend(part.geoms)
        out = unary_union(polygon_parts) if polygon_parts else None
    return out


def main() -> None:
    cfg = _load_cfg()
    pol = cfg.get("policy") or {}
    paths = cfg["paths"]

    ap = argparse.ArgumentParser(description="研究4: 高风险筛选 + 12海里政策区生成")
    ap.add_argument("--spri-csv", default=str(pol.get("spri_csv", "")))
    ap.add_argument("--covariates-csv",
                    default=os.path.join(paths["covariates_dir"],
                                         "city_covariates_2019_2028.csv"))
    ap.add_argument("--world-shp", default=str(paths["world_shp"]))
    ap.add_argument("--top-pct", type=float, default=float(pol.get("top_pct", 0.20)))
    ap.add_argument("--top-n", type=int, default=int(pol.get("top_n_countries", 10)))
    ap.add_argument("--nm", type=float, default=float(pol.get("territorial_nm", 12.0)))
    ap.add_argument("--simplify-m", type=float,
                    default=float(pol.get("simplify_m", 250.0)))
    ap.add_argument("--out-gpkg", default=str(pol.get(
        "zones_gpkg", os.path.join(HERE, "outputs", "candidate_policy_zones.gpkg"))))
    ap.add_argument("--out-nodes-csv", default=str(pol.get(
        "trigger_nodes_csv", os.path.join(HERE, "outputs", "policy_trigger_nodes.csv"))))
    ap.add_argument("--countries", default="",
                    help="调试: 逗号分隔 iso3, 跳过 SPRI 筛选直接指定实施单元")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_gpkg)), exist_ok=True)

    if args.countries.strip():
        zones = {c.strip().upper(): [c.strip().upper()]
                 for c in args.countries.split(",") if c.strip()}
        trigger = pd.DataFrame(columns=["zone_id", "node_id", "country",
                                        "iso3", "SPRI", "rank"])
        print(f"[debug] 直接指定实施单元: {list(zones)}")
    else:
        if not os.path.isfile(args.spri_csv):
            raise SystemExit(
                f"SPRI 表不存在: {args.spri_csv}\n"
                "研究 3 尚未产出时可用 --countries CHN,VNM 调试几何生成。")
        print(f"[1/3] 高风险筛选: SPRI 前 {args.top_pct:.0%}, 前 {args.top_n} 国")
        trigger, zones = screen_high_risk(
            args.spri_csv, args.covariates_csv, args.top_pct, args.top_n,
            pol.get("blocs") or {})
        trigger = (trigger.dropna(subset=["zone_id"])
                   [["zone_id", "node_id", "country", "iso3", "SPRI", "rank"]])

    print(f"[2/3] 生成 12 海里政策区几何({args.nm} nmi = {args.nm*NM_TO_M/1000:.3f} km)")
    rows = []
    for zid, iso3s in zones.items():
        print(f"  - {zid} ({','.join(iso3s)}) ...", flush=True)
        geom = build_zone_geometry(args.world_shp, iso3s, args.nm, args.simplify_m)
        if geom is None or geom.is_empty:
            print(f"    ⚠ {zid} 几何为空, 跳过")
            continue
        n_trig = int((trigger["zone_id"] == zid).sum()) if len(trigger) else 0
        rows.append({"zone_id": zid, "iso3s": ",".join(iso3s),
                     "n_trigger_nodes": n_trig, "geometry": geom})
        area_km2 = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs("EPSG:6933").area.iloc[0] / 1e6
        print(f"    带面积 {area_km2:,.0f} km²  触发节点 {n_trig}")

    if not rows:
        raise SystemExit("无任何有效政策区")
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    valid_zone_ids = set(gdf["zone_id"].astype(str))
    trigger = trigger.loc[trigger["zone_id"].astype(str).isin(valid_zone_ids)].copy()
    gdf.to_file(args.out_gpkg, driver="GPKG")
    trigger.to_csv(args.out_nodes_csv, index=False, encoding="utf-8-sig")
    print(f"[3/3] 写出 {args.out_gpkg} ({len(gdf)} 单元) 与 {args.out_nodes_csv} "
          f"({len(trigger)} 触发节点)")


if __name__ == "__main__":
    main()
