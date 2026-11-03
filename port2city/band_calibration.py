# -*- coding: utf-8 -*-
"""面积带宽校准（finalize 之后的定点修复阶段）：只处理出带节点，保证节点互斥。

不重跑 L1/L2/L3；读取现有 gpkg + mapping，定点修改。

四条确定性规则（与用户确认的方案一致）：
  A  整国单元过小（岛国/城市国家/属地，adm_level 含 0）
       → 保留 + band_exception='island_state'（方案 3.6.1 设计内例外）
  B  次级单元过小（<500 km²）
       → 与同国最近节点（面间距 ≤30 km，与集水区宽度一致）合并为组合节点；
         无邻近对象 → 保留 + band_exception='urban_core'
  C1 过大（>80,000 km²）且其港口可下钻到同国更细单元（距港口点 ≤25 km）
       → 逐港改派（带内优先，score=|log10(A/1e4)| 最小）；清空的粗节点删除
  C2 过大且 GADM 无更细层级（沙特/格陵兰/利比亚 admin1 到底）或下钻无候选
       → 保留 + band_exception='gadm_resolution'
全局两两包含检查保证修复后节点互斥。

用法： python band_calibration.py plan   # 只输出 band_fix_plan.csv 供审阅
       python band_calibration.py apply  # 执行并改写 gpkg / shp / mapping
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import yaml  # noqa: E402
from l2_gis import load_world, ensure_row_areas, point_to_geom_km  # noqa: E402

OUT = os.path.join(HERE, "outputs")
AREA_CACHE = os.path.join(HERE, "world_row_areas.parquet")
PLAN_CSV = os.path.join(OUT, "band_fix_plan.csv")

AMIN, AMAX, TARGET = 500.0, 80000.0, 10000.0
MERGE_DIST_KM = 30.0
DRILL_TOL_KM = 25.0
EQ = "EPSG:6933"


def load_config() -> dict:
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def score(a: float) -> float:
    return abs(np.log10(max(a, 1e-9) / TARGET))


def drill_port(world, iso3: str, lon: float, lat: float, node_lv: int):
    """找港口点 ≤25km 的同国更细单元；带内优先 score 最小。

    返回 (level, gid, name, area_km2, dist_km) 或 None。
    """
    pt = Point(lon, lat)
    sub_idx = np.where(world["GID_0"].values == iso3)[0]
    if len(sub_idx) == 0:
        return None
    minx = world["minx"].values[sub_idx]; maxx = world["maxx"].values[sub_idx]
    miny = world["miny"].values[sub_idx]; maxy = world["maxy"].values[sub_idx]
    near = sub_idx[(minx < lon + 1.2) & (maxx > lon - 1.2)
                   & (miny < lat + 1.2) & (maxy > lat - 1.2)]
    cands: dict[str, tuple] = {}
    for ridx in near:
        for lv in range(node_lv + 1, 6):
            g = world[f"GID_{lv}"].values[ridx]
            nm = world[f"NAME_{lv}"].values[ridx]
            if not isinstance(g, str) or not g or not isinstance(nm, str) or not nm:
                break
            if g in cands:
                continue
            geom = world.geometry.values[ridx]
            if geom is None:
                continue
            d = point_to_geom_km(pt, geom)
            if d <= DRILL_TOL_KM:
                rows = np.where(world[f"GID_{lv}"].values == g)[0]
                a = float(world["area_km2"].values[rows].sum())
                cands[g] = (lv, g, str(nm), a, d)
    if not cands:
        return None
    in_band = [c for c in cands.values() if AMIN <= c[3] <= AMAX]
    pool = in_band if in_band else list(cands.values())
    return min(pool, key=lambda c: (score(c[3]), c[4]))


# ---------------------------------------------------------------------------
# 诊断与计划
# ---------------------------------------------------------------------------
def classify(cities: gpd.GeoDataFrame, world, mapping: pd.DataFrame) -> pd.DataFrame:
    proj = cities.to_crs(EQ)
    plan = []

    small = cities[cities.area_km2 < AMIN]
    big = cities[cities.area_km2 > AMAX]

    for idx, r in small.iterrows():
        if "0" in str(r.adm_level).split(";"):
            plan.append({"node_id": r.node_id, "area_km2": r.area_km2,
                         "category": "A", "action": "keep+flag island_state",
                         "detail": ""})
            continue
        me = proj.loc[idx].geometry
        # 合并目标排除过大节点（并入巨面只会放大口径问题）
        cand = proj[(proj.iso3 == r.iso3) & (proj.index != idx)
                    & (proj.area_km2 <= AMAX)]
        if len(cand):
            d = cand.geometry.distance(me) / 1000.0
            j = d.idxmin()
            if d[j] <= MERGE_DIST_KM:
                plan.append({"node_id": r.node_id, "area_km2": r.area_km2,
                             "category": "B", "action": "merge",
                             "detail": f"into={cities.loc[j].node_id}; gap_km={d[j]:.1f}"})
                continue
        plan.append({"node_id": r.node_id, "area_km2": r.area_km2,
                     "category": "B", "action": "keep+flag urban_core",
                     "detail": "no_neighbor<=30km"})

    for idx, r in big.iterrows():
        node_lv = max(int(x) for x in str(r.adm_level).split(";"))
        ports = mapping[mapping.node_id == r.node_id]
        drills = []
        for _, p in ports.iterrows():
            tgt = drill_port(world, r.iso3, float(p.Lon_port), float(p.Lat_port), node_lv)
            if tgt is not None:
                drills.append((p.Port_Name, *tgt))
        if drills:
            det = "; ".join(f"{pn}->{nm}(L{lv},{a:.0f}km2,d={d:.1f}km)"
                            for pn, lv, g, nm, a, d in drills)
            plan.append({"node_id": r.node_id, "area_km2": r.area_km2,
                         "category": "C1", "action": "drill_ports", "detail": det})
        else:
            plan.append({"node_id": r.node_id, "area_km2": r.area_km2,
                         "category": "C2", "action": "keep+flag gadm_resolution",
                         "detail": "no_finer_unit_within_25km"})
    return pd.DataFrame(plan)


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------
def apply_plan(cities: gpd.GeoDataFrame, mapping: pd.DataFrame,
               plan: pd.DataFrame, world):
    cities = cities.set_index("node_id", drop=False).copy()
    cities["band_exception"] = ""
    mapping = mapping.copy()
    changed_nodes: set[str] = set()

    for _, p in plan.iterrows():
        if p.action.startswith("keep+flag") and p.node_id in cities.index:
            cities.loc[p.node_id, "band_exception"] = p.action.split()[-1]

    # ---- C1 下钻 ----
    for _, p in plan[plan.action == "drill_ports"].iterrows():
        if p.node_id not in cities.index:
            continue
        node_lv = max(int(x) for x in str(cities.loc[p.node_id, "adm_level"]).split(";"))
        iso = cities.loc[p.node_id, "iso3"]
        country = cities.loc[p.node_id, "country"]
        ports = mapping[mapping.node_id == p.node_id]
        for _, prt in ports.iterrows():
            tgt = drill_port(world, iso, float(prt.Lon_port), float(prt.Lat_port), node_lv)
            if tgt is None:
                continue
            lv, gid, name, area, _d = tgt
            new_id = f"{country}__{name}".replace(" ", "_")
            if new_id not in cities.index:
                rows = np.where(world[f"GID_{lv}"].values == gid)[0]
                geom = unary_union(world.geometry.values[rows.tolist()])
                cities.loc[new_id] = {
                    "node_id": new_id, "gid": gid, "adm_level": str(lv),
                    "city": name, "country": country, "iso3": iso,
                    "area_km2": round(area, 1), "n_ports": 0, "ports": "",
                    "rep_lon": float(prt.Lon_port), "rep_lat": float(prt.Lat_port),
                    "geometry": geom, "band_exception": "",
                }
            sel = mapping.Port_Name == prt.Port_Name
            mapping.loc[sel, ["node_id", "final_gid", "final_level", "final_name"]] = \
                [new_id, gid, lv, name]
            mapping.loc[sel, "note"] = (str(mapping.loc[sel, "note"].iloc[0])
                                        + "; band_drill").strip("; ")
            changed_nodes.add(new_id)
        if not len(mapping[mapping.node_id == p.node_id]):
            cities = cities.drop(p.node_id)
            print(f"  C1: 删除空壳节点 {p.node_id}")

    # ---- B 合并（apply 时按当前节点状态实时重算最近邻，规避 C1 删除/链式改名）----
    def nearest_eligible(node_id: str) -> str | None:
        geo = gpd.GeoSeries(cities.geometry, crs="EPSG:4326").to_crs(EQ)
        me = geo.loc[node_id]
        iso = cities.loc[node_id, "iso3"]
        mask = ((cities["iso3"] == iso) & (cities.index != node_id)
                & (cities["area_km2"] <= AMAX))
        cand = geo[mask.values]
        if not len(cand):
            return None
        d = cand.distance(me) / 1000.0
        j = d.idxmin()
        return j if d[j] <= MERGE_DIST_KM else None

    for _, p in plan[plan.action == "merge"].iterrows():
        if p.node_id not in cities.index:
            continue
        target_id = nearest_eligible(p.node_id)
        if target_id is None:
            cities.loc[p.node_id, "band_exception"] = "urban_core"
            continue
        s, t = cities.loc[p.node_id], cities.loc[target_id]
        new_city = f"{t.city}-{s.city}"
        new_id = f"{t.country}__{new_city}".replace(" ", "_")
        merged = t.copy()
        merged["node_id"] = new_id
        merged["city"] = new_city
        merged["gid"] = ";".join(sorted(set(str(t.gid).split(";")) | set(str(s.gid).split(";"))))
        merged["adm_level"] = ";".join(sorted(set(str(t.adm_level).split(";"))
                                              | set(str(s.adm_level).split(";"))))
        merged["area_km2"] = round(float(t.area_km2) + float(s.area_km2), 1)
        merged["n_ports"] = int(t.n_ports) + int(s.n_ports)
        merged["ports"] = ";".join(sorted(set(str(t.ports).split(";"))
                                          | set(str(s.ports).split(";"))))
        merged["geometry"] = unary_union([t.geometry, s.geometry])
        merged["band_exception"] = ""
        cities = cities.drop([p.node_id, target_id])
        cities.loc[new_id] = merged
        sel = mapping.node_id.isin([p.node_id, target_id])
        mapping.loc[sel, "node_id"] = new_id
        lp = mapping.loc[mapping.node_id == new_id]
        cities.loc[new_id, "rep_lon"] = float(lp.Lon_port.mean())
        cities.loc[new_id, "rep_lat"] = float(lp.Lat_port.mean())
        changed_nodes.add(new_id)
        print(f"  B: {p.node_id} + {target_id} → {new_id}")

    # ---- 清理无名节点（GADM 细层 NAME 为空产生的 'nan' 城市名）：
    #      与同国等几何兄弟节点合并，否则回退用 gid 作名 ----
    bad_idx = [i for i in cities.index
               if str(cities.loc[i, "city"]).strip().lower() in ("nan", "none", "")]
    for i in bad_idx:
        me = cities.loc[i]
        twin = None
        for j in cities.index:
            if j == i or cities.loc[j, "iso3"] != me.iso3:
                continue
            if str(cities.loc[j, "city"]).strip().lower() in ("nan", "none", ""):
                continue
            gj = cities.loc[j, "geometry"]
            if gj is not None and abs(cities.loc[j, "area_km2"] - me.area_km2) < 1.0 \
                    and gj.buffer(1e-6).contains(me.geometry.representative_point()):
                twin = j
                break
        if twin is not None:
            mapping.loc[mapping.node_id == me.node_id, "node_id"] = cities.loc[twin, "node_id"]
            cities.loc[twin, "n_ports"] = int(cities.loc[twin, "n_ports"]) + int(me.n_ports)
            cities.loc[twin, "ports"] = ";".join(sorted(
                set(str(cities.loc[twin, "ports"]).split(";")) | set(str(me.ports).split(";"))))
            cities = cities.drop(i)
            changed_nodes.add(cities.loc[twin, "node_id"])
            print(f"  清理: 无名节点 {me.node_id} → 并入等几何节点 {cities.loc[twin, 'node_id']}")
        else:
            new_name = str(me.gid).split(";")[0]
            new_id = f"{me.country}__{new_name}".replace(" ", "_")
            mapping.loc[mapping.node_id == me.node_id, "node_id"] = new_id
            cities.loc[i, ["node_id", "city"]] = [new_id, new_name]
            print(f"  清理: 无名节点改用 gid 命名 → {new_id}")

    cities = gpd.GeoDataFrame(cities.reset_index(drop=True), geometry="geometry")
    if cities.crs is None:
        cities = cities.set_crs("EPSG:4326")

    # ---- 全局两两包含检查（1km 简化加速；covers 谓词不会误报共享边界的邻区）----
    proj = cities.to_crs(EQ)
    simp = proj.geometry.simplify(1000.0).buffer(0)
    sidx = simp.sindex
    nested = []
    for i, g in enumerate(simp.values):
        if g is None or g.is_empty:
            continue
        for j in sidx.query(g, predicate="covers"):
            if i != j and proj.iso3.values[i] == proj.iso3.values[j]:
                nested.append((cities.node_id.values[i], cities.node_id.values[j]))
    if nested:
        print(f"⚠ 仍存在 {len(nested)} 对嵌套节点(粗含细): {nested[:8]}")
    else:
        print("✓ 互斥检查通过：无节点嵌套")

    return cities, mapping, sorted(changed_nodes)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "plan"
    cfg = load_config()
    cities = gpd.read_file(cfg["paths"]["coastal_cities_gpkg"])
    mapping = pd.read_csv(cfg["paths"]["port_city_mapping_csv"], encoding="utf-8-sig")
    world = load_world(cfg["paths"]["world_shp"])
    world = ensure_row_areas(world, AREA_CACHE)

    plan = classify(cities, world, mapping)
    plan.to_csv(PLAN_CSV, index=False, encoding="utf-8-sig")
    print(f"\n计划已写出: {PLAN_CSV}")
    print(plan.groupby(["category", plan.action.str.split().str[0]]).size().to_string())
    if mode == "plan":
        return

    cities2, mapping2, changed = apply_plan(cities, mapping, plan, world)
    print(f"\n节点数: {len(cities)} → {len(cities2)}  几何变化节点: {len(changed)}")

    a = cities2["area_km2"]
    in_band = (a >= AMIN) & (a <= AMAX)
    flagged = cities2["band_exception"].fillna("") != ""
    unflagged = cities2[~in_band & ~flagged]
    print(f"带宽内: {in_band.mean():.1%}  出带未标注: {len(unflagged)}")
    if len(unflagged):
        # 合并/下钻后仍出带的，按方向补标注
        for i, r in unflagged.iterrows():
            cities2.loc[i, "band_exception"] = (
                "merged_still_small" if r.area_km2 < AMIN else "drill_still_large")
        print(unflagged[["node_id", "area_km2"]].to_string(index=False))

    gpkg = cfg["paths"]["coastal_cities_gpkg"]
    if os.path.exists(gpkg):
        os.remove(gpkg)
    cities2.to_file(gpkg, layer="coastal_cities", driver="GPKG")
    shp = os.path.join(OUT, "coastal_cities.shp")
    cities2.rename(columns={"band_exception": "band_exc"}).to_file(shp, encoding="utf-8")
    mapping2.to_csv(cfg["paths"]["port_city_mapping_csv"], index=False, encoding="utf-8-sig")
    with open(os.path.join(OUT, "band_changed_nodes.json"), "w", encoding="utf-8") as f:
        json.dump(changed, f, ensure_ascii=False, indent=1)
    print("已改写 gpkg / shp / mapping；几何变化节点清单 band_changed_nodes.json")


if __name__ == "__main__":
    main()
