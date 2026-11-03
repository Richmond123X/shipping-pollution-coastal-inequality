# -*- coding: utf-8 -*-
"""Port2City 主流程：L1 缓存 → L2 GIS → L3 仲裁批次 → 节点生成与输出。

用法：
  python run_port2city.py l2        # 跑 L2 匹配，写 mapping 草稿 + L3 待仲裁批次
  python run_port2city.py finalize  # 合并 L3 仲裁（若有）→ 生成城市节点面与全部输出

输出（outputs/）：
  port_city_mapping.csv   港口级审计表（SI 表）
  coastal_cities.gpkg     权威城市节点面（node_id/gid/adm_level/city/country/iso3/...）
  coastal_cities.shp      兼容版（字段名 ≤10 字符 + UTF-8 .cpg）
  review_queue.csv        残余人工复核清单
  qc_report.md            质量控制统计
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
import yaml
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from l2_gis import L2Matcher, load_world, ensure_row_areas  # noqa: E402
from l1_batches import load_cache, save_cache, port_key  # noqa: E402

OUT = os.path.join(HERE, "outputs")
L3_PENDING = os.path.join(HERE, "l3_pending")
L3_RESULTS = os.path.join(HERE, "l3_results")
DRAFT_CSV = os.path.join(OUT, "port_city_mapping_draft.csv")
AREA_CACHE = os.path.join(HERE, "world_row_areas.parquet")


def load_config() -> dict:
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_inputs(cfg):
    ports = pd.read_csv(cfg["paths"]["coastal_ports_csv"])
    cache = load_cache(cfg)
    missing = [r["Port_Name"] for _, r in ports.iterrows()
               if port_key(r["Port_Name"], r["Country_Region"]) not in cache]
    if missing:
        raise RuntimeError(f"{len(missing)} 个港口无 L1 判定（先跑 l1_batches.py）: {missing[:5]}")
    return ports, cache


# ---------------------------------------------------------------------------
# Stage 1: L2 匹配
# ---------------------------------------------------------------------------
def stage_l2() -> None:
    cfg = load_config()
    ports, cache = _load_inputs(cfg)

    world = load_world(cfg["paths"]["world_shp"])
    world = ensure_row_areas(world, AREA_CACHE, cfg["port2city"]["equal_area_crs"])
    matcher = L2Matcher(world, cfg)

    rows = []
    n = len(ports)
    for i, r in ports.iterrows():
        k = port_key(r["Port_Name"], r["Country_Region"])
        l1 = cache[k]
        res = matcher.match(r["Port_Name"], float(r["Lon_port"]), float(r["Lat_port"]),
                            l1["city"], l1.get("alt_names", []), l1["country"])
        rows.append({
            "Port_Name": r["Port_Name"],
            "Country_Region": r["Country_Region"],
            "Lon_port": r["Lon_port"], "Lat_port": r["Lat_port"],
            "n_berths": r["n_berths"],
            "l1_city": l1["city"], "l1_country": l1["country"],
            "method": res.method, "hit_level": res.hit_level,
            "final_gid": res.final_gid, "final_level": res.final_level,
            "final_name": res.final_name, "area_km2": round(res.area_km2, 1),
            "dist_km": round(res.dist_km, 2) if np.isfinite(res.dist_km) else "",
            "confidence": res.confidence, "needs_l3": res.needs_l3, "note": res.note,
            "chain": json.dumps([(lv, g, nm, round(a, 1)) for lv, g, nm, a, _ in res.chain],
                                ensure_ascii=False),
        })
        if (i + 1) % 200 == 0 or i + 1 == n:
            print(f"  L2 匹配 {i + 1}/{n}")

    os.makedirs(OUT, exist_ok=True)
    draft = pd.DataFrame(rows)
    draft.to_csv(DRAFT_CSV, index=False, encoding="utf-8-sig")
    print(f"草稿已写出: {DRAFT_CSV}")
    print(draft["method"].value_counts().to_string())
    print(draft["confidence"].value_counts().to_string())

    # 生成 L3 仲裁批次（带候选层级链）
    l3 = draft[draft["needs_l3"]].reset_index(drop=True)
    os.makedirs(L3_PENDING, exist_ok=True)
    for f in glob.glob(os.path.join(L3_PENDING, "batch_*.json")):
        os.remove(f)
    bs = int(cfg["port2city"]["l1_batch_size"])
    for b in range(0, len(l3), bs):
        bid = b // bs + 1
        items = []
        for _, x in l3.iloc[b:b + bs].iterrows():
            items.append({
                "port_name": x["Port_Name"], "country_region": x["Country_Region"],
                "lon": x["Lon_port"], "lat": x["Lat_port"],
                "l1_city": x["l1_city"], "l1_country": x["l1_country"],
                "l2_method": x["method"], "l2_note": x["note"],
                "l2_choice": {"gid": x["final_gid"], "level": int(x["final_level"]),
                              "name": x["final_name"], "area_km2": float(x["area_km2"])},
                "candidates": json.loads(x["chain"]),
            })
        with open(os.path.join(L3_PENDING, f"batch_{bid:03d}.json"), "w", encoding="utf-8") as f:
            json.dump({"batch_id": bid, "cases": items}, f, ensure_ascii=False, indent=1)
    print(f"L3 待仲裁: {len(l3)} 港 → {os.path.join(L3_PENDING)}")


# ---------------------------------------------------------------------------
# Stage 2: 合并 L3 + 节点生成
# ---------------------------------------------------------------------------
def merge_l3(draft: pd.DataFrame, world: gpd.GeoDataFrame) -> pd.DataFrame:
    """把 l3_results/*.json 的仲裁结果回写进 mapping（覆盖 final_gid/level/name）。"""
    files = sorted(glob.glob(os.path.join(L3_RESULTS, "batch_*_result.json")))
    if not files:
        print("无 L3 仲裁结果（l3_results 为空），沿用 L2 结果")
        return draft
    gid_name: dict[tuple[int, str], str] = {}
    decisions = {}
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            res = json.load(f)
        for it in res.get("results", []):
            key = (it["port_name"], it["country_region"])
            decisions[key] = it
    n_apply, n_review = 0, 0
    draft = draft.set_index(["Port_Name", "Country_Region"], drop=False)
    for key, it in decisions.items():
        if key not in draft.index:
            continue
        if it.get("decision") == "accept_l2":
            draft.loc[key, "confidence"] = "medium"
            draft.loc[key, "note"] = (str(draft.loc[key, "note"]) + "; l3_accepted").strip("; ")
            n_apply += 1
        elif it.get("decision") == "choose" and it.get("gid"):
            lv = int(it["level"])
            draft.loc[key, ["final_gid", "final_level", "final_name"]] = [
                it["gid"], lv, it.get("name", "")]
            draft.loc[key, "area_km2"] = float(it.get("area_km2", np.nan))
            draft.loc[key, "confidence"] = "medium"
            draft.loc[key, "method"] = "llm_arbitrated"
            n_apply += 1
        else:  # review
            draft.loc[key, "confidence"] = "low"
            draft.loc[key, "note"] = (str(draft.loc[key, "note"]) + "; l3_review").strip("; ")
            n_review += 1
    print(f"L3 合并: 采纳 {n_apply}, 转人工 {n_review}")
    return draft.reset_index(drop=True)


def stage_finalize() -> None:
    cfg = load_config()
    p2c = cfg["port2city"]
    draft = pd.read_csv(DRAFT_CSV, encoding="utf-8-sig")
    world = load_world(cfg["paths"]["world_shp"])
    world = ensure_row_areas(world, AREA_CACHE, p2c["equal_area_crs"])

    draft = merge_l3(draft, world)
    ok = draft[draft["final_gid"].astype(str).str.len() > 0].copy()
    bad = draft[draft["final_gid"].astype(str).str.len() == 0]

    # ---- 组合节点登记表：成员城市 final_name 改写为组合节点名 ----
    merge_map = {}
    for m in p2c.get("node_merges", []) or []:
        for member in m["members"]:
            merge_map[(m["country"], member)] = m["node"]

    # ---- 按 (country, final_gid) 聚合为节点 ----
    gid0_of = {}
    iso3_of = dict(world[["NAME_0", "GID_0"]].drop_duplicates().itertuples(index=False, name=None))
    nodes = {}
    port_rows = []
    for _, x in ok.iterrows():
        lv, gid = int(x["final_level"]), x["final_gid"]
        rows = np.where((world[f"GID_{lv}" if lv > 0 else "GID_0"].values == gid))[0]
        country = str(world["NAME_0"].values[rows[0]])
        iso3 = str(world["GID_0"].values[rows[0]])
        city = str(x["final_name"])
        merged_node = merge_map.get((country, city))
        node_city = merged_node or city
        node_id = f"{country}__{node_city}".replace(" ", "_")
        if node_id not in nodes:
            nodes[node_id] = {
                "node_id": node_id, "city": node_city, "country": country, "iso3": iso3,
                "gids": set(), "levels": set(), "row_idx": set(),
                "ports": [], "port_lons": [], "port_lats": [], "merged": bool(merged_node),
            }
        nd = nodes[node_id]
        nd["gids"].add(gid)
        nd["levels"].add(lv)
        nd["row_idx"].update(rows.tolist())
        nd["ports"].append(str(x["Port_Name"]))
        nd["port_lons"].append(float(x["Lon_port"]))
        nd["port_lats"].append(float(x["Lat_port"]))
        port_rows.append({**x.to_dict(), "node_id": node_id, "iso3": iso3, "country": country})

    # ---- dissolve 节点面、组装 GeoDataFrame ----
    print(f"生成 {len(nodes)} 个城市节点面（dissolve）...")
    recs = []
    for i, (node_id, nd) in enumerate(sorted(nodes.items())):
        idx = sorted(nd["row_idx"])
        geom = unary_union(world.geometry.values[idx])
        area = float(world["area_km2"].values[idx].sum())
        recs.append({
            "node_id": node_id, "gid": ";".join(sorted(nd["gids"])),
            "adm_level": ";".join(str(v) for v in sorted(nd["levels"])),
            "city": nd["city"], "country": nd["country"], "iso3": nd["iso3"],
            "area_km2": round(area, 1), "n_ports": len(nd["ports"]),
            "ports": ";".join(sorted(nd["ports"])),
            "rep_lon": float(np.mean(nd["port_lons"])),
            "rep_lat": float(np.mean(nd["port_lats"])),
            "geometry": geom,
        })
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(nodes)}")
    cities = gpd.GeoDataFrame(recs, crs="EPSG:4326")

    # 组合节点同国校验
    for m in p2c.get("node_merges", []) or []:
        nid = f"{m['country']}__{m['node']}".replace(" ", "_")
        if nid in set(cities["node_id"]):
            assert (cities.loc[cities.node_id == nid, "iso3"].nunique() == 1), f"组合节点跨国: {nid}"

    # ---- 输出 ----
    os.makedirs(OUT, exist_ok=True)
    gpkg = cfg["paths"]["coastal_cities_gpkg"]
    cities.to_file(gpkg, layer="coastal_cities", driver="GPKG")
    shp = os.path.join(OUT, "coastal_cities.shp")
    shp_df = cities.rename(columns={"adm_level": "adm_lvl", "area_km2": "area_km2",
                                    "n_ports": "n_ports", "rep_lon": "rep_lon",
                                    "rep_lat": "rep_lat"})
    shp_df.to_file(shp, encoding="utf-8")

    mapping = pd.DataFrame(port_rows).drop(columns=["chain"])
    mapping.to_csv(cfg["paths"]["port_city_mapping_csv"], index=False, encoding="utf-8-sig")

    review = pd.concat([
        bad,
        draft[draft["note"].astype(str).str.contains("l3_review|country_mismatch", na=False)],
    ]).drop_duplicates(subset=["Port_Name", "Country_Region"])
    review.to_csv(os.path.join(OUT, "review_queue.csv"), index=False, encoding="utf-8-sig")

    # ---- QC 报告 ----
    qc = []
    qc.append("# Port2City QC 报告\n")
    qc.append(f"- 港口总数: {len(draft)}；成功映射: {len(ok)}；失败: {len(bad)}")
    qc.append(f"- 城市节点数: {len(cities)}；国家数: {cities['iso3'].nunique()}")
    qc.append(f"\n## 匹配方式分布\n{mapping['method'].value_counts().to_string()}")
    nearest_share = (mapping["method"] == "spatial_nearest").mean()
    qc.append(f"\n- nearest 兜底占比: {nearest_share:.2%}" +
              ("  ⚠ 超过 5%，需在 SI 专门报告" if nearest_share > 0.05 else ""))
    qc.append(f"\n## 置信度分布\n{mapping['confidence'].value_counts().to_string()}")
    a = cities["area_km2"]
    qc.append("\n## 节点面积分布 (km²)")
    qc.append(f"- min={a.min():.0f}, P25={a.quantile(.25):.0f}, 中位={a.median():.0f}, "
              f"P75={a.quantile(.75):.0f}, max={a.max():.0f}")
    in_band = ((a >= p2c["area_min_km2"]) & (a <= p2c["area_max_km2"])).mean()
    qc.append(f"- 带宽内 [{p2c['area_min_km2']:.0f}, {p2c['area_max_km2']:.0f}] 占比: {in_band:.1%}")
    hist, edges = np.histogram(np.log10(a.clip(lower=1)), bins=12)
    qc.append("- log10(面积) 直方图: " + ", ".join(
        f"[{10**edges[i]:.0f}-{10**edges[i+1]:.0f}]:{hist[i]}" for i in range(len(hist))))
    qc.append(f"\n## 各国节点数 TOP15\n{cities['country'].value_counts().head(15).to_string()}")
    qc.append(f"\n- 人工复核队列: {len(review)} 条 (review_queue.csv)")
    with open(os.path.join(OUT, "qc_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(qc))
    print("\n".join(qc))
    print(f"\n输出完成: {gpkg}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "l2"
    {"l2": stage_l2, "finalize": stage_finalize}[stage]()
