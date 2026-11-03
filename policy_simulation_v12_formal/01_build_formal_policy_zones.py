"""Build the unique V12 X=50% policy treatment zones.

The selected set is fixed before policy effects are calculated: the first 22
formal Max-p regions, which cover 54.30% of global coastal population and map
to 75 unique policy-responsible states.  Each selected GADM geographic unit is
buffered 12 nautical miles from its ocean-facing coastline; land is removed
and geographic-unit belts are dissolved to the formal region identifier.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import make_valid, union_all
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union


ROOT = Path(r"E:\transfer justice")
CLUSTER_DIR = ROOT / "policy_clustering_v12_formal"
OUT_DIR = ROOT / "policy_simulation_v12_formal"
INPUT_DIR = OUT_DIR / "inputs"
TABLE_DIR = OUT_DIR / "tables"

MEMBERSHIP_FILE = CLUSTER_DIR / "tables" / "coastal_regions_geo_membership.csv"
REGION_FILE = CLUSTER_DIR / "tables" / "coastal_regions_summary.csv"
STATE_FILE = CLUSTER_DIR / "tables" / "top50pct_policy_countries.csv"
GEOMETRY_CACHE = CLUSTER_DIR / "cache" / "admin0_coastlines_2km_v3.gpkg"

ZONE_FILE = INPUT_DIR / "formal_policy_zones_50pct.gpkg"
SELECTED_REGION_FILE = TABLE_DIR / "selected_regions_50pct.csv"
SELECTED_GEO_FILE = TABLE_DIR / "selected_geo_units_50pct.csv"
SELECTED_STATE_FILE = TABLE_DIR / "selected_policy_states_50pct.csv"
QC_FILE = TABLE_DIR / "policy_zone_geometry_qc.csv"
MANIFEST_FILE = OUT_DIR / "formal_policy_manifest.json"

X_PCT = 50
MAX_REGION_RANK = 22
EXPECTED_POLICY_STATES = 75
ACTUAL_POP_SHARE = 0.5429617956079309
BUFFER_M = 12.0 * 1852.0

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def polygonal_only(geometry):
    if geometry is None or geometry.is_empty:
        return None
    geometry = make_valid(geometry)
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    parts = []
    for part in getattr(geometry, "geoms", []):
        if isinstance(part, Polygon):
            parts.append(part)
        elif isinstance(part, MultiPolygon):
            parts.extend(part.geoms)
    return unary_union(parts) if parts else None


def local_aeqd(geometry) -> str:
    minx, miny, maxx, maxy = geometry.bounds
    if maxx - minx > 180:
        raise ValueError("geometry crosses antimeridian and requires manual handling")
    lon0 = (minx + maxx) / 2.0
    lat0 = (miny + maxy) / 2.0
    return (
        f"+proj=aeqd +lat_0={lat0:.8f} +lon_0={lon0:.8f} "
        "+datum=WGS84 +units=m +no_defs"
    )


def build_geo_belt(
    geo_iso3: str,
    coast_geometry,
    land_wgs84: gpd.GeoDataFrame,
) -> tuple[object | None, dict]:
    audit = {
        "geo_iso3": geo_iso3,
        "status": "pending",
        "area_km2": 0.0,
        "n_components": 0,
        "is_valid": False,
        "note": "",
    }
    if coast_geometry is None or coast_geometry.is_empty:
        audit.update(status="failed", note="empty coastline")
        return None, audit

    try:
        aeqd = local_aeqd(coast_geometry)
    except ValueError as exc:
        audit.update(status="failed", note=str(exc))
        return None, audit

    minx, miny, maxx, maxy = coast_geometry.bounds
    pad_deg = BUFFER_M / 100_000.0 + 0.35
    query = box(minx - pad_deg, miny - pad_deg, maxx + pad_deg, maxy + pad_deg)
    idx = land_wgs84.sindex.query(query, predicate="intersects")
    nearby = land_wgs84.iloc[idx]
    if nearby.empty:
        audit.update(status="failed", note="no nearby land polygons")
        return None, audit

    coast_local = gpd.GeoSeries([coast_geometry], crs="EPSG:4326").to_crs(aeqd).iloc[0]
    land_local = nearby.to_crs(aeqd)
    land_parts = []
    for geometry in land_local.geometry.values:
        repaired = polygonal_only(geometry)
        if repaired is not None and not repaired.is_empty:
            land_parts.append(repaired)
    if not land_parts:
        audit.update(status="failed", note="nearby land could not be repaired")
        return None, audit
    # GADM contains a small number of self-touching rings.  A one-metre precision
    # grid is negligible relative to the 22,224 m policy belt and prevents GEOS
    # side-location conflicts during the local land dissolve.
    land_union = union_all(land_parts, grid_size=1.0)
    belt_local = polygonal_only(coast_local.buffer(BUFFER_M, resolution=8).difference(land_union))
    if belt_local is None or belt_local.is_empty:
        audit.update(status="failed", note="empty marine belt after land subtraction")
        return None, audit

    belt_wgs84 = gpd.GeoSeries([belt_local], crs=aeqd).to_crs("EPSG:4326").iloc[0]
    belt_wgs84 = polygonal_only(belt_wgs84)
    if belt_wgs84 is None or belt_wgs84.is_empty:
        audit.update(status="failed", note="empty belt after reprojection")
        return None, audit

    area_km2 = gpd.GeoSeries([belt_wgs84], crs="EPSG:4326").to_crs("EPSG:6933").area.iloc[0] / 1e6
    n_components = 1 if isinstance(belt_wgs84, Polygon) else len(belt_wgs84.geoms)
    audit.update(
        status="ok",
        area_km2=float(area_km2),
        n_components=int(n_components),
        is_valid=bool(belt_wgs84.is_valid),
    )
    return belt_wgs84, audit


def main() -> None:
    for directory in (INPUT_DIR, TABLE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    membership = pd.read_csv(MEMBERSHIP_FILE)
    selected_geo = membership.loc[membership["region_rank"].le(MAX_REGION_RANK)].copy()
    selected_geo = selected_geo.sort_values(["region_rank", "geo_iso3"])
    selected_regions = pd.read_csv(REGION_FILE)
    selected_regions = selected_regions.loc[selected_regions["rank"].le(MAX_REGION_RANK)].copy()
    selected_regions = selected_regions.sort_values("rank")
    selected_states = pd.read_csv(STATE_FILE).sort_values(
        ["first_region_rank", "policy_country_spri", "policy_country"],
        ascending=[True, False, True],
    )

    if selected_regions.shape[0] != 22:
        raise RuntimeError(f"Expected 22 selected regions, found {len(selected_regions)}")
    if selected_states["policy_iso3"].nunique() != EXPECTED_POLICY_STATES:
        raise RuntimeError(
            f"Expected {EXPECTED_POLICY_STATES} policy states, "
            f"found {selected_states['policy_iso3'].nunique()}"
        )
    if abs(selected_regions.iloc[-1]["cumulative_population_share"] - ACTUAL_POP_SHARE) > 1e-9:
        raise RuntimeError("Stored X=50% population coverage differs from formal manifest")

    polygons = gpd.read_file(GEOMETRY_CACHE, layer="admin0_polygon").to_crs("EPSG:4326")
    coasts = gpd.read_file(GEOMETRY_CACHE, layer="admin0_coast").to_crs("EPSG:4326")
    coast_lookup = coasts.set_index("iso3")["geometry"].to_dict()
    missing_coasts = sorted(set(selected_geo["geo_iso3"]) - set(coast_lookup))
    if missing_coasts:
        raise RuntimeError(f"Selected geographic units missing coastline geometry: {missing_coasts}")

    geo_rows = []
    audits = []
    for row in selected_geo.itertuples(index=False):
        print(f"[belt] {row.region_id} {row.geo_iso3} {row.geo_country}", flush=True)
        belt, audit = build_geo_belt(row.geo_iso3, coast_lookup[row.geo_iso3], polygons)
        audit.update(region_id=row.region_id, region_rank=int(row.region_rank))
        audits.append(audit)
        if belt is None:
            continue
        geo_rows.append(
            {
                "zone_id": row.region_id,
                "region_rank": int(row.region_rank),
                "geo_iso3": row.geo_iso3,
                "geo_country": row.geo_country,
                "policy_iso3": row.policy_iso3,
                "policy_country": row.policy_country,
                "geometry": belt,
            }
        )

    geo_zones = gpd.GeoDataFrame(geo_rows, crs="EPSG:4326")
    audit = pd.DataFrame(audits).sort_values(["region_rank", "geo_iso3"])
    failed = audit.loc[audit["status"].ne("ok")]
    if not failed.empty:
        failed.to_csv(QC_FILE, index=False, encoding="utf-8-sig")
        raise RuntimeError(
            "One or more selected geographic units failed policy-belt construction: "
            + ", ".join(failed["geo_iso3"])
        )

    region_rows = []
    for region_id, group in geo_zones.groupby("zone_id", sort=True):
        geometry = polygonal_only(unary_union(group.geometry.values))
        region_info = selected_regions.loc[selected_regions["region_id"].eq(region_id)].iloc[0]
        policy_states = sorted(group["policy_iso3"].dropna().unique())
        region_rows.append(
            {
                "zone_id": region_id,
                "region_rank": int(region_info["rank"]),
                "region_spri": float(region_info["region_spri"]),
                "population_share": float(region_info["population_share_global"]),
                "cumulative_population_share": float(region_info["cumulative_population_share"]),
                "n_geo_units": int(group["geo_iso3"].nunique()),
                "n_policy_states": int(len(policy_states)),
                "geo_iso3s": ",".join(sorted(group["geo_iso3"].unique())),
                "policy_iso3s": ",".join(policy_states),
                "geometry": geometry,
            }
        )
    region_zones = gpd.GeoDataFrame(region_rows, crs="EPSG:4326").sort_values("region_rank")
    region_zones["area_km2"] = region_zones.to_crs("EPSG:6933").area / 1e6

    # 正式排放计算需要在保持区域整体实施范围不变的同时，把结果精确拆到
    # “入选区域 × 政策责任国”。同一责任国映射到多个相邻地理单元时先 dissolve，
    # 避免同国海岸带内部重叠；跨国边界附近的窄小交叠仍由点赋值函数稳定取一个
    # 单元，因此全局总量不会重复计算。区域图继续使用 policy_regions。
    unit_rows = []
    for (region_id, policy_iso3, policy_country), group in geo_zones.groupby(
        ["zone_id", "policy_iso3", "policy_country"], sort=True
    ):
        region_rank = int(group["region_rank"].iloc[0])
        geo_members = sorted(group["geo_iso3"].astype(str).unique())
        unit_rows.append({
            "zone_id": f"{region_id}__{policy_iso3}",
            "region_id": region_id,
            "region_rank": region_rank,
            "policy_iso3": policy_iso3,
            "policy_country": policy_country,
            "geo_iso3s": ",".join(geo_members),
            "n_geo_units": len(geo_members),
            "geometry": polygonal_only(unary_union(group.geometry.values)),
        })
    policy_units = gpd.GeoDataFrame(unit_rows, crs="EPSG:4326").sort_values(
        ["region_rank", "policy_iso3"]
    )
    policy_units["area_km2"] = policy_units.to_crs("EPSG:6933").area / 1e6

    if ZONE_FILE.exists():
        ZONE_FILE.unlink()
    region_zones.to_file(ZONE_FILE, layer="policy_regions", driver="GPKG")
    geo_zones.to_file(ZONE_FILE, layer="policy_geo_units", driver="GPKG")
    policy_units.to_file(ZONE_FILE, layer="policy_units", driver="GPKG")

    selected_regions.to_csv(SELECTED_REGION_FILE, index=False, encoding="utf-8-sig")
    selected_geo.to_csv(SELECTED_GEO_FILE, index=False, encoding="utf-8-sig")
    selected_states.to_csv(SELECTED_STATE_FILE, index=False, encoding="utf-8-sig")
    audit.to_csv(QC_FILE, index=False, encoding="utf-8-sig")

    manifest = {
        "status": "formal_v12_policy_treatment_definition",
        "x_pct": X_PCT,
        "selected_region_count": int(len(region_zones)),
        "selected_geo_unit_count": int(selected_geo["geo_iso3"].nunique()),
        "selected_policy_state_count": int(selected_states["policy_iso3"].nunique()),
        "actual_coastal_population_share": ACTUAL_POP_SHARE,
        "territorial_nm": 12.0,
        "buffer_m": BUFFER_M,
        "geometry_source": str(GEOMETRY_CACHE),
        "cluster_membership": str(MEMBERSHIP_FILE),
        "policy_zone_file": str(ZONE_FILE),
        "layers": ["policy_regions", "policy_geo_units", "policy_units"],
        "n_policy_units": int(len(policy_units)),
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[complete] regions={len(region_zones)} geo_units={selected_geo.geo_iso3.nunique()} "
        f"policy_states={selected_states.policy_iso3.nunique()} "
        f"area_km2={region_zones.area_km2.sum():,.0f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
