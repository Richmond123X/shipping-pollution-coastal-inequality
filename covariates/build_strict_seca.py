# -*- coding: utf-8 -*-
"""Build an auditable city-year indicator for strict 0.1% sulfur ECAs.

The indicator deliberately excludes China's 0.5% domestic control areas.  It
uses the detailed ECA polygons rather than the coarse ``*_All.shp`` bounding
boxes that are used by the legacy AC3 pipeline.

Outputs
-------
outputs/strict_seca_zone_overlap.csv
    Time-invariant spatial overlap diagnostics by city and ECA zone.
outputs/strict_seca_0p1_city_year.csv
    City-year status, active-zone overlap area/share, and contributing zones.
"""
from __future__ import annotations

import os

import geopandas as gpd
import pandas as pd

from common import OUT, YEARS, load_config


# Detailed polygon, effective date for a zone-wide 0.1% sulfur requirement.
# North_America_ECA.shp already contains Hawaii and the US Caribbean ECA.
# Korea is dated 2022 here to match the emissions inventory's all-operation
# treatment; the earlier berth-only transition is not equivalent to a city-wide
# annual SECA exposure indicator.
STRICT_SECA_REGISTRY = [
    ("Baltic_sea_ECA.shp", "Baltic Sea", "2015-01"),
    ("North_sea_ECA.shp", "North Sea", "2015-01"),
    ("North_America_ECA.shp", "North America incl. Hawaii/US Caribbean", "2015-01"),
    ("South_Korea_ECA.shp", "South Korea port ECAs", "2022-01"),
    ("Mediterranean.shp", "Mediterranean Sea", "2025-05"),
]


def _union_valid_large_polygons(gdf: gpd.GeoDataFrame, equal_area_crs: str):
    """Return a cleaned union, dropping sub-1 km2 source artifacts."""
    projected = gdf.to_crs(equal_area_crs)
    projected = projected[projected.geometry.notna() & ~projected.geometry.is_empty].copy()
    projected = projected[projected.geometry.area.ge(1_000_000)].copy()
    if projected.empty:
        return None
    return projected.geometry.make_valid().union_all()


def main() -> None:
    cfg = load_config()
    equal_area_crs = cfg["port2city"]["equal_area_crs"]
    catch = gpd.read_file(cfg["paths"]["city_catchments_gpkg"], layer="catchments")
    catch_eq = catch.to_crs(equal_area_crs)
    catch_area_km2 = catch_eq.geometry.area / 1_000_000.0
    eca_dir = cfg["paths"]["eca_shp_dir"]

    overlap_rows: list[dict] = []
    for shp, zone, effective_month in STRICT_SECA_REGISTRY:
        path = os.path.join(eca_dir, shp)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing detailed strict-SECA polygon: {path}")
        print(f"Reading detailed polygon: {shp}", flush=True)
        zone_gdf = gpd.read_file(path)
        zone_geom = _union_valid_large_polygons(zone_gdf, equal_area_crs)
        if zone_geom is None:
            raise ValueError(f"No usable polygon remains in {path}")
        candidates = list(catch_eq.sindex.query(zone_geom, predicate="intersects"))
        area_lookup: dict[int, float] = {}
        if candidates:
            intersections = catch_eq.geometry.iloc[candidates].intersection(zone_geom)
            areas = intersections.area.to_numpy(float) / 1_000_000.0
            area_lookup = dict(zip(candidates, areas))
        positive = 0
        for idx, row in catch.iterrows():
            overlap_km2 = max(float(area_lookup.get(idx, 0.0)), 0.0)
            if overlap_km2 > 1e-6:
                positive += 1
            overlap_rows.append({
                "node_id": row.node_id,
                "iso3": row.iso3,
                "zone": zone,
                "effective_month": effective_month,
                "overlap_area_km2": overlap_km2,
                "catchment_overlap_share": min(
                    overlap_km2 / float(catch_area_km2.loc[idx]), 1.0
                ),
            })
        print(f"  cities with positive overlap: {positive}", flush=True)

    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(
        os.path.join(OUT, "strict_seca_zone_overlap.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    annual_rows: list[dict] = []
    for year in YEARS:
        year_end = f"{year}-12"
        active = overlap[overlap.effective_month.le(year_end)].copy()
        for node_id, group in active.groupby("node_id", sort=False):
            overlap_km2 = min(
                group.overlap_area_km2.sum(),
                float(catch_area_km2.loc[catch.index[catch.node_id.eq(node_id)][0]]),
            )
            positive = group[group.overlap_area_km2.gt(1e-6)]
            annual_rows.append({
                "node_id": node_id,
                "year": year,
                "strict_seca": int(overlap_km2 > 1e-6),
                "strict_seca_overlap_km2": overlap_km2,
                "strict_seca_catchment_share": min(
                    overlap_km2 /
                    float(catch_area_km2.loc[catch.index[catch.node_id.eq(node_id)][0]]),
                    1.0,
                ),
                "strict_seca_zones": "; ".join(positive.zone.tolist()),
            })

    annual = pd.DataFrame(annual_rows)
    annual.to_csv(
        os.path.join(OUT, "strict_seca_0p1_city_year.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    counts = annual.groupby("year").strict_seca.sum().astype(int)
    print("Strict 0.1% SECA city counts by year:", counts.to_dict(), flush=True)
    print("China 0.5% control areas are excluded by construction.", flush=True)


if __name__ == "__main__":
    main()
