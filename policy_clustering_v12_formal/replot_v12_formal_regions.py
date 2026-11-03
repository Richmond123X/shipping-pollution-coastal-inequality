"""Replot the formal v12 regions from cached geometry and finalized membership."""

from pathlib import Path

import geopandas as gpd
import pandas as pd

import maxp_clustering_core as base


ROOT = Path(r"E:\transfer justice\policy_clustering_v12_formal")
CACHE = ROOT / "cache" / "admin0_coastlines_2km_v3.gpkg"
MEMBERSHIP = ROOT / "tables" / "coastal_regions_geo_membership.csv"
EXCLUDED = ROOT / "tables" / "isolated_and_infeasible_geo_units.csv"
OUTPUT = ROOT / "figures" / "v12_formal_raw_regions_scientific.png"


def main() -> None:
    polygons = gpd.read_file(CACHE, layer="admin0_polygon").to_crs(4326)
    membership = pd.read_csv(MEMBERSHIP).rename(columns={"geo_iso3": "iso3"})
    excluded = pd.read_csv(EXCLUDED)
    included_isos = set(membership["iso3"]) | set(excluded["iso3"])
    polygons = polygons.loc[polygons["iso3"].isin(included_isos), ["iso3", "geometry"]]

    base.plot_regions(polygons, membership, excluded, output_path=OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
