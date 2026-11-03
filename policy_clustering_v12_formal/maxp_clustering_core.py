"""Shared algorithm and geometry utilities for formal v12 regionalization.

The published ``spopt.region.MaxPHeuristic`` accepts one additive threshold.
Research plan v12 instead specifies an exact logical-OR constraint:

    coastal population >= 2% of the global coastal population
    OR coastline length >= 5,000 km.

The formal runner uses an auditable extension of the Max-p heuristic: randomized
construction starts maximize p under the exact OR constraint, followed by a
connectivity-preserving local search that minimizes within-region SPRI SSE.

Run ``run_v12_formal_clustering.py`` rather than executing this module directly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path


ROOT = Path(r"E:\transfer justice")
FORMAL_DIR = ROOT / "policy_clustering_v12_formal"
LOCAL_DEPS = FORMAL_DIR / "_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyogrio
import shapely
from matplotlib.lines import Line2D
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon
from shapely.ops import split as split_geometry
from shapely.ops import transform as transform_geometry
from sklearn.metrics import adjusted_rand_score


PLAN = ROOT / "research_plan_v12.md"
RISK_FILE = (
    ROOT
    / "results_v10_reconstructed_2019_2027"
    / "data"
    / "SPRI_city_annual_2019_2025.parquet"
)
COVARIATE_FILE = ROOT / "covariates" / "outputs" / "city_covariates_2019_2028.csv"
CITY_GPKG = ROOT / "port2city" / "outputs" / "coastal_cities.gpkg"
WORLD_FINE = Path(r"C:\Users\86176\Desktop\Global_ECA\Global_shp\World.shp")
WORLD_ADMIN1 = Path(
    r"C:\Users\86176\Desktop\Global_ECA\Global_shp\world_province_new.shp"
)
NATURAL_EARTH_ADMIN0 = Path(
    r"C:\Users\86176\Desktop\shipping pollution transfer\Data\ne_110m_admin_0_countries.shp"
)

CACHE_DIR = FORMAL_DIR / "cache"
TABLE_DIR = FORMAL_DIR / "tables"
FIGURE_DIR = FORMAL_DIR / "figures"
REPORT_DIR = FORMAL_DIR / "report"

COUNTRY_CACHE = CACHE_DIR / "admin0_coastlines_2km_v3.gpkg"
POLYGON_STAGE_CACHE = CACHE_DIR / "admin0_polygons_2km_stage.gpkg"
COUNTRY_POLYGON_LAYER = "admin0_polygon"
COUNTRY_COAST_LAYER = "admin0_coast"

SEED = 20260822
N_STARTS = 200
COAST_NEIGHBOR_KM = 400.0
POLICY_BELT_CONNECT_KM = 2 * 22.224
COAST_THRESHOLD_KM = 5_000.0
POP_THRESHOLD_SHARE = 0.02
SIMPLIFY_TOLERANCE_M = 2_000.0
UNION_GRID_M = 100.0


# Operational parent assignment for the formal regionalization. Some entries are not
# uncontroversial sovereignty assignments; those are explicitly flagged in the
# audit output rather than silently treated as settled political facts.
PARENT_ISO = {
    "ABW": "NLD",
    "ALA": "FIN",
    "ASM": "USA",
    "BES": "NLD",
    "BLM": "FRA",
    "BMU": "GBR",
    "CCK": "AUS",
    "COK": "NZL",
    "CUW": "NLD",
    "CXR": "AUS",
    "CYM": "GBR",
    "ESH": "MAR",
    "FLK": "GBR",
    "FRO": "DNK",
    "GGY": "GBR",
    "GIB": "GBR",
    "GLP": "FRA",
    "GRL": "DNK",
    "GUF": "FRA",
    "GUM": "USA",
    "HKG": "CHN",
    "IMN": "GBR",
    "IOT": "GBR",
    "JEY": "GBR",
    "MAC": "CHN",
    "MAF": "FRA",
    "MNP": "USA",
    "MSR": "GBR",
    "MTQ": "FRA",
    "MYT": "FRA",
    "NCL": "FRA",
    "NFK": "AUS",
    "NIU": "NZL",
    "PRI": "USA",
    "PYF": "FRA",
    "REU": "FRA",
    "SGS": "GBR",
    "SJM": "NOR",
    "SPM": "FRA",
    "SXM": "NLD",
    "TCA": "GBR",
    "TKL": "NZL",
    "UMI": "USA",
    "VGB": "GBR",
    "VIR": "USA",
    "WLF": "FRA",
    "XNC": "TUR",
}

POLITICAL_REVIEW = {
    "ABW",
    "COK",
    "CUW",
    "ESH",
    "FRO",
    "GGY",
    "GRL",
    "HKG",
    "IMN",
    "JEY",
    "MAC",
    "NIU",
    "SXM",
    "XNC",
}


def ensure_dirs() -> None:
    for directory in (CACHE_DIR, TABLE_DIR, FIGURE_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def safe_union(geometries, grid_size: float = UNION_GRID_M):
    """Union geometries after validity repair, with a fixed precision grid."""

    geoms = np.asarray([g for g in geometries if g is not None and not g.is_empty])
    if len(geoms) == 0:
        return None
    geoms = shapely.make_valid(geoms)
    try:
        return shapely.union_all(geoms, grid_size=grid_size)
    except shapely.errors.GEOSException:
        repaired = shapely.buffer(geoms, 0)
        return shapely.union_all(repaired, grid_size=grid_size)


def polygonal_only(geometry):
    """Recursively retain only polygonal members of a possibly mixed result."""

    if geometry is None or geometry.is_empty:
        return GeometryCollection()
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    polygon_parts = []
    stack = list(shapely.get_parts(geometry))
    while stack:
        part = stack.pop()
        if isinstance(part, (Polygon, MultiPolygon)):
            polygon_parts.append(part)
        elif (
            part is not None
            and not part.is_empty
            and part.geom_type == "GeometryCollection"
        ):
            stack.extend(shapely.get_parts(part))
    if not polygon_parts:
        return GeometryCollection()
    return shapely.union_all(np.asarray(polygon_parts), grid_size=UNION_GRID_M)


def build_admin0_coast_cache(rebuild: bool = False) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Create simplified GADM admin-0 polygons and true outer coastlines.

    The 339k-row multi-level World.shp is too expensive to dissolve repeatedly.
    Its sibling admin-1 layer is derived from the same GADM geography and has
    only 3,668 polygons. It is simplified at 2 km for the reproducible formal
    clustering geometry cache.
    """

    if COUNTRY_CACHE.exists() and not rebuild:
        polygons = gpd.read_file(COUNTRY_CACHE, layer=COUNTRY_POLYGON_LAYER)
        coasts = gpd.read_file(COUNTRY_CACHE, layer=COUNTRY_COAST_LAYER)
        return polygons, coasts

    if POLYGON_STAGE_CACHE.exists() and not rebuild:
        polygons = gpd.read_file(POLYGON_STAGE_CACHE)
        polygons["geometry"] = polygons.geometry.apply(polygonal_only)
    else:
        print("[geometry] reading GADM admin-1 source", flush=True)
        admin1 = gpd.read_file(WORLD_ADMIN1, columns=["NAME_0", "geometry"]).to_crs(6933)
        admin1["geometry"] = admin1.geometry.simplify(
            SIMPLIFY_TOLERANCE_M, preserve_topology=True
        )

        metadata = pyogrio.read_dataframe(
            WORLD_FINE,
            columns=["GID_0", "NAME_0"],
            read_geometry=False,
        ).drop_duplicates()
        name_to_iso = dict(zip(metadata["NAME_0"], metadata["GID_0"], strict=False))

        print("[geometry] dissolving to admin-0", flush=True)
        rows = []
        for index, (name, group) in enumerate(admin1.groupby("NAME_0", sort=True), start=1):
            geometry = polygonal_only(safe_union(group.geometry.values))
            rows.append({"iso3": name_to_iso.get(name), "country": name, "geometry": geometry})
            if index % 50 == 0:
                print(f"[geometry] dissolved {index} countries", flush=True)
        polygons = gpd.GeoDataFrame(rows, geometry="geometry", crs=6933)
        if polygons["iso3"].isna().any():
            missing = polygons.loc[polygons["iso3"].isna(), "country"].tolist()
            raise RuntimeError(f"Admin-0 ISO mapping failed: {missing}")
        polygons.to_file(POLYGON_STAGE_CACHE, driver="GPKG")

    print("[geometry] extracting coastline with Natural Earth ocean-side mask", flush=True)
    # Dissolving separately simplified GADM countries can leave sub-kilometre
    # slivers at shared borders, causing inland borders to appear in the global
    # union boundary.  Use a coarse independent land mask only to classify which
    # GADM boundary segments face the ocean; retain the GADM lines themselves for
    # coastline length and distance calculations.
    ne = gpd.read_file(NATURAL_EARTH_ADMIN0, columns=["ISO_A3", "geometry"]).to_crs(6933)
    ne_land = polygonal_only(safe_union(ne.geometry.values))
    if ne_land.is_empty:
        raise RuntimeError("Natural Earth land mask contains no polygonal geometry")
    ocean_side_corridor = ne_land.boundary.buffer(25_000)
    ne_iso = set(ne["ISO_A3"].dropna())
    coast_rows = []
    for row in polygons.itertuples(index=False):
        if row.geometry is None or row.geometry.is_empty:
            coastline = GeometryCollection()
        else:
            boundary = row.geometry.boundary
            if boundary is None:
                coastline = GeometryCollection()
            elif row.iso3 not in ne_iso:
                # Natural Earth 1:110m omits many small islands and microstates.
                # Every such unit that reaches the policy regionalization is coastal,
                # so its complete GADM boundary is the best available coastline.
                coastline = boundary
            else:
                coastline = boundary.intersection(ocean_side_corridor)
                if coastline is None:
                    coastline = GeometryCollection()
        coast_rows.append(
            {
                "iso3": row.iso3,
                "country": row.country,
                "coast_km": coastline.length / 1_000,
                "geometry": coastline,
            }
        )
    coasts = gpd.GeoDataFrame(coast_rows, geometry="geometry", crs=6933)

    polygons.to_file(COUNTRY_CACHE, layer=COUNTRY_POLYGON_LAYER, driver="GPKG")
    coasts.to_file(COUNTRY_CACHE, layer=COUNTRY_COAST_LAYER, driver="GPKG")
    return polygons, coasts


def load_city_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    risk = pd.read_parquet(RISK_FILE)
    risk = risk.loc[risk["year"].eq(2025)].copy()
    cov = pd.read_csv(COVARIATE_FILE, usecols=["node_id", "year", "S1_pop"])
    cov = cov.loc[cov["year"].eq(2025), ["node_id", "S1_pop"]]
    cities = risk.merge(cov, on="node_id", how="left", validate="one_to_one")
    if cities["S1_pop"].isna().any():
        raise RuntimeError("2025 population is missing for one or more risk nodes")
    cities["source_iso3"] = cities["iso3"]
    cities["parent_iso3"] = cities["iso3"].replace(PARENT_ISO)
    return cities, risk


def make_parent_geographies(
    polygons: gpd.GeoDataFrame,
    coasts: gpd.GeoDataFrame,
    parent_isos: set[str],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    polygons = polygons.copy()
    coasts = coasts.copy()
    polygons["parent_iso3"] = polygons["iso3"].replace(PARENT_ISO)
    coasts["parent_iso3"] = coasts["iso3"].replace(PARENT_ISO)

    poly_rows = []
    coast_rows = []
    for parent_iso in sorted(parent_isos):
        pg = polygons.loc[polygons["parent_iso3"].eq(parent_iso)]
        cg = coasts.loc[coasts["parent_iso3"].eq(parent_iso)]
        if pg.empty or cg.empty:
            raise RuntimeError(f"Missing GADM geography for parent country {parent_iso}")
        poly_rows.append(
            {
                "iso3": parent_iso,
                "source_geographies": ";".join(sorted(pg["iso3"].tolist())),
                "geometry": safe_union(pg.geometry.values),
            }
        )
        coast_geometry = safe_union(cg.geometry.values)
        coast_rows.append(
            {
                "iso3": parent_iso,
                "source_geographies": ";".join(sorted(cg["iso3"].tolist())),
                "coast_km": coast_geometry.length / 1_000,
                "geometry": coast_geometry,
            }
        )
    parent_polygons = gpd.GeoDataFrame(poly_rows, geometry="geometry", crs=6933)
    parent_coasts = gpd.GeoDataFrame(coast_rows, geometry="geometry", crs=6933)
    return parent_polygons, parent_coasts


def aggregate_country_attributes(cities: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_names = (
        cities[["source_iso3", "country"]]
        .drop_duplicates()
        .sort_values(["source_iso3", "country"])
    )
    parent_names = {
        iso: group.loc[group["source_iso3"].eq(iso), "country"].iloc[0]
        if group["source_iso3"].eq(iso).any()
        else iso
        for iso, group in cities.groupby("parent_iso3")
    }
    # Prefer the parent country's own name when it occurs in the city data.
    for parent_iso, group in cities.groupby("parent_iso3"):
        own = group.loc[group["source_iso3"].eq(parent_iso), "country"]
        if not own.empty:
            parent_names[parent_iso] = own.iloc[0]

    valid_risk = cities.loc[cities["oversized_flag"].fillna(0).eq(0)].copy()
    risk_agg = valid_risk.groupby("parent_iso3").agg(
        country_spri=("SPRI", "mean"),
        n_risk_nodes=("node_id", "nunique"),
        spri_median=("SPRI", "median"),
        spri_q25=("SPRI", lambda x: x.quantile(0.25)),
        spri_q75=("SPRI", lambda x: x.quantile(0.75)),
        spri_min=("SPRI", "min"),
        spri_max=("SPRI", "max"),
    )
    pop_agg = cities.groupby("parent_iso3").agg(
        coastal_population=("S1_pop", "sum"),
        n_all_nodes=("node_id", "nunique"),
    )
    country = risk_agg.join(pop_agg, how="outer").reset_index().rename(
        columns={"parent_iso3": "iso3"}
    )
    country["country"] = country["iso3"].map(parent_names)

    audit = source_names.rename(columns={"source_iso3": "source_iso3", "country": "source_country"})
    audit["parent_iso3"] = audit["source_iso3"].replace(PARENT_ISO)
    audit["changed"] = audit["source_iso3"].ne(audit["parent_iso3"])
    audit["political_review_required"] = audit["source_iso3"].isin(POLITICAL_REVIEW)
    audit["n_nodes"] = audit["source_iso3"].map(cities.groupby("source_iso3")["node_id"].nunique())
    audit["coastal_population"] = audit["source_iso3"].map(cities.groupby("source_iso3")["S1_pop"].sum())
    return country, audit


def build_adjacency(
    country: pd.DataFrame,
    parent_polygons: gpd.GeoDataFrame,
    parent_coasts: gpd.GeoDataFrame,
) -> tuple[nx.Graph, pd.DataFrame]:
    coast_columns = ["iso3", "geometry"]
    if "coast_km" not in country.columns:
        coast_columns.append("coast_km")
    c = country.merge(
        parent_coasts[coast_columns].rename(columns={"geometry": "coast_geometry"}),
        on="iso3",
        how="left",
        validate="one_to_one",
    ).merge(
        parent_polygons[["iso3", "geometry"]].rename(columns={"geometry": "polygon_geometry"}),
        on="iso3",
        how="left",
        validate="one_to_one",
    )
    if c[["coast_geometry", "polygon_geometry", "coast_km"]].isna().any().any():
        raise RuntimeError("Country geography did not merge completely")

    graph = nx.Graph()
    for row in c.itertuples(index=False):
        graph.add_node(row.iso3)

    # A second 5 km line simplification makes the global 400 km distance query
    # tractable. The source polygons were already simplified at 2 km, so this
    # only affects adjacency, not city attributes. Global distances must remain
    # continuous across ±180°. EPSG:6933 planar distances create threshold
    # errors and cannot represent a true global shortest-coast distance, so the
    # simplified vertices are transformed to 3-D unit-sphere coordinates.
    print("[adjacency] simplifying coastline geometries for distance queries", flush=True)
    c["coast_distance_geometry"] = [
        shapely.simplify(geom, 5_000, preserve_topology=False)
        for geom in c["coast_geometry"]
    ]
    print("[adjacency] querying candidate neighbors", flush=True)
    coast_geoms = c["coast_distance_geometry"].tolist()
    coast_xy = [shapely.get_coordinates(geom) for geom in coast_geoms]
    transformer = Transformer.from_crs(parent_coasts.crs, 4326, always_xy=True)
    coast_coords = []
    for xy in coast_xy:
        lon, lat = transformer.transform(xy[:, 0], xy[:, 1])
        lon, lat = np.deg2rad(lon), np.deg2rad(lat)
        coast_coords.append(np.column_stack([
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ]))
    coast_trees = [cKDTree(coords) if len(coords) else None for coords in coast_coords]
    by_iso = c.set_index("iso3")
    edge_rows = []
    earth_radius_km = 6_371.0088
    # Euclidean chord corresponding to the 400-km great-circle cutoff.
    max_chord = 2.0 * math.sin(COAST_NEIGHBOR_KM / (2.0 * earth_radius_km))
    for left_pos, right_pos in combinations(range(len(c)), 2):
        left = c.iloc[left_pos]
        right = c.iloc[right_pos]
        left_iso = left["iso3"]
        right_iso = right["iso3"]
        pair = tuple(sorted((left_iso, right_iso)))
        left_coords = coast_coords[left_pos]
        right_coords = coast_coords[right_pos]
        if len(left_coords) <= len(right_coords):
            distances, _ = coast_trees[right_pos].query(
                left_coords, k=1, distance_upper_bound=max_chord
            )
        else:
            distances, _ = coast_trees[left_pos].query(
                right_coords, k=1, distance_upper_bound=max_chord
            )
        min_chord = float(np.min(distances))
        if not np.isfinite(min_chord) or min_chord >= max_chord:
            continue
        distance_km = float(
            earth_radius_km * 2.0 * math.asin(min(1.0, min_chord / 2.0))
        )
        # With country coastlines, a shared coastal land border has zero
        # endpoint distance.  The 5 km category also absorbs simplification
        # error at the GADM boundary.
        land_adjacent = distance_km <= 5.0
        shared_border_km = np.nan
        adjacency_type = (
            "coast_contact_or_shared_border"
            if land_adjacent
            else "sea_under_400km"
        )
        graph.add_edge(
            left_iso,
            right_iso,
            distance_km=distance_km,
            shared_border_km=shared_border_km,
            adjacency_type=adjacency_type,
        )
        edge_rows.append(
            {
                "iso3_a": pair[0],
                "country_a": by_iso.loc[pair[0], "country"],
                "iso3_b": pair[1],
                "country_b": by_iso.loc[pair[1], "country"],
                "distance_km": distance_km,
                "shared_border_km": shared_border_km,
                "adjacency_type": adjacency_type,
                "distance_method": (
                    "5km_simplified_coast_vertices_unit_sphere_"
                    "chord_to_great_circle"
                ),
                "policy_belts_touch": bool(
                    land_adjacent or distance_km <= POLICY_BELT_CONNECT_KM
                ),
                "long_sea_link_flag": bool(
                    (not land_adjacent) and distance_km > POLICY_BELT_CONNECT_KM
                ),
            }
        )
    edges = pd.DataFrame(edge_rows).sort_values(["iso3_a", "iso3_b"])
    return graph, edges


def region_feasible(nodes: set[str], country_lookup: pd.DataFrame, pop_threshold: float) -> bool:
    rows = country_lookup.loc[list(nodes)]
    return bool(
        rows["coastal_population"].sum() >= pop_threshold
        or rows["coast_km"].sum() >= COAST_THRESHOLD_KM
    )


def region_wss(nodes: set[str], country_lookup: pd.DataFrame) -> float:
    values = country_lookup.loc[list(nodes), "country_spri"].to_numpy(float)
    return float(((values - values.mean()) ** 2).sum())


def construct_component_regions(
    component_nodes: set[str],
    graph: nx.Graph,
    country_lookup: pd.DataFrame,
    pop_threshold: float,
    rng: np.random.Generator,
) -> list[set[str]]:
    unassigned = set(component_nodes)
    regions: list[set[str]] = []
    enclaves: set[str] = set()
    seed_order = sorted(component_nodes)
    rng.shuffle(seed_order)

    for seed in seed_order:
        if seed not in unassigned:
            continue
        cluster = {seed}
        unassigned.remove(seed)
        frontier = set(graph.neighbors(seed)) & unassigned
        while not region_feasible(cluster, country_lookup, pop_threshold) and frontier:
            mean_spri = country_lookup.loc[list(cluster), "country_spri"].mean()
            ordered = sorted(
                frontier,
                key=lambda node: (
                    abs(country_lookup.loc[node, "country_spri"] - mean_spri),
                    node,
                ),
            )
            top_n = min(2, len(ordered))
            chosen = ordered[int(rng.integers(top_n))]
            cluster.add(chosen)
            unassigned.remove(chosen)
            frontier.remove(chosen)
            frontier |= (set(graph.neighbors(chosen)) & unassigned)
        if region_feasible(cluster, country_lookup, pop_threshold):
            regions.append(cluster)
        else:
            enclaves |= cluster

    enclaves |= unassigned
    while enclaves:
        options = []
        for node in sorted(enclaves):
            for region_id, region in enumerate(regions):
                if any(neighbor in region for neighbor in graph.neighbors(node)):
                    increase = region_wss(region | {node}, country_lookup) - region_wss(
                        region, country_lookup
                    )
                    options.append((increase, node, region_id))
        if not options:
            raise RuntimeError(
                "Enclave assignment failed despite a feasible connected component"
            )
        options.sort(key=lambda item: (item[0], item[1], item[2]))
        _, node, region_id = options[0]
        regions[region_id].add(node)
        enclaves.remove(node)
    return regions


def improve_regions(
    regions: list[set[str]],
    graph: nx.Graph,
    country_lookup: pd.DataFrame,
    pop_threshold: float,
) -> list[set[str]]:
    """Greedy boundary moves preserving contiguity and the exact OR threshold."""

    improved = True
    while improved:
        improved = False
        membership = {
            node: region_id for region_id, region in enumerate(regions) for node in region
        }
        best = None
        for donor_id, donor in enumerate(regions):
            if len(donor) <= 1:
                continue
            donor_subgraph = graph.subgraph(donor)
            articulation = set(nx.articulation_points(donor_subgraph))
            for node in sorted(donor):
                if node in articulation:
                    continue
                donor_after = donor - {node}
                if not region_feasible(donor_after, country_lookup, pop_threshold):
                    continue
                recipient_ids = {
                    membership[n]
                    for n in graph.neighbors(node)
                    if membership[n] != donor_id
                }
                for recipient_id in sorted(recipient_ids):
                    recipient = regions[recipient_id]
                    before = region_wss(donor, country_lookup) + region_wss(
                        recipient, country_lookup
                    )
                    after = region_wss(donor_after, country_lookup) + region_wss(
                        recipient | {node}, country_lookup
                    )
                    delta = after - before
                    candidate = (delta, node, donor_id, recipient_id)
                    if delta < -1e-10 and (best is None or candidate < best):
                        best = candidate
        if best is not None:
            _, node, donor_id, recipient_id = best
            regions[donor_id].remove(node)
            regions[recipient_id].add(node)
            improved = True
    return regions


def solve_maxp_or(
    graph: nx.Graph,
    country: pd.DataFrame,
    n_starts: int,
) -> tuple[list[set[str]], pd.DataFrame, pd.DataFrame]:
    lookup = country.set_index("iso3")
    pop_threshold = country["coastal_population"].sum() * POP_THRESHOLD_SHARE

    isolated_nodes = sorted(nx.isolates(graph))
    cluster_graph = graph.copy()
    cluster_graph.remove_nodes_from(isolated_nodes)

    feasible_components = []
    infeasible_components = []
    components = sorted(
        (set(component) for component in nx.connected_components(cluster_graph)),
        key=lambda component: (min(component), len(component)),
    )
    for component in components:
        if region_feasible(set(component), lookup, pop_threshold):
            feasible_components.append(set(component))
        else:
            infeasible_components.append(set(component))
            cluster_graph.remove_nodes_from(component)

    excluded_rows = []
    for iso in isolated_nodes:
        excluded_rows.append({"iso3": iso, "reason": "no_neighbor_under_400km", "component_id": None})
    for component_id, component in enumerate(infeasible_components, start=1):
        for iso in sorted(component):
            excluded_rows.append(
                {
                    "iso3": iso,
                    "reason": "connected_component_fails_both_size_thresholds",
                    "component_id": component_id,
                }
            )

    run_rows = []
    solutions = []
    for run in range(n_starts):
        rng = np.random.default_rng(SEED + run)
        regions: list[set[str]] = []
        for component in feasible_components:
            regions.extend(
                construct_component_regions(
                    component, cluster_graph, lookup, pop_threshold, rng
                )
            )
        objective = sum(region_wss(region, lookup) for region in regions)
        run_rows.append({"run": run + 1, "seed": SEED + run, "p": len(regions), "wss": objective})
        solutions.append(regions)
        if (run + 1) % 10 == 0:
            print(f"[max-p] completed {run + 1}/{n_starts} construction starts", flush=True)

    runs = pd.DataFrame(run_rows)
    best_p = runs["p"].max()
    best_idx = runs.loc[runs["p"].eq(best_p), "wss"].idxmin()
    best_regions = improve_regions(
        solutions[int(best_idx)], cluster_graph, lookup, pop_threshold
    )
    improved_wss = sum(region_wss(region, lookup) for region in best_regions)

    ordered_nodes = sorted(cluster_graph.nodes())
    label_arrays = []
    for regions in solutions:
        membership = {node: rid for rid, reg in enumerate(regions) for node in reg}
        label_arrays.append([membership[node] for node in ordered_nodes])
    ari_values = []
    for a, b in combinations(range(len(label_arrays)), 2):
        ari_values.append(adjusted_rand_score(label_arrays[a], label_arrays[b]))
    stability = pd.DataFrame(
        [
            {
                "n_solutions": len(label_arrays),
                "n_pairs": len(ari_values),
                "ari_mean": float(np.mean(ari_values)),
                "ari_median": float(np.median(ari_values)),
                "ari_min": float(np.min(ari_values)),
                "ari_max": float(np.max(ari_values)),
                "best_p": int(best_p),
                "best_run": int(runs.loc[best_idx, "run"]),
                "best_construction_wss": float(runs.loc[best_idx, "wss"]),
                "best_improved_wss": float(improved_wss),
            }
        ]
    )
    excluded = pd.DataFrame(excluded_rows)
    if not excluded.empty:
        excluded = excluded.merge(country[["iso3", "country", "country_spri", "coastal_population", "coast_km"]], on="iso3", how="left")
    return best_regions, runs, stability, excluded


def canonicalize_regions(
    regions: list[set[str]],
    country: pd.DataFrame,
    cities: pd.DataFrame,
    graph: nx.Graph,
    edges: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lookup = country.set_index("iso3")
    city_valid = cities.loc[cities["oversized_flag"].fillna(0).eq(0)].copy()
    region_records = []
    member_records = []

    raw_summaries = []
    for region in regions:
        member_cities = city_valid.loc[city_valid["parent_iso3"].isin(region)]
        all_cities = cities.loc[cities["parent_iso3"].isin(region)]
        region_spri = member_cities["SPRI"].mean()
        raw_summaries.append((region, region_spri, member_cities, all_cities))
    raw_summaries.sort(key=lambda item: (-item[1], sorted(item[0])))

    total_population = cities["S1_pop"].sum()
    cumulative_population = 0.0
    for rank, (region, region_spri, member_cities, all_cities) in enumerate(raw_summaries, start=1):
        region_id = f"R{rank:02d}"
        members = sorted(region)
        population = lookup.loc[members, "coastal_population"].sum()
        coast_km = lookup.loc[members, "coast_km"].sum()
        cumulative_population += population

        policy_graph = nx.Graph()
        policy_graph.add_nodes_from(members)
        region_edges = edges.loc[
            edges["iso3_a"].isin(members) & edges["iso3_b"].isin(members)
        ].copy()
        for edge in region_edges.itertuples(index=False):
            if edge.policy_belts_touch:
                policy_graph.add_edge(edge.iso3_a, edge.iso3_b)
        policy_components = list(nx.connected_components(policy_graph))
        component_population = [
            lookup.loc[list(component), "coastal_population"].sum()
            for component in policy_components
        ]
        largest_policy_component_share = (
            max(component_population) / population if population > 0 else np.nan
        )
        graph_sub = graph.subgraph(members)
        graph_edges = list(graph_sub.edges(data=True))
        sea_edges = [d for _, _, d in graph_edges if d["adjacency_type"] == "sea_under_400km"]
        long_sea_edges = [
            d for d in sea_edges if d["distance_km"] > POLICY_BELT_CONNECT_KM
        ]
        source_geographies = lookup.loc[members, "source_geographies"].str.split(";").explode()
        territory_sources = [iso for iso in source_geographies if PARENT_ISO.get(iso, iso) != iso]

        record = {
            "region_id": region_id,
            "rank": rank,
            "region_spri": region_spri,
            "n_countries": len(members),
            "n_city_nodes": member_cities["node_id"].nunique(),
            "coastal_population": population,
            "population_share_global": population / total_population,
            "cumulative_population_share": cumulative_population / total_population,
            "coast_km": coast_km,
            "meets_population_threshold": population >= total_population * POP_THRESHOLD_SHARE,
            "meets_coast_threshold": coast_km >= COAST_THRESHOLD_KM,
            "city_spri_median": member_cities["SPRI"].median(),
            "city_spri_q25": member_cities["SPRI"].quantile(0.25),
            "city_spri_q75": member_cities["SPRI"].quantile(0.75),
            "city_spri_iqr": member_cities["SPRI"].quantile(0.75)
            - member_cities["SPRI"].quantile(0.25),
            "city_spri_min": member_cities["SPRI"].min(),
            "city_spri_max": member_cities["SPRI"].max(),
            "n_graph_edges": len(graph_edges),
            "n_sea_edges": len(sea_edges),
            "n_long_sea_edges": len(long_sea_edges),
            "max_internal_neighbor_distance_km": max(
                [d["distance_km"] for _, _, d in graph_edges], default=0.0
            ),
            "policy_belt_country_components": len(policy_components),
            "largest_policy_component_population_share": largest_policy_component_share,
            "n_mapped_territory_sources": len(territory_sources),
            "members": ";".join(members),
            "member_names": ";".join(lookup.loc[members, "country"].tolist()),
        }
        region_records.append(record)
        for iso in members:
            member_records.append(
                {
                    "region_id": region_id,
                    "region_rank": rank,
                    "iso3": iso,
                    **lookup.loc[iso].to_dict(),
                }
            )

    region_summary = pd.DataFrame(region_records)
    membership = pd.DataFrame(member_records)
    cutoff = region_summary[
        [
            "region_id",
            "rank",
            "region_spri",
            "n_countries",
            "coastal_population",
            "population_share_global",
            "cumulative_population_share",
            "policy_belt_country_components",
            "largest_policy_component_population_share",
        ]
    ].copy()
    return region_summary, membership, cutoff


def make_x_threshold_lookup(region_summary: pd.DataFrame) -> pd.DataFrame:
    """Show how nominal X targets map to indivisible ranked regions."""
    ranked = region_summary.sort_values("rank").reset_index(drop=True)
    records = []
    for target_pct in range(10, 100, 10):
        target = target_pct / 100.0
        reached = ranked.loc[ranked["cumulative_population_share"].ge(target)]
        cutoff_row = reached.iloc[0] if not reached.empty else ranked.iloc[-1]
        records.append(
            {
                "target_x_pct": target_pct,
                "selected_region_count": int(cutoff_row["rank"]),
                "last_selected_region": cutoff_row["region_id"],
                "last_selected_region_spri": cutoff_row["region_spri"],
                "achieved_population_share_pct": 100
                * cutoff_row["cumulative_population_share"],
                "overshoot_percentage_points": 100
                * cutoff_row["cumulative_population_share"]
                - target_pct,
            }
        )
    return pd.DataFrame(records)


def split_antimeridian_for_plot(geometry):
    """Split dateline-crossing polygons so map rendering cannot draw world-spanning bands."""
    if geometry is None or geometry.is_empty:
        return geometry
    min_x, _, max_x, _ = geometry.bounds
    if max_x - min_x <= 180:
        return geometry

    def shift_to_360(x, y, z=None):
        shifted_x = np.where(np.asarray(x) < 0, np.asarray(x) + 360, x)
        return (shifted_x, y) if z is None else (shifted_x, y, z)

    def shift_to_180(x, y, z=None):
        shifted_x = np.asarray(x) - 360
        return (shifted_x, y) if z is None else (shifted_x, y, z)

    shifted = transform_geometry(shift_to_360, shapely.make_valid(geometry))
    cutter = LineString([(180, -90), (180, 90)])
    try:
        pieces = list(shapely.get_parts(split_geometry(shifted, cutter)))
    except (ValueError, TypeError):
        pieces = list(shapely.get_parts(shifted))

    restored = []
    for piece in pieces:
        if piece is None or piece.is_empty:
            continue
        if piece.representative_point().x >= 180:
            piece = transform_geometry(shift_to_180, piece)
        restored.append(piece)
    if not restored:
        return GeometryCollection()
    return shapely.make_valid(shapely.union_all(restored))


def scientific_region_colors(region_ids: list[str]) -> dict[str, tuple[float, ...]]:
    """Return a fresh, low-saturation categorical palette with separated neighbours."""
    import seaborn as sns

    n_colors = max(len(region_ids), 2)
    base_palette = sns.husl_palette(n_colors, h=0.03, s=0.62, l=0.64)
    # A coprime stride avoids an artificial rainbow gradient by region rank.
    stride = 23 if math.gcd(23, n_colors) == 1 else 19
    while math.gcd(stride, n_colors) != 1:
        stride += 2
    ordered = [base_palette[(index * stride) % n_colors] for index in range(n_colors)]
    return {region_id: ordered[index] for index, region_id in enumerate(region_ids)}


def prepare_region_plot_geometries(plot: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Repair dateline seams in cached geometries without changing membership."""
    plot = plot.copy()
    plot["geometry"] = plot.geometry.apply(split_antimeridian_for_plot)

    # The equal-area admin-0 cache leaves a horizontal seam in Russia after
    # round-tripping across the dateline. Rebuild only this plotting geometry
    # from the original WGS84 admin-1 source; clustering inputs remain untouched.
    if WORLD_ADMIN1.exists() and plot["iso3"].eq("RUS").any():
        russia = gpd.read_file(WORLD_ADMIN1, where="NAME_0 = 'Russia'").to_crs(4326)
        russia_parts = russia.geometry.apply(split_antimeridian_for_plot)
        russia_geometry = shapely.make_valid(shapely.union_all(russia_parts.values))
        for index in plot.index[plot["iso3"].eq("RUS")]:
            plot.at[index, "geometry"] = russia_geometry
    return plot


def plot_regions(
    parent_polygons: gpd.GeoDataFrame,
    membership: pd.DataFrame,
    excluded: pd.DataFrame,
    output_path: Path | None = None,
) -> None:
    plot = parent_polygons.merge(
        membership[["iso3", "region_id"]], on="iso3", how="left"
    ).to_crs(4326)
    plot = prepare_region_plot_geometries(plot)
    if not excluded.empty:
        reason_map = excluded.set_index("iso3")["reason"]
        plot["excluded_reason"] = plot["iso3"].map(reason_map)

    region_ids = sorted(plot["region_id"].dropna().unique())
    colors = scientific_region_colors(region_ids)
    fig, ax = plt.subplots(figsize=(16, 8.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for region_id, group in plot.dropna(subset=["region_id"]).groupby("region_id"):
        group.plot(ax=ax, color=colors[region_id], edgecolor="white", linewidth=0.22)
    excluded_plot = plot.loc[plot["region_id"].isna()]
    if not excluded_plot.empty:
        excluded_plot.plot(ax=ax, color="#D4DCE3", edgecolor="white", linewidth=0.2)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_axis_off()
    ax.set_title(
        "V12 coastal Max-p regionalization: formal result",
        fontsize=17,
        color="#24364B",
        pad=12,
    )
    ax.text(
        0.01,
        0.015,
        "Graph adjacency: coastal land contact or coast distance <400 km.\nGray: excluded isolated/infeasible components. Colors are region IDs, not policy selection.",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#536579",
        va="bottom",
    )
    if output_path is None:
        output_path = FIGURE_DIR / "v12_maxp_raw_regions.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_ranking(region_summary: pd.DataFrame) -> None:
    d = region_summary.sort_values("rank")
    x = np.arange(len(d))
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12.5, 8.2),
        gridspec_kw={"height_ratios": [1.35, 1]},
        sharex=True,
    )
    yerr = np.vstack(
        [
            d["region_spri"] - d["city_spri_q25"],
            d["city_spri_q75"] - d["region_spri"],
        ]
    )
    yerr = np.maximum(yerr, 0)
    ax1.errorbar(
        x,
        d["region_spri"],
        yerr=yerr,
        fmt="o",
        color="#2171B5",
        ecolor="#9ECAE1",
        capsize=2.5,
        markersize=5,
    )
    ax1.set_ylabel("Region city-equal SPRI (2025)")
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_title("Region risk ranking and cumulative coastal-population coverage")

    ax2.step(
        x,
        100 * d["cumulative_population_share"],
        where="mid",
        color="#D95F0E",
        linewidth=2,
    )
    ax2.scatter(x, 100 * d["cumulative_population_share"], color="#D95F0E", s=24)
    ax2.set_ylabel("Cumulative coastal population (%)")
    ax2.set_xlabel("Regions ordered by descending SPRI")
    ax2.set_xticks(x)
    ax2.set_xticklabels(d["region_id"], rotation=60, ha="right")
    ax2.set_ylim(0, 103)
    ax2.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "v12_region_ranking_cumulative_population.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(
    country: pd.DataFrame,
    graph: nx.Graph,
    edges: pd.DataFrame,
    region_summary: pd.DataFrame,
    excluded: pd.DataFrame,
    stability: pd.DataFrame,
    territory_audit: pd.DataFrame,
    x_lookup: pd.DataFrame,
) -> None:
    long_links = edges.loc[edges["long_sea_link_flag"]]
    fragmented = region_summary.loc[region_summary["policy_belt_country_components"].gt(1)]
    wide_iqr = region_summary.nlargest(min(5, len(region_summary)), "city_spri_iqr")
    politically_sensitive = territory_audit.loc[
        territory_audit["political_review_required"] & territory_audit["changed"]
    ]
    singleton_count = int(region_summary["n_countries"].eq(1).sum())
    starts = pd.read_csv(TABLE_DIR / "maxp_100_start_results.csv")
    p_min = int(starts["p"].min())
    p_max = int(starts["p"].max())
    n_best_p = int(starts["p"].eq(p_max).sum())

    lines = [
        "# V12 policy clustering: automatic geographic audit",
        "",
        "## Verdict",
        "",
        "**The current raw clustering is not realistic enough to define X or enter the formal policy simulation.** The 'cluster first, then rank' idea is still useful, but the spatial unit and contiguity rules must be revised before choosing X.",
        "",
        "The main conflict is structural: Max-p maximizes the number of feasible regions, while the policy concept requires broad, coherent ECA-like sea areas. The coastline arm of the OR threshold makes many large-coast countries feasible as singleton regions, and whole-country parent nodes connect unrelated ocean basins through overseas territories or multiple separated coasts.",
        "",
        "## Execution summary",
        "",
        f"- Parent-country units entering adjacency: {len(country)}",
        f"- Adjacency edges: {graph.number_of_edges()}",
        f"- Raw Max-p regions: {len(region_summary)}",
        f"- Excluded isolated/infeasible units: {len(excluded)}",
        f"- Best p across 100 starts: {int(stability.iloc[0]['best_p'])}",
        f"- Mean pairwise adjusted Rand index: {stability.iloc[0]['ari_mean']:.3f}",
        f"- Region-count range across starts: {p_min}–{p_max}; only {n_best_p} start(s) reached p={p_max}",
        f"- Singleton regions in the selected solution: {singleton_count}",
        "",
        "## Direct implications for realism",
        "",
        f"- {len(long_links)} adjacency edges bridge more than 44.448 km of open water, so the corresponding national 12-nautical-mile belts do not physically touch.",
        f"- {len(fragmented)} of {len(region_summary)} regions contain more than one country-level 12-nautical-mile policy-belt component.",
        f"- {len(politically_sensitive)} parent assignments require an explicit legal/political review before formal use.",
        "- Graph contiguity therefore must not be described as geometric continuity of the 12-nautical-mile treatment polygons.",
        "- Indivisible sovereign nodes can connect different sea basins. Examples in this solution include Russia linking European and Pacific coasts, France linking several oceans, and the Netherlands linking Europe and the Caribbean.",
        "- Cape Verde and Maldives are isolated under the 400 km rule and are excluded from regional ranking despite non-trivial SPRI, which conflicts with the environmental-justice purpose of the exercise.",
        "",
        "## Consequence for selecting X",
        "",
        "Because regions are indivisible, nominal X targets can overshoot sharply; moreover, these coverage cutoffs inherit the unrealistic raw regionalization. `tables/nominal_X_coverage_lookup.csv` records the exact target-to-coverage mapping, but it should be treated as a diagnostic rather than a basis for choosing X.",
        "",
    ]
    for row in x_lookup.itertuples(index=False):
        lines.append(
            f"- X={row.target_x_pct}%: select {row.selected_region_count} regions; achieved {row.achieved_population_share_pct:.2f}% (overshoot {row.overshoot_percentage_points:.2f} percentage points)."
        )

    lines.extend(
        [
            "",
            "## Required design revision before rerunning",
            "",
            "1. Cluster connected coast segments or sea-basin units, not indivisible sovereign countries; attach national policy responsibility after the marine regions are formed.",
            "2. Keep overseas territories and spatially separated national coast segments distinct during clustering. This prevents a mother country from acting as a bridge across oceans.",
            "3. Constrain adjacency to the same sea basin and use a water-path/geographic-strait rule rather than straight-line coastal distance alone.",
            "4. Replace the maximize-p objective with a fixed/selected number of marine regions or another compactness/coherence objective; otherwise the algorithm is rewarded for producing singleton regions.",
            "5. Retain isolated island states in risk reporting and define a separate island-policy category instead of dropping them from the ranking.",
            "",
            "Only after the revised regions pass the map audit should the ranked cumulative-population curve be used to determine X.",
            "",
            "## Regions with fragmented country-level policy belts",
            "",
        ]
    )
    if fragmented.empty:
        lines.append("None.")
    else:
        for row in fragmented.itertuples(index=False):
            lines.append(
                f"- {row.region_id}: {row.policy_belt_country_components} components; largest contains {100*row.largest_policy_component_population_share:.1f}% of regional coastal population; members: {row.member_names}"
            )

    lines.extend(["", "## Regions with the widest internal city-SPRI dispersion", ""])
    for row in wide_iqr.itertuples(index=False):
        lines.append(
            f"- {row.region_id}: IQR={row.city_spri_iqr:.2f}, range={row.city_spri_min:.2f}–{row.city_spri_max:.2f}; members: {row.member_names}"
        )

    lines.extend(
        [
            "",
            "## Method constraint discovered during execution",
            "",
            "`spopt.region.MaxPHeuristic` accepts one additive threshold only. It cannot directly express the v12 requirement `population >= 2% OR coastline >= 5,000 km`. The formal implementation uses an auditable exact-OR extension with the same randomized grow/assign logic and connectivity-preserving local improvement. The research plan should not claim that the unmodified PySAL class implements the OR constraint.",
            "",
            "## Status",
            "",
            "This is a raw algorithmic result. No manual reassignment has been applied. `tables/adjacency_edges.csv`, `tables/territory_parent_assignment_audit.csv`, and the global map are the evidence base for the required manual geographic audit.",
        ]
    )
    (REPORT_DIR / "reality_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-geometry", action="store_true")
    parser.add_argument("--n-starts", type=int, default=N_STARTS)
    args = parser.parse_args()
    ensure_dirs()
    started = time.time()

    polygons, coasts = build_admin0_coast_cache(rebuild=args.rebuild_geometry)
    cities, _ = load_city_inputs()
    country, territory_audit = aggregate_country_attributes(cities)
    parent_polygons, parent_coasts = make_parent_geographies(
        polygons, coasts, set(country["iso3"])
    )
    country = country.merge(
        parent_coasts[["iso3", "coast_km", "source_geographies"]],
        on="iso3",
        how="left",
        validate="one_to_one",
    )
    print(
        f"[data] {len(cities)} cities -> {len(country)} parent-country units; global coastal population={country.coastal_population.sum():,.0f}",
        flush=True,
    )

    graph, edges = build_adjacency(country, parent_polygons, parent_coasts)
    print(
        f"[adjacency] {graph.number_of_edges()} edges; {nx.number_connected_components(graph)} graph components; {len(list(nx.isolates(graph)))} isolates",
        flush=True,
    )
    best_regions, runs, stability, excluded = solve_maxp_or(
        graph, country, args.n_starts
    )
    region_summary, membership, cutoff = canonicalize_regions(
        best_regions, country, cities, graph, edges
    )
    x_lookup = make_x_threshold_lookup(region_summary)

    territory_audit.to_csv(TABLE_DIR / "territory_parent_assignment_audit.csv", index=False, encoding="utf-8-sig")
    edges.to_csv(TABLE_DIR / "adjacency_edges.csv", index=False, encoding="utf-8-sig")
    runs.to_csv(TABLE_DIR / "maxp_100_start_results.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(TABLE_DIR / "maxp_solution_stability.csv", index=False, encoding="utf-8-sig")
    excluded.to_csv(TABLE_DIR / "isolated_and_infeasible_units.csv", index=False, encoding="utf-8-sig")
    membership.drop(columns=["polygon_geometry", "coast_geometry"], errors="ignore").to_csv(
        TABLE_DIR / "coastal_regions_country_membership.csv", index=False, encoding="utf-8-sig"
    )
    region_summary.to_csv(TABLE_DIR / "coastal_regions_summary.csv", index=False, encoding="utf-8-sig")
    cutoff.to_csv(TABLE_DIR / "region_ranking_cumulative_population.csv", index=False, encoding="utf-8-sig")
    x_lookup.to_csv(TABLE_DIR / "nominal_X_coverage_lookup.csv", index=False, encoding="utf-8-sig")

    plot_regions(parent_polygons, membership, excluded)
    plot_ranking(region_summary)
    write_report(
        country,
        graph,
        edges,
        region_summary,
        excluded,
        stability,
        territory_audit,
        x_lookup,
    )

    manifest = {
        "plan": str(PLAN),
        "run_timestamp": pd.Timestamp.now().isoformat(),
        "runtime_seconds": time.time() - started,
        "inputs": {
            "risk": str(RISK_FILE),
            "covariates": str(COVARIATE_FILE),
            "city_geometry": str(CITY_GPKG),
            "country_geometry_source": str(WORLD_ADMIN1),
            "country_metadata_source": str(WORLD_FINE),
            "ocean_facing_boundary_mask": str(NATURAL_EARTH_ADMIN0),
        },
        "parameters": {
            "year": 2025,
            "country_spri_aggregation": "city equal mean after oversized_flag exclusion",
            "region_spri_aggregation": "direct city equal mean after oversized_flag exclusion",
            "coastal_neighbor_distance_km": COAST_NEIGHBOR_KM,
            "population_threshold_share": POP_THRESHOLD_SHARE,
            "coastline_threshold_km": COAST_THRESHOLD_KM,
            "threshold_logic": "OR",
            "n_random_starts": args.n_starts,
            "seed_start": SEED,
            "geometry_simplification_m": SIMPLIFY_TOLERANCE_M,
            "geometry_union_grid_m": UNION_GRID_M,
            "adjacency_distance_method": "approximately 5-km sampled GADM ocean-facing coastline vertices; Euclidean chord distance",
        },
        "implementation_note": "Custom exact-OR Max-p heuristic because spopt MaxPHeuristic exposes one additive threshold only.",
        "outputs": [str(p.relative_to(FORMAL_DIR)) for p in FORMAL_DIR.rglob("*") if p.is_file() and "_deps" not in p.parts],
    }
    (FORMAL_DIR / "core_diagnostic_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[complete] regions={len(region_summary)}; excluded={len(excluded)}; runtime={time.time()-started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit("Run run_v12_formal_clustering.py for the formal result.")
