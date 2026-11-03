"""Run the formal v12 coastal regionalization used by policy simulations.

Overseas territories remain separate geographic units during clustering and
are mapped to their policy-responsible parent state after regionalization.
The exact-OR Max-p heuristic uses 200 deterministic-order random starts.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


ROOT = Path(r"E:\transfer justice")
OUT_DIR = ROOT / "policy_clustering_v12_formal"
TABLE_DIR = OUT_DIR / "tables"
FIGURE_DIR = OUT_DIR / "figures"
REPORT_DIR = OUT_DIR / "report"
sys.path.insert(0, str(OUT_DIR))

import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import maxp_clustering_core as base  # noqa: E402


N_STARTS = 200


def ensure_dirs() -> None:
    for path in (OUT_DIR, TABLE_DIR, FIGURE_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def aggregate_geo_units(
    cities: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate risk by original GADM/ISO geography, retaining parent mapping."""

    parent_country, _ = base.aggregate_country_attributes(cities)
    parent_names = parent_country.set_index("iso3")["country"].to_dict()
    parent_spri = parent_country.set_index("iso3")["country_spri"].to_dict()
    parent_population = parent_country.set_index("iso3")["coastal_population"].to_dict()

    valid = cities.loc[cities["oversized_flag"].fillna(0).eq(0)].copy()
    risk_agg = valid.groupby("source_iso3").agg(
        country_spri=("SPRI", "mean"),
        n_risk_nodes=("node_id", "nunique"),
        spri_median=("SPRI", "median"),
        spri_q25=("SPRI", lambda x: x.quantile(0.25)),
        spri_q75=("SPRI", lambda x: x.quantile(0.75)),
        spri_min=("SPRI", "min"),
        spri_max=("SPRI", "max"),
    )
    pop_agg = cities.groupby("source_iso3").agg(
        coastal_population=("S1_pop", "sum"),
        n_all_nodes=("node_id", "nunique"),
    )
    names = (
        cities[["source_iso3", "country"]]
        .drop_duplicates("source_iso3")
        .set_index("source_iso3")["country"]
    )
    units = risk_agg.join(pop_agg, how="outer").reset_index().rename(
        columns={"source_iso3": "iso3"}
    )
    units["country"] = units["iso3"].map(names)
    units["source_geographies"] = units["iso3"]
    units["policy_iso3"] = units["iso3"].replace(base.PARENT_ISO)
    units["policy_country"] = units["policy_iso3"].map(parent_names).fillna(
        units["policy_iso3"]
    )

    audit = units[
        [
            "iso3",
            "country",
            "policy_iso3",
            "policy_country",
            "n_all_nodes",
            "coastal_population",
        ]
    ].rename(columns={"iso3": "geo_iso3", "country": "geo_country"})
    audit["changed"] = audit["geo_iso3"].ne(audit["policy_iso3"])
    audit["political_review_required"] = audit["geo_iso3"].isin(
        base.POLITICAL_REVIEW
    )

    policy = pd.DataFrame(
        {
            "policy_iso3": sorted(parent_names),
        }
    )
    policy["policy_country"] = policy["policy_iso3"].map(parent_names)
    policy["policy_country_spri"] = policy["policy_iso3"].map(parent_spri)
    policy["policy_coastal_population"] = policy["policy_iso3"].map(
        parent_population
    )
    return units, audit, policy


def source_geographies(
    polygons: gpd.GeoDataFrame,
    coasts: gpd.GeoDataFrame,
    unit_isos: set[str],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    source_polygons = polygons.loc[polygons["iso3"].isin(unit_isos)].copy()
    source_coasts = coasts.loc[coasts["iso3"].isin(unit_isos)].copy()
    missing_polygon = sorted(unit_isos - set(source_polygons["iso3"]))
    missing_coast = sorted(unit_isos - set(source_coasts["iso3"]))
    if missing_polygon or missing_coast:
        raise RuntimeError(
            f"Missing source geography: polygons={missing_polygon}, coasts={missing_coast}"
        )
    source_polygons = source_polygons[["iso3", "geometry"]]
    source_coasts = source_coasts[["iso3", "coast_km", "geometry"]]
    source_coasts["source_geographies"] = source_coasts["iso3"]
    return source_polygons, source_coasts


def solve_maxp_or_with_feasible_isolates(
    graph: nx.Graph,
    units: pd.DataFrame,
    n_starts: int,
) -> tuple[list[set[str]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Exact-OR heuristic retaining size-feasible isolates as singleton regions."""

    lookup = units.set_index("iso3")
    pop_threshold = units["coastal_population"].sum() * base.POP_THRESHOLD_SHARE
    all_isolates = sorted(nx.isolates(graph))
    feasible_isolates = [
        iso
        for iso in all_isolates
        if base.region_feasible({iso}, lookup, pop_threshold)
    ]
    excluded_isolates = sorted(set(all_isolates) - set(feasible_isolates))

    cluster_graph = graph.copy()
    cluster_graph.remove_nodes_from(all_isolates)
    components = sorted(
        (set(component) for component in nx.connected_components(cluster_graph)),
        key=lambda component: (min(component), len(component)),
    )
    feasible_components: list[set[str]] = []
    infeasible_components: list[set[str]] = []
    for component in components:
        if base.region_feasible(component, lookup, pop_threshold):
            feasible_components.append(component)
        else:
            infeasible_components.append(component)
            cluster_graph.remove_nodes_from(component)

    excluded_rows = [
        {"iso3": iso, "reason": "no_neighbor_and_fails_both_size_thresholds", "component_id": None}
        for iso in excluded_isolates
    ]
    for component_id, component in enumerate(infeasible_components, start=1):
        for iso in sorted(component):
            excluded_rows.append(
                {
                    "iso3": iso,
                    "reason": "connected_component_fails_both_size_thresholds",
                    "component_id": component_id,
                }
            )

    fixed_singletons = [{iso} for iso in feasible_isolates]
    run_rows = []
    solutions: list[list[set[str]]] = []
    for run in range(n_starts):
        rng = np.random.default_rng(base.SEED + run)
        regions = [set(region) for region in fixed_singletons]
        for component in feasible_components:
            regions.extend(
                base.construct_component_regions(
                    component, cluster_graph, lookup, pop_threshold, rng
                )
            )
        objective = sum(base.region_wss(region, lookup) for region in regions)
        run_rows.append(
            {"run": run + 1, "seed": base.SEED + run, "p": len(regions), "wss": objective}
        )
        solutions.append(regions)
        if (run + 1) % 10 == 0:
            print(f"[max-p] completed {run + 1}/{n_starts} construction starts", flush=True)

    runs = pd.DataFrame(run_rows)
    best_p = int(runs["p"].max())
    best_idx = runs.loc[runs["p"].eq(best_p), "wss"].idxmin()
    variable_best = [
        set(region)
        for region in solutions[int(best_idx)]
        if not (len(region) == 1 and next(iter(region)) in feasible_isolates)
    ]
    improved_variable = base.improve_regions(
        variable_best, cluster_graph, lookup, pop_threshold
    )
    best_regions = [set(region) for region in fixed_singletons] + improved_variable
    improved_wss = sum(base.region_wss(region, lookup) for region in best_regions)

    ordered_nodes = sorted(set(cluster_graph.nodes()) | set(feasible_isolates))
    label_arrays = []
    for regions in solutions:
        membership = {
            node: region_id
            for region_id, region in enumerate(regions)
            for node in region
        }
        label_arrays.append([membership[node] for node in ordered_nodes])
    ari_values = []
    for left, right in base.combinations(range(len(label_arrays)), 2):
        ari_values.append(
            base.adjusted_rand_score(label_arrays[left], label_arrays[right])
        )
    stability = pd.DataFrame(
        [
            {
                "n_solutions": len(label_arrays),
                "n_pairs": len(ari_values),
                "ari_mean": float(np.mean(ari_values)),
                "ari_median": float(np.median(ari_values)),
                "ari_min": float(np.min(ari_values)),
                "ari_max": float(np.max(ari_values)),
                "best_p": best_p,
                "best_run": int(runs.loc[best_idx, "run"]),
                "best_construction_wss": float(runs.loc[best_idx, "wss"]),
                "best_improved_wss": float(improved_wss),
                "n_feasible_isolate_singletons": len(feasible_isolates),
            }
        ]
    )
    excluded = pd.DataFrame(excluded_rows)
    if not excluded.empty:
        excluded = excluded.merge(
            units[
                [
                    "iso3",
                    "country",
                    "country_spri",
                    "coastal_population",
                    "coast_km",
                    "policy_iso3",
                    "policy_country",
                ]
            ],
            on="iso3",
            how="left",
        )
    return best_regions, runs, stability, excluded


def add_policy_membership(
    membership: pd.DataFrame,
    region_summary: pd.DataFrame,
    audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = audit.set_index("geo_iso3")
    membership = membership.rename(columns={"iso3": "geo_iso3", "country": "geo_country"})
    membership["policy_iso3"] = membership["geo_iso3"].map(mapping["policy_iso3"])
    membership["policy_country"] = membership["geo_iso3"].map(
        mapping["policy_country"]
    )

    policy_rows = []
    for region_id, group in membership.groupby("region_id", sort=False):
        policy_pairs = (
            group[["policy_iso3", "policy_country"]]
            .drop_duplicates()
            .sort_values(["policy_iso3", "policy_country"])
        )
        policy_rows.append(
            {
                "region_id": region_id,
                "n_geo_units": group["geo_iso3"].nunique(),
                "n_policy_countries": policy_pairs["policy_iso3"].nunique(),
                "geo_members": ";".join(sorted(group["geo_iso3"].unique())),
                "geo_member_names": ";".join(sorted(group["geo_country"].unique())),
                "policy_members": ";".join(policy_pairs["policy_iso3"]),
                "policy_member_names": ";".join(policy_pairs["policy_country"]),
                "n_mapped_territory_units": int(
                    group["geo_iso3"].ne(group["policy_iso3"]).sum()
                ),
            }
        )
    policy_summary = pd.DataFrame(policy_rows)
    region_summary = region_summary.merge(
        policy_summary, on="region_id", how="left", validate="one_to_one"
    )
    region_summary = region_summary.rename(columns={"n_countries": "n_geo_units_original"})
    return membership, region_summary


def selected_policy_table(
    membership: pd.DataFrame,
    region_summary: pd.DataFrame,
    policy: pd.DataFrame,
    target: float,
    rank30: int,
) -> tuple[pd.DataFrame, int, float]:
    cutoff = (
        region_summary.loc[region_summary["cumulative_population_share"].ge(target)]
        .sort_values("rank")
        .iloc[0]
    )
    rank = int(cutoff["rank"])
    selected = membership.loc[membership["region_rank"].le(rank)].copy()
    rows = []
    for policy_iso3, group in selected.groupby("policy_iso3", sort=True):
        rows.append(
            {
                "policy_iso3": policy_iso3,
                "policy_country": group["policy_country"].iloc[0],
                "selected_regions": ";".join(
                    sorted(group["region_id"].unique(), key=lambda x: int(x[1:]))
                ),
                "selected_geo_units": ";".join(sorted(group["geo_iso3"].unique())),
                "first_region_rank": int(group["region_rank"].min()),
            }
        )
    result = pd.DataFrame(rows).merge(
        policy, on=["policy_iso3", "policy_country"], how="left"
    )
    result["selection_band"] = np.where(
        result["first_region_rank"].le(rank30),
        "top_30_core",
        "additional_30_to_50",
    )
    result = result.sort_values(
        ["first_region_rank", "policy_country_spri", "policy_country"],
        ascending=[True, False, True],
    )
    return result, rank, float(cutoff["cumulative_population_share"])


def plot_selected_units(
    polygons: gpd.GeoDataFrame,
    membership: pd.DataFrame,
    cutoff_rank: int,
    rank30: int,
    achieved_share: float,
    output: Path,
) -> None:
    selected = membership.loc[membership["region_rank"].le(cutoff_rank), [
        "geo_iso3", "region_rank"
    ]].drop_duplicates()
    plot = polygons.merge(selected, left_on="iso3", right_on="geo_iso3", how="left").to_crs(4326)
    plot["band"] = np.select(
        [plot["region_rank"].le(rank30), plot["region_rank"].le(cutoff_rank)],
        ["top30", "additional"],
        default="not_selected",
    )
    colors = {"not_selected": "#D9DEE4", "top30": "#2B6CB0", "additional": "#E6862A"}
    fig, ax = plt.subplots(figsize=(15.8, 8.3), facecolor="white")
    ax.set_facecolor("#F7FAFC")
    for band in ["not_selected", "top30", "additional"]:
        layer = plot.loc[plot["band"].eq(band)]
        if not layer.empty:
            layer.plot(ax=ax, color=colors[band], edgecolor="white", linewidth=0.28)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-58, 85)
    ax.set_axis_off()
    target = 30 if cutoff_rank == rank30 else 50
    ax.set_title(
        f"Optimized v12 selection at nominal {target}% coastal-population coverage",
        fontsize=17,
        pad=14,
    )
    handles = [mpatches.Patch(facecolor=colors["top30"], label="Top-30% core")]
    if cutoff_rank > rank30:
        handles.append(
            mpatches.Patch(facecolor=colors["additional"], label="Additional units to 50%")
        )
    handles.append(mpatches.Patch(facecolor=colors["not_selected"], label="Not selected"))
    ax.legend(handles=handles, loc="lower left", frameon=False, ncol=len(handles))
    ax.text(
        0.985,
        0.025,
        f"Actual cumulative coastal-population coverage: {100*achieved_share:.2f}%",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="#3F4A56",
    )
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(
    units: pd.DataFrame,
    graph: nx.Graph,
    region_summary: pd.DataFrame,
    excluded: pd.DataFrame,
    stability: pd.DataFrame,
    runs: pd.DataFrame,
    top30: pd.DataFrame,
    rank30: int,
    share30: float,
    top50: pd.DataFrame,
    rank50: int,
    share50: float,
) -> None:
    p_counts = runs.groupby("p").size().to_dict()
    fragmented = int(region_summary["policy_belt_country_components"].gt(1).sum())
    singleton = int(region_summary["n_geo_units"].eq(1).sum())
    mapped_regions = int(region_summary["n_mapped_territory_units"].gt(0).sum())

    lines = [
        "# V12 政策模拟正式聚类结果",
        "",
        "## 执行口径",
        "",
        "- 海外属地在聚类阶段保持为独立 GADM/ISO 地理单元，区划完成后再映射政策责任国。",
        "- 400 km 邻接、沿海人口 2% 或海岸线 5,000 km 的双门槛、城市等权 SPRI 与区域累计人口截断均未改变。海岸最短距离采用约 5 km 简化顶点的单位球弦长转大圆距离，以保持日期变更线两侧连续。",
        "- 精确 OR Max-p 启发式使用 200 个固定种子，遍历顺序确定化。",
        "",
        "## 主要结果",
        "",
        f"- 输入地理单元：{len(units)}；政策责任国：{units['policy_iso3'].nunique()}。",
        f"- 邻接边：{graph.number_of_edges()}；原始区域：{len(region_summary)}；孤立/不可行单元：{len(excluded)}。",
        f"- 最优 p：{int(stability.iloc[0]['best_p'])}；达到最优 p 的初始解：{int(runs['p'].eq(runs['p'].max()).sum())}/200。",
        f"- 平均 ARI：{stability.iloc[0]['ari_mean']:.3f}（中位数 {stability.iloc[0]['ari_median']:.3f}）。",
        f"- 单一地理单元区域：{singleton}；含映射属地单元的区域：{mapped_regions}；含多个 12 海里国别政策带分量的区域：{fragmented}。",
        f"- p 分布：{p_counts}。",
        "",
        "## 累计人口截断",
        "",
        f"- 前30%：纳入 {rank30} 个区域、{len(top30)} 个去重政策责任国，实际覆盖 {100*share30:.2f}%。",
        f"- 前50%：纳入 {rank50} 个区域、{len(top50)} 个去重政策责任国，实际覆盖 {100*share50:.2f}%。",
        "",
        "## 正式结果的使用边界",
        "",
        "该区划作为后续政策模拟的唯一聚类输入。Max-p 最大化区域数的性质未改变，400 km 图连通也不等同于 12 海里政策带几何无缝相接；正式政策解释仍须结合逐区域人工核实记录。",
    ]
    (REPORT_DIR / "formal_clustering_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    ensure_dirs()
    started = time.time()
    base.FIGURE_DIR = FIGURE_DIR

    polygons, coasts = base.build_admin0_coast_cache(rebuild=False)
    cities, _ = base.load_city_inputs()
    units, audit, policy = aggregate_geo_units(cities)
    source_polygons, source_coasts = source_geographies(
        polygons, coasts, set(units["iso3"])
    )
    units = units.merge(
        source_coasts[["iso3", "coast_km", "source_geographies"]],
        on="iso3",
        how="left",
        validate="one_to_one",
        suffixes=("", "_geometry"),
    )
    if "source_geographies_geometry" in units:
        units["source_geographies"] = units["source_geographies_geometry"]
        units = units.drop(columns=["source_geographies_geometry"])

    print(
        f"[data] {len(cities)} cities -> {len(units)} geographic units -> {units.policy_iso3.nunique()} policy states",
        flush=True,
    )
    graph, edges = base.build_adjacency(units, source_polygons, source_coasts)
    print(
        f"[adjacency] {graph.number_of_edges()} edges; {nx.number_connected_components(graph)} components; {len(list(nx.isolates(graph)))} isolates",
        flush=True,
    )
    regions, runs, stability, excluded = solve_maxp_or_with_feasible_isolates(
        graph, units, N_STARTS
    )

    cluster_cities = cities.copy()
    cluster_cities["parent_iso3"] = cluster_cities["source_iso3"]
    region_summary, membership, cutoff = base.canonicalize_regions(
        regions, units, cluster_cities, graph, edges
    )
    membership, region_summary = add_policy_membership(
        membership, region_summary, audit
    )
    x_lookup = base.make_x_threshold_lookup(region_summary)
    rank30 = int(
        region_summary.loc[region_summary["cumulative_population_share"].ge(0.30), "rank"].min()
    )
    top30, rank30, share30 = selected_policy_table(
        membership, region_summary, policy, 0.30, rank30
    )
    top50, rank50, share50 = selected_policy_table(
        membership, region_summary, policy, 0.50, rank30
    )

    audit.to_csv(TABLE_DIR / "geo_unit_policy_state_mapping.csv", index=False, encoding="utf-8-sig")
    edges.to_csv(TABLE_DIR / "adjacency_edges.csv", index=False, encoding="utf-8-sig")
    runs.to_csv(TABLE_DIR / "maxp_200_start_results.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(TABLE_DIR / "maxp_solution_stability.csv", index=False, encoding="utf-8-sig")
    excluded.to_csv(TABLE_DIR / "isolated_and_infeasible_geo_units.csv", index=False, encoding="utf-8-sig")
    membership.to_csv(TABLE_DIR / "coastal_regions_geo_membership.csv", index=False, encoding="utf-8-sig")
    region_summary.to_csv(TABLE_DIR / "coastal_regions_summary.csv", index=False, encoding="utf-8-sig")
    cutoff.to_csv(TABLE_DIR / "region_ranking_cumulative_population.csv", index=False, encoding="utf-8-sig")
    x_lookup.to_csv(TABLE_DIR / "nominal_X_coverage_lookup.csv", index=False, encoding="utf-8-sig")
    top30.to_csv(TABLE_DIR / "top30pct_policy_countries.csv", index=False, encoding="utf-8-sig")
    top50.to_csv(TABLE_DIR / "top50pct_policy_countries.csv", index=False, encoding="utf-8-sig")

    plot_membership = membership.rename(columns={"geo_iso3": "iso3"})
    base.plot_regions(source_polygons, plot_membership, excluded)
    (FIGURE_DIR / "v12_maxp_raw_regions.png").replace(
        FIGURE_DIR / "v12_formal_raw_regions.png"
    )
    base.plot_ranking(region_summary)
    (FIGURE_DIR / "v12_region_ranking_cumulative_population.png").replace(
        FIGURE_DIR / "v12_formal_region_ranking_cumulative_population.png"
    )
    plot_selected_units(
        polygons,
        membership,
        rank30,
        rank30,
        share30,
        FIGURE_DIR / "v12_formal_top30pct_geo_units.png",
    )
    plot_selected_units(
        polygons,
        membership,
        rank50,
        rank30,
        share50,
        FIGURE_DIR / "v12_formal_top50pct_geo_units.png",
    )
    write_report(
        units,
        graph,
        region_summary,
        excluded,
        stability,
        runs,
        top30,
        rank30,
        share30,
        top50,
        rank50,
        share50,
    )

    manifest = {
        "status": "formal_policy_simulation_input",
        "plan": str(ROOT / "research_plan_v12.md"),
        "run_timestamp": pd.Timestamp.now().isoformat(),
        "runtime_seconds": time.time() - started,
        "method": "overseas territories remain separate geographic units until post-cluster policy-state mapping",
        "inputs": {
            "risk": str(base.RISK_FILE),
            "covariates": str(base.COVARIATE_FILE),
            "city_geometry": str(base.CITY_GPKG),
            "country_geometry_cache": str(base.COUNTRY_CACHE),
        },
        "parameters": {
            "year": 2025,
            "n_random_starts": N_STARTS,
            "seed_start": base.SEED,
            "coastal_neighbor_distance_km": base.COAST_NEIGHBOR_KM,
            "coastal_distance_method": (
                "5km_simplified_coast_vertices_unit_sphere_"
                "chord_to_great_circle"
            ),
            "population_threshold_share": base.POP_THRESHOLD_SHARE,
            "coastline_threshold_km": base.COAST_THRESHOLD_KM,
            "threshold_logic": "OR",
            "traversal_order": "deterministic sorted order",
        },
        "result_summary": {
            "n_geographic_units": int(len(units)),
            "n_policy_states": int(units["policy_iso3"].nunique()),
            "n_regions": int(len(region_summary)),
            "n_excluded_geo_units": int(len(excluded)),
            "top30_region_count": int(rank30),
            "top30_policy_state_count": int(len(top30)),
            "top30_actual_population_share": float(share30),
            "top50_region_count": int(rank50),
            "top50_policy_state_count": int(len(top50)),
            "top50_actual_population_share": float(share50),
        },
    }
    (OUT_DIR / "formal_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[complete] regions={len(region_summary)}; excluded={len(excluded)}; "
        f"top30_states={len(top30)}; top50_states={len(top50)}; runtime={time.time()-started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
