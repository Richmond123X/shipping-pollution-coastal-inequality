# -*- coding: utf-8 -*-
"""绘制 V12 政策模拟正式独立 PNG（不输出 PDF、不拼接面板）。"""
from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from map_display_geometry import prepare_display_regions


ROOT = Path(r"E:\transfer justice\policy_simulation_v12_formal")
CLUSTER = Path(r"E:\transfer justice\policy_clustering_v12_formal")
TABLES = ROOT / "tables"
FIG = ROOT / "figures"
ZONE = ROOT / "inputs" / "formal_policy_zones_50pct.gpkg"
ADMIN = CLUSTER / "cache" / "admin0_coastlines_2km_v3.gpkg"
SELECT_GEO = TABLES / "selected_geo_units_50pct.csv"
SELECT_REG = TABLES / "selected_regions_50pct.csv"
RANK_ALL = CLUSTER / "tables" / "region_ranking_cumulative_population.csv"
MEMBERSHIP = CLUSTER / "tables" / "coastal_regions_geo_membership.csv"
ISOLATED = CLUSTER / "tables" / "isolated_and_infeasible_geo_units.csv"
REGION_NAMES = TABLES / "table_s14_region_names.csv"

INK = "#2B2B2B"
GRID = "#DFE3E8"
BLUE = "#3A6EB4"
TEAL = "#2B999F"
ORANGE = "#F99871"
RED = "#D5706B"
PURPLE = "#8775B4"
GREEN = "#59A376"
LIGHT_BLUE = "#4E89B5"
GREY = "#B7C1C9"
NETWORK_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "established_network",
    ["#312555", "#5A2D7A", "#923788", "#CF648C", "#F99871", "#FADBA7"],
)
REGION_MAP_COLORS = [
    "#F1F7C9", "#C7E9B4", "#7FCDBB",
    "#41B6C4", "#2C7FB8", "#253494",
]
REGION_CMAP = mpl.colors.ListedColormap(REGION_MAP_COLORS, name="regional_classes")
MAP_LABEL_POS = {
    "R01": (7, 8), "R02": (-20, 16), "R03": (48, 8), "R04": (24, -10),
    "R05": (-67, 17), "R06": (31, -35), "R07": (69, 27), "R08": (80, 17),
    "R09": (104, 12), "R10": (44, 25), "R11": (-50, -12), "R12": (39, 11),
    "R13": (31, 43), "R14": (-91, 10), "R15": (95, 25), "R16": (110, 20),
    "R17": (0, 31), "R18": (-77, -15), "R19": (125, -10), "R20": (-7, 43),
    "R21": (115, 4), "R22": (145, 43),
}
SCENARIO_COLORS = {
    "S1": BLUE, "S2": PURPLE, "S3(alpha*)": TEAL,
    "S4": GREY, "S5": RED, "S6": LIGHT_BLUE, "S7": GREEN,
}


def style() -> None:
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 10.5, "axes.labelsize": 11,
        "axes.titlesize": 11.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9.2, "axes.linewidth": 0.7, "axes.edgecolor": INK,
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.grid": False, "grid.color": GRID,
        "legend.frameon": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": 330, "savefig.bbox": "tight", "figure.dpi": 130,
    })


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=330, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def quantile_norm(values, n_classes: int = 6):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    bounds = np.quantile(arr, np.linspace(0, 1, n_classes + 1))
    for i in range(1, len(bounds)):
        if bounds[i] <= bounds[i - 1]:
            bounds[i] = np.nextafter(bounds[i - 1], np.inf)
    return mpl.colors.BoundaryNorm(bounds, REGION_CMAP.N, clip=True), bounds


def fig7a_map() -> None:
    land = gpd.read_file(ADMIN, layer="admin0_polygon").to_crs(4326)
    membership = pd.read_csv(MEMBERSHIP)[["geo_iso3", "region_id", "region_rank"]]
    plot = land.merge(membership, left_on="iso3", right_on="geo_iso3", how="left")
    selected_ids = set(pd.read_csv(SELECT_REG)["region_id"].astype(str))
    region_rank = pd.read_csv(RANK_ALL).set_index("region_id")
    region_poly = (plot.dropna(subset=["region_id"])
                   .dissolve(by="region_id", as_index=False))
    selected_poly = region_poly[region_poly["region_id"].isin(selected_ids)].copy()
    selected_poly["region_spri"] = selected_poly["region_id"].map(region_rank["region_spri"])
    selected_poly = prepare_display_regions(selected_poly)
    norm, bounds = quantile_norm(selected_poly["region_spri"])

    fig = plt.figure(figsize=(10.0, 5.7))
    ax = fig.add_axes([0.02, 0.13, 0.84, 0.76], projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="white", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#F1F3F5",
                   edgecolor="#C1C9CF", linewidth=0.20, zorder=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#CCD3D8",
                   linewidth=0.16, zorder=0.7)
    for row in selected_poly.itertuples():
        ax.add_geometries([row.display_fill], crs=ccrs.PlateCarree(),
                          facecolor=REGION_CMAP(norm(float(row.region_spri))),
                          edgecolor="none", zorder=2)
        ax.add_geometries([row.display_outline], crs=ccrs.PlateCarree(),
                          facecolor="none", edgecolor="#607079",
                          linewidth=0.34, zorder=2.4)
    isolated = pd.read_csv(ISOLATED)
    iso_pts = land[land["iso3"].isin(isolated["iso3"])].copy()
    if len(iso_pts):
        rp = iso_pts.geometry.representative_point()
        ax.scatter(rp.x, rp.y, marker="D", s=15, facecolor="none",
                   edgecolor=INK, linewidth=0.65, transform=ccrs.PlateCarree(), zorder=5)
    for r in selected_poly.itertuples():
        p = r.geometry.representative_point()
        tx, ty = MAP_LABEL_POS.get(r.region_id, (p.x, p.y))
        xp, yp = ax.projection.transform_point(p.x, p.y, ccrs.PlateCarree())
        xt, yt = ax.projection.transform_point(tx, ty, ccrs.PlateCarree())
        ax.annotate(
            f"{r.region_id}", xy=(xp, yp), xytext=(xt, yt),
            fontsize=7.6, fontweight="bold", ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                      edgecolor="none", alpha=0.82),
            arrowprops=dict(arrowstyle="-", color="#66717C", lw=0.42,
                            shrinkA=1.0, shrinkB=1.0),
        )
    ax.spines["geo"].set_edgecolor("#AEB7BE")
    ax.spines["geo"].set_linewidth(0.65)
    ax.set_title("Formal policy regions", fontsize=14.0, pad=10)
    ax.legend(handles=[
        Line2D([], [], marker="D", linestyle="none", markerfacecolor="none",
               markeredgecolor=INK, markersize=4.5, label="Infeasible isolated unit"),
    ], loc="lower left", bbox_to_anchor=(0.01, 0.005), frameon=False)
    cax = fig.add_axes([0.885, 0.27, 0.020, 0.46])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=REGION_CMAP), cax=cax,
                      boundaries=bounds, ticks=bounds, spacing="proportional")
    cb.set_label("Regional SPRI", fontsize=10.5)
    cb.ax.tick_params(labelsize=9.2)
    fig.text(0.50, 0.045,
             "22 selected regions   ·   75 policy-responsible states   ·   54.30% of global coastal population",
             ha="center", fontsize=10.2, color="#5E676D")
    save(fig, "fig7a_formal_policy_treatment_map.png")


def fig7a2_ranking() -> None:
    d = pd.read_csv(SELECT_REG).sort_values("rank", ascending=False)
    names = pd.read_csv(REGION_NAMES)[["region_id", "region_name_en"]]
    d = d.merge(names, on="region_id", how="left", validate="1:1")
    d["region_label"] = d["region_id"] + "  " + d["region_name_en"]
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8.8, 8.4))
    ax.hlines(y, d["city_spri_q25"], d["city_spri_q75"], color="#AEB7C0", lw=2.2)
    ax.scatter(d["region_spri"], y, s=36, color=BLUE, edgecolor="white", lw=0.45, zorder=3)
    ax.set_yticks(y, d["region_label"])
    ax.tick_params(axis="y", labelsize=7.6)
    ax.set_xlabel("City-equal regional SPRI")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    despine(ax)
    ax.text(0.99, 0.01, "Horizontal bars: within-region city IQR",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#66717C")
    fig.subplots_adjust(left=0.29, right=0.98, bottom=0.09, top=0.98)
    save(fig, "fig7a2_selected_region_risk_ranking.png")


def fig7a3_cumulative() -> None:
    d = pd.read_csv(RANK_ALL).sort_values("rank")
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    ax.plot(d["rank"], 100 * d["cumulative_population_share"], color=BLUE,
            lw=2.2, marker="o", ms=3.2)
    ax.axhline(50, color=RED, lw=1.0, ls="--")
    ax.axvline(22, color=INK, lw=0.9, ls=":")
    y22 = float(d.loc[d["rank"] == 22, "cumulative_population_share"].iloc[0]) * 100
    ax.scatter([22], [y22], color=ORANGE, s=58, edgecolor="white", zorder=4)
    ax.annotate(f"22 regions\n{y22:.2f}%", xy=(22, y22), xytext=(27, y22 - 10),
                arrowprops=dict(arrowstyle="-", color="#66717C", lw=0.8), fontsize=9.5)
    ax.set_xlabel("Number of regions included in descending SPRI order")
    ax.set_ylabel("Cumulative share of global coastal population (%)")
    ax.set_xlim(1, int(d["rank"].max()))
    ax.set_ylim(0, 102)
    ax.grid(axis="y", color=GRID, lw=0.6)
    despine(ax)
    save(fig, "fig7a3_cumulative_population_selection.png")


def fig7b_ops_scan() -> None:
    d = pd.read_csv(TABLES / "ops_alpha_scan.csv")
    k = pd.read_csv(TABLES / "ops_knee_results.csv")
    fig, ax = plt.subplots(figsize=(9.4, 5.9))
    x = 100 * d["alpha"]
    ax.plot(x, d["gross_weighted_reduction_g"] / 1e9, color=BLUE, marker="o",
            lw=2.2, label="Gross ship-side reduction")
    if d["net_weighted_reduction_g"].notna().any():
        pm_included = (
            "net_composite_includes_pm" in d
            and d["net_composite_includes_pm"].fillna(False).all()
        )
        if not pm_included and "net_comparable_gross_reduction_g" in d:
            ax.plot(
                x, d["net_comparable_gross_reduction_g"] / 1e9,
                color="#7CA7A9", marker="none", ls="--", lw=1.25,
                label="Gross on SO$_2$+NOx comparable basis",
            )
        ax.plot(x, d["net_weighted_reduction_g"] / 1e9, color=TEAL, marker="s",
                lw=1.8, label=(
                    "Net reduction after grid emissions"
                    if pm_included else "Net reduction (SO$_2$+NOx basis; PM unavailable)"
                ))
    ax2 = ax.twinx()
    ax2.plot(x, d["exposure_reduction"] / 1e12, color=ORANGE, marker="^",
             lw=1.8, label="Population-weighted exposure reduction")
    kg = k[k["criterion"] == "gross_weighted_reduction_g"]
    selected_col = "selected_alpha" if "selected_alpha" in kg else "kneedle_alpha"
    if len(kg) and np.isfinite(kg[selected_col].iloc[0]):
        a = 100 * float(kg[selected_col].iloc[0])
        ax.axvline(a, color=RED, ls="--", lw=1.0)
        ax.text(a + 1.2, ax.get_ylim()[1] * 0.91, r"$\alpha^*$", color=RED)
    elif len(kg):
        ax.text(
            0.02, 0.96, f"No unique knee; reported: {kg['reported'].iloc[0]}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9, color=RED,
        )
    ax.set_xlabel("Shore-power use among eligible calls (%)")
    ax.set_ylabel("Weighted emission reduction (kt-equivalent)")
    ax2.set_ylabel("Exposure reduction proxy (×10$^{12}$)")
    ax.set_xticks(x)
    ax.grid(axis="y", color=GRID, lw=0.6)
    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(handles + h2, labels + l2, loc="lower right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    save(fig, "fig7b_ops_use_scan_and_knee.png")


def fig7c_efficiency_scatter() -> None:
    d = pd.read_csv(TABLES / "table_6_policy_scenario_summary.csv")
    d = d[d["row_type"] == "scenario"].copy()
    d["emission_pct"] = 100 * d["global_weighted_reduction_share"]
    d["exposure_pct"] = 100 * d["delta_pwe"] / d["global_baseline_pwe"]
    d["amplification"] = d["exposure_pct"] / d["emission_pct"]
    d["vulnerable_scale"] = d["vulnerable_benefit_absolute"] / 1e14
    norm = mpl.colors.Normalize(
        vmin=np.floor(d["amplification"].min()),
        vmax=np.ceil(d["amplification"].max()),
    )
    max_vuln = max(float(d["vulnerable_scale"].max()), 1e-9)
    fig, ax = plt.subplots(figsize=(8.8, 5.7))

    # Iso-amplification rays turn the sparse point cloud into an interpretable
    # policy-efficiency frontier without inventing additional observations.
    xmax, ymax = 2.85, 18.5
    for ratio in [5, 7, 9]:
        x_end = min(xmax, ymax / ratio)
        xx = np.linspace(0, x_end, 80)
        ax.plot(xx, ratio * xx, color="#C9D0D5", lw=0.8,
                ls=(0, (3, 3)), zorder=0)
        ax.text(x_end - 0.03, ratio * x_end - 0.35, f"{ratio}×",
                ha="right", va="top", fontsize=8.8, color="#7A858C")

    label_offsets = {
        "S1": (7, -13), "S2": (7, 6), "S3(alpha*)": (7, -13),
        "S4": (-30, 8), "S5": (-32, -17), "S6": (7, 6), "S7": (8, 5),
    }
    for r in d.itertuples():
        size = 70 + 255 * np.sqrt(float(r.vulnerable_scale) / max_vuln)
        marker = "*" if r.scenario == "S7" else ("D" if r.scenario in {"S4", "S5", "S6"} else "o")
        ax.scatter(r.emission_pct, r.exposure_pct, s=size, marker=marker,
                   color=NETWORK_CMAP(norm(r.amplification)),
                   edgecolor="white", lw=0.85, zorder=3)
        ox, oy = label_offsets.get(r.scenario, (6, 5))
        ax.annotate(f"{r.scenario}\n{r.amplification:.1f}×",
                    (r.emission_pct, r.exposure_pct), xytext=(ox, oy),
                    textcoords="offset points", fontsize=9.0,
                    fontweight="bold" if r.scenario == "S7" else "normal")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Global composite-emission reduction (%)")
    ax.set_ylabel("Global population-exposure reduction (%)")
    ax.set_title("Exposure amplification and vulnerable-population scale of policy scenarios",
                 pad=10)
    ax.grid(color=GRID, lw=0.6)
    despine(ax)
    cax = fig.add_axes([0.88, 0.27, 0.018, 0.44])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=NETWORK_CMAP), cax=cax)
    cb.set_label("Exposure / emission amplification", fontsize=10.2)
    cb.ax.tick_params(labelsize=9.0)
    size_handles = []
    for value in [0.5, 1.5, 2.5]:
        size = 70 + 255 * np.sqrt(value / max_vuln)
        size_handles.append(Line2D([], [], marker="o", linestyle="none",
                                   markerfacecolor="#FADBA7", markeredgecolor="white",
                                   markersize=np.sqrt(size), label=f"{value:.1f}"))
    ax.legend(handles=size_handles, title=r"Vulnerable benefit ($10^{14}$ PWE-proxy)",
              loc="upper left", ncol=3, columnspacing=0.8, handletextpad=0.2,
              fontsize=8.4, title_fontsize=8.8)
    fig.subplots_adjust(left=0.13, right=0.85, bottom=0.15, top=0.88)
    save(fig, "fig7c_emission_exposure_efficiency_scatter.png")


def fig7d_distribution() -> None:
    g = pd.read_csv(TABLES / "table_s15_quartile_effects.csv")
    g = g[g["grouping"] == "gdp_quartile"].copy()
    t6 = pd.read_csv(TABLES / "table_6_policy_scenario_summary.csv")
    order = t6.loc[t6["row_type"] == "scenario", "source_scenario"].tolist()
    labels = t6.loc[t6["row_type"] == "scenario", ["source_scenario", "scenario"]]
    label_map = dict(zip(labels["source_scenario"], labels["scenario"]))
    g = g[g["scenario"].isin(order)]
    quartiles = ["Q1", "Q2", "Q3", "Q4"]
    width = 0.1
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(10.2, 5.9))
    for j, sc in enumerate(order):
        z = g[g["scenario"] == sc].set_index("quartile").reindex(quartiles)
        ax.bar(x + (j - (len(order)-1)/2) * width, 100 * z["reduction_rate"],
               width=width, color=SCENARIO_COLORS.get(label_map.get(sc, sc), "#82909D"),
               label=label_map.get(sc, sc))
    ax.set_xticks(x, ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    ax.set_ylabel("Population-weighted exposure reduction rate (%)")
    ax.set_xlabel("2019 city grid-cell GDP per capita quartile")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(ncol=min(4, len(order)), loc="upper center", bbox_to_anchor=(0.5, -0.14))
    despine(ax)
    save(fig, "fig7d1_gdp_quartile_reduction_rates.png")

    fig, ax = plt.subplots(figsize=(9.6, 5.9))
    left = np.zeros(len(order))
    qcolors = ["#DCE8F5", "#A9C4E2", "#6D99C8", "#315F99"]
    for q, color in zip(quartiles, qcolors):
        vals = []
        for sc in order:
            z = g[(g["scenario"] == sc) & (g["quartile"] == q)]
            vals.append(float(z["distribution_share"].iloc[0]) if len(z) else 0.0)
        vals = 100 * np.asarray(vals)
        ax.barh(np.arange(len(order)), vals, left=left, color=color, label=q)
        left += vals
    ax.set_yticks(np.arange(len(order)), [label_map.get(sc, sc) for sc in order])
    ax.set_xlabel("Share of total exposure reduction (%)")
    ax.set_xlim(0, 100)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.18))
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "fig7d2_gdp_quartile_reduction_distribution.png")


def main() -> None:
    style()
    fig7a_map()
    fig7a2_ranking()
    fig7a3_cumulative()
    # 情景结果生成后才画 7b--7d；前三幅可先独立复核政策范围。
    if (TABLES / "ops_alpha_scan.csv").is_file():
        fig7b_ops_scan()
    if (TABLES / "table_6_policy_scenario_summary.csv").is_file():
        fig7c_efficiency_scatter()
    if (TABLES / "table_s15_quartile_effects.csv").is_file():
        fig7d_distribution()
    # The former country × pollutant × mode heatmap is intentionally retired.
    # Its spatial replacement is generated by 06a_make_overlap_figures.py as
    # figS6_s7_policy_unit_pollutant_mode_map.png.
    print(f"[complete] figures={FIG}")


if __name__ == "__main__":
    main()
