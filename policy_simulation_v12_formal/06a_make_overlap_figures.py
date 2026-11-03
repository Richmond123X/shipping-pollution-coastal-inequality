# -*- coding: utf-8 -*-
"""Build map-based supplementary policy-zone diagnostics (Figures S5-S6)."""
from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(r"E:\transfer justice\policy_simulation_v12_formal")
ZONE = ROOT / "inputs" / "formal_policy_zones_50pct.gpkg"
TABLE = ROOT / "tables" / "figure_s5_policy_zone_pairwise_overlap_data.csv"
S6_DATA = ROOT / "tables" / "figure_s6_s7_country_pollutant_mode_data.csv"
FIG = ROOT / "figures"

INK = "#2B2B2B"
MUTED = "#747E86"
LAND = "#F1F3F5"
BORDER = "#B7C0C8"
BLUE = "#3A6EB4"
LIGHT_BLUE = "#8EBEDA"
CORAL = "#D5706B"
PM = "#F6C98C"
NOX = "#9CD1B1"
POLLUTANT_COLORS = {"SO2": LIGHT_BLUE, "PM": PM, "NOx": NOX}
MODE_MARKERS = {"Berth": "o", "Sea": "D", "Manoeuvring": "^", "Anchored": "s"}
DPI = 330
SIZE = (9.0, 5.7)
NETWORK_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "established_network",
    ["#312555", "#5A2D7A", "#923788", "#CF648C", "#F99871", "#FADBA7"],
)


def style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10.5,
        "axes.titlesize": 14,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "text.color": INK,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def world_axis(fig: plt.Figure, rect=(0.02, 0.20, 0.96, 0.70)):
    ax = fig.add_axes(rect, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor=LAND, edgecolor=BORDER, linewidth=0.26, zorder=0)
    ax.add_feature(cfeature.BORDERS, edgecolor="#CBD2D7", linewidth=0.22, zorder=1)
    ax.coastlines(color=BORDER, linewidth=0.30, zorder=1)
    ax.spines["geo"].set_edgecolor(BORDER)
    ax.spines["geo"].set_linewidth(0.55)
    return ax


def panel_tag(fig: plt.Figure, tag: str) -> None:
    fig.text(0.018, 0.965, tag, ha="left", va="top", fontsize=15.5, fontweight="bold")


def horizontal_colorbar(fig: plt.Figure, cmap, norm, label: str,
                        rect=(0.32, 0.105, 0.40, 0.025)):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal")
    cb.set_label(label, labelpad=4)
    cb.outline.set_linewidth(0.7)
    return cb


def policy_units() -> gpd.GeoDataFrame:
    units = gpd.read_file(ZONE, layer="policy_units").copy()
    # In the formal GPKG, zone_id is already the unique Rxx__ISO3 policy-unit key.
    units["unit_id"] = units["zone_id"].astype(str)
    return units.sort_values(["region_rank", "policy_iso3"]).reset_index(drop=True)


def compute_overlap() -> tuple[gpd.GeoDataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    units_ll = policy_units()
    units = units_ll.to_crs(6933)
    units["area_km2"] = units.geometry.area / 1e6
    rows: list[dict] = []
    geoms = []
    for i in range(len(units)):
        a = units.iloc[i]
        aminx, aminy, amaxx, amaxy = a.geometry.bounds
        for j in range(i + 1, len(units)):
            b = units.iloc[j]
            bminx, bminy, bmaxx, bmaxy = b.geometry.bounds
            if amaxx < bminx or bmaxx < aminx or amaxy < bminy or bmaxy < aminy:
                continue
            inter = a.geometry.intersection(b.geometry)
            area = float(inter.area / 1e6) if not inter.is_empty else 0.0
            if area <= 1e-9:
                continue
            rows.append({
                "unit_a": a.unit_id, "region_a": a.region_id,
                "policy_iso3_a": a.policy_iso3, "policy_country_a": a.policy_country,
                "unit_b": b.unit_id, "region_b": b.region_id,
                "policy_iso3_b": b.policy_iso3, "policy_country_b": b.policy_country,
                "area_a_km2": float(a.area_km2), "area_b_km2": float(b.area_km2),
                "intersection_km2": area,
                "share_of_a": area / float(a.area_km2),
                "share_of_b": area / float(b.area_km2),
                "share_of_smaller": area / min(float(a.area_km2), float(b.area_km2)),
                "same_selected_region": bool(a.region_id == b.region_id),
            })
            geoms.append(inter)
    raw_pairs = pd.DataFrame(rows)
    intersections = gpd.GeoDataFrame(raw_pairs.copy(), geometry=geoms, crs=6933)
    intersections = intersections.sort_values("intersection_km2", ascending=False).reset_index(drop=True)
    pairs = pd.DataFrame(intersections.drop(columns="geometry")).reset_index(drop=True)
    intersections = intersections.to_crs(4326)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(TABLE, index=False, encoding="utf-8-sig")
    return units_ll, pairs, intersections


def representative_points(units: gpd.GeoDataFrame) -> pd.DataFrame:
    projected = units.to_crs(6933)
    points = gpd.GeoDataFrame(
        units[["unit_id", "region_id", "policy_iso3", "policy_country"]].copy(),
        geometry=projected.geometry.representative_point(), crs=6933).to_crs(4326)
    points["lon"] = points.geometry.x
    points["lat"] = points.geometry.y
    return points.drop(columns="geometry")


def add_policy_outlines(ax, units: gpd.GeoDataFrame, color="#6F808A", lw=0.50, zorder=2):
    ax.add_geometries(units.geometry, crs=ccrs.PlateCarree(), facecolor="none",
                      edgecolor=color, linewidth=lw, zorder=zorder)


def draw_s5a(units: gpd.GeoDataFrame, pairs: pd.DataFrame,
             intersections: gpd.GeoDataFrame) -> None:
    fig = plt.figure(figsize=SIZE)
    ax = world_axis(fig)
    add_policy_outlines(ax, units, color="#70828B", lw=0.44, zorder=2)
    values = np.log10(1 + intersections["intersection_km2"].to_numpy(float))
    cmap = NETWORK_CMAP
    norm = mpl.colors.Normalize(vmin=0, vmax=max(float(values.max()), 1.0))
    for geom, value in zip(intersections.geometry, values):
        ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                          facecolor=cmap(norm(value)), edgecolor=CORAL,
                          linewidth=0.25, alpha=0.88, zorder=3)

    cent = intersections.to_crs(6933)
    cent = gpd.GeoDataFrame(intersections.drop(columns="geometry"),
                            geometry=cent.geometry.representative_point(), crs=6933).to_crs(4326)
    top = cent.nlargest(6, "intersection_km2").reset_index(drop=True)
    sizes = 13 + 13 * np.sqrt(np.clip(cent["intersection_km2"].to_numpy(float), 0, None) /
                              max(float(cent["intersection_km2"].max()), 1e-9))
    ax.scatter(cent.geometry.x, cent.geometry.y, s=sizes,
               color=[cmap(norm(v)) for v in values], edgecolor="white", linewidth=0.38,
               transform=ccrs.PlateCarree(), zorder=4)
    ranked_lines = ["Largest intersections"]
    for number, r in top.iterrows():
        rank = number + 1
        ax.text(r.geometry.x, r.geometry.y, str(rank), transform=ccrs.PlateCarree(),
                ha="center", va="center", fontsize=7.4, fontweight="bold", color=INK,
                bbox=dict(boxstyle="circle,pad=0.16", facecolor="white",
                          edgecolor=INK, linewidth=0.55), zorder=6)
        ranked_lines.append(
            f"{rank}  {r.policy_iso3_a}-{r.policy_iso3_b}   {r.intersection_km2:,.0f} km²")
    ax.text(0.055, 0.75, "\n".join(ranked_lines), transform=ax.transAxes,
            ha="left", va="top", fontsize=8.4, linespacing=1.35,
            bbox=dict(facecolor="white", edgecolor="#C9D0D5", linewidth=0.55,
                      alpha=0.88, pad=4.0), zorder=7)
    fig.suptitle("Spatial extent and magnitude of pairwise policy-zone intersections",
                 x=0.52, y=0.94, fontsize=14.0)
    panel_tag(fig, "(S5a)")
    horizontal_colorbar(fig, cmap, norm, r"Intersection area, log$_{10}$(1 + km$^2$)")
    fig.text(0.985, 0.018, f"Non-zero intersecting pairs: {len(pairs):,}",
             ha="right", fontsize=9.2, color=MUTED)
    fig.savefig(FIG / "figS5a_policy_zone_intersection_area_map.png", dpi=DPI,
                facecolor="white")
    plt.close(fig)


def unit_overlap_metrics(units: gpd.GeoDataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    records = []
    for unit in units["unit_id"]:
        a = pairs.loc[pairs["unit_a"] == unit, ["share_of_a", "unit_b", "intersection_km2"]].copy()
        a.columns = ["share", "partner", "intersection_km2"]
        b = pairs.loc[pairs["unit_b"] == unit, ["share_of_b", "unit_a", "intersection_km2"]].copy()
        b.columns = ["share", "partner", "intersection_km2"]
        z = pd.concat([a, b], ignore_index=True)
        if z.empty:
            records.append({"unit_id": unit, "max_overlap_share": 0.0,
                            "main_partner": "", "n_partners": 0})
        else:
            row = z.loc[z["share"].idxmax()]
            records.append({"unit_id": unit, "max_overlap_share": 100 * float(row["share"]),
                            "main_partner": row["partner"], "n_partners": len(z)})
    return pd.DataFrame(records)


def draw_s5b(units: gpd.GeoDataFrame, pairs: pd.DataFrame) -> None:
    metrics = unit_overlap_metrics(units, pairs)
    plot = units.merge(metrics, on="unit_id", how="left")
    points = representative_points(units).set_index("unit_id")
    cmap = NETWORK_CMAP
    vmax = max(20.0, float(np.ceil(plot["max_overlap_share"].quantile(0.98) / 5) * 5))
    norm = mpl.colors.Normalize(0, vmax)
    fig = plt.figure(figsize=SIZE)
    ax = world_axis(fig)
    for _, r in plot.iterrows():
        ax.add_geometries([r.geometry], crs=ccrs.PlateCarree(),
                          facecolor=cmap(norm(r.max_overlap_share)),
                          edgecolor="#6F808A", linewidth=0.42, alpha=0.94, zorder=2)
    link_data = pairs.nlargest(15, "share_of_smaller")
    for r in link_data.itertuples():
        if r.unit_a not in points.index or r.unit_b not in points.index:
            continue
        pa, pb = points.loc[r.unit_a], points.loc[r.unit_b]
        ax.plot([pa.lon, pb.lon], [pa.lat, pb.lat], transform=ccrs.Geodetic(),
                color=CORAL, alpha=0.52, linewidth=0.45 + 2.1 * r.share_of_smaller,
                zorder=3)
    top = plot.nlargest(7, "max_overlap_share").merge(
        points.reset_index()[["unit_id", "lon", "lat"]], on="unit_id")
    for _, r in top.iterrows():
        ax.scatter(r.lon, r.lat, s=27, color=CORAL, edgecolor="white", linewidth=0.55,
                   transform=ccrs.PlateCarree(), zorder=4)
        ax.annotate(f"{r.policy_iso3}  {r.max_overlap_share:.1f}%",
                    xy=(r.lon, r.lat), xytext=(4, 5), textcoords="offset points",
                    transform=ccrs.PlateCarree(), fontsize=8.3, fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.7),
                    zorder=5)
    fig.suptitle("Policy-unit overlap intensity and principal spatial connections",
                 x=0.52, y=0.94, fontsize=14.0)
    panel_tag(fig, "(S5b)")
    horizontal_colorbar(fig, cmap, norm,
                        "Maximum intersection as share of own policy-unit area (%)")
    fig.savefig(FIG / "figS5b_policy_zone_overlap_share_map.png", dpi=DPI,
                facecolor="white")
    plt.close(fig)


def s6_summary() -> pd.DataFrame:
    raw = pd.read_csv(S6_DATA)
    total = raw.groupby("unit_id", as_index=False)["reduction_g"].sum()
    total["total_kt_eq"] = total["reduction_g"] / 1e9
    pollutant = raw.groupby(["unit_id", "Pollutant"], as_index=False)["reduction_g"].sum()
    pollutant = pollutant.loc[pollutant.groupby("unit_id")["reduction_g"].idxmax(),
                              ["unit_id", "Pollutant"]].rename(columns={"Pollutant": "dominant_pollutant"})
    mode = raw.groupby(["unit_id", "Mode"], as_index=False)["reduction_g"].sum()
    mode = mode.loc[mode.groupby("unit_id")["reduction_g"].idxmax(),
                    ["unit_id", "Mode"]].rename(columns={"Mode": "dominant_mode"})
    return total.merge(pollutant, on="unit_id").merge(mode, on="unit_id")


def draw_s6(units: gpd.GeoDataFrame) -> None:
    summary = s6_summary()
    points = representative_points(units)
    plot = units.merge(summary, on="unit_id", how="left").merge(points, on=[
        "unit_id", "region_id", "policy_iso3", "policy_country"], how="left")
    plot["total_kt_eq"] = plot["total_kt_eq"].fillna(0)
    values = np.log10(1 + plot["total_kt_eq"].to_numpy(float))
    cmap = NETWORK_CMAP
    norm = mpl.colors.Normalize(0, max(float(values.max()), 1.0))
    fig = plt.figure(figsize=SIZE)
    ax = world_axis(fig)
    for _, r in plot.iterrows():
        ax.add_geometries([r.geometry], crs=ccrs.PlateCarree(),
                          facecolor=cmap(norm(np.log10(1 + r.total_kt_eq))),
                          edgecolor="#6F808A", linewidth=0.45, alpha=0.94, zorder=2)

    for mode, marker in MODE_MARKERS.items():
        for pollutant, color in POLLUTANT_COLORS.items():
            z = plot[(plot["dominant_mode"] == mode) &
                     (plot["dominant_pollutant"] == pollutant)]
            if z.empty:
                continue
            ax.scatter(z["lon"], z["lat"], s=42, marker=marker, color=color,
                       edgecolor=INK, linewidth=0.55, alpha=0.96,
                       transform=ccrs.PlateCarree(), zorder=4)
    top = plot.nlargest(9, "total_kt_eq")
    label_offsets = {
        "IDN": (6, -12), "JPN": (6, -11), "MYS": (-8, 10),
        "TUR": (6, 8), "BRA": (-6, -12), "IND": (-8, -13),
        "ZAF": (6, -12), "VNM": (-8, 9), "ESP": (6, 7),
    }
    for _, r in top.iterrows():
        ox, oy = label_offsets.get(r.policy_iso3, (6, 7))
        ax.annotate(f"{r.policy_iso3}  {r.total_kt_eq:.1f}", xy=(r.lon, r.lat),
                    xytext=(ox, oy), textcoords="offset points",
                    transform=ccrs.PlateCarree(), ha="left" if ox > 0 else "right",
                    fontsize=8.2, fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.7),
                    zorder=5)
    pollutant_handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=7,
                                 markerfacecolor=POLLUTANT_COLORS[p], markeredgecolor=INK,
                                 label=p) for p in ["SO2", "PM", "NOx"]]
    mode_handles = [Line2D([0], [0], marker=MODE_MARKERS[m], linestyle="none", markersize=7,
                           markerfacecolor="white", markeredgecolor=INK, label=m)
                    for m in ["Berth", "Sea", "Manoeuvring", "Anchored"]]
    leg1 = ax.legend(handles=pollutant_handles, title="Dominant pollutant",
                     loc="lower left", bbox_to_anchor=(0.015, 0.025), ncol=3,
                     frameon=False, handletextpad=0.35, columnspacing=0.8)
    ax.add_artist(leg1)
    ax.legend(handles=mode_handles, title="Dominant operating mode",
              loc="lower right", bbox_to_anchor=(0.99, 0.025), ncol=2,
              frameon=False, handletextpad=0.35, columnspacing=0.8)
    fig.suptitle("Spatial composition of S7 reductions across policy units",
                 x=0.52, y=0.94, fontsize=14.0)
    panel_tag(fig, "(S6)")
    horizontal_colorbar(fig, cmap, norm, r"Total S7 reduction, log$_{10}$(1 + kt-equivalent)")
    fig.savefig(FIG / "figS6_s7_policy_unit_pollutant_mode_map.png", dpi=DPI,
                facecolor="white")
    plt.close(fig)


def main() -> None:
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    units, pairs, intersections = compute_overlap()
    draw_s5a(units, pairs, intersections)
    draw_s5b(units, pairs)
    draw_s6(units)
    print(f"[complete] units={len(units)} nonzero_pairs={len(pairs)} map_figures=3", flush=True)


if __name__ == "__main__":
    main()
