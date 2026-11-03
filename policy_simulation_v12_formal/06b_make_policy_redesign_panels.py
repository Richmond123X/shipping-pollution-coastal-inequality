# -*- coding: utf-8 -*-
"""Draw formal policy panels in the approved v10 visual language."""
from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

from map_display_geometry import prepare_display_regions


ROOT = Path(r"E:\transfer justice\policy_simulation_v12_formal")
CLUSTER = Path(r"E:\transfer justice\policy_clustering_v12_formal")
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
ADMIN = CLUSTER / "cache" / "admin0_coastlines_2km_v3.gpkg"
MEMBERSHIP = CLUSTER / "tables" / "coastal_regions_geo_membership.csv"
SELECTED_REGIONS = TABLES / "selected_regions_50pct.csv"
CITY_EFFECT_PANEL = (
    ROOT / "intermediate" / "scenarios_2024" / "city_scenario_effect_panel.parquet"
)
ALIGNMENT_TABLE = TABLES / "table_s19_region_risk_benefit_alignment.csv"

PANEL_SIZE = (8.8, 5.7)
DPI = 330

# Established-panel visual family.
INK = "#2B2B2B"
MUTED = "#747E86"
GRID = "#D9E0E5"
LAND = "#F1F3F5"
OCEAN = "#FFFFFF"
BLUE = "#3A6EB4"
LIGHT_BLUE = "#4E89B5"
GREEN = "#59A376"
CORAL = "#D5706B"
PURPLE = "#8775B4"
TEAL = "#2B999F"
GREY = "#B8C0C8"
PALE_GREY = "#E9EDF0"

# Pollutant colors reproduce established Figure 2a.
SO2 = "#8EBEDA"
PM = "#F6C98C"
NOX = "#9CD1B1"
# Discrete ColorBrewer YlGnBu classes. Quantile boundaries put a comparable
# number of the 22 selected regions in each class, avoiding a single dominant
# hue when the underlying distribution is strongly right-skewed.
REGION_MAP_COLORS = [
    "#F1F7C9", "#C7E9B4", "#7FCDBB",
    "#41B6C4", "#2C7FB8", "#253494",
]
REGION_CMAP = mpl.colors.ListedColormap(REGION_MAP_COLORS, name="regional_classes")
RATE_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "policy_rate", ["#EDF4F8", "#B7D4E4", "#77A9C8", "#3B6FB6"])
QUARTILE_COLORS = [SO2, NOX, PM, BLUE]
ALIGNMENT_CMAP = REGION_CMAP

SCENARIO_ORDER = ["S1", "S2", "S3(alpha*)", "S4", "S5", "S6", "S7"]
SCENARIO_SOURCE = ["S1", "S2", "S3_040", "S4", "S5", "S6", "S7"]
SCENARIO_LABELS = {
    "S1": "S1  Sulfur",
    "S2": "S2  NO$_x$",
    "S3(alpha*)": "S3  Shore power (40%)",
    "S4": "S4  Sulfur + NO$_x$",
    "S5": "S5  Sulfur + shore power",
    "S6": "S6  NO$_x$ + shore power",
    "S7": "S7  All three measures",
}
MAP_LABEL_OFFSETS = {
    "R03": (25, 14), "R04": (-25, -12),
    "R08": (-12, 11), "R11": (-13, -13), "R13": (5, 13),
    "R16": (17, 4), "R19": (-12, -14), "R20": (-20, 9),
    "R21": (-24, 12), "R22": (18, 11),
}

CREATED: list[dict[str, str]] = []


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12.1,
        "axes.labelsize": 13.2,
        "axes.titlesize": 14.0,
        "xtick.labelsize": 11.3,
        "ytick.labelsize": 11.1,
        "legend.fontsize": 10.8,
        "axes.linewidth": 0.85,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "grid.color": GRID,
        "grid.linewidth": 0.65,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": DPI,
    })


def save_panel(fig: plt.Figure, filename: str, group: str, title: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / filename
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    CREATED.append({"group": group, "file": filename, "title": title})


def panel_label(fig: plt.Figure, letter: str) -> None:
    fig.text(0.020, 0.970, f"({letter})", ha="left", va="top",
             fontsize=16.0, fontweight="bold", color=INK)


def clean_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.65)
    ax.set_axisbelow(True)


def quantile_norm(values, n_classes: int = 6):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    bounds = np.quantile(arr, np.linspace(0, 1, n_classes + 1))
    for i in range(1, len(bounds)):
        if bounds[i] <= bounds[i - 1]:
            bounds[i] = np.nextafter(bounds[i - 1], np.inf)
    return mpl.colors.BoundaryNorm(bounds, REGION_CMAP.N, clip=True), bounds


def add_single_combo_divider(ax: plt.Axes) -> None:
    ax.axhline(2.5, color="#8A949B", lw=0.8, ls=(0, (3, 3)), zorder=0)


def scenario_table() -> pd.DataFrame:
    d = pd.read_csv(TABLES / "table_6_policy_scenario_summary.csv")
    d = d[d["row_type"].eq("scenario")].copy().set_index("scenario")
    missing = set(SCENARIO_ORDER) - set(d.index)
    if missing:
        raise RuntimeError(f"Missing main scenarios in Table 6: {sorted(missing)}")
    return d.reindex(SCENARIO_ORDER).reset_index()


def region_geometry() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    land = gpd.read_file(ADMIN, layer="admin0_polygon").to_crs(4326)
    membership = pd.read_csv(MEMBERSHIP)[["geo_iso3", "region_id"]].drop_duplicates("geo_iso3")
    selected = set(pd.read_csv(SELECTED_REGIONS)["region_id"].astype(str))
    joined = land.merge(membership, left_on="iso3", right_on="geo_iso3", how="left")
    joined["selected"] = joined["region_id"].astype(str).isin(selected)
    regions = (joined[joined["selected"] & joined["region_id"].notna()]
               .dissolve(by="region_id", as_index=False))
    regions = prepare_display_regions(regions)
    return joined, regions


def base_world_map(fig: plt.Figure,
                   rect=(0.025, 0.155, 0.95, 0.735)) -> plt.Axes:
    ax = fig.add_axes(rect, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=LAND,
                   edgecolor="#C1C9CF", linewidth=0.20, zorder=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#CCD3D8",
                   linewidth=0.16, zorder=0.7)
    ax.spines["geo"].set_edgecolor("#AEB7BE")
    ax.spines["geo"].set_linewidth(0.65)
    return ax


def add_region_geometries(ax: plt.Axes, regions: gpd.GeoDataFrame,
                          values: pd.Series, cmap, norm) -> None:
    for row in regions.itertuples():
        value = float(values.get(str(row.region_id), np.nan))
        color = cmap(norm(value)) if np.isfinite(value) else "#E3E8EA"
        ax.add_geometries([row.display_fill], crs=ccrs.PlateCarree(),
                          facecolor=color, edgecolor="none", zorder=2)
        ax.add_geometries([row.display_outline], crs=ccrs.PlateCarree(),
                          facecolor="none", edgecolor="#607079",
                          linewidth=0.34, zorder=2.4)


def annotate_region_ids(ax: plt.Axes, regions: gpd.GeoDataFrame,
                        exclude: set[str] | None = None,
                        fontsize: float = 7.4) -> None:
    """Directly identify every selected region without a separate lookup map."""
    exclude = exclude or set()
    for row in regions.itertuples():
        rid = str(row.region_id)
        if rid in exclude:
            continue
        p = row.geometry.representative_point()
        xp, yp = ax.projection.transform_point(p.x, p.y, ccrs.PlateCarree())
        dx, dy = MAP_LABEL_OFFSETS.get(rid, (0, 0))
        # Large offsets are intended for top-value callouts; keep ordinary IDs
        # close to their polygons and only nudge dense Asian/Mediterranean areas.
        dx = np.clip(dx * 0.30, -8, 8)
        dy = np.clip(dy * 0.30, -6, 6)
        ax.annotate(
            rid, xy=(xp, yp), xytext=(dx, dy), textcoords="offset points",
            ha="center", va="center", fontsize=fontsize, fontweight="bold",
            color=INK, zorder=5,
            bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                      edgecolor="none", alpha=0.78),
        )


def annotate_map_regions(ax: plt.Axes, regions: gpd.GeoDataFrame,
                         data: pd.DataFrame, value_col: str, n: int = 3,
                         suffix: str = "%") -> None:
    top = data.nlargest(n, value_col)
    shapes = regions.set_index("region_id")
    for row in top.itertuples():
        if row.region_id not in shapes.index:
            continue
        p = shapes.loc[row.region_id, "geometry"].representative_point()
        xp, yp = ax.projection.transform_point(p.x, p.y, ccrs.PlateCarree())
        dx, dy = MAP_LABEL_OFFSETS.get(str(row.region_id), (7, 8))
        value = float(getattr(row, value_col))
        ax.annotate(
            f"{row.region_id}  {value:.1f}{suffix}", xy=(xp, yp),
            xytext=(dx, dy), textcoords="offset points", fontsize=10.0,
            fontweight="bold", ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="square,pad=0.13", facecolor="white",
                      edgecolor="none", alpha=0.82),
            arrowprops=dict(arrowstyle="-", color="#687981", lw=0.65),
        )


def regional_pollutant_data() -> pd.DataFrame:
    d = pd.read_csv(TABLES / "figure_s6_s7_country_pollutant_mode_data.csv")
    out = d.groupby(["region_id", "Pollutant"], as_index=False)["reduction_g"].sum()
    out["share_pct"] = 100 * out["reduction_g"] / out.groupby(
        "Pollutant")["reduction_g"].transform("sum")
    return out


def draw_pollutant_map(pollutant: str, letter: str, filename: str) -> None:
    _, regions = region_geometry()
    all_data = regional_pollutant_data()
    d = all_data[all_data["Pollutant"].eq(pollutant)].copy()
    norm, bounds = quantile_norm(d["share_pct"])
    cmap = REGION_CMAP
    values = d.set_index("region_id")["share_pct"]
    fig = plt.figure(figsize=PANEL_SIZE)
    ax = base_world_map(fig)
    add_region_geometries(ax, regions, values, cmap, norm)
    top_ids = set(d.nlargest(3, "share_pct")["region_id"].astype(str))
    annotate_region_ids(ax, regions, exclude=top_ids)
    annotate_map_regions(ax, regions, d, "share_pct", n=3)
    pollutant_label = {"SO2": "SO$_2$", "NOx": "NO$_x$"}.get(pollutant, pollutant)
    ax.set_title(f"Regional distribution of S7 {pollutant_label} reduction",
                 fontsize=14.0, pad=10)
    panel_label(fig, letter)
    cax = fig.add_axes([0.27, 0.102, 0.46, 0.028])
    tick_bounds = bounds[[0, 2, 4, 6]]
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=cax, orientation="horizontal", boundaries=bounds,
                      ticks=tick_bounds, spacing="uniform")
    cb.set_label(
        f"Share of total S7 {pollutant_label} reduction (%; six quantile classes)",
        fontsize=11.5)
    cb.ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
    cb.ax.tick_params(labelsize=10.4)
    save_panel(fig, filename, "Figure 7", f"S7 {pollutant} regional reduction map")


def draw_region_metric_map(*, value_col: str, scale: float, letter: str,
                           filename: str, title: str, colorbar_label: str,
                           annotation_suffix: str, manifest_title: str) -> None:
    """Draw a region-level S7 quantity with the same map grammar as Figure 7."""
    _, regions = region_geometry()
    t7 = pd.read_csv(TABLES / "table_7_selected_regions_countries.csv")
    d = t7[t7["row_type"].eq("region_subtotal")].copy()
    d["map_value"] = pd.to_numeric(d[value_col], errors="coerce") / scale
    d = d[np.isfinite(d["map_value"])].copy()
    norm, bounds = quantile_norm(d["map_value"])
    values = d.set_index("region_id")["map_value"]

    fig = plt.figure(figsize=PANEL_SIZE)
    ax = base_world_map(fig)
    add_region_geometries(ax, regions, values, REGION_CMAP, norm)
    top_ids = set(d.nlargest(3, "map_value")["region_id"].astype(str))
    annotate_region_ids(ax, regions, exclude=top_ids)
    annotate_map_regions(
        ax, regions, d, "map_value", n=3, suffix=annotation_suffix)
    ax.set_title(title, fontsize=14.0, pad=10)
    panel_label(fig, letter)

    cax = fig.add_axes([0.27, 0.102, 0.46, 0.028])
    tick_bounds = bounds[[0, 2, 4, 6]]
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=REGION_CMAP),
        cax=cax, orientation="horizontal", boundaries=bounds,
        ticks=tick_bounds, spacing="uniform")
    cb.set_label(colorbar_label, fontsize=11.5)
    cb.ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
    cb.ax.tick_params(labelsize=10.4)
    save_panel(fig, filename, "Figure 8", manifest_title)


def fig7d_pollutant_rates() -> None:
    d = scenario_table()
    y = np.arange(len(d), dtype=float)
    series = [
        ("SO2_policy_zone_reduction_rate", "SO$_2$", SO2, -0.24),
        ("PM_policy_zone_reduction_rate", "PM", PM, 0.00),
        ("NOx_policy_zone_reduction_rate", "NO$_x$", NOX, 0.24),
    ]
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    for col, label, color, offset in series:
        values = 100 * pd.to_numeric(d[col], errors="coerce").fillna(0).to_numpy(float)
        bars = ax.barh(y + offset, values, height=0.19, color=color,
                       edgecolor="white", linewidth=0.55, label=label, zorder=2)
        for bar, value in zip(bars, values):
            if value <= 0.04:
                continue
            ax.text(value + 0.8, bar.get_y() + bar.get_height()/2, f"{value:.1f}",
                    ha="left", va="center", fontsize=10.6, color=INK)
    ax.set_yticks(y, [SCENARIO_LABELS[s] for s in SCENARIO_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, 92)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.set_xlabel("Reduction relative to selected policy-zone baseline (%)")
    ax.set_title("Pollutant-specific effectiveness of individual and combined measures", pad=11)
    add_single_combo_divider(ax)
    clean_axis(ax, "x")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3,
              columnspacing=1.8, handlelength=1.7)
    fig.subplots_adjust(left=0.31, right=0.97, bottom=0.24, top=0.86)
    panel_label(fig, "d")
    fig.text(0.98, 0.035, "S3: shore power = 40%; S5–S7: shore power = 100%",
             ha="right", fontsize=10.1, color=MUTED)
    save_panel(fig, "fig7d_scenario_pollutant_reduction_rates.png", "Figure 7",
               "Scenario pollutant reduction rates")


def combination_accounting_data() -> tuple[list[tuple[str, str, str, str]], np.ndarray]:
    t6 = pd.read_csv(TABLES / "table_6_policy_scenario_summary.csv")
    main = t6[t6["row_type"].eq("scenario")].set_index("scenario")
    net = t6[t6["row_type"].eq("country_net")].set_index("scenario")
    effects = pd.read_csv(TABLES / "scenario_effect_metrics_all.csv").set_index("scenario")
    ops100 = float(effects.loc["S3_100", "exact_weighted_reduction_g"]) / 1e9
    rows = [
        ("S5  S1 + shore power", "S1", "S5", "S5 net"),
        ("S6  S2 + shore power", "S2", "S6", "S6 net"),
        ("S7  S4 + shore power", "S4", "S7", "S7 net"),
    ]
    values = []
    for _, base_sc, combo_sc, net_sc in rows:
        base = float(main.loc[base_sc, "exact_weighted_reduction_g"]) / 1e9
        gross = float(main.loc[combo_sc, "exact_weighted_reduction_g"]) / 1e9
        net_value = (float(net.loc[net_sc, "global_weighted_reduction_share"]) *
                     float(net.loc[net_sc, "global_baseline_weighted_g"]) / 1e9)
        realized = gross - base
        overlap = max(base + ops100 - gross, 0)
        values.append([base, realized, overlap, gross, net_value, ops100])
    return rows, np.asarray(values, float)


def fig7e_combination_accounting() -> None:
    rows, v = combination_accounting_data()
    base, realized, overlap, gross, net_total, ops100 = v.T
    centers = np.arange(len(rows), dtype=float) * 1.35
    potential = base + ops100
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    ax.barh(centers - 0.12, base, height=0.38, color=LIGHT_BLUE,
            label="Existing policy effect", zorder=2)
    ax.barh(centers - 0.12, realized, left=base, height=0.38, color=GREEN,
            label="Realized shore-power marginal", zorder=2)
    for i, ov in enumerate(overlap):
        if ov > 0.05:
            ax.barh(centers[i] - 0.12, ov, left=gross[i], height=0.38,
                    facecolor="#F8E6E4", edgecolor=CORAL, linewidth=0.9,
                    hatch="////", label="Removed overlap" if i == 0 else None, zorder=2)
    for i, y0 in enumerate(centers):
        ax.hlines(y0 + 0.28, 0, net_total[i], color="#98A5AD", lw=1.25, zorder=1)
        ax.scatter(net_total[i], y0 + 0.28, marker="D", s=76, color=BLUE,
                   edgecolor="white", linewidth=0.8, zorder=4,
                   label="Net after grid emissions" if i == 0 else None)
        ax.vlines(gross[i], y0 - 0.34, y0 + 0.05, color=INK, lw=0.85, zorder=4)
        ax.text(base[i] / 2, y0 - 0.12, f"{base[i]:.1f}", ha="center", va="center",
                fontsize=10.0, color="white", fontweight="bold")
        ax.text(base[i] + realized[i] / 2, y0 - 0.12, f"+{realized[i]:.1f}",
                ha="center", va="center", fontsize=10.0, color=INK, fontweight="bold")
        ax.text(gross[i], y0 - 0.43, f"Gross {gross[i]:.1f}", ha="center", va="bottom",
                fontsize=10.0, color=INK)
        ax.text(net_total[i] + 7, y0 + 0.28, f"Net {net_total[i]:.1f}",
                ha="left", va="center", fontsize=10.0, color=BLUE)
        if overlap[i] > 0.05:
            ax.text(gross[i] + overlap[i] + 7, y0 - 0.12,
                    f"Overlap {overlap[i]:.1f}", ha="left", va="center",
                    fontsize=9.6, color=CORAL)
    ax.set_yticks(centers, [row[0] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, potential.max() * 1.24)
    ax.set_xlabel("Exact weighted reduction in policy waters (kt-equivalent)")
    ax.set_title("Policy-combination accounting: marginal gain, overlap, and grid penalty", pad=11)
    clean_axis(ax, "x")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2,
              columnspacing=1.1, handletextpad=0.5)
    fig.subplots_adjust(left=0.31, right=0.97, bottom=0.29, top=0.84)
    panel_label(fig, "e")
    fig.text(0.98, 0.035, f"Standalone shore power at 100%: {ops100[0]:.1f} kt-equivalent",
             ha="right", fontsize=10.0, color=MUTED)
    save_panel(fig, "fig7e_combination_marginal_overlap_net.png", "Figure 7",
               "Combination marginal contribution and overlap")


def fig7f_ops_returns() -> None:
    d = pd.read_csv(TABLES / "ops_alpha_scan.csv").sort_values("alpha")
    k = pd.read_csv(TABLES / "ops_knee_results.csv")
    x = 100 * d["alpha"].to_numpy(float)
    gross = d["gross_weighted_reduction_g"].to_numpy(float) / 1e9
    net = d["net_weighted_reduction_g"].to_numpy(float) / 1e9
    marginal = np.diff(np.r_[0.0, gross])
    gross_knee = k[k["criterion"].eq("gross_weighted_reduction_g")]
    alpha_star = float(gross_knee.iloc[0].get("selected_alpha", 0.4)) if len(gross_knee) else 0.4
    a_pct = 100 * alpha_star
    row = d.iloc[(d["alpha"] - alpha_star).abs().argmin()]

    fig = plt.figure(figsize=PANEL_SIZE)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.06,
                          left=0.14, right=0.97, bottom=0.18, top=0.86)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax)
    ax.fill_between(x, net, gross, color="#DCE9F0", alpha=0.75,
                    label="Grid-emission penalty", zorder=1)
    ax.plot(x, gross, color=BLUE, marker="o", ms=6.8, lw=2.3,
            label="Gross ship-side reduction", zorder=3)
    ax.plot(x, net, color=GREEN, marker="s", ms=6.2, lw=2.1,
            label="Net reduction after grid emissions", zorder=3)
    ax.axvline(a_pct, color=CORAL, ls="--", lw=1.15, zorder=2)
    ax.annotate(
        f"Selected knee: {a_pct:.0f}% use\n{100*row.energy_share:.1f}% of OPS demand covered",
        xy=(a_pct, row.gross_weighted_reduction_g/1e9), xytext=(48, 40),
        arrowprops=dict(arrowstyle="-", color=CORAL, lw=0.9),
        fontsize=10.0, color=CORAL, ha="left", va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.2))
    ax.text(101, gross[-1], f"{gross[-1]:.1f}", ha="left", va="center",
            fontsize=10.1, color=BLUE)
    ax.text(101, net[-1], f"{net[-1]:.1f}", ha="left", va="center",
            fontsize=10.1, color=GREEN)
    ax.set_ylim(0, max(gross) * 1.20)
    ax.set_ylabel("Cumulative reduction\n(kt-equivalent)")
    ax.set_title("Shore-power deployment and diminishing marginal returns", pad=11)
    ax.legend(loc="upper left", ncol=3, columnspacing=1.0, handlelength=2.0)
    clean_axis(ax, "y")
    ax.tick_params(labelbottom=False)

    bars = ax2.bar(x, marginal, width=6.4, color=PM, edgecolor="white",
                   linewidth=0.55, zorder=2)
    ax2.axvline(a_pct, color=CORAL, ls="--", lw=1.15, zorder=3)
    for bar, value, x_value in zip(bars, marginal, x):
        x_label = bar.get_x() + bar.get_width()/2
        ha = "center"
        if abs(x_value - a_pct) < 1e-8:
            x_label += 2.2
            ha = "left"
        ax2.text(x_label, value + 0.40, f"{value:.1f}",
                 ha=ha, va="bottom", fontsize=9.8, color=INK)
    ax2.set_xlim(5, 106)
    ax2.set_ylim(0, max(marginal) * 1.27)
    ax2.set_xticks(x)
    ax2.set_xlabel("Shore-power use among eligible calls (%)")
    ax2.set_ylabel("Marginal gain")
    clean_axis(ax2, "y")
    panel_label(fig, "f")
    save_panel(fig, "fig7f_ops_diminishing_returns.png", "Figure 7",
               "Shore-power diminishing returns")


def fig8a_regional_heterogeneity() -> None:
    _, regions = region_geometry()
    t7 = pd.read_csv(TABLES / "table_7_selected_regions_countries.csv")
    d = t7[t7["row_type"].eq("region_subtotal")].copy()
    d["exposure_rate_pct"] = 100 * pd.to_numeric(d["s7_exposure_reduction_rate"], errors="coerce")
    values = d.set_index("region_id")["exposure_rate_pct"]
    norm, bounds = quantile_norm(d["exposure_rate_pct"])
    cmap = REGION_CMAP
    fig = plt.figure(figsize=PANEL_SIZE)
    ax = base_world_map(fig)
    add_region_geometries(ax, regions, values, cmap, norm)
    top_ids = set(d.nlargest(3, "exposure_rate_pct")["region_id"].astype(str))
    annotate_region_ids(ax, regions, exclude=top_ids)
    annotate_map_regions(ax, regions, d, "exposure_rate_pct", n=3)
    ax.set_title("Regional S7 population-exposure reduction rates",
                 fontsize=14.0, pad=10)
    panel_label(fig, "a")
    cax = fig.add_axes([0.27, 0.102, 0.46, 0.028])
    tick_bounds = bounds[[0, 2, 4, 6]]
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=cax, orientation="horizontal", boundaries=bounds,
                      ticks=tick_bounds, spacing="uniform")
    cb.set_label(
        "Regional S7 exposure-proxy reduction rate (%; six quantile classes)",
        fontsize=11.2)
    cb.ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
    cb.ax.tick_params(labelsize=10.4)
    save_panel(fig, "fig8a_s7_regional_exposure_heterogeneity_map.png", "Figure 8",
               "S7 regional exposure-reduction rates")


def fig8b_global_emission_exposure() -> None:
    d = scenario_table()
    emission = 100 * d["global_weighted_reduction_share"].to_numpy(float)
    exposure = 100 * d["delta_pwe"].to_numpy(float) / d["global_baseline_pwe"].to_numpy(float)
    amplification = np.divide(exposure, emission, out=np.full_like(exposure, np.nan), where=emission > 0)
    y = np.arange(len(d), dtype=float)
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    ax.barh(y - 0.16, emission, height=0.26, color=SO2,
            label="Emission reduction", zorder=2)
    ax.barh(y + 0.16, exposure, height=0.26, color=PM,
            label="Exposure reduction", zorder=2)
    for yy, xe, xp, amp in zip(y, emission, exposure, amplification):
        ax.text(xe + 0.18, yy - 0.16, f"{xe:.2f}", ha="left", va="center", fontsize=10.0)
        ax.text(xp + 0.18, yy + 0.16, f"{xp:.2f}", ha="left", va="center", fontsize=10.0)
        ax.text(19.05, yy, f"{amp:.1f}×", ha="center", va="center",
                fontsize=10.2, color=NOX, fontweight="bold")
    ax.text(19.05, -0.34, "Amplification", ha="center", va="bottom",
            fontsize=9.3, color=NOX, fontweight="bold")
    ax.plot([], [], linestyle="none", marker="D", markersize=6.2, color=NOX,
            label="Exposure / emission")
    ax.set_yticks(y, [SCENARIO_LABELS[s] for s in SCENARIO_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, 20.4)
    ax.set_xlabel("Reduction relative to the corresponding global baseline (%)")
    ax.set_title("Global emission reduction and population-exposure amplification", pad=11)
    add_single_combo_divider(ax)
    clean_axis(ax, "x")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3,
              columnspacing=1.0, handletextpad=0.5)
    fig.subplots_adjust(left=0.31, right=0.97, bottom=0.24, top=0.86)
    panel_label(fig, "b")
    save_panel(fig, "fig8b_global_emission_exposure_reduction.png", "Figure 8",
               "Global emission and exposure reduction")


def fig8c_exposure_efficiency() -> None:
    d = scenario_table()
    metrics = [
        ("Catchment reduction", d["city_weighted_reduction_g"].to_numpy(float) / 1e9,
         "kt-eq", lambda x: f"{x:.0f}"),
        ("Exposure benefit", d["delta_pwe"].to_numpy(float) / 1e14,
         r"$10^{14}$ PWE-proxy", lambda x: f"{x:.1f}"),
        ("Exposure efficiency", d["exposure_efficiency"].to_numpy(float),
         "benefit per reduction", lambda x: f"{x:,.0f}"),
        ("CI improvement", -100 * d["ci_pwe_change"].to_numpy(float),
         r"$-100\,\Delta CI$", lambda x: f"{x:.2f}"),
    ]
    y = np.arange(len(d), dtype=float)
    fig = plt.figure(figsize=PANEL_SIZE)
    gs = fig.add_gridspec(1, 4, left=0.30, right=0.98, bottom=0.18, top=0.75,
                          wspace=0.54)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    bar_colors = np.array(["#9CBBD1"] * len(d), dtype=object)
    bar_colors[3:6] = "#76A2C3"
    bar_colors[-1] = BLUE
    for j, (title, values, unit, formatter) in enumerate(metrics):
        ax = axes[j]
        ax.barh(y, values, height=0.56, color=bar_colors, edgecolor="white",
                linewidth=0.55, zorder=2)
        xmax = max(float(np.nanmax(values)), 1e-9)
        ax.set_xlim(0, xmax * 1.23)
        for value, yy in zip(values, y):
            ax.text(value + xmax * 0.025, yy, formatter(value), ha="left", va="center",
                    fontsize=9.7, color=INK)
        ax.set_title(f"{title}\n({unit})", fontsize=10.8, pad=8)
        ax.invert_yaxis()
        ax.axhline(2.5, color="#8A949B", lw=0.75, ls=(0, (3, 3)), zorder=0)
        clean_axis(ax, "x")
        ax.tick_params(axis="x", labelsize=9.5)
        if j == 0:
            ax.set_yticks(y, [SCENARIO_LABELS[s] for s in SCENARIO_ORDER])
        else:
            ax.set_yticks(y, [])
    fig.suptitle("Scale, exposure efficiency, and equity performance of policy scenarios",
                 x=0.64, y=0.895, fontsize=14.0)
    panel_label(fig, "c")
    save_panel(fig, "fig8c_exposure_efficiency.png", "Figure 8",
               "Multidimensional exposure-efficiency profile")


def fig8d_vulnerable_share() -> None:
    d = scenario_table()
    values = 100 * d["vulnerable_benefit_share"].to_numpy(float)
    baseline = 100 * float(d.loc[0, "global_baseline_vulnerable_pwe"]) / float(
        d.loc[0, "global_baseline_pwe"])
    absolute = d["vulnerable_benefit_absolute"].to_numpy(float) / 1e14
    y = np.arange(len(d), dtype=float)
    delta = values - baseline
    colors = np.array([GREEN if value >= 0 else CORAL for value in delta])
    fig = plt.figure(figsize=PANEL_SIZE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.70, 0.90], wspace=0.25,
                          left=0.30, right=0.97, bottom=0.20, top=0.78)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax)

    ax.axvline(baseline, color=INK, lw=1.0, ls=(0, (3, 3)), zorder=1)
    ax.hlines(y, baseline, values, color=colors, linewidth=5.2, alpha=0.78, zorder=2)
    ax.scatter(values, y, s=82, color=colors, edgecolor="white", linewidth=0.8, zorder=4)
    for value, change, yy in zip(values, delta, y):
        ax.text(value + 0.07, yy,
                f"{value:.2f}%  ({change:+.2f} pp)",
                ha="left", va="center",
                fontsize=9.6, color=INK)
    ax.set_yticks(y, [SCENARIO_LABELS[s] for s in SCENARIO_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(17.15, 20.85)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.set_xlabel("Vulnerable-benefit share (%)")
    ax.set_title("Distributional direction", fontsize=11.5, pad=9)
    ax.text(baseline, 1.02, f"Baseline {baseline:.2f}%", ha="center", va="bottom",
            transform=ax.get_xaxis_transform(), fontsize=9.2, color=MUTED)
    add_single_combo_divider(ax)
    clean_axis(ax, "x")

    bars = ax2.barh(y, absolute, height=0.52, color=LIGHT_BLUE,
                    edgecolor="white", linewidth=0.55, zorder=2)
    for bar, value in zip(bars, absolute):
        ax2.text(value + max(absolute) * 0.025,
                 bar.get_y() + bar.get_height()/2, f"{value:.2f}",
                 ha="left", va="center", fontsize=9.6)
    ax2.set_xlim(0, max(absolute) * 1.24)
    ax2.set_xlabel(r"Absolute benefit ($10^{14}$ PWE)")
    ax2.set_title("Absolute scale", fontsize=11.5, pad=9)
    ax2.tick_params(axis="y", left=False, labelleft=False)
    add_single_combo_divider(ax2)
    clean_axis(ax2, "x")

    fig.suptitle("Vulnerable-population benefits: direction and scale",
                 x=0.63, y=0.91, fontsize=14.0)
    handles = [Patch(facecolor=GREEN, label="Above baseline share"),
               Patch(facecolor=CORAL, label="Below baseline share")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.64, 0.035), ncol=2)
    panel_label(fig, "d")
    save_panel(fig, "fig8d_vulnerable_population_benefit_share.png", "Figure 8",
               "Vulnerable-population benefit share")


def gdp_quartile_data(value_col: str) -> tuple[np.ndarray, np.ndarray]:
    g = pd.read_csv(TABLES / "table_s15_quartile_effects.csv")
    g = g[g["grouping"].eq("gdp_quartile") & g["scenario"].isin(SCENARIO_SOURCE)].copy()
    quartiles = ["Q1", "Q2", "Q3", "Q4"]
    matrix = np.full((len(SCENARIO_SOURCE), len(quartiles)), np.nan)
    for i, scenario in enumerate(SCENARIO_SOURCE):
        row = g[g["scenario"].eq(scenario)].set_index("quartile").reindex(quartiles)
        matrix[i] = row[value_col].to_numpy(float)
    ratio = matrix[:, 0] / matrix[:, 3]
    return matrix, ratio


def fig8e_quartile_rate_heatmap() -> None:
    matrix, ratio = gdp_quartile_data("reduction_rate")
    values = 100 * matrix
    vmax = 5 * np.ceil(np.nanmax(values) / 5)
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    im = ax.imshow(values, aspect="auto", cmap=RATE_CMAP, norm=norm,
                   interpolation="nearest")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            rgba = RATE_CMAP(norm(values[i, j]))
            luminance = 0.2126*rgba[0] + 0.7152*rgba[1] + 0.0722*rgba[2]
            color = "white" if luminance < 0.58 else INK
            ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center",
                    fontsize=10.6, color=color, fontweight="bold")
        cell_color = "#F8E6E4" if ratio[i] >= 1 else "#E7EFF6"
        text_color = CORAL if ratio[i] >= 1 else BLUE
        ax.add_patch(Rectangle((3.72, i - 0.48), 1.05, 0.96,
                               facecolor=cell_color, edgecolor="white", lw=0.8))
        arrow = "↑" if ratio[i] >= 1 else "↓"
        ax.text(4.245, i, f"{ratio[i]:.2f}× {arrow}", ha="center", va="center",
                fontsize=10.4, color=text_color, fontweight="bold")
    ax.axvline(3.60, color="#AAB8BE", lw=0.9)
    ax.set_xlim(-0.5, 4.82)
    ax.set_xticks([0, 1, 2, 3, 4.245], ["Q1\nlowest", "Q2", "Q3", "Q4\nhighest", "Q1/Q4"])
    ax.set_yticks(np.arange(len(SCENARIO_ORDER)), [SCENARIO_LABELS[s] for s in SCENARIO_ORDER])
    ax.tick_params(length=0)
    ax.set_title("Exposure-proxy reduction rates across 2019 GDP-per-capita quartiles", pad=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = fig.add_axes([0.34, 0.086, 0.40, 0.025])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Exposure-proxy reduction rate (%)", fontsize=11.0)
    cb.ax.tick_params(labelsize=9.9)
    fig.subplots_adjust(left=0.30, right=0.97, bottom=0.17, top=0.85)
    panel_label(fig, "e")
    save_panel(fig, "fig8e_gdp_quartile_reduction_rate_heatmap.png", "Figure 8",
               "GDP-quartile exposure reduction rate")


def fig8f_quartile_distribution() -> None:
    matrix, _ = gdp_quartile_data("distribution_share")
    values = 100 * matrix
    y = np.arange(len(SCENARIO_ORDER), dtype=float)
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    left = np.zeros(len(y))
    for j, (q, color) in enumerate(zip(["Q1", "Q2", "Q3", "Q4"], QUARTILE_COLORS)):
        width = values[:, j]
        ax.barh(y, width, left=left, height=0.62, color=color, edgecolor="white",
                linewidth=0.65, label=q)
        for i, (x0, w) in enumerate(zip(left, width)):
            if w >= 4.0:
                ax.text(x0 + w/2, i, f"{w:.1f}", ha="center", va="center",
                        fontsize=9.7, color=INK, fontweight="bold")
        left += width
    ax.set_yticks(y, [SCENARIO_LABELS[s] for s in SCENARIO_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of total exposure-proxy reduction (%)")
    ax.set_title("Distribution of policy benefits across GDP-per-capita quartiles", pad=11)
    clean_axis(ax, "x")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=4,
              columnspacing=1.6, handletextpad=0.45)
    fig.subplots_adjust(left=0.31, right=0.97, bottom=0.24, top=0.86)
    panel_label(fig, "f")
    save_panel(fig, "fig8f_gdp_quartile_reduction_distribution.png", "Figure 8",
               "GDP-quartile distribution of exposure benefits")


def region_risk_benefit_alignment_data() -> pd.DataFrame:
    """Match each region's vulnerable benefit share to its baseline risk share."""
    panel = pd.read_parquet(CITY_EFFECT_PANEL)
    required = {
        "scenario", "node_id", "iso3", "base_pwe", "reduction_pwe",
        "S2_vuln_share",
    }
    missing = required - set(panel.columns)
    if missing:
        raise RuntimeError(f"City policy panel is missing columns: {sorted(missing)}")
    d = panel[panel["scenario"].eq("S7")].copy()
    if d["S2_vuln_share"].isna().any():
        raise RuntimeError("S7 city panel contains missing vulnerable-population shares")

    membership = pd.read_csv(MEMBERSHIP)[["geo_iso3", "region_id"]].drop_duplicates(
        "geo_iso3")
    selected = set(pd.read_csv(SELECTED_REGIONS)["region_id"].astype(str))
    d = d.merge(
        membership, left_on="iso3", right_on="geo_iso3", how="inner",
        validate="m:1")
    d = d[d["region_id"].astype(str).isin(selected)].copy()
    d["baseline_vulnerable_pwe"] = (
        pd.to_numeric(d["base_pwe"], errors="coerce")
        * pd.to_numeric(d["S2_vuln_share"], errors="coerce")
    )
    d["s7_vulnerable_benefit"] = (
        pd.to_numeric(d["reduction_pwe"], errors="coerce")
        * pd.to_numeric(d["S2_vuln_share"], errors="coerce")
    )

    out = d.groupby("region_id", as_index=False).agg(
        n_city_nodes=("node_id", "nunique"),
        baseline_pwe=("base_pwe", "sum"),
        s7_exposure_reduction=("reduction_pwe", "sum"),
        baseline_vulnerable_pwe=("baseline_vulnerable_pwe", "sum"),
        s7_vulnerable_benefit=("s7_vulnerable_benefit", "sum"),
    )
    if set(out["region_id"].astype(str)) != selected:
        absent = selected - set(out["region_id"].astype(str))
        raise RuntimeError(f"Selected regions missing from city policy panel: {sorted(absent)}")
    if (out[["baseline_vulnerable_pwe", "s7_vulnerable_benefit"]] <= 0).any().any():
        raise RuntimeError("Risk-benefit alignment requires positive regional totals")

    metadata = pd.read_csv(TABLES / "table_7_selected_regions_countries.csv")
    metadata = metadata[metadata["row_type"].eq("region_subtotal")][[
        "region_id", "region_rank", "region_name_en", "region_name_zh",
        "spri_median", "country_baseline_pwe", "s7_exposure_reduction",
    ]].rename(columns={"s7_exposure_reduction": "table7_exposure_reduction"})
    out = out.merge(metadata, on="region_id", how="left", validate="1:1")
    relerr = np.abs(
        out["s7_exposure_reduction"] - out["table7_exposure_reduction"]
    ) / np.maximum(out["table7_exposure_reduction"].to_numpy(float), 1.0)
    if float(np.nanmax(relerr)) > 1e-10:
        raise RuntimeError(
            f"Regional exposure aggregation does not reproduce Table 7: {relerr.max():.3e}")

    total_risk = float(out["baseline_vulnerable_pwe"].sum())
    total_benefit = float(out["s7_vulnerable_benefit"].sum())
    out["vulnerable_risk_share"] = out["baseline_vulnerable_pwe"] / total_risk
    out["vulnerable_benefit_share"] = out["s7_vulnerable_benefit"] / total_benefit
    out["vulnerable_exposure_reduction_rate"] = (
        out["s7_vulnerable_benefit"] / out["baseline_vulnerable_pwe"])
    out["alignment_ratio"] = (
        out["vulnerable_benefit_share"] / out["vulnerable_risk_share"])
    out["log2_alignment"] = np.log2(out["alignment_ratio"])
    out["selected_region_benchmark_rate"] = total_benefit / total_risk
    out["alignment_class"] = pd.cut(
        out["alignment_ratio"],
        bins=[-np.inf, 0.8, 0.95, 1.05, 1.2, np.inf],
        labels=[
            "substantially_under_aligned", "moderately_under_aligned",
            "approximately_proportional", "moderately_above_proportional",
            "substantially_above_proportional",
        ],
    ).astype(str)
    out = out.drop(columns=["table7_exposure_reduction"]).sort_values("region_rank")
    out.to_csv(ALIGNMENT_TABLE, index=False, encoding="utf-8-sig")
    return out


def fig8g_risk_benefit_alignment() -> None:
    _, regions = region_geometry()
    d = region_risk_benefit_alignment_data()
    values = d.set_index("region_id")["alignment_ratio"]
    ratio_bounds = np.asarray([0.50, 0.70, 0.85, 1.00, 1.10, 1.25, 1.40])
    norm = mpl.colors.BoundaryNorm(ratio_bounds, ALIGNMENT_CMAP.N, clip=True)

    fig = plt.figure(figsize=PANEL_SIZE)
    ax = base_world_map(fig)
    add_region_geometries(ax, regions, values, ALIGNMENT_CMAP, norm)
    extremes = pd.concat([
        d.nsmallest(2, "alignment_ratio"),
        d.nlargest(2, "alignment_ratio"),
    ]).drop_duplicates("region_id")
    extreme_ids = set(extremes["region_id"].astype(str))
    annotate_region_ids(ax, regions, exclude=extreme_ids)

    shapes = regions.set_index("region_id")
    alignment_offsets = {
        "R04": (-27, -13), "R12": (-13, 16),
        "R15": (17, 13), "R21": (-27, 14),
    }
    for row in extremes.itertuples(index=False):
        rid = str(row.region_id)
        if rid not in shapes.index:
            continue
        p = shapes.loc[rid, "geometry"].representative_point()
        xp, yp = ax.projection.transform_point(p.x, p.y, ccrs.PlateCarree())
        dx, dy = alignment_offsets.get(rid, MAP_LABEL_OFFSETS.get(rid, (8, 9)))
        ax.annotate(
            f"{rid}  {row.alignment_ratio:.2f}×", xy=(xp, yp),
            xytext=(dx, dy), textcoords="offset points", fontsize=10.0,
            fontweight="bold", ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="square,pad=0.13", facecolor="white",
                      edgecolor="none", alpha=0.84),
            arrowprops=dict(arrowstyle="-", color="#687981", lw=0.65),
        )

    ax.set_title("Risk-adjusted alignment of S7 vulnerable-population benefits",
                 fontsize=14.0, pad=10)
    panel_label(fig, "g")
    cax = fig.add_axes([0.25, 0.102, 0.50, 0.028])
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=ALIGNMENT_CMAP),
        cax=cax, orientation="horizontal", boundaries=ratio_bounds,
        ticks=ratio_bounds, spacing="uniform")
    cb.ax.set_xticklabels([f"{value:.2f}×" for value in ratio_bounds])
    cb.set_label(
        "Policy-benefit share / baseline vulnerable-risk share "
        "(1 = proportional)", fontsize=11.2)
    cb.ax.tick_params(labelsize=10.2)
    save_panel(
        fig, "fig8g_s7_risk_adjusted_policy_benefit_alignment_map.png",
        "Figure 8", "S7 risk-adjusted policy-benefit alignment")


def main() -> None:
    apply_style()
    required = [
        TABLES / "figure_s6_s7_country_pollutant_mode_data.csv",
        TABLES / "table_6_policy_scenario_summary.csv",
        TABLES / "scenario_effect_metrics_all.csv",
        TABLES / "ops_alpha_scan.csv",
        TABLES / "ops_knee_results.csv",
        TABLES / "table_7_selected_regions_countries.csv",
        TABLES / "table_s15_quartile_effects.csv",
        CITY_EFFECT_PANEL,
        ADMIN, MEMBERSHIP, SELECTED_REGIONS,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing formal-figure inputs:\n" + "\n".join(missing))

    draw_pollutant_map("SO2", "a", "fig7a_s7_so2_regional_reduction_map.png")
    draw_pollutant_map("PM", "b", "fig7b_s7_pm_regional_reduction_map.png")
    draw_pollutant_map("NOx", "c", "fig7c_s7_nox_regional_reduction_map.png")
    fig7d_pollutant_rates()
    fig7e_combination_accounting()
    fig7f_ops_returns()

    fig8a_regional_heterogeneity()
    fig8b_global_emission_exposure()
    fig8c_exposure_efficiency()
    fig8d_vulnerable_share()
    fig8e_quartile_rate_heatmap()
    fig8f_quartile_distribution()
    fig8g_risk_benefit_alignment()
    draw_region_metric_map(
        value_col="s7_exposure_reduction", scale=1e14, letter="h",
        filename="fig8h_s7_regional_population_exposure_reduction_map.png",
        title="Regional distribution of S7 population-exposure reduction",
        colorbar_label=r"S7 population-exposure reduction ($10^{14}$ PWE-proxy; six quantile classes)",
        annotation_suffix=r" $\times10^{14}$",
        manifest_title="S7 regional population-exposure reduction")

    manifest = pd.DataFrame(CREATED)
    manifest["path"] = manifest["file"].map(lambda name: str(FIGURES / name))
    manifest["bytes"] = manifest["file"].map(lambda name: (FIGURES / name).stat().st_size)
    manifest.to_csv(TABLES / "policy_redesigned_figure_manifest.csv",
                    index=False, encoding="utf-8-sig")
    print(f"[complete] standalone_panels={len(manifest)} figures={FIGURES}")


if __name__ == "__main__":
    main()
