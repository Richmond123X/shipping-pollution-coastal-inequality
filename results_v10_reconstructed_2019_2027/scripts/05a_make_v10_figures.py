# -*- coding: utf-8 -*-
"""Generate the complete v10 figure set from the 2019--2025 run.

The visual language follows the previously approved v10 panels: serif type,
quiet grids, restrained blue/red/grey accents, separate PNG panels, and a
single combined Figure 1.  Data-processing provenance belongs in the audit
tables, not in figure titles or legends.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import cartopy.crs as ccrs
import cartopy.feature as cfeature


ROOT = Path(os.environ.get("TRANSFER_JUSTICE_ROOT", r"E:\transfer justice"))
SOURCE_SCRIPTS = ROOT / "results_v10_empirical_risk_2019_2026" / "scripts"
if str(SOURCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SOURCE_SCRIPTS))
import common as C  # noqa: E402

C.apply_style()

YEARS = C.OBS_YEARS
LATEST = C.LATEST_OBS_YEAR
FORECAST_YEARS = C.FORECAST_YEARS

QUAD_COLORS = {
    "emerging": C.ACCENT["purple"],
    "established_rising": C.ACCENT["red"],
    "established_stable_declining": "#8F99A3",
    "low_current_low_momentum": "#D5DBE0",
}
QUAD_LABELS = {
    "emerging": "Emerging risk",
    "established_rising": "Established and rising",
    "established_stable_declining": "Established, stable/declining",
    "low_current_low_momentum": "Low current · low momentum",
}


def out_path(group: str, name: str) -> Path:
    path = C.FIGURES / "v10_new" / group / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save(fig, group: str, name: str) -> None:
    fig.savefig(out_path(group, name), dpi=330, bbox_inches="tight")
    plt.close(fig)


def base_map(ax, land="#F1F3F5", edge="#B7C0C8") -> None:
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor=land,
                   edgecolor=edge, linewidth=0.28, zorder=0)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), edgecolor=edge,
                   linewidth=0.22, zorder=0.2)
    ax.set_global()
    ax.spines["geo"].set_edgecolor(C.INK)
    ax.spines["geo"].set_linewidth(0.65)


def soft_axes(ax, axis="y") -> None:
    C.despine(ax)
    C.soft_grid(ax, axis=axis)


def marker_sizes(values, lo=8.0, hi=95.0) -> np.ndarray:
    x = np.asarray(values, float)
    z = np.log1p(np.maximum(x, 0))
    q1, q99 = np.nanquantile(z[np.isfinite(z)], [0.01, 0.99])
    if q99 <= q1:
        return np.full_like(z, (lo + hi) / 2)
    return lo + (hi - lo) * np.clip((z - q1) / (q99 - q1), 0, 1)


def city_coords() -> pd.DataFrame:
    annual = pd.read_parquet(C.DATA / f"emission_city_annual_{C.OBS_STEM}.parquet")
    return annual[["node_id", "city", "country", "iso3", "rep_lon", "rep_lat"]].drop_duplicates("node_id")


def edge_segments(edges: pd.DataFrame, coords: pd.DataFrame, n=1200):
    pos = coords.set_index("node_id")[["rep_lon", "rep_lat"]]
    d = edges.nlargest(min(n, len(edges)), "w_t").copy()
    d = d.join(pos, on="From_City").rename(columns={"rep_lon": "x1", "rep_lat": "y1"})
    d = d.join(pos, on="To_City").rename(columns={"rep_lon": "x2", "rep_lat": "y2"})
    d = d.dropna(subset=["x1", "y1", "x2", "y2"])
    # Straight projected segments are intentionally omitted across the date line.
    d = d[np.abs(d.x2 - d.x1) <= 180]
    return np.stack([
        d[["x1", "y1"]].to_numpy(float),
        d[["x2", "y2"]].to_numpy(float),
    ], axis=1), d.w_t.to_numpy(float)


def add_routes(ax, edges, coords, n=1200, alpha=0.08) -> None:
    seg, weight = edge_segments(edges, coords, n=n)
    if len(seg) == 0:
        return
    lw = 0.12 + 1.05 * np.clip(
        (np.log1p(weight) - np.nanquantile(np.log1p(weight), 0.02)) /
        max(np.nanquantile(np.log1p(weight), 0.98) - np.nanquantile(np.log1p(weight), 0.02), 1e-9),
        0, 1,
    )
    lc = LineCollection(seg, colors="#7899B0", linewidths=lw, alpha=alpha,
                        transform=ccrs.PlateCarree(), zorder=1)
    ax.add_collection(lc)


def figure1_topology() -> None:
    metrics = pd.read_csv(C.TABLES / "Table_3_network_metrics.csv")
    fig = plt.figure(figsize=(11.4, 5.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.32], wspace=0.42, hspace=0.5)
    specs = [
        ("total_edge_composite_t", "Total network emission flux", "Health-weighted tonnes", C.ACCENT["blue"]),
        ("city_emission_gini", "City-emission concentration", "Gini coefficient", C.ACCENT["orange"]),
        ("leiden_weighted_modularity", "Regional modularity", "Weighted modularity $Q$", C.ACCENT["green"]),
        ("weighted_assortativity", "Strength assortativity", "Weighted $r$", C.ACCENT["purple"]),
    ]
    for letter, (key, title, ylabel, color), pos in zip("abcd", specs, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        ax = fig.add_subplot(gs[pos])
        ax.plot(metrics.year, metrics[key], color=color, lw=1.8, marker="o",
                ms=5.2, mec="white", mew=0.6, zorder=3)
        soft_axes(ax)
        ax.set_title(title, fontsize=9.6, pad=6)
        ax.set_ylabel(ylabel, fontsize=8.6)
        ax.set_xticks(YEARS)
        ax.margins(x=0.05, y=0.18)
        C.panel_label(ax, letter, dx=-0.15, dy=1.02)

    ax = fig.add_subplot(gs[:, 2], projection=ccrs.Robinson())
    base_map(ax)
    ax.plot(metrics.centroid_lon, metrics.centroid_lat, color=C.INK, lw=1.2,
            transform=ccrs.Geodetic(), zorder=3)
    sc = ax.scatter(metrics.centroid_lon, metrics.centroid_lat, c=metrics.year,
                    cmap="viridis", s=68, edgecolor="white", linewidth=0.8,
                    transform=ccrs.PlateCarree(), zorder=4)
    for row in metrics.itertuples():
        ax.text(row.centroid_lon + 2.3, row.centroid_lat + 1.0, str(int(row.year)),
                fontsize=7.5, fontweight="bold", transform=ccrs.PlateCarree(), zorder=5)
    ax.set_title("Emission centre-of-mass position", fontsize=9.6, pad=6)
    C.panel_label(ax, "e", dx=-0.08, dy=1.02)
    fig.suptitle(f"Global ship-emission network topology and spatial evolution, {YEARS[0]}–{YEARS[-1]}",
                 y=1.0, fontsize=11.5, fontweight="bold")
    save(fig, "figure_1", "Figure_1_network_topology.png")


def figure2_emission_evolution(annual: pd.DataFrame, coords: pd.DataFrame) -> None:
    wide = annual.pivot(index="node_id", columns="year", values="composite_t")
    slope = wide[YEARS].apply(lambda row: C.theil_sen_log_slope(row.to_numpy(float), YEARS), axis=1)
    change = (100 * np.expm1(slope)).replace([np.inf, -np.inf], np.nan).clip(-30, 30)
    latest = annual[annual.year.eq(LATEST)].copy().set_index("node_id")
    latest["change"] = change
    latest = latest.reset_index().dropna(subset=["rep_lon", "rep_lat", "change"])
    edges = pd.read_parquet(C.DATA / f"network_edges_{LATEST}.parquet")

    fig = plt.figure(figsize=(9.0, 4.55))
    ax = fig.add_subplot(111, projection=ccrs.Robinson())
    base_map(ax)
    add_routes(ax, edges, coords, n=1800, alpha=0.07)
    norm = TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30)
    sc = ax.scatter(latest.rep_lon, latest.rep_lat, c=latest.change,
                    s=marker_sizes(latest.composite_t, 7, 68), cmap="RdBu_r", norm=norm,
                    alpha=0.92, edgecolor="white", linewidth=0.35,
                    transform=ccrs.PlateCarree(), zorder=3)
    cb = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.025)
    cb.set_label("Theil–Sen annual change in city composite emission (% yr$^{-1}$)")
    ax.set_title(f"City-level emission evolution and shipping connections, {YEARS[0]}–{LATEST}", pad=8)
    legend = [
        Line2D([0], [0], color="#7899B0", lw=0.5, label="Lower route emission"),
        Line2D([0], [0], color="#7899B0", lw=2.0, label="Higher route emission"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#C4CBD1", markersize=5,
               markeredgecolor="none", label=f"Lower {LATEST} city emission"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#C4CBD1", markersize=10,
               markeredgecolor="none", label=f"Higher {LATEST} city emission"),
    ]
    ax.legend(handles=legend, loc="lower left", title="Route flux and city level", fontsize=7.8, title_fontsize=8.2)
    save(fig, "figure_2", "fig2a_city_emission_network_change.png")

    for i, year in enumerate(YEARS[1:], start=1):
        prev = annual[annual.year.eq(year - 1)][["node_id", "composite_t"]].rename(columns={"composite_t": "prev"})
        cur = annual[annual.year.eq(year)].merge(prev, on="node_id", how="left")
        cur["delta"] = cur.composite_t - cur.prev
        vmax = np.nanquantile(np.abs(cur.delta), 0.98)
        fig = plt.figure(figsize=(8.45, 4.2))
        ax = fig.add_subplot(111, projection=ccrs.Robinson())
        base_map(ax)
        sc = ax.scatter(cur.rep_lon, cur.rep_lat, c=cur.delta, s=marker_sizes(np.abs(cur.delta), 6, 60),
                        cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
                        alpha=0.9, edgecolor="white", linewidth=0.3,
                        transform=ccrs.PlateCarree(), zorder=2)
        cb = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.025)
        cb.set_label("Annual change in composite emission (t)")
        ax.set_title(f"City emission change, {year} minus {year - 1}", pad=8)
        save(fig, "figure_2", f"fig2b{i}_city_emission_change_{year}_minus_{year - 1}.png")


def aggregate_test_prediction():
    z = np.load(C.MODELS / "test_2025_predictions.npz", allow_pickle=True)
    weights = np.tile(np.array([2.0, 5.0, 1.0]), 4)
    pred = (z["pred"] * weights).sum(axis=2).sum(axis=0) / 1e6
    truth = (z["truth"] * weights).sum(axis=2).sum(axis=0) / 1e6
    return z["nodes"].astype(str), truth, pred, str(z["selected_model"])


def figure4_forecast(annual: pd.DataFrame, coords: pd.DataFrame) -> None:
    perf = pd.read_csv(C.TABLES / "Table_4_forecast_model_performance.csv")
    order = ["gwnet", "dcrnn", "agcrn", "HA", "LSTM", "ARIMA_restricted"]
    d = perf.set_index("model").reindex(order).dropna(subset=["WAPE_mean_pct"]).reset_index()
    names = {"gwnet": "Graph WaveNet", "dcrnn": "DCRNN", "agcrn": "AGCRN",
             "HA": "HA", "LSTM": "LSTM", "ARIMA_restricted": "Restricted ARIMA"}
    colors = [C.ACCENT["red"] if bool(x) else (C.ACCENT["blue"] if role == "GNN candidate" else "#A6AEB5")
              for x, role in zip(d.formal_selected, d.role)]
    fig, ax = plt.subplots(figsize=(7.55, 4.35))
    bars = ax.bar(np.arange(len(d)), d.WAPE_mean_pct, color=colors, width=0.72, zorder=2)
    ax.set_xticks(np.arange(len(d)), [names.get(x, x) for x in d.model], rotation=19, ha="right")
    ax.set_ylabel(f"{LATEST} test WAPE (%)")
    soft_axes(ax)
    for bar, value in zip(bars, d.WAPE_mean_pct):
        ax.text(bar.get_x() + bar.get_width()/2, value + max(d.WAPE_mean_pct)*0.018,
                f"{value:.1f}", ha="center", va="bottom")
    ax.legend(handles=[mpatches.Patch(color=C.ACCENT["red"], label="Selected formal GNN"),
                       mpatches.Patch(color=C.ACCENT["blue"], label="Other formal candidates"),
                       mpatches.Patch(color="#A6AEB5", label="Comparison only")],
              loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    ax.margins(y=0.16)
    save(fig, "figure_4", "fig4a_forecast_model_wape.png")

    nodes, truth, pred, model = aggregate_test_prediction()
    valid = np.isfinite(truth) & np.isfinite(pred) & (truth > 0) & (pred > 0)
    r = pearsonr(np.log1p(truth[valid]), np.log1p(pred[valid])).statistic
    wape = 100 * np.abs(pred[valid] - truth[valid]).sum() / truth[valid].sum()
    lo = min(truth[valid].min(), pred[valid].min()) * 0.85
    hi = max(truth[valid].max(), pred[valid].max()) * 1.15
    fig, ax = plt.subplots(figsize=(6.1, 5.45))
    ax.scatter(truth[valid], pred[valid], s=20, alpha=0.48, color="#6A9ED4", edgecolor="none")
    ax.plot([lo, hi], [lo, hi], "--", color=C.INK, lw=1.35)
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi),
           xlabel=f"Observed {LATEST} city composite emission (t)",
           ylabel=f"Predicted {LATEST} city composite emission (t)")
    ax.grid(True, color=C.GRID, lw=0.6)
    pretty_model = "Graph WaveNet" if model == "gwnet" else model.upper()
    ax.text(0.04, 0.96, f"{pretty_model}\nr = {r:.2f}; annual-composite WAPE = {wape:.1f}%",
            transform=ax.transAxes, va="top")
    save(fig, "figure_4", f"fig4b_observed_vs_predicted_{LATEST}.png")

    forecast = pd.read_csv(C.DATA / f"forecast_city_annual_{C.FORECAST_STEM}.csv").merge(coords, on="node_id", how="left")
    for letter, year in zip(["c", "d"], FORECAST_YEARS):
        fy = forecast[forecast.year.eq(year)].dropna(subset=["rep_lon", "rep_lat"])
        vmin = max(np.nanquantile(fy.composite_mean_t, 0.01), 1e-6)
        vmax = np.nanquantile(fy.composite_mean_t, 0.99)
        fig = plt.figure(figsize=(8.45, 4.2))
        ax = fig.add_subplot(111, projection=ccrs.Robinson())
        base_map(ax)
        sc = ax.scatter(fy.rep_lon, fy.rep_lat, c=fy.composite_mean_t,
                        s=marker_sizes(fy.composite_mean_t, 7, 68), cmap="YlOrRd",
                        norm=LogNorm(vmin=vmin, vmax=vmax), alpha=0.88,
                        edgecolor="white", linewidth=0.3,
                        transform=ccrs.PlateCarree(), zorder=2)
        cb = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.025)
        cb.set_label("Forecast city composite emission (t)")
        ax.set_title(f"City-level ship-emission forecast, {year}", pad=8)
        save(fig, "figure_4", f"fig4{letter}_forecast_map_{year}.png")

    obs = annual.groupby("year").composite_t.sum().reindex(YEARS) / 1e6
    zp = np.load(C.MODELS / f"forecast_paths_{C.FORECAST_STEM}.npz", allow_pickle=True)
    weights = np.tile(np.array([2.0, 5.0, 1.0]), 4)
    monthly = (zp["paths"] * weights).sum(axis=3).sum(axis=2)
    annual_paths = np.stack([monthly[:, :12].sum(axis=1), monthly[:, 12:].sum(axis=1)], axis=1) / 1e6
    fm = np.nanmean(annual_paths, axis=0)
    flo, fhi = np.nanquantile(annual_paths, [0.025, 0.975], axis=0)
    fig, ax = plt.subplots(figsize=(7.55, 4.3))
    ax.plot(YEARS, obs, color=C.ACCENT["blue"], marker="o", lw=1.8, label="Observed")
    ax.plot([LATEST, *FORECAST_YEARS], [obs.loc[LATEST], *fm], color=C.ACCENT["red"], marker="o", lw=1.8,
            label="Formal forecast")
    ax.fill_between(FORECAST_YEARS, flo, fhi, color=C.ACCENT["red"], alpha=0.18, label="95% interval")
    ax.axvline(LATEST + 0.5, color="#7B848C", ls="--", lw=1.0)
    ax.set_xticks(YEARS + FORECAST_YEARS)
    ax.set_ylabel("Global composite emission (million t)")
    soft_axes(ax)
    ax.legend(loc="best", ncol=3)
    ax.set_title("Observed global path and formal forecast")
    save(fig, "figure_4", "fig4e_global_forecast_path.png")


def figure5_risk(risk: pd.DataFrame) -> None:
    legend_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=QUAD_COLORS[k],
                             markeredgecolor="none", markersize=6, label=QUAD_LABELS[k])
                      for k in QUAD_COLORS]
    for i, year in enumerate(YEARS, start=1):
        d = risk[risk.year.eq(year)].dropna(subset=["rep_lon", "rep_lat"])
        fig = plt.figure(figsize=(9.1, 4.9))
        ax = fig.add_subplot(111, projection=ccrs.Robinson())
        base_map(ax)
        for q in QUAD_COLORS:
            s = d[d.quadrant.eq(q)]
            ax.scatter(s.rep_lon, s.rep_lat, s=marker_sizes(s.SPRI, 7, 60),
                       color=QUAD_COLORS[q], alpha=0.82, edgecolor="white", linewidth=0.3,
                       transform=ccrs.PlateCarree(), zorder=2)
        ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=4)
        ax.set_title(f"Spatial distribution of city risk quadrants, {year}", pad=8)
        save(fig, "figure_5", f"fig5a{i}_risk_quadrant_map_{year}.png")

    latest = risk[risk.year.eq(LATEST)].copy()
    latest["current_pct"] = latest.current_level.rank(pct=True) * 100
    latest["future_pct"] = latest.future_direction.rank(pct=True) * 100
    fig, ax = plt.subplots(figsize=(9.1, 5.35))
    ax.axvspan(0, 50, color=C.ACCENT["purple"], alpha=0.045)
    ax.axvspan(50, 100, color=C.ACCENT["red"], alpha=0.045)
    ax.scatter(latest.current_pct, latest.future_pct, s=15, color="#D6DCE1", alpha=0.68, edgecolor="none")
    ax.axvline(50, color="#737C84", ls="--", lw=1.0)
    ax.axhline(50, color="#737C84", ls="--", lw=1.0)
    hi = latest.nlargest(8, "SPRI")
    for row in hi.itertuples():
        ax.scatter(row.current_pct, row.future_pct, s=55, color=QUAD_COLORS[row.quadrant],
                   edgecolor="white", linewidth=0.8, zorder=4)
        ax.annotate(row.city, (row.current_pct, row.future_pct), xytext=(5, 4),
                    textcoords="offset points", fontsize=7.2)
    ax.text(3, 94, "Emerging risk\nlow current · high momentum", color=C.ACCENT["purple"],
            fontsize=11, fontweight="bold", va="top")
    ax.text(97, 94, "Established and rising", color="#A44642", fontsize=11,
            fontweight="bold", ha="right", va="top")
    ax.text(97, 5, "Established, stable/declining", color="#7D868E", fontsize=10,
            ha="right", va="bottom")
    ax.text(3, 5, "Low current · low momentum", color="#7D868E", fontsize=10, va="bottom")
    ax.set(xlim=(0, 100), ylim=(0, 100),
           xlabel=f"Current exposure percentile ({LATEST})",
           ylabel=f"Future-direction percentile ({LATEST} anchor)")
    ax.grid(True, color=C.GRID, lw=0.6)
    save(fig, "figure_5", "fig5b_current_momentum_quadrant.png")

    ranks = risk.pivot(index="node_id", columns="year", values="rank_full")
    ranks["rise"] = ranks[YEARS[0]] - ranks[LATEST]
    ranks["decline"] = ranks[LATEST] - ranks[YEARS[0]]
    rising = ranks[ranks[LATEST] <= 300].nlargest(6, "rise").index
    declining = ranks[ranks[YEARS[0]] <= 300].nlargest(6, "decline").index
    selected = list(dict.fromkeys([*rising, *declining]))
    meta = risk[risk.year.eq(LATEST)].set_index("node_id")
    fig, ax = plt.subplots(figsize=(9.2, 5.45))
    label_rows = []
    for node in selected:
        vals = ranks.loc[node, YEARS].to_numpy(float)
        rise = ranks.loc[node, "rise"] > 0
        color = C.ACCENT["blue"] if rise else "#66717A"
        ax.plot(YEARS, vals, color=color, lw=1.55, ls="-" if rise else "--", marker="o", ms=3.8)
        row = meta.loc[node]
        arrow = "↑" if rise else "↓"
        delta = abs(int(round(ranks.loc[node, "rise"] if rise else ranks.loc[node, "decline"])))
        label_rows.append({"node": node, "end": vals[-1], "text": f"{row.city} ({row.country}, {arrow}{delta})",
                           "color": color})
    # Allocate right-hand labels with a minimum vertical gap and connect them
    # back to the true 2025 endpoint.  This keeps the selected rank paths intact.
    label_rows.sort(key=lambda r: r["end"])
    label_y = []
    for item in label_rows:
        label_y.append(max(item["end"], (label_y[-1] + 34) if label_y else item["end"]))
    overflow = max(label_y[-1] - 1420, 0) if label_y else 0
    label_y = [y - overflow for y in label_y]
    for i in range(len(label_y) - 2, -1, -1):
        label_y[i] = min(label_y[i], label_y[i + 1] - 34)
    for item, ylab in zip(label_rows, label_y):
        ax.plot([LATEST, LATEST + 0.08], [item["end"], ylab], color=item["color"], lw=0.7)
        ax.text(LATEST + 0.1, ylab, item["text"], va="center", fontsize=7.0)
    ax.invert_yaxis()
    ax.set_xticks(YEARS)
    ax.set_xlim(YEARS[0] - 0.15, LATEST + 1.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Full-SPRI rank (1 = highest risk)")
    ax.grid(True, axis="y", color=C.GRID, lw=0.6)
    ax.legend(handles=[Line2D([0], [0], color=C.ACCENT["blue"], lw=1.8, label="Rank rising"),
                       Line2D([0], [0], color="#66717A", lw=1.8, ls="--", label="Rank declining")],
              loc="upper center", bbox_to_anchor=(0.58, 1.1), ncol=2)
    save(fig, "figure_5", "fig5c_risk_rank_trajectories.png")

    top = latest.nlargest(18, "SPRI").sort_values("SPRI")
    comps = top[["level_component", "history_component", "future_component"]].clip(lower=0)
    comps = comps.div(comps.sum(axis=1), axis=0) * 100
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    left = np.zeros(len(top))
    for col, label, color in [("level_component", "Current level", C.ACCENT["blue"]),
                              ("history_component", "Historical trajectory", C.ACCENT["orange"]),
                              ("future_component", "Future direction", C.ACCENT["purple"])]:
        ax.barh(y, comps[col], left=left, color=color, height=0.72, label=label)
        left += comps[col].to_numpy(float)
    ax.set_yticks(y, top.city)
    ax.set_xlabel("Share of temporal risk signal (%)")
    ax.set_xlim(0, 100)
    C.despine(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    ax.set_title(f"Temporal component decomposition of leading risk cities, {LATEST}")
    save(fig, "figure_5", "fig5d_temporal_component_decomposition.png")


def figure6_network(risk: pd.DataFrame) -> None:
    d = pd.read_csv(C.DATA / f"network_position_risk_{LATEST}.csv")
    x = np.log1p(d.betweenness_backbone)
    rho, p = spearmanr(x, d.SPRI, nan_policy="omit")
    fig, ax = plt.subplots(figsize=(7.3, 5.15))
    ax.scatter(x, d.SPRI, color="#5B91CB", alpha=0.55, s=20, edgecolor="none")
    ok = np.isfinite(x) & np.isfinite(d.SPRI)
    coef = np.polyfit(x[ok], d.loc[ok, "SPRI"], 1)
    xx = np.linspace(x[ok].min(), x[ok].max(), 100)
    ax.plot(xx, np.polyval(coef, xx), color=C.ACCENT["red"], lw=1.7)
    ax.text(0.04, 0.95, f"Spearman ρ = {rho:.3f}\np = {p:.2g}", transform=ax.transAxes, va="top")
    ax.set_xlabel("log(1 + backbone betweenness centrality)")
    ax.set_ylabel(f"SPRI ({LATEST})")
    ax.grid(True, color=C.GRID, lw=0.6)
    save(fig, "figure_6", "fig6a_network_centrality_risk.png")

    biggest = d.groupby("module").size().nlargest(8).index
    med = d[d.module.isin(biggest)].groupby("module").SPRI.median().sort_values()
    groups = [d.loc[d.module.eq(m), "SPRI"].dropna().to_numpy(float) for m in med.index]
    fig, ax = plt.subplots(figsize=(7.8, 5.15))
    bp = ax.boxplot(groups, patch_artist=True, widths=0.6, showfliers=False,
                    medianprops=dict(color=C.ACCENT["orange"], lw=1.4))
    colors = plt.cm.viridis(np.linspace(0.16, 0.82, len(groups)))
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color); box.set_alpha(0.72)
    ax.set_xticks(np.arange(1, len(groups) + 1), [f"M{int(m)}" for m in med.index])
    ax.set_xlabel("Leiden community (largest eight, ordered by median risk)")
    ax.set_ylabel(f"SPRI ({LATEST})")
    C.soft_grid(ax)
    C.despine(ax)
    save(fig, "figure_6", "fig6b_network_module_risk_distribution.png")

    flow = pd.read_csv(C.TABLES / "Table_6B_environmental_injustice_flow.csv")
    fig, ax = plt.subplots(figsize=(7.6, 4.45))
    for col, label, color in [("positive_unfair_flow", "Positive unfair flow", C.ACCENT["red"]),
                              ("reverse_flow", "Reverse flow", C.ACCENT["blue"]),
                              ("net_directional_flow", "Net directional flow", C.ACCENT["purple"])]:
        ax.plot(flow.year, flow[col] / 1e9, marker="o", lw=1.8, color=color, label=label)
    ax.axhline(0, color="#7B848C", lw=0.8)
    ax.set_xticks(YEARS)
    ax.set_ylabel("Health-weighted network flow (billion units)")
    soft_axes(ax)
    ax.legend(loc="best")
    ax.set_title("Environmental-injustice flow through the shipping network")
    save(fig, "figure_6", "fig6c_environmental_injustice_flow.png")


def supplementary_s1(annual: pd.DataFrame) -> None:
    global_y = annual.groupby("year")[["SO2_t", "PM_t", "NOx_t"]].sum().reindex(YEARS)
    idx = global_y.div(global_y.iloc[0]) * 100
    fig, ax = plt.subplots(figsize=(7.5, 4.35))
    for col, label, color in [("SO2_t", "SO$_2$", C.ACCENT["orange"]),
                              ("PM_t", "PM", C.ACCENT["red"]),
                              ("NOx_t", "NO$_x$", C.ACCENT["blue"])]:
        ax.plot(YEARS, idx[col], marker="o", lw=1.8, color=color, label=label)
    ax.axhline(100, color="#7B848C", ls="--", lw=0.8)
    ax.set_xticks(YEARS)
    ax.set_ylabel(f"Global emission index ({YEARS[0]} = 100)")
    soft_axes(ax)
    ax.legend(ncol=3)
    save(fig, "supplementary", "figS1a_annual_pollutant_index.png")

    cols = ["composite_ship_container_t", "composite_ship_general_cargo_t",
            "composite_ship_oil_tanker_t", "composite_ship_bulk_carrier_t"]
    labels = ["Container", "General cargo", "Oil tanker", "Bulk carrier"]
    colors = [C.ACCENT["blue"], C.ACCENT["teal"], C.ACCENT["orange"], C.ACCENT["purple"]]
    d = annual.groupby("year")[cols].sum().reindex(YEARS)
    share = d.div(d.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(7.6, 4.45))
    ax.stackplot(YEARS, *[share[c] for c in cols], labels=labels, colors=colors, alpha=0.84)
    ax.set_xticks(YEARS)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of composite emission (%)")
    C.despine(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=4)
    save(fig, "supplementary", "figS1b_shiptype_emission_share.png")


def main() -> None:
    annual = pd.read_parquet(C.DATA / f"emission_city_annual_{C.OBS_STEM}.parquet")
    risk = pd.read_parquet(C.DATA / f"SPRI_city_annual_{C.OBS_STEM}.parquet")
    coords = city_coords()
    figure1_topology()
    figure2_emission_evolution(annual, coords)
    figure4_forecast(annual, coords)
    figure5_risk(risk)
    figure6_network(risk)
    supplementary_s1(annual)
    print("v10 figure generator completed")


if __name__ == "__main__":
    main()
