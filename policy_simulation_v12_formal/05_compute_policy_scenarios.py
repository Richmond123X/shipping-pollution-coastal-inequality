# -*- coding: utf-8 -*-
"""由 V12 正式 AIS 增量表构造 S1--S7、岸电扫描及效果指标。"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from policy_run_config import (
    BASELINE_MONTHLY_ROOT as RESULTS_MONTHLY,
    CONSOLIDATED_ROOT as CON, ROOT, SCENARIO_ROOT as OUT,
    SIMULATION_YEAR, TRADE_YEAR,
)


COVARIATES = Path(r"E:\transfer justice\covariates\outputs\city_covariates_2019_2028.csv")
PORT_MAP = Path(r"E:\transfer justice\port2city\outputs\port_city_mapping.csv")
TRADE = Path(r"E:\transfer justice\empirical_trade_emissions\data_processed\trade_country_annual.parquet")
INCOME = Path(r"E:\transfer justice\empirical_trade_emissions\data_processed\world_bank_income_fy19.csv")
UN_REGION = Path(r"E:\transfer justice\empirical_trade_emissions\data_processed\un_m49_region_groups.csv")
GRID = ROOT / "inputs" / "grid_pollution_intensity_2022.csv"
SELECTED_GEO = ROOT / "tables" / "selected_geo_units_50pct.csv"
TABLES = ROOT / "tables"

SHIP_TYPES = ["Bulk carrier", "General cargo", "Oil tanker", "Container"]
POLS = ["SO2", "PM", "NOx"]
MODES = ["Berth", "Anchored", "Manoeuvring", "Sea"]
# 研究方案唯一口径：PM:SO2:NOx = 5:2:1。
WEIGHTS = {"SO2": 2.0, "PM": 5.0, "NOx": 1.0}
ALPHAS = [i / 10 for i in range(1, 11)]


def _load_baseline_long() -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    true_parts = []
    for month in range(1, 13):
        ym = f"{SIMULATION_YEAR}{month:02d}"
        for st in SHIP_TYPES:
            d = RESULTS_MONTHLY / ym / st
            p = d / f"city_emissions_{st}_{ym}.csv"
            q = d / f"true_totals_{st}_{ym}.csv"
            if not p.is_file() or not q.is_file():
                raise FileNotFoundError(
                    f"{SIMULATION_YEAR} 基线缺失: {p if not p.is_file() else q}"
                )
            x = pd.read_csv(p)
            recs = []
            for pol in POLS:
                for mode in MODES:
                    c = f"{pol}_{mode}"
                    recs.append(pd.DataFrame({
                        "node_id": x["node_id"].astype(str),
                        "Ship_Type": st,
                        "Mode": mode,
                        "Pollutant": pol,
                        "baseline_g": pd.to_numeric(x[c], errors="coerce").fillna(0.0),
                    }))
            parts.append(pd.concat(recs, ignore_index=True))
            t = pd.read_csv(q)
            true_parts.append(pd.DataFrame({
                "Ship_Type": st,
                "Pollutant": POLS,
                "baseline_g": [float(t[p].sum()) for p in POLS],
            }))
    city = pd.concat(parts, ignore_index=True).groupby(
        ["node_id", "Ship_Type", "Mode", "Pollutant"], as_index=False
    )["baseline_g"].sum()
    true = pd.concat(true_parts, ignore_index=True).groupby(
        ["Ship_Type", "Pollutant"], as_index=False
    )["baseline_g"].sum()
    return city, true


def _load_covariates() -> pd.DataFrame:
    c = pd.read_csv(COVARIATES)
    c["year"] = pd.to_numeric(c["year"], errors="coerce")
    c19 = c[c["year"] == 2019].copy()
    c_current = c[c["year"] == SIMULATION_YEAR].copy()
    keep19 = ["node_id", "AC1", "doctor_per_10k", "S2_vuln_share"]
    keep25 = ["node_id", "city", "country", "iso3", "pop_density_km2",
              "S1_pop", "S2_vuln_share"]
    out = c_current[keep25].merge(
        c19[keep19].rename(columns={
            "AC1": "gdp_pc_2019",
            "doctor_per_10k": "doctor_per_10k_2019",
            "S2_vuln_share": "vuln_share_2019",
        }),
        on="node_id", how="left", validate="1:1",
    )
    for var, outcol, ascending in (
        ("gdp_pc_2019", "gdp_quartile", True),
        ("doctor_per_10k_2019", "doctor_quartile", True),
        ("vuln_share_2019", "vuln_quartile", True),
    ):
        valid = out[var].notna()
        # rank(method=first) 只用于打破完全相同值，切点仍由 2019 全部城市决定。
        out.loc[valid, outcol] = pd.qcut(
            out.loc[valid, var].rank(method="first"), 4,
            labels=["Q1", "Q2", "Q3", "Q4"],
        ).astype(str).values
    return out


def _area_city_long() -> tuple[pd.DataFrame, pd.DataFrame]:
    city = pd.read_parquet(CON / f"city_policy_delta_{SIMULATION_YEAR}.parquet")
    totals = pd.read_parquet(CON / f"policy_delta_totals_{SIMULATION_YEAR}.parquet")
    mode_map = {
        "Berth_Class": "Berth", "Anchored_Class": "Anchored",
        "Manoeuvring_Class": "Manoeuvring", "Sea_Class": "Sea",
    }
    rows = []
    for scenario, pol, col in (
        ("S1", "SO2", "D_SO2_SECA"),
        ("S1", "PM", "D_PM_SECA"),
        ("S2", "NOx", "D_NOx_NECA"),
    ):
        rows.append(pd.DataFrame({
            "scenario": scenario,
            "node_id": city["node_id"].astype(str),
            "Ship_Type": city["Ship_Type"].astype(str),
            "Mode": city["Mode_Class"].map(mode_map),
            "Pollutant": pol,
            "reduction_g": pd.to_numeric(city[col], errors="coerce").fillna(0.0),
        }))
    city_long = pd.concat(rows, ignore_index=True)
    city_long = city_long[city_long["reduction_g"] != 0]

    exact_rows = []
    for scenario, pol, col in (
        ("S1", "SO2", "D_SO2_SECA"),
        ("S1", "PM", "D_PM_SECA"),
        ("S2", "NOx", "D_NOx_NECA"),
    ):
        exact_rows.append({
            "scenario": scenario,
            "Pollutant": pol,
            "reduction_g": float(totals[col].sum()),
        })
    exact = pd.DataFrame(exact_rows)
    return city_long, exact


def _prepare_ops() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calls = pd.read_parquet(CON / f"ops_calls_container_{SIMULATION_YEAR}.parquet")
    call_city = pd.read_parquet(
        CON / f"ops_call_city_container_{SIMULATION_YEAR}.parquet"
    )
    pmap = pd.read_csv(PORT_MAP, encoding="utf-8-sig")
    pm = (pmap[["Port_Name", "iso3", "country"]]
          .drop_duplicates("Port_Name")
          .rename(columns={"iso3": "geo_iso3", "country": "geo_country"}))
    # 海外属地按正式区划的政策责任国核算电网替代排放和逐国结果；港口的
    # 实际地理代码另行保留。其余地理单元均为 geo_iso3 == policy_iso3。
    gm = pd.read_csv(SELECTED_GEO)[
        ["geo_iso3", "policy_iso3", "policy_country"]
    ].drop_duplicates()
    if gm["geo_iso3"].duplicated().any():
        raise ValueError("正式地理单元到政策责任国映射不是多对一")
    pm = pm.merge(gm, on="geo_iso3", how="left", validate="m:1")
    pm["iso3"] = pm["policy_iso3"].fillna(pm["geo_iso3"])
    pm["country"] = pm["policy_country"].fillna(pm["geo_country"])
    calls = calls.merge(pm, on="Port_Name", how="left", validate="m:1")
    calls["eligible"] = (
        (pd.to_numeric(calls["Gross_Tonnage"], errors="coerce") > 5000)
        & (pd.to_numeric(calls["Strict_Berth_Hours"], errors="coerce") >= 2.0)
        & (calls["Ship_Type"].astype(str).str.casefold() == "container")
    )
    eligible = calls[calls["eligible"]].copy()
    eligible = eligible.sort_values(
        ["Port_Name", "OPS_Energy_kWh", "Call_ID"],
        ascending=[True, False, True],
    )
    eligible["rank_in_port"] = eligible.groupby("Port_Name").cumcount() + 1
    eligible["n_eligible_port"] = eligible.groupby("Port_Name")["Call_ID"].transform("size")
    call_city = call_city.merge(
        eligible[["Call_ID", "rank_in_port", "n_eligible_port", "iso3"]],
        on="Call_ID", how="inner", validate="m:1",
    )
    return calls, eligible, call_city


def _selected_ops(eligible: pd.DataFrame, alpha: float) -> pd.DataFrame:
    n_take = np.ceil(alpha * eligible["n_eligible_port"]).astype(int)
    return eligible[eligible["rank_in_port"] <= n_take].copy()


def _ops_city_long(call_city: pd.DataFrame, selected_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = call_city[call_city["Call_ID"].isin(selected_ids)].copy()
    gross = []
    overlap = []
    for pol, bcol, dcol in (
        ("SO2", "Berth_SO2_g", "D_SO2_SECA"),
        ("PM", "Berth_PM_g", "D_PM_SECA"),
        ("NOx", "Berth_NOx_g", "D_NOx_NECA"),
    ):
        gross.append(pd.DataFrame({
            "node_id": x["node_id"], "Ship_Type": "Container", "Mode": "Berth",
            "Pollutant": pol, "reduction_g": x[bcol],
        }))
        overlap.append(pd.DataFrame({
            "node_id": x["node_id"], "Ship_Type": "Container", "Mode": "Berth",
            "Pollutant": pol, "overlap_g": x[dcol],
        }))
    g = pd.concat(gross, ignore_index=True).groupby(
        ["node_id", "Ship_Type", "Mode", "Pollutant"], as_index=False
    )["reduction_g"].sum()
    o = pd.concat(overlap, ignore_index=True).groupby(
        ["node_id", "Ship_Type", "Mode", "Pollutant"], as_index=False
    )["overlap_g"].sum()
    return g, o


def _combine_city(area: pd.DataFrame, eligible: pd.DataFrame,
                  call_city: pd.DataFrame) -> pd.DataFrame:
    keys = ["node_id", "Ship_Type", "Mode", "Pollutant"]
    s1 = area[area["scenario"] == "S1"].groupby(keys, as_index=False)["reduction_g"].sum()
    s2 = area[area["scenario"] == "S2"].groupby(keys, as_index=False)["reduction_g"].sum()

    def add(label: str, d: pd.DataFrame, out: list[pd.DataFrame]) -> None:
        z = d.copy()
        z.insert(0, "scenario", label)
        out.append(z)

    out = []
    add("S1", s1, out)
    add("S2", s2, out)
    s4 = pd.concat([s1, s2]).groupby(keys, as_index=False)["reduction_g"].sum()
    add("S4", s4, out)

    for alpha in ALPHAS:
        selected = _selected_ops(eligible, alpha)
        ids = set(selected["Call_ID"])
        ops, overlap = _ops_city_long(call_city, ids)
        add(f"S3_{int(alpha*100):03d}", ops, out)
        if alpha == 1.0:
            # 岸电优先：面状政策 + 靠泊基线 - 同一靠泊已含的面状 delta。
            ov = overlap.rename(columns={"overlap_g": "reduction_g"})
            for label, base, pols in (
                ("S5", s1, {"SO2", "PM"}),
                ("S6", s2, {"NOx"}),
                ("S7", s4, {"SO2", "PM", "NOx"}),
            ):
                relevant_ov = ov[ov["Pollutant"].isin(pols)].copy()
                combo = pd.concat([
                    base.assign(_sign=1.0),
                    ops.assign(_sign=1.0),
                    relevant_ov.assign(_sign=-1.0),
                ], ignore_index=True)
                combo["reduction_g"] *= combo.pop("_sign")
                combo = combo.groupby(keys, as_index=False)["reduction_g"].sum()
                add(label, combo, out)
    return pd.concat(out, ignore_index=True)


def _combine_exact(area_exact: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    out = [area_exact.copy()]
    s1 = area_exact[area_exact["scenario"] == "S1"].set_index("Pollutant")["reduction_g"]
    s2 = area_exact[area_exact["scenario"] == "S2"].set_index("Pollutant")["reduction_g"]
    out.append(pd.DataFrame({
        "scenario": "S4",
        "Pollutant": POLS,
        "reduction_g": [float(s1.get(p, 0.0)) + float(s2.get(p, 0.0)) for p in POLS],
    }))
    for alpha in ALPHAS:
        selected = _selected_ops(eligible, alpha)
        gross = {
            p: float(selected[f"Berth_{p}_g"].sum()) for p in POLS
        }
        overlap = {
            "SO2": float(selected["D_SO2_SECA"].sum()),
            "PM": float(selected["D_PM_SECA"].sum()),
            "NOx": float(selected["D_NOx_NECA"].sum()),
        }
        out.append(pd.DataFrame({
            "scenario": f"S3_{int(alpha*100):03d}",
            "Pollutant": POLS,
            "reduction_g": [gross[p] for p in POLS],
        }))
        if alpha == 1.0:
            for label, a1, a2 in (("S5", True, False), ("S6", False, True), ("S7", True, True)):
                vals = {}
                for p in POLS:
                    area = (float(s1.get(p, 0.0)) if a1 else 0.0) + (
                        float(s2.get(p, 0.0)) if a2 else 0.0
                    )
                    ov = overlap[p] if ((a1 and p in {"SO2", "PM"}) or (a2 and p == "NOx")) else 0.0
                    vals[p] = area + gross[p] - ov
                out.append(pd.DataFrame({
                    "scenario": label, "Pollutant": POLS,
                    "reduction_g": [vals[p] for p in POLS],
                }))
    exact = pd.concat(out, ignore_index=True)
    return exact.groupby(["scenario", "Pollutant"], as_index=False)["reduction_g"].sum()


def _combine_exact_detail(policy_totals: pd.DataFrame,
                          eligible: pd.DataFrame) -> pd.DataFrame:
    """守恒口径的情景×船型×工况×污染物分解。"""
    mode_map = {
        "Berth_Class": "Berth", "Anchored_Class": "Anchored",
        "Manoeuvring_Class": "Manoeuvring", "Sea_Class": "Sea",
    }
    keys = ["Ship_Type", "Mode", "Pollutant"]
    area_rows = []
    for scenario, pol, col in (
        ("S1", "SO2", "D_SO2_SECA"),
        ("S1", "PM", "D_PM_SECA"),
        ("S2", "NOx", "D_NOx_NECA"),
    ):
        area_rows.append(pd.DataFrame({
            "scenario": scenario,
            "Ship_Type": policy_totals["Ship_Type"].astype(str),
            "Mode": policy_totals["Mode_Class"].map(mode_map),
            "Pollutant": pol,
            "reduction_g": pd.to_numeric(policy_totals[col], errors="coerce").fillna(0.0),
        }))
    area = pd.concat(area_rows, ignore_index=True).groupby(
        ["scenario", *keys], as_index=False
    )["reduction_g"].sum()
    s1 = area[area["scenario"].eq("S1")].drop(columns="scenario")
    s2 = area[area["scenario"].eq("S2")].drop(columns="scenario")
    out = [area]
    s4 = pd.concat([s1, s2], ignore_index=True).groupby(keys, as_index=False)[
        "reduction_g"
    ].sum()
    out.append(s4.assign(scenario="S4"))
    for alpha in ALPHAS:
        selected = _selected_ops(eligible, alpha)
        ops = pd.DataFrame({
            "Ship_Type": "Container", "Mode": "Berth",
            "Pollutant": POLS,
            "reduction_g": [float(selected[f"Berth_{p}_g"].sum()) for p in POLS],
        })
        overlap = pd.DataFrame({
            "Ship_Type": "Container", "Mode": "Berth",
            "Pollutant": POLS,
            "reduction_g": [
                float(selected["D_SO2_SECA"].sum()),
                float(selected["D_PM_SECA"].sum()),
                float(selected["D_NOx_NECA"].sum()),
            ],
        })
        out.append(ops.assign(scenario=f"S3_{int(alpha * 100):03d}"))
        if alpha == 1.0:
            for label, base, pols in (
                ("S5", s1, {"SO2", "PM"}),
                ("S6", s2, {"NOx"}),
                ("S7", s4, {"SO2", "PM", "NOx"}),
            ):
                ov = overlap[overlap["Pollutant"].isin(pols)].copy()
                combo = pd.concat([
                    base.assign(_sign=1.0), ops.assign(_sign=1.0),
                    ov.assign(_sign=-1.0),
                ], ignore_index=True)
                combo["reduction_g"] *= combo.pop("_sign")
                combo = combo.groupby(keys, as_index=False)["reduction_g"].sum()
                out.append(combo.assign(scenario=label))
    detail = pd.concat(out, ignore_index=True)
    return detail[["scenario", *keys, "reduction_g"]].groupby(
        ["scenario", *keys], as_index=False
    )["reduction_g"].sum()


def _knee(curve: pd.DataFrame, ycol: str) -> dict:
    c = curve.sort_values("alpha").copy()
    x = c["alpha"].to_numpy(float)
    y = c[ycol].to_numpy(float)
    if len(c) < 3 or not np.isfinite(y).all() or y[-1] <= 0:
        return {"criterion": ycol, "kneedle_alpha": np.nan,
                "auxiliary_alpha": np.nan, "selected_alpha": np.nan,
                "adjacent_stable": False, "perturbed_knees": "",
                "reported": "no clear knee"}

    def raw_kneedle(xx: np.ndarray, yy: np.ndarray) -> float:
        # 累计减排曲线的自然原点是 (0,0)，即使正式扫描从 10% 起也必须纳入
        # 归一化，否则首档被误当作新的原点并机械推迟拐点。
        xx = np.r_[0.0, xx]
        yy = np.r_[0.0, yy]
        if yy[-1] <= 0 or xx[-1] <= 0:
            return np.nan
        diff = yy / yy[-1] - xx / xx[-1]
        if len(diff) <= 2:
            return np.nan
        idx0 = int(np.argmax(diff[1:-1]) + 1)
        return float(xx[idx0])

    k_alpha = raw_kneedle(x, y)
    increments = np.diff(np.r_[0.0, y])
    threshold = increments[0] / 3.0
    aux_idx = np.where(increments <= threshold)[0]
    aux_alpha = float(x[aux_idx[0]]) if len(aux_idx) else np.nan

    k_idx = int(np.argmin(np.abs(x - k_alpha))) if np.isfinite(k_alpha) else -1
    perturb = []
    for drop_idx in (k_idx - 1, k_idx, k_idx + 1):
        if 0 <= drop_idx < len(x):
            keep = np.ones(len(x), dtype=bool)
            keep[drop_idx] = False
            perturb.append(raw_kneedle(x[keep], y[keep]))
    finite_perturb = [a for a in perturb if np.isfinite(a)]
    adjacent_stable = bool(
        finite_perturb
        and all(abs(a - k_alpha) <= 0.1 + 1e-12 for a in finite_perturb)
    )
    selected_alpha = np.nan
    if (np.isfinite(aux_alpha) and np.isfinite(k_alpha)
            and abs(aux_alpha - k_alpha) <= 0.1 + 1e-12
            and adjacent_stable):
        reported = f"{k_alpha:.1f}"
        selected_alpha = k_alpha
    elif np.isfinite(aux_alpha):
        reported = f"{min(k_alpha, aux_alpha):.1f}-{max(k_alpha, aux_alpha):.1f}"
    else:
        reported = "no clear knee"
    return {"criterion": ycol, "kneedle_alpha": k_alpha,
            "auxiliary_alpha": aux_alpha, "selected_alpha": selected_alpha,
            "adjacent_stable": adjacent_stable,
            "perturbed_knees": ";".join(f"{a:.1f}" for a in finite_perturb),
            "reported": reported}


def _ci_equal(y: pd.Series, rank_var: pd.Series) -> float:
    d = pd.DataFrame({"y": y, "rank": rank_var}).dropna()
    d = d[d["y"] >= 0].sort_values("rank")
    n = len(d)
    mu = float(d["y"].mean()) if n else 0.0
    if n == 0 or mu == 0:
        return np.nan
    r = (np.arange(n) + 0.5) / n
    return float(2.0 * np.mean(d["y"].to_numpy() * r) / mu - 1.0)


def _metrics(city_scn: pd.DataFrame, exact: pd.DataFrame,
             baseline_city: pd.DataFrame, true_base: pd.DataFrame,
             cov: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_np = baseline_city.groupby(["node_id", "Pollutant"], as_index=False)["baseline_g"].sum()
    base_np["weighted"] = base_np["baseline_g"] * base_np["Pollutant"].map(WEIGHTS)
    base_w = base_np.groupby("node_id", as_index=False)["weighted"].sum().rename(
        columns={"weighted": "base_weighted_g"})
    base_w = cov.merge(base_w, on="node_id", how="left")
    base_w["base_weighted_g"] = base_w["base_weighted_g"].fillna(0.0)
    base_w["base_pwe"] = base_w["base_weighted_g"] * base_w["pop_density_km2"].fillna(0.0)

    red_np = city_scn.groupby(["scenario", "node_id", "Pollutant"], as_index=False)["reduction_g"].sum()
    red_np["weighted"] = red_np["reduction_g"] * red_np["Pollutant"].map(WEIGHTS)
    red_w = red_np.groupby(["scenario", "node_id"], as_index=False)["weighted"].sum().rename(
        columns={"weighted": "reduction_weighted_g"})
    panel = pd.MultiIndex.from_product(
        [sorted(city_scn["scenario"].unique()), base_w["node_id"]],
        names=["scenario", "node_id"],
    ).to_frame(index=False).merge(base_w, on="node_id", how="left").merge(
        red_w, on=["scenario", "node_id"], how="left")
    panel["reduction_weighted_g"] = panel["reduction_weighted_g"].fillna(0.0)
    over = panel["reduction_weighted_g"] - panel["base_weighted_g"]
    if (over > np.maximum(1e-3, panel["base_weighted_g"] * 1e-8)).any():
        bad = panel.loc[over.nlargest(10).index]
        raise RuntimeError(f"城市削减量超过基线:\n{bad.to_string(index=False)}")
    panel["reduction_weighted_g"] = np.minimum(
        panel["reduction_weighted_g"], panel["base_weighted_g"])
    panel["reduction_pwe"] = (
        panel["reduction_weighted_g"] * panel["pop_density_km2"].fillna(0.0)
    )
    panel["post_pwe"] = panel["base_pwe"] - panel["reduction_pwe"]

    exact_w = exact.assign(
        weighted=lambda d: d["reduction_g"] * d["Pollutant"].map(WEIGHTS)
    ).groupby("scenario", as_index=False)["weighted"].sum().rename(
        columns={"weighted": "exact_weighted_reduction_g"})
    true_w = float((true_base["baseline_g"] * true_base["Pollutant"].map(WEIGHTS)).sum())
    rows = []
    for sc, g in panel.groupby("scenario"):
        delta_pwe = float(g["reduction_pwe"].sum())
        delta_city = float(g["reduction_weighted_g"].sum())
        vuln_abs = float((g["reduction_pwe"] * g["S2_vuln_share"].fillna(0.0)).sum())
        q = g.groupby("gdp_quartile", observed=False).agg(
            delta=("reduction_pwe", "sum"), base=("base_pwe", "sum")
        )
        qrate = q["delta"] / q["base"].replace(0, np.nan)
        rows.append({
            "scenario": sc,
            "city_weighted_reduction_g": delta_city,
            "delta_pwe": delta_pwe,
            "exposure_efficiency": delta_pwe / delta_city if delta_city else np.nan,
            "vulnerable_benefit_share": vuln_abs / delta_pwe if delta_pwe else np.nan,
            "vulnerable_benefit_absolute": vuln_abs,
            "gdp_q1_reduction_rate": qrate.get("Q1", np.nan),
            "gdp_q4_reduction_rate": qrate.get("Q4", np.nan),
            "q1_q4_rate_ratio": (
                qrate.get("Q1", np.nan) / qrate.get("Q4", np.nan)
                if pd.notna(qrate.get("Q4", np.nan))
                and float(qrate.get("Q4", np.nan)) != 0.0 else np.nan
            ),
            "ci_pwe_baseline": _ci_equal(g["base_pwe"], g["gdp_pc_2019"]),
            "ci_pwe_post": _ci_equal(g["post_pwe"], g["gdp_pc_2019"]),
        })
    summary = pd.DataFrame(rows).merge(exact_w, on="scenario", how="left")
    summary["global_baseline_weighted_g"] = true_w
    true_by_pol = true_base.groupby("Pollutant")["baseline_g"].sum()
    for pol in POLS:
        summary[f"global_baseline_{pol}_g"] = float(true_by_pol.get(pol, 0.0))
    summary["global_baseline_pwe"] = float(base_w["base_pwe"].sum())
    summary["global_baseline_vulnerable_pwe"] = float(
        (base_w["base_pwe"] * base_w["S2_vuln_share"].fillna(0.0)).sum()
    )
    summary["global_weighted_reduction_share"] = (
        summary["exact_weighted_reduction_g"] / true_w
    )
    summary["ci_pwe_change"] = summary["ci_pwe_post"] - summary["ci_pwe_baseline"]

    groups = []
    for var in ("gdp_quartile", "doctor_quartile", "vuln_quartile"):
        z = panel.groupby(["scenario", var], observed=False).agg(
            reduction_pwe=("reduction_pwe", "sum"), base_pwe=("base_pwe", "sum")
        ).reset_index().rename(columns={var: "quartile"})
        z["grouping"] = var
        z["reduction_rate"] = z["reduction_pwe"] / z["base_pwe"].replace(0, np.nan)
        z["distribution_share"] = z["reduction_pwe"] / z.groupby(
            "scenario")["reduction_pwe"].transform("sum").replace(0, np.nan)
        groups.append(z)
    return summary, panel, pd.concat(groups, ignore_index=True)


def _ops_scan(eligible: pd.DataFrame, call_city: pd.DataFrame,
              cov: pd.DataFrame, grid: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    country_rows = []
    density = cov.set_index("node_id")["pop_density_km2"].fillna(0.0)
    for alpha in ALPHAS:
        s = _selected_ops(eligible, alpha)
        ids = set(s["Call_ID"])
        gross = sum(
            WEIGHTS[p] * float(s[f"Berth_{p}_g"].sum()) for p in POLS
        )
        cc = call_city[call_city["Call_ID"].isin(ids)]
        node_comp = sum(
            WEIGHTS[p] * cc.groupby("node_id")[f"Berth_{p}_g"].sum()
            for p in POLS
        )
        exposure = float((node_comp * density.reindex(node_comp.index).fillna(0.0)).sum())
        demand = float(s["OPS_Energy_kWh"].sum())
        dwt = float(s["DWT"].sum())
        rec = {
            "alpha": alpha,
            "nominal_calls": len(s),
            "actual_call_share": len(s) / len(eligible) if len(eligible) else 0.0,
            "ops_energy_kwh": demand,
            "energy_share": demand / eligible["OPS_Energy_kWh"].sum() if len(eligible) else 0.0,
            "dwt_sum": dwt,
            "dwt_share": dwt / eligible["DWT"].sum() if len(eligible) else 0.0,
            "gross_weighted_reduction_g": gross,
            "exposure_reduction": exposure,
            "net_weighted_reduction_g": np.nan,
            "net_comparable_gross_reduction_g": np.nan,
            "net_composite_includes_pm": False,
            "net_composite_basis": np.nan,
        }
        if grid is not None:
            sum_cols = ["OPS_Energy_kWh"] + [f"Berth_{p}_g" for p in POLS]
            cg = s.groupby("iso3", as_index=False)[sum_cols].sum().merge(
                grid, on="iso3", how="left", validate="m:1")
            missing = cg[cg[["SO2_g_per_kWh", "NOx_g_per_kWh"]].isna().any(axis=1)]
            if len(missing):
                raise RuntimeError(f"入选岸电国家缺电网强度: {sorted(missing.iso3.astype(str))}")
            corrected = cg["OPS_Energy_kWh"] / 0.92
            cg["grid_SO2_g"] = corrected * cg["SO2_g_per_kWh"]
            cg["grid_NOx_g"] = corrected * cg["NOx_g_per_kWh"]
            cg["net_SO2_g"] = cg["Berth_SO2_g"] - cg["grid_SO2_g"]
            cg["net_NOx_g"] = cg["Berth_NOx_g"] - cg["grid_NOx_g"]
            grid_weighted = 2 * cg["grid_SO2_g"] + cg["grid_NOx_g"]
            net_basis_gross = (
                2 * float(s["Berth_SO2_g"].sum())
                + float(s["Berth_NOx_g"].sum())
            )
            # PM 只有明确口径可比时才并入净综合量。
            pm_ok = False
            if "PM_comparable" in cg:
                pm_flag = cg["PM_comparable"]
                if pm_flag.dtype == object:
                    pm_flag = pm_flag.astype(str).str.strip().str.casefold().isin(
                        {"true", "1", "yes", "y"}
                    )
                pm_ok = bool(pm_flag.fillna(False).all())
            if pm_ok:
                cg["grid_PM_g"] = corrected * cg["PM_g_per_kWh"]
                cg["net_PM_g"] = cg["Berth_PM_g"] - cg["grid_PM_g"]
                grid_weighted += 5 * cg["grid_PM_g"]
                net_basis_gross += 5 * float(s["Berth_PM_g"].sum())
                rec["net_composite_includes_pm"] = True
            else:
                cg["grid_PM_g"] = np.nan
                cg["net_PM_g"] = np.nan
            rec["net_weighted_reduction_g"] = net_basis_gross - float(grid_weighted.sum())
            rec["net_comparable_gross_reduction_g"] = net_basis_gross
            rec["net_composite_basis"] = (
                "SO2+PM+NOx (2:5:1)" if pm_ok else "SO2+NOx (2:1; PM unavailable)"
            )
            cg["alpha"] = alpha
            cg["grid_weighted_emission_g"] = grid_weighted
            country_rows.append(cg)
        rows.append(rec)
    return pd.DataFrame(rows), (
        pd.concat(country_rows, ignore_index=True) if country_rows else pd.DataFrame()
    )


def _grouped_ops_knees(eligible: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按污染物、2019 收入组和联合国区域报告岸电毛效应拐点。"""
    income = pd.read_csv(INCOME)[["iso3", "income_group_2019"]]
    region = pd.read_csv(UN_REGION)[["iso3", "region_group_5"]]
    e = eligible.merge(income, on="iso3", how="left", validate="m:1").merge(
        region, on="iso3", how="left", validate="m:1"
    )
    e["income_group_2019"] = e["income_group_2019"].fillna("Missing")
    e["region_group_5"] = e["region_group_5"].fillna("Missing")
    rows = []
    for alpha in ALPHAS:
        s = _selected_ops(e, alpha)
        for pol in POLS:
            rows.append({
                "grouping": "Pollutant", "group": pol, "alpha": alpha,
                "gross_reduction_g": float(s[f"Berth_{pol}_g"].sum()),
            })
        s = s.copy()
        s["weighted_g"] = sum(
            WEIGHTS[p] * pd.to_numeric(s[f"Berth_{p}_g"], errors="coerce").fillna(0.0)
            for p in POLS
        )
        for grouping, col in (
            ("Income group (2019)", "income_group_2019"),
            ("UN region", "region_group_5"),
        ):
            for group, value in s.groupby(col, dropna=False)["weighted_g"].sum().items():
                rows.append({
                    "grouping": grouping, "group": group, "alpha": alpha,
                    "gross_reduction_g": float(value),
                })
    scan = pd.DataFrame(rows)
    keys = scan[["grouping", "group"]].drop_duplicates()
    full = keys.assign(_key=1).merge(
        pd.DataFrame({"alpha": ALPHAS, "_key": 1}), on="_key"
    ).drop(columns="_key")
    scan = full.merge(scan, on=["grouping", "group", "alpha"], how="left")
    scan["gross_reduction_g"] = scan["gross_reduction_g"].fillna(0.0)
    knees = []
    for (grouping, group), d in scan.groupby(["grouping", "group"], sort=True):
        rec = _knee(d, "gross_reduction_g")
        rec.update({"grouping": grouping, "group": group})
        knees.append(rec)
    return scan, pd.DataFrame(knees)


def _build_table6(metrics: pd.DataFrame, exact: pd.DataFrame,
                  knee: dict, policy_totals: pd.DataFrame,
                  country_grid: pd.DataFrame | None = None) -> pd.DataFrame:
    selected_alpha = knee.get("selected_alpha", np.nan)
    alpha_star = (
        selected_alpha if np.isfinite(selected_alpha)
        else knee.get("kneedle_alpha", np.nan)
    )
    s3 = f"S3_{int(round(alpha_star * 100)):03d}" if np.isfinite(alpha_star) else None
    s3_label = (
        "S3(alpha*)" if np.isfinite(selected_alpha)
        else "S3(kneedle; no unique alpha*)"
    )
    main = ["S1", "S2"] + ([s3] if s3 else []) + ["S4", "S5", "S6", "S7"]
    ep = exact.pivot(index="scenario", columns="Pollutant", values="reduction_g")
    denom = {
        "SO2": float(policy_totals["B_SO2"].sum()),
        "PM": float(policy_totals["B_PM"].sum()),
        "NOx": float(policy_totals["B_NOx"].sum()),
    }
    metric_idx = metrics.set_index("scenario")

    def row_for(sc: str, label: str | None = None) -> dict:
        r = {"row_type": "scenario", "scenario": label or sc,
             "source_scenario": sc}
        for pol in POLS:
            grams = float(ep.loc[sc, pol]) if sc in ep.index and pol in ep.columns else 0.0
            r[f"{pol}_reduction_t"] = grams / 1e6
            r[f"{pol}_policy_zone_reduction_rate"] = grams / denom[pol] if denom[pol] else np.nan
        if sc in metric_idx.index:
            r.update(metric_idx.loc[sc].to_dict())
        return r

    base_ref = metric_idx.iloc[0] if len(metric_idx) else pd.Series(dtype=float)
    baseline = {
        "row_type": "baseline", "scenario": "Baseline",
        "source_scenario": "Baseline",
        "city_weighted_reduction_g": 0.0,
        "delta_pwe": 0.0,
        "vulnerable_benefit_absolute": 0.0,
        "global_weighted_reduction_share": 0.0,
        "global_baseline_weighted_g": base_ref.get("global_baseline_weighted_g", np.nan),
        "global_baseline_pwe": base_ref.get("global_baseline_pwe", np.nan),
        "global_baseline_vulnerable_pwe": base_ref.get(
            "global_baseline_vulnerable_pwe", np.nan),
    }
    for pol in POLS:
        baseline[f"{pol}_policy_zone_baseline_t"] = denom[pol] / 1e6
        baseline[f"{pol}_reduction_t"] = 0.0
        baseline[f"{pol}_policy_zone_reduction_rate"] = 0.0
    rows = [baseline]
    rows.extend(row_for(sc, s3_label if sc == s3 else sc) for sc in main)
    table = pd.DataFrame(rows)
    for pol in POLS:
        table[f"{pol}_policy_zone_baseline_t"] = denom[pol] / 1e6
    table["global_baseline_pwe"] = base_ref.get("global_baseline_pwe", np.nan)
    table["global_baseline_vulnerable_pwe"] = base_ref.get(
        "global_baseline_vulnerable_pwe", np.nan)

    # 岸电在三种面状政策背景下的边际贡献。线性量做差，比例以同一分母做差；
    # 效率和受益份额由做差后的分子、分母重新计算。
    for label, hi, lo in (("S5-S1", "S5", "S1"),
                          ("S6-S2", "S6", "S2"),
                          ("S7-S4", "S7", "S4")):
        h, l = table[table["source_scenario"] == hi].iloc[0], table[
            table["source_scenario"] == lo].iloc[0]
        rec = {"row_type": "ops_marginal", "scenario": label,
               "source_scenario": label}
        linear = [c for c in table.columns if c.endswith("_reduction_t")
                  or c.endswith("_reduction_rate")
                  or c in {"city_weighted_reduction_g", "delta_pwe",
                           "vulnerable_benefit_absolute",
                           "exact_weighted_reduction_g",
                           "global_weighted_reduction_share"}]
        for c in linear:
            rec[c] = float(h.get(c, 0.0)) - float(l.get(c, 0.0))
        den_city = rec.get("city_weighted_reduction_g", 0.0)
        den_pwe = rec.get("delta_pwe", 0.0)
        rec["exposure_efficiency"] = (
            rec.get("delta_pwe", 0.0) / den_city if den_city else np.nan)
        rec["vulnerable_benefit_share"] = (
            rec.get("vulnerable_benefit_absolute", 0.0) / den_pwe
            if den_pwe else np.nan)
        rec["gdp_q1_reduction_rate"] = float(h.get("gdp_q1_reduction_rate", 0)) - float(
            l.get("gdp_q1_reduction_rate", 0))
        rec["gdp_q4_reduction_rate"] = float(h.get("gdp_q4_reduction_rate", 0)) - float(
            l.get("gdp_q4_reduction_rate", 0))
        rec["q1_q4_rate_ratio"] = rec["gdp_q1_reduction_rate"] / rec[
            "gdp_q4_reduction_rate"] if rec["gdp_q4_reduction_rate"] else np.nan
        table = pd.concat([table, pd.DataFrame([rec])], ignore_index=True)

    # 电网替代排放只具有国家层面的空间含义。这里先逐国计算再求和，
    # Table 6 的 net 行不填城市暴露、脆弱人群或分位组指标。
    if country_grid is not None and len(country_grid):
        for source, alpha, label in (
            (s3, alpha_star, (
                "S3(alpha*) net" if np.isfinite(selected_alpha)
                else "S3(kneedle; no unique alpha*) net"
            )),
            ("S5", 1.0, "S5 net"),
            ("S6", 1.0, "S6 net"),
            ("S7", 1.0, "S7 net"),
        ):
            if source is None or not np.isfinite(alpha):
                continue
            gross_row = table.loc[table["source_scenario"].eq(source)]
            if gross_row.empty:
                continue
            gross_row = gross_row.iloc[0]
            cg = country_grid[np.isclose(country_grid["alpha"], alpha)]
            rec = {
                "row_type": "country_net", "scenario": label,
                "source_scenario": f"{source}_net",
                "global_baseline_weighted_g": base_ref.get(
                    "global_baseline_weighted_g", np.nan),
                "global_baseline_pwe": base_ref.get("global_baseline_pwe", np.nan),
                "global_baseline_vulnerable_pwe": base_ref.get(
                    "global_baseline_vulnerable_pwe", np.nan),
                "net_composite_includes_pm": bool(
                    cg.get("grid_PM_g", pd.Series(dtype=float)).notna().all()
                ) if len(cg) else False,
            }
            for pol in POLS:
                rec[f"{pol}_policy_zone_baseline_t"] = denom[pol] / 1e6
                grid_col = f"grid_{pol}_g"
                grid_g = float(cg[grid_col].sum(min_count=1)) if grid_col in cg else np.nan
                gross_g = float(gross_row.get(f"{pol}_reduction_t", 0.0)) * 1e6
                net_g = gross_g - grid_g if np.isfinite(grid_g) else np.nan
                rec[f"{pol}_reduction_t"] = net_g / 1e6 if np.isfinite(net_g) else np.nan
                rec[f"{pol}_policy_zone_reduction_rate"] = (
                    net_g / denom[pol] if np.isfinite(net_g) and denom[pol] else np.nan
                )
            weighted_net = sum(
                WEIGHTS[p] * rec[f"{p}_reduction_t"] * 1e6
                for p in POLS if np.isfinite(rec[f"{p}_reduction_t"])
            )
            rec["exact_weighted_reduction_g"] = weighted_net
            included = [p for p in POLS if np.isfinite(rec[f"{p}_reduction_t"])]
            rec["net_composite_basis"] = "+".join(included)
            true_w = sum(
                WEIGHTS[p] * float(base_ref.get(f"global_baseline_{p}_g", np.nan))
                for p in included
            )
            rec["global_weighted_reduction_share"] = (
                weighted_net / true_w if np.isfinite(true_w) and true_w else np.nan
            )
            table = pd.concat([table, pd.DataFrame([rec])], ignore_index=True)
    # 所有行（包括后追加的边际贡献与净效应）共用同一组参照分母。
    for pol in POLS:
        table[f"{pol}_policy_zone_baseline_t"] = denom[pol] / 1e6
    table["global_baseline_weighted_g"] = base_ref.get(
        "global_baseline_weighted_g", np.nan
    )
    table["global_baseline_pwe"] = base_ref.get("global_baseline_pwe", np.nan)
    table["global_baseline_vulnerable_pwe"] = base_ref.get(
        "global_baseline_vulnerable_pwe", np.nan
    )
    for pol in POLS:
        table[f"global_baseline_{pol}_g"] = base_ref.get(
            f"global_baseline_{pol}_g", np.nan
        )
    table["alpha_star_kneedle"] = knee.get("kneedle_alpha", np.nan)
    table["alpha_star_selected"] = selected_alpha
    table["alpha_star_reported"] = knee.get("reported")
    return table


def _supplementary_tables(city_panel: pd.DataFrame, metrics: pd.DataFrame,
                          calls: pd.DataFrame, eligible: pd.DataFrame) -> None:
    metrics[["scenario", "ci_pwe_baseline", "ci_pwe_post", "ci_pwe_change"]].to_csv(
        TABLES / "table_s16_ci_pwe_policy_change.csv", index=False,
        encoding="utf-8-sig")

    trade = pd.read_parquet(TRADE)
    trade = trade[trade["year"] == TRADE_YEAR][["iso3", "q_total_tonnes"]]
    cp = city_panel.groupby(["scenario", "iso3"], as_index=False).agg(
        base_pwe=("base_pwe", "sum"), reduction_pwe=("reduction_pwe", "sum"))
    cp["post_pwe"] = cp["base_pwe"] - cp["reduction_pwe"]
    cp = cp.merge(trade, on="iso3", how="left", validate="m:1")
    cp["base_pwe_per_trade_tonne"] = cp["base_pwe"] / cp["q_total_tonnes"]
    cp["post_pwe_per_trade_tonne"] = cp["post_pwe"] / cp["q_total_tonnes"]
    cp["intensity_reduction_rate"] = 1 - (
        cp["post_pwe_per_trade_tonne"] / cp["base_pwe_per_trade_tonne"])
    cp = cp.merge(pd.read_csv(INCOME), on="iso3", how="left", validate="m:1")
    cp.to_csv(TABLES / "table_s17_country_trade_exposure_intensity.csv",
              index=False, encoding="utf-8-sig")

    port = calls.groupby(["iso3", "country", "Port_Name"], dropna=False, as_index=False).agg(
        all_container_calls=("Call_ID", "size"),
        calls_berth_ge2h=("Strict_Berth_Hours", lambda s: int((s >= 2).sum())),
        gt_known_calls=("Gross_Tonnage", lambda s: int(s.notna().sum())),
        total_ops_energy_kwh=("OPS_Energy_kWh", "sum"),
    )
    elig_port = eligible.groupby(["iso3", "Port_Name"], dropna=False, as_index=False).agg(
        eligible_calls=("Call_ID", "size"),
        eligible_ops_energy_kwh=("OPS_Energy_kWh", "sum"),
        eligible_dwt=("DWT", "sum"),
    )
    port = port.merge(elig_port, on=["iso3", "Port_Name"], how="left")
    for c in ("eligible_calls", "eligible_ops_energy_kwh", "eligible_dwt"):
        port[c] = port[c].fillna(0)
    ind = calls.assign(
        _indeterminate=(calls["Gross_Tonnage"].isna()
                        & (calls["Strict_Berth_Hours"] >= 2))
    ).groupby(["iso3", "Port_Name"], dropna=False, as_index=False).agg(
        indeterminate_gt_calls_ge2h=("_indeterminate", "sum"))
    port = port.merge(ind, on=["iso3", "Port_Name"], how="left")
    port.to_csv(TABLES / "table_s18_ops_effective_subset_by_port.csv",
                index=False, encoding="utf-8-sig")

    rounding_rows = []
    for alpha in ALPHAS:
        for port_name, g in eligible.groupby("Port_Name", dropna=False):
            n = int(len(g))
            exact_target = alpha * n
            selected_n = int(math.ceil(exact_target))
            rounding_rows.append({
                "alpha": alpha, "Port_Name": port_name,
                "iso3": g["iso3"].iloc[0] if "iso3" in g else np.nan,
                "n_eligible_calls": n, "exact_target_calls": exact_target,
                "selected_calls": selected_n,
                "rounding_excess_calls": selected_n - exact_target,
                "actual_port_call_share": selected_n / n if n else np.nan,
                "share_bias": selected_n / n - alpha if n else np.nan,
            })
    pd.DataFrame(rounding_rows).to_csv(
        TABLES / "table_s18_ops_rounding_by_port_alpha.csv",
        index=False, encoding="utf-8-sig",
    )

    selected = pd.read_csv(TABLES / "selected_policy_states_50pct.csv")
    country = calls.groupby("iso3", dropna=False, as_index=False).agg(
        all_container_calls=("Call_ID", "size"),
        gt_known_calls=("Gross_Tonnage", lambda s: int(s.notna().sum())),
    )
    ec = eligible.groupby("iso3", as_index=False).agg(
        eligible_calls=("Call_ID", "size"),
        eligible_ops_energy_kwh=("OPS_Energy_kWh", "sum"))
    country = selected.merge(country, left_on="policy_iso3", right_on="iso3", how="left").merge(
        ec, left_on="policy_iso3", right_on="iso3", how="left", suffixes=("", "_eligible"))
    for c in ("all_container_calls", "gt_known_calls", "eligible_calls", "eligible_ops_energy_kwh"):
        country[c] = country[c].fillna(0)
    country["zero_eligible_calls"] = country["eligible_calls"] == 0
    country.to_csv(TABLES / "table_s18_ops_effective_subset_by_country.csv",
                   index=False, encoding="utf-8-sig")

    # 岸电有效子集（至少一次符合条件挂靠的实施国）单独给出 S3 各档效果。
    effective_geo = set(eligible["geo_iso3"].dropna().astype(str))
    ep = city_panel[city_panel["iso3"].astype(str).isin(effective_geo)].copy()
    eff = ep[ep["scenario"].str.startswith("S3_")].groupby(
        "scenario", as_index=False
    ).agg(
        effective_subset_baseline_weighted_g=("base_weighted_g", "sum"),
        effective_subset_reduction_weighted_g=("reduction_weighted_g", "sum"),
        effective_subset_baseline_pwe=("base_pwe", "sum"),
        effective_subset_reduction_pwe=("reduction_pwe", "sum"),
    )
    eff["emission_reduction_rate"] = (
        eff["effective_subset_reduction_weighted_g"]
        / eff["effective_subset_baseline_weighted_g"].replace(0, np.nan)
    )
    eff["exposure_reduction_rate"] = (
        eff["effective_subset_reduction_pwe"]
        / eff["effective_subset_baseline_pwe"].replace(0, np.nan)
    )
    eff["n_effective_geo_states"] = len(effective_geo)
    eff.to_csv(TABLES / "table_s18_ops_effective_subset_metrics.csv",
               index=False, encoding="utf-8-sig")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-missing-grid", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    required = [
        CON / f"city_policy_delta_{SIMULATION_YEAR}.parquet",
        CON / f"policy_delta_totals_{SIMULATION_YEAR}.parquet",
        CON / f"ops_calls_container_{SIMULATION_YEAR}.parquet",
        CON / f"ops_call_city_container_{SIMULATION_YEAR}.parquet",
    ]
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"04 合并产物未就绪: {missing}")

    baseline_city, true_base = _load_baseline_long()
    cov = _load_covariates()
    area_city, area_exact = _area_city_long()
    calls, eligible, call_city = _prepare_ops()
    city_scn = _combine_city(area_city, eligible, call_city)
    exact = _combine_exact(area_exact, eligible)
    policy_totals = pd.read_parquet(
        CON / f"policy_delta_totals_{SIMULATION_YEAR}.parquet"
    )
    exact_detail = _combine_exact_detail(policy_totals, eligible)
    detail_check = exact_detail.groupby(
        ["scenario", "Pollutant"], as_index=False
    )["reduction_g"].sum().merge(
        exact, on=["scenario", "Pollutant"], how="outer",
        suffixes=("_detail", "_exact"), validate="1:1",
    ).fillna(0.0)
    mismatch = (
        detail_check["reduction_g_detail"] - detail_check["reduction_g_exact"]
    ).abs()
    if (mismatch > np.maximum(1e-4, detail_check["reduction_g_exact"].abs() * 1e-10)).any():
        raise RuntimeError(
            "船型×工况分解与情景守恒总量不一致:\n"
            + detail_check.loc[mismatch.nlargest(10).index].to_string(index=False)
        )

    grid = pd.read_csv(GRID) if GRID.is_file() else None
    if grid is None and not args.allow_missing_grid:
        raise FileNotFoundError(
            f"正式净效应缺少 2022 官方电网强度表: {GRID}；"
            "可用 --allow-missing-grid 先生成毛效应。"
        )
    scan, country_grid = _ops_scan(eligible, call_city, cov, grid)
    grouped_scan, grouped_knees = _grouped_ops_knees(eligible)
    knees = [_knee(scan, "gross_weighted_reduction_g")]
    if grid is not None:
        knees.append(_knee(scan, "net_weighted_reduction_g"))
    metrics, city_panel, groups = _metrics(
        city_scn, exact, baseline_city, true_base, cov)
    table6 = _build_table6(metrics, exact, knees[0], policy_totals, country_grid)

    city_scn.to_parquet(OUT / "city_scenario_reductions_long.parquet", index=False)
    exact.to_csv(TABLES / "scenario_exact_reductions_by_pollutant.csv",
                 index=False, encoding="utf-8-sig")
    exact_detail.to_csv(
        TABLES / "scenario_exact_reductions_ship_mode_pollutant.csv",
        index=False, encoding="utf-8-sig",
    )
    metrics.to_csv(TABLES / "scenario_effect_metrics_all.csv",
                   index=False, encoding="utf-8-sig")
    city_panel.to_parquet(OUT / "city_scenario_effect_panel.parquet", index=False)
    groups.to_csv(TABLES / "table_s15_quartile_effects.csv",
                  index=False, encoding="utf-8-sig")
    scan.to_csv(TABLES / "ops_alpha_scan.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(knees).to_csv(TABLES / "ops_knee_results.csv",
                              index=False, encoding="utf-8-sig")
    grouped_scan.to_csv(TABLES / "ops_grouped_alpha_scan.csv",
                        index=False, encoding="utf-8-sig")
    grouped_knees.to_csv(TABLES / "ops_grouped_knee_results.csv",
                         index=False, encoding="utf-8-sig")
    table6.to_csv(TABLES / "table_6_policy_scenario_summary.csv",
                  index=False, encoding="utf-8-sig")
    _supplementary_tables(city_panel, metrics, calls, eligible)
    if len(country_grid):
        country_grid.to_csv(TABLES / "country_ops_grid_emissions_by_alpha.csv",
                            index=False, encoding="utf-8-sig")
    calls.to_parquet(OUT / "ops_calls_with_eligibility.parquet", index=False)
    print(
        f"[complete] scenarios={city_scn.scenario.nunique()} cities={cov.node_id.nunique()} "
        f"calls={len(calls):,} eligible={len(eligible):,} grid={'yes' if grid is not None else 'no'}"
    )


if __name__ == "__main__":
    main()
