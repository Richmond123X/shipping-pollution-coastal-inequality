# -*- coding: utf-8 -*-
"""审查 V12 S1--S7 所需输入，并输出可审计的数据就绪表。"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import geopandas as gpd
import pandas as pd

from policy_run_config import (
    AIS_DIR as AIS, BASELINE_MONTHLY_ROOT, GT_CROSSWALK, ROOT,
    SIMULATION_YEAR, WORK,
)


TABLES = ROOT / "tables"
FORMAL_TYPES = ["Bulk carrier", "General cargo", "Oil tanker", "Container"]


def add(rows: list[dict], item: str, needed_for: str, status: str,
        evidence: str, requires_raw_ais: bool) -> None:
    rows.append({
        "input_or_output": item, "needed_for": needed_for, "status": status,
        "evidence": evidence, "requires_raw_single_ship_ais": requires_raw_ais,
    })


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    rows = []
    months = [f"{SIMULATION_YEAR}{x:02d}" for x in range(1, 13)]
    dbs = [AIS / f"ais_{m}.db" for m in months]
    missing = [p.name for p in dbs if not p.is_file()]
    add(rows, f"{SIMULATION_YEAR} monthly AIS SQLite", "S1-S7 point inventory and calls",
        "ready" if not missing else "blocked",
        f"12 target months; missing={missing}", True)
    prev_boundary = f"{SIMULATION_YEAR - 1}12"
    next_boundary = f"{SIMULATION_YEAR + 1}01"
    boundary_present = {
        prev_boundary: (AIS / f"ais_{prev_boundary}.db").is_file(),
        next_boundary: (AIS / f"ais_{next_boundary}.db").is_file(),
    }
    add(rows, f"Adjacent boundary months ({prev_boundary} / {next_boundary})",
        "Optional cross-year interpolation only", "documented_exclusion",
        f"availability={boundary_present}; formal inventory is strictly "
        f"{SIMULATION_YEAR}-01..{SIMULATION_YEAR}-12; missing boundary months are not totals.",
        True)

    parts = []
    required_ship_cols = {
        "MMSI", "Ship_Type", "Build_Year", "DWT", "Fuel_Type", "Engine_Type",
        "ME_NOx_noECA", "ME_NOx_ECA",
    }
    schema_missing = set()
    for path in dbs:
        if not path.is_file():
            continue
        conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(ship_info)").fetchall()}
        schema_missing |= required_ship_cols - cols
        d = pd.read_sql_query(
            "SELECT MMSI, Ship_Type, Build_Year, DWT, Fuel_Type, Engine_Type, "
            "ME_NOx_noECA, ME_NOx_ECA FROM ship_info "
            "WHERE Ship_Type IN (?,?,?,?)", conn, params=FORMAL_TYPES,
        )
        conn.close()
        parts.append(d)
    ships = pd.concat(parts, ignore_index=True).drop_duplicates("MMSI", keep="first")
    coverage = {
        c: float(ships[c].notna().mean()) for c in (
            "Build_Year", "DWT", "Fuel_Type", "Engine_Type", "ME_NOx_noECA", "ME_NOx_ECA"
        )
    }
    add(rows, "AIS ship_info technical attributes", "S1, S2, S4-S7",
        "ready" if not schema_missing else "blocked",
        f"unique formal ships={len(ships):,}; missing_columns={sorted(schema_missing)}; coverage={coverage}",
        True)

    gt_path = GT_CROSSWALK
    gt_total = 0
    matched = 0
    if gt_path.is_file():
        gt = pd.read_csv(gt_path)
        gt_total = len(gt)
        gt_col = "GT" if "GT" in gt.columns else "Gross_Tonnage"
        matched = int(pd.to_numeric(gt[gt_col], errors="coerce").notna().sum())
        add(rows, "Direct container gross-tonnage crosswalk", "S3, S5-S7 eligibility",
            "ready_with_indeterminate_calls",
            f"container MMSIs={len(gt):,}; GT matched={matched:,} ({matched/len(gt):.2%}); missing GT is not imputed from DWT",
            False)
    else:
        add(rows, "Direct container gross-tonnage crosswalk", "S3, S5-S7 eligibility",
            "blocked", str(gt_path), False)

    zone = ROOT / "inputs" / "formal_policy_zones_50pct.gpkg"
    try:
        regions = gpd.read_file(zone, layer="policy_regions")
        units = gpd.read_file(zone, layer="policy_units")
        ok = len(regions) == 22 and len(units) == 75 and regions.geometry.is_valid.all() and units.geometry.is_valid.all()
        evidence = f"regions={len(regions)}; policy units={len(units)}; valid={bool(ok)}"
    except Exception as exc:
        ok, evidence = False, repr(exc)
    add(rows, "Formal X=50% 12-nm policy zones", "S1-S7 treatment scope",
        "ready" if ok else "blocked", evidence, False)

    baseline_missing = []
    for ym in months:
        for st in FORMAL_TYPES:
            d = BASELINE_MONTHLY_ROOT / ym / st
            for name in (f"city_emissions_{st}_{ym}.csv", f"true_totals_{st}_{ym}.csv"):
                if not (d / name).is_file():
                    baseline_missing.append(str(d / name))
    add(rows, f"{SIMULATION_YEAR} baseline city inventory and conservation totals",
        "Table 6 denominators and exposure effects",
        "ready" if not baseline_missing else "blocked",
        f"expected=96 files; missing={len(baseline_missing)}", False)

    static = {
        "City covariates": WORK / "covariates" / "outputs" / "city_covariates_2019_2028.csv",
        "Strict SECA city-year": WORK / "covariates" / "outputs" / "strict_seca_0p1_city_year.csv",
        "Port-city mapping": WORK / "port2city" / "outputs" / "port_city_mapping.csv",
        "SPRI 2019-2025": WORK / "results_v10_reconstructed_2019_2027" / "data" / "SPRI_city_annual_2019_2025.parquet",
        "Income groups": WORK / "empirical_trade_emissions" / "data_processed" / "world_bank_income_fy19.csv",
        "UN M49 regions": WORK / "empirical_trade_emissions" / "data_processed" / "un_m49_region_groups.csv",
        "2024 trade mass": WORK / "empirical_trade_emissions" / "data_processed" / "trade_country_annual.parquet",
    }
    for item, path in static.items():
        add(rows, item, "Table 6/7 distribution diagnostics",
            "ready" if path.is_file() else "blocked", str(path), False)

    grid = ROOT / "inputs" / "grid_pollution_intensity_2022.csv"
    grid_complete = 0
    grid_total = 75
    grid_detail = str(grid)
    grid_status = "external_input_pending"
    if grid.is_file():
        gd = pd.read_csv(grid)
        required_grid = {
            "iso3", "SO2_g_per_kWh", "NOx_g_per_kWh", "PM_g_per_kWh",
            "PM_comparable", "coverage_status",
        }
        if required_grid.issubset(gd.columns) and gd["iso3"].nunique() == grid_total:
            grid_complete = int(gd["coverage_status"].eq("complete").sum())
            grid_status = ("ready" if grid_complete == grid_total
                           else "ready_with_documented_missing")
            miss = gd.loc[~gd["coverage_status"].eq("complete"), "iso3"].tolist()
            grid_detail = f"complete={grid_complete}/{grid_total}; missing={miss}; {grid}"
        else:
            grid_status = "blocked"
            grid_detail = f"schema_or_country_coverage_invalid; {grid}"
    add(rows, "Official country grid SO2/NOx/PM intensity",
        "Net S3/S5/S6/S7 only",
        grid_status, grid_detail, False)

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "policy_input_readiness.csv", index=False, encoding="utf-8-sig")

    gt_note = (
        f"GT direct/crosswalk coverage={matched}/{gt_total} "
        f"({matched/gt_total:.2%}); missing GT remains indeterminate and is not imputed"
        if gt_total else "GT crosswalk missing"
    )
    common_area = (
        f"{SIMULATION_YEAR} point AIS location/time/operating mode + ship_info technical fields + "
        "current ECA/NECA baseline + formal 12-nm policy zones"
    )
    common_ops = (
        "single-vessel container port calls + strict berth duration + direct GT + "
        "call-level berth energy/emissions + 2022 country grid intensity"
    )
    scenarios = pd.DataFrame([
        {
            "scenario": "S1", "policy": "Sulfur-control class",
            "required_data": common_area,
            "data_status": "ready", "raw_single_ship_ais_required": True,
            "aggregated_network_sufficient": False,
            "reason": "Point-level 12-nm membership, current sulfur baseline and fuel-dependent SO2/PM counterfactual cannot be reconstructed from aggregated edges.",
        },
        {
            "scenario": "S2", "policy": "NOx Tier-III class",
            "required_data": common_area + " + Build_Year + ME NOx factors",
            "data_status": "ready", "raw_single_ship_ais_required": True,
            "aggregated_network_sufficient": False,
            "reason": "Eligibility and current-NECA marginal effect depend jointly on vessel build year and each AIS point's location/time.",
        },
        {
            "scenario": "S3", "policy": "OPS 10%-100% scan",
            "required_data": common_ops,
            "data_status": "ready_with_indeterminate_gt",
            "raw_single_ship_ais_required": True,
            "aggregated_network_sufficient": False,
            "reason": "Exact call boundaries, >=2-hour exemption, within-port demand ranking and ceil(alpha*n_p) selection require raw vessel trajectories; " + gt_note,
        },
        {
            "scenario": "S4", "policy": "Sulfur + NOx",
            "required_data": common_area,
            "data_status": "ready", "raw_single_ship_ais_required": True,
            "aggregated_network_sufficient": False,
            "reason": "Requires the same point-level S1 and S2 deltas and exact pollutant-wise combination.",
        },
        {
            "scenario": "S5", "policy": "Sulfur + OPS at 100%",
            "required_data": common_area + " + " + common_ops,
            "data_status": "ready_with_indeterminate_gt",
            "raw_single_ship_ais_required": True,
            "aggregated_network_sufficient": False,
            "reason": "Call-level OPS must be applied first and its berth overlap with S1 subtracted exactly; " + gt_note,
        },
        {
            "scenario": "S6", "policy": "NOx + OPS at 100%",
            "required_data": common_area + " + " + common_ops,
            "data_status": "ready_with_indeterminate_gt",
            "raw_single_ship_ais_required": True,
            "aggregated_network_sufficient": False,
            "reason": "Call-level OPS must be applied first and its berth overlap with S2 subtracted exactly; " + gt_note,
        },
        {
            "scenario": "S7", "policy": "Sulfur + NOx + OPS at 100%",
            "required_data": common_area + " + " + common_ops,
            "data_status": "ready_with_indeterminate_gt",
            "raw_single_ship_ais_required": True,
            "aggregated_network_sufficient": False,
            "reason": "All three point/call-level mechanisms and exact overlap accounting are required; " + gt_note,
        },
    ])
    scenarios.to_csv(
        TABLES / "policy_scenario_data_feasibility.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "critical_blockers_excluding_grid_net_effect": int(out["status"].eq("blocked").sum()),
        "grid_net_effect_pending": grid_complete < grid_total,
        "grid_net_effect_complete_policy_states": grid_complete,
        "grid_net_effect_total_policy_states": grid_total,
        "raw_single_ship_ais_required": True,
        "reason": (
            "S1/S2 require point location, time, fuel and build year; S3 requires exact berth calls, "
            "strict berth duration and call-level energy. Aggregated network edges cannot reconstruct these."
        ),
    }
    (TABLES / "policy_input_readiness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
