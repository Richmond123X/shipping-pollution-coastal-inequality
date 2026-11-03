# -*- coding: utf-8 -*-
"""Prepare, but do not execute, the 2025 policy-simulation contract."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


OUT = Path(os.environ.get(
    "V10_OUT", r"E:\transfer justice\results_v10_reconstructed_2019_2027"
))
MONTHLY = Path(os.environ.get("V10_MONTHLY_ROOT", str(OUT / "reconstructed_monthly")))
POLICY_INPUTS = OUT / "policy_inputs"
POLICY_INPUTS.mkdir(parents=True, exist_ok=True)

SHIP_TYPES = ["Container", "General Cargo", "Oil Tanker", "Bulk Carrier"]
REQUIRED = ["node_id", "year", "E1", "E2", "E3", "S1", "S2",
            "AC1", "AC2", "AC3", "AC4", "AC5"]


def build_panel() -> pd.DataFrame:
    empirical = pd.read_parquet(OUT / "data" / "city_empirical_panel.parquet")
    risk = pd.read_parquet(OUT / "data" / "SPRI_city_annual_2019_2025.parquet")
    observed = empirical[[
        "node_id", "year", "composite_t", "PWE", "S1_pop", "S2_vuln_share",
        "AC1", "AC2", "AC3", "AC4", "AC5",
    ]].merge(
        risk[["node_id", "year", "E3_now_raw"]],
        on=["node_id", "year"], how="left", validate="one_to_one",
    )
    panel = pd.DataFrame({
        "node_id": observed.node_id,
        "year": observed.year.astype(int),
        "E1": observed.composite_t * 1e6,
        "E2": observed.PWE * 1e6,
        "E3": observed.E3_now_raw,
        "S1": observed.S1_pop,
        "S2": observed.S2_vuln_share,
        "AC1": observed.AC1,
        "AC2": observed.AC2,
        "AC3": observed.AC3,
        "AC4": observed.AC4,
        "AC5": observed.AC5,
    })

    forecast = pd.read_csv(OUT / "data" / "forecast_city_annual_2026_2027.csv")
    fixed = panel[panel.year.eq(2025)].drop(columns=["year", "E1"]).set_index("node_id")
    future = forecast[["node_id", "year", "composite_mean_t"]].copy()
    future["E1"] = future.pop("composite_mean_t") * 1e6
    future = future.join(fixed, on="node_id", validate="many_to_one")
    panel = pd.concat([panel, future[REQUIRED]], ignore_index=True)
    panel = panel.sort_values(["year", "node_id"], kind="mergesort").reset_index(drop=True)
    return panel


def audit_inventory() -> list[str]:
    missing = []
    for month in range(1, 13):
        ym = f"2025{month:02d}"
        for ship in SHIP_TYPES:
            file = MONTHLY / ym / ship / f"city_emissions_{ship}_{ym}.csv"
            if not file.is_file():
                missing.append(str(file))
    return missing


def main() -> None:
    panel = build_panel()
    if list(panel.columns) != REQUIRED:
        raise RuntimeError(f"Unexpected policy panel schema: {panel.columns.tolist()}")
    if set(panel.year.unique()) != set(range(2019, 2028)):
        raise RuntimeError("Policy panel does not cover 2019-2027")
    if panel.duplicated(["node_id", "year"]).any():
        raise RuntimeError("Duplicate node-year keys in policy panel")
    counts = panel.groupby("year").node_id.nunique()
    if counts.nunique() != 1 or counts.iloc[0] != 1425:
        raise RuntimeError(f"Policy panel is not balanced: {counts.to_dict()}")
    if panel[REQUIRED[2:]].isna().any().any():
        missing = panel[REQUIRED[2:]].isna().sum()
        raise RuntimeError(f"Missing policy panel values: {missing[missing.gt(0)].to_dict()}")
    missing_inventory = audit_inventory()
    if missing_inventory:
        raise RuntimeError(f"Incomplete 2025 baseline inventory: {missing_inventory[:8]}")

    panel_path = POLICY_INPUTS / "spri_panel_2019_2027.csv"
    panel.to_csv(panel_path, index=False, encoding="utf-8-sig")
    manifest = {
        "status": "ready_for_policy_simulation",
        "policy_simulation_executed": False,
        "policy_year": 2025,
        "results_root_argument": str(MONTHLY),
        "spri_panel_argument": str(panel_path),
        "panel_years": list(range(2019, 2028)),
        "panel_nodes": int(panel.node_id.nunique()),
        "panel_rows": int(len(panel)),
        "baseline_city_files_2025": 48,
        "future_exposure_source": "formal Graph WaveNet forecasts for 2026-2027",
        "policy_delta_contract": (
            "Policy-delta files are intentionally absent because the policy run was deferred. "
            "After a future policy emission run, rerun 00_reconstruct_monthly.py so 2025 "
            "policy deltas receive the same 1.25 result-space factor as the baseline."
        ),
    }
    with open(POLICY_INPUTS / "policy_input_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"policy inputs ready: {panel_path} ({len(panel):,} rows)")


if __name__ == "__main__":
    main()
