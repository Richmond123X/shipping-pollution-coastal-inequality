#!/usr/bin/env python3
"""Create fixed 2019 income, UN M49 region, and SECA-exposure mappings."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd


INCOME_LABELS = {
    "H": "High income",
    "UM": "Upper middle income",
    "LM": "Lower middle income",
    "L": "Low income",
}


def world_bank_fy19(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Country Analytical History", header=None)
    header_rows = raw.index[raw.iloc[:, 1].astype(str).eq("Bank's fiscal year:")].tolist()
    if len(header_rows) != 1:
        raise RuntimeError(f"Could not identify unique fiscal-year header in {path}")
    header_row = header_rows[0]
    fy_columns = {str(value).strip(): index for index, value in raw.iloc[header_row].items()}
    if "FY19" not in fy_columns:
        raise RuntimeError("FY19 is absent from World Bank historical classification")
    fy19_col = fy_columns["FY19"]
    out = raw.iloc[header_row + 1 :, [0, 1, fy19_col]].copy()
    out.columns = ["iso3", "economy", "income_code_2019"]
    out["iso3"] = out["iso3"].astype(str).str.upper().str.strip()
    out = out.loc[out["iso3"].str.fullmatch(r"[A-Z]{3}", na=False)].copy()
    out["income_code_2019"] = out["income_code_2019"].astype(str).str.strip()
    out["income_group_2019"] = out["income_code_2019"].map(INCOME_LABELS)
    out = out.loc[out["income_group_2019"].notna()].copy()
    if out["iso3"].duplicated().any():
        raise RuntimeError("Duplicate ISO3 in World Bank FY19 classification")
    return out.sort_values("iso3", ignore_index=True)


def un_m49_groups(path: Path) -> pd.DataFrame:
    tables = pd.read_html(path)
    if not tables:
        raise RuntimeError(f"No HTML tables found in {path}")
    raw = tables[0].copy()
    raw = raw.rename(
        columns={
            "Country or Area": "economy_m49",
            "ISO-alpha3 Code": "iso3",
            "Region Name": "region_name",
            "Sub-region Name": "subregion_name",
        }
    )
    raw["iso3"] = raw["iso3"].astype("string").str.upper().str.strip()
    out = raw[["iso3", "economy_m49", "region_name", "subregion_name"]].dropna(subset=["iso3"])

    def collapse(row: pd.Series) -> str | None:
        region = row["region_name"]
        subregion = row["subregion_name"]
        if region == "Europe":
            return "Europe"
        if region == "Americas":
            return "Americas"
        if region in {"Africa", "Oceania"}:
            return "Africa and Oceania"
        if region == "Asia" and subregion in {"Eastern Asia", "South-eastern Asia"}:
            return "East and Southeast Asia"
        if region == "Asia" and subregion in {"Southern Asia", "Western Asia", "Central Asia"}:
            return "South and West Asia"
        return None

    out["region_group_5"] = out.apply(collapse, axis=1)
    # Antarctica has no M49 region assignment and is outside the study universe.
    unassigned = out.loc[out["region_group_5"].isna(), "iso3"].tolist()
    if set(unassigned) - {"ATA"}:
        bad = out.loc[out["region_group_5"].isna(), ["iso3", "region_name", "subregion_name"]]
        raise RuntimeError(f"Unassigned M49 regions:\n{bad}")
    out = out.loc[out["region_group_5"].notna()].copy()
    # BACI code 490 is used as the Taiwan proxy. Taiwan has no row in UN M49;
    # the analytical placement below is explicit and kept separate from source rows.
    taiwan = pd.DataFrame(
        [
            {
                "iso3": "TWN",
                "economy_m49": "Taiwan (BACI 490 analytical proxy)",
                "region_name": "Asia",
                "subregion_name": "Eastern Asia",
                "region_group_5": "East and Southeast Asia",
            }
        ]
    )
    out = pd.concat([out, taiwan], ignore_index=True)
    if out["iso3"].duplicated().any():
        raise RuntimeError("Duplicate ISO3 in M49 mapping")
    return out.sort_values("iso3", ignore_index=True)


def seca_exposure(workspace: Path) -> pd.DataFrame:
    monthly = pd.read_csv(workspace / "covariates" / "outputs" / "seca_neca_monthly.csv")
    nodes = gpd.read_file(
        workspace / "port2city" / "outputs" / "city_catchments_30km.gpkg",
        layer="catchments",
        columns=["node_id", "iso3", "area_km2"],
    )
    nodes = pd.DataFrame(nodes.drop(columns="geometry"))
    nodes["original_iso3"] = nodes["iso3"].astype(str).str.upper().str.strip()
    nodes["iso3"] = nodes["original_iso3"].replace({"XNC": "CYP"})
    base = monthly.loc[monthly["month"].astype(str).str.startswith("2019-")].copy()
    stability = base.groupby("node_id")["seca"].nunique()
    if (stability > 1).any():
        raise RuntimeError("2019 SECA classification changes within year for some nodes")
    base = base.drop_duplicates("node_id")[["node_id", "seca"]]
    joined = nodes.merge(base, on="node_id", how="left", validate="one_to_one")
    if joined["seca"].isna().any():
        raise RuntimeError("SECA classification missing for coastal nodes")
    joined["seca_area_proxy"] = joined["seca"] * joined["area_km2"]
    grouped = joined.groupby("iso3", as_index=False).agg(
        coastal_node_count=("node_id", "nunique"),
        seca_exposed_node_count_2019=("seca", "sum"),
        admin_area_km2=("area_km2", "sum"),
        seca_exposed_admin_area_km2_2019=("seca_area_proxy", "sum"),
    )
    grouped["seca_exposed_node_share_2019"] = (
        grouped["seca_exposed_node_count_2019"] / grouped["coastal_node_count"]
    )
    grouped["seca_exposed_admin_area_share_2019"] = (
        grouped["seca_exposed_admin_area_km2_2019"] / grouped["admin_area_km2"]
    )
    grouped["seca_measure_note"] = (
        "proxy: share of study nodes whose 30km catchment intersects a SECA; "
        "not exact polygon coverage"
    )
    return grouped.sort_values("iso3", ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    raw = workspace / "empirical_trade_emissions" / "data_raw"
    output = workspace / "empirical_trade_emissions" / "data_processed"
    qc = workspace / "empirical_trade_emissions" / "results" / "qc"
    output.mkdir(parents=True, exist_ok=True)
    qc.mkdir(parents=True, exist_ok=True)

    income = world_bank_fy19(raw / "world_bank" / "OGHIST.xlsx")
    regions = un_m49_groups(raw / "un_m49" / "UN_M49_overview.html")
    seca = seca_exposure(workspace)
    groups = income.merge(regions, on="iso3", how="outer", validate="one_to_one").merge(
        seca, on="iso3", how="outer", validate="one_to_one"
    )
    income.to_csv(output / "world_bank_income_fy19.csv", index=False, encoding="utf-8-sig")
    regions.to_csv(output / "un_m49_region_groups.csv", index=False, encoding="utf-8-sig")
    seca.to_csv(output / "seca_exposure_proxy_2019.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(output / "country_classifications.csv", index=False, encoding="utf-8-sig")

    summary = {
        "world_bank_fy19_economies": int(len(income)),
        "m49_economies_plus_taiwan_proxy": int(len(regions)),
        "study_coastal_economies_with_seca_proxy": int(len(seca)),
        "world_bank_source_sheet": "Country Analytical History",
        "world_bank_source_column": "FY19",
        "taiwan_region_override": "TWN -> Eastern Asia -> East and Southeast Asia",
        "north_cyprus_country_override": "XNC -> CYP",
        "seca_control_limitation": "node-intersection proxy, not exact catchment polygon coverage share",
    }
    (qc / "classification_build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
