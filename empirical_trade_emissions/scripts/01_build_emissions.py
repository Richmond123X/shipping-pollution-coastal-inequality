#!/usr/bin/env python3
"""Build the two annual emissions outcomes required by research_plan_v8 §4.6.

Model 1 uses city OD edge emissions and maps endpoints to countries. Model 2
uses the documented fallback: city-catchment totals summed within country.
All source values are grams; outputs are tonnes. SO2, PM and NOx are added
without health weights, across four ship types and twelve months.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


YEARS = tuple(range(2019, 2025))
POLLUTANTS = ("SO2", "PM", "NOx")
SHIP_DIRS = ("Container", "General cargo", "Oil tanker", "Bulk carrier")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_node_map(catchments: Path) -> pd.DataFrame:
    layer = "catchments"
    geo = gpd.read_file(catchments, layer=layer, columns=["node_id", "iso3"])
    out = geo[["node_id", "iso3"]].copy()
    out["node_id"] = out["node_id"].astype("string")
    out["original_iso3"] = out["iso3"].astype("string").str.upper().str.strip()
    # GADM uses XNC for Northern Cyprus, which is not an ISO/BACI economy code.
    # Its three ports are reported under Cyprus in the source port table, so the
    # empirical country unit is harmonised to CYP. The original code is retained
    # for audit. No other territory is collapsed into a sovereign state.
    out["iso3"] = out["original_iso3"].replace({"XNC": "CYP"})
    if out["node_id"].duplicated().any():
        duplicate = out.loc[out["node_id"].duplicated(False)].sort_values("node_id")
        raise RuntimeError(f"node_id is not unique in catchments:\n{duplicate.head(20)}")
    if out["iso3"].isna().any():
        raise RuntimeError("Missing iso3 values in catchments")
    return out[["node_id", "original_iso3", "iso3"]]


def build_edges(
    results_root: Path, node_map: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], list[dict]]:
    mapper = node_map.set_index("node_id")["iso3"]
    annual_parts: list[pd.DataFrame] = []
    monthly_qc: list[dict] = []
    mapping_qc: list[dict] = []

    for year in YEARS:
        for month in range(1, 13):
            ym = f"{year}{month:02d}"
            for ship in SHIP_DIRS:
                for pollutant in POLLUTANTS:
                    path = results_root / ym / ship / f"city_edge_list_{pollutant}_{ship}_{ym}.csv"
                    if not path.exists():
                        raise FileNotFoundError(path)
                    frame = pd.read_csv(
                        path,
                        usecols=["From_City", "To_City", "Total_Emission"],
                        dtype={"From_City": "string", "To_City": "string", "Total_Emission": "float64"},
                    )
                    if not np.isfinite(frame["Total_Emission"]).all():
                        raise RuntimeError(f"Non-finite edge emission in {path}")
                    if (frame["Total_Emission"] < 0).any():
                        raise RuntimeError(f"Negative edge emission in {path}")
                    frame["i_iso3"] = frame["From_City"].map(mapper)
                    frame["j_iso3"] = frame["To_City"].map(mapper)
                    mapped = frame["i_iso3"].notna() & frame["j_iso3"].notna()
                    grams_all = float(frame["Total_Emission"].sum())
                    grams_mapped = float(frame.loc[mapped, "Total_Emission"].sum())
                    mapping_qc.append(
                        {
                            "year_month": ym,
                            "ship_type": ship,
                            "pollutant": pollutant,
                            "rows": len(frame),
                            "rows_mapped": int(mapped.sum()),
                            "row_mapping_rate": float(mapped.mean()) if len(frame) else 1.0,
                            "grams_all": grams_all,
                            "grams_mapped": grams_mapped,
                            "emission_mapping_rate": grams_mapped / grams_all if grams_all else 1.0,
                        }
                    )
                    if not mapped.all():
                        missing_nodes = sorted(
                            set(frame.loc[frame["i_iso3"].isna(), "From_City"].dropna())
                            | set(frame.loc[frame["j_iso3"].isna(), "To_City"].dropna())
                        )
                        raise RuntimeError(f"Unmapped city-edge endpoints in {path}: {missing_nodes[:20]}")
                    grouped = (
                        frame.groupby(["i_iso3", "j_iso3"], observed=True, as_index=False)["Total_Emission"]
                        .sum()
                        .rename(columns={"Total_Emission": f"e_{pollutant.lower()}_grams"})
                    )
                    grouped["year"] = year
                    grouped["year_month"] = ym
                    annual_parts.append(grouped)
                    monthly_qc.append(
                        {
                            "outcome": "edge",
                            "year_month": ym,
                            "ship_type": ship,
                            "pollutant": pollutant,
                            "source_rows": len(frame),
                            "emission_tonnes": grams_mapped / 1e6,
                            "source_file": str(path.resolve()),
                        }
                    )

    edge_monthly = pd.concat(annual_parts, ignore_index=True)
    value_cols = [f"e_{p.lower()}_grams" for p in POLLUTANTS]
    edge_monthly = edge_monthly.groupby(
        ["i_iso3", "j_iso3", "year", "year_month"], as_index=False
    )[value_cols].sum()
    edge_monthly = edge_monthly.loc[edge_monthly["i_iso3"] != edge_monthly["j_iso3"]].copy()
    for col in value_cols:
        edge_monthly[col.replace("_grams", "_tonnes")] = edge_monthly.pop(col) / 1e6
    component_cols = [f"e_{p.lower()}_tonnes" for p in POLLUTANTS]
    edge_monthly["e_total_tonnes"] = edge_monthly[component_cols].sum(axis=1)
    edge = edge_monthly.groupby(["i_iso3", "j_iso3", "year"], as_index=False)[component_cols].sum()
    edge["e_total_tonnes"] = edge[component_cols].sum(axis=1)
    edge = edge.sort_values(["i_iso3", "j_iso3", "year"], ignore_index=True)
    edge_monthly = edge_monthly.sort_values(
        ["i_iso3", "j_iso3", "year_month"], ignore_index=True
    )
    return edge, edge_monthly, monthly_qc, mapping_qc


def build_catchments(
    results_root: Path, node_map: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], list[dict]]:
    mapper = node_map.set_index("node_id")["iso3"]
    pieces: list[pd.DataFrame] = []
    monthly_qc: list[dict] = []
    identity_qc: list[dict] = []

    total_cols = [f"{pollutant}_Total" for pollutant in POLLUTANTS]
    mode_cols = {
        pollutant: [f"{pollutant}_{mode}" for mode in ("Berth", "Anchored", "Manoeuvring", "Sea")]
        for pollutant in POLLUTANTS
    }
    usecols = ["node_id", "YearMonth", "Ship_Type"] + total_cols + [
        col for cols in mode_cols.values() for col in cols
    ]

    for year in YEARS:
        for month in range(1, 13):
            ym = f"{year}{month:02d}"
            for ship in SHIP_DIRS:
                path = results_root / ym / ship / f"city_emissions_{ship}_{ym}.csv"
                if not path.exists():
                    raise FileNotFoundError(path)
                frame = pd.read_csv(path, usecols=usecols)
                if frame["node_id"].duplicated().any():
                    raise RuntimeError(f"Duplicate node_id rows in {path}")
                if set(frame["YearMonth"].astype(str)) != {ym}:
                    raise RuntimeError(f"YearMonth mismatch in {path}")
                frame["iso3"] = frame["node_id"].map(mapper)
                if frame["iso3"].isna().any():
                    missing = sorted(frame.loc[frame["iso3"].isna(), "node_id"].astype(str).unique())
                    raise RuntimeError(f"Unmapped city emissions in {path}: {missing[:20]}")
                for pollutant in POLLUTANTS:
                    total = frame[f"{pollutant}_Total"].to_numpy(dtype=float)
                    parts = frame[mode_cols[pollutant]].sum(axis=1).to_numpy(dtype=float)
                    max_abs = float(np.max(np.abs(total - parts))) if len(frame) else 0.0
                    scale = float(np.max(np.abs(total))) if len(frame) else 0.0
                    identity_qc.append(
                        {
                            "year_month": ym,
                            "ship_type": ship,
                            "pollutant": pollutant,
                            "max_abs_difference_grams": max_abs,
                            "max_relative_difference": max_abs / scale if scale else 0.0,
                        }
                    )
                    if max_abs > max(1e-5, scale * 1e-12):
                        raise RuntimeError(f"Mode totals do not equal {pollutant}_Total in {path}")
                grouped = frame.groupby("iso3", observed=True, as_index=False)[total_cols].sum()
                grouped["year"] = year
                grouped["year_month"] = ym
                pieces.append(grouped)
                for pollutant in POLLUTANTS:
                    monthly_qc.append(
                        {
                            "outcome": "catchment_city_sum",
                            "year_month": ym,
                            "ship_type": ship,
                            "pollutant": pollutant,
                            "source_rows": len(frame),
                            "emission_tonnes": float(frame[f"{pollutant}_Total"].sum() / 1e6),
                            "source_file": str(path.resolve()),
                        }
                    )

    catchment_monthly = pd.concat(pieces, ignore_index=True)
    catchment_monthly = catchment_monthly.groupby(
        ["iso3", "year", "year_month"], as_index=False
    )[total_cols].sum()
    for pollutant in POLLUTANTS:
        catchment_monthly[f"e_{pollutant.lower()}_tonnes"] = (
            catchment_monthly.pop(f"{pollutant}_Total") / 1e6
        )
    component_cols = [f"e_{p.lower()}_tonnes" for p in POLLUTANTS]
    catchment_monthly["e_total_tonnes"] = catchment_monthly[component_cols].sum(axis=1)
    catchment = catchment_monthly.groupby(["iso3", "year"], as_index=False)[component_cols].sum()
    catchment["e_total_tonnes"] = catchment[component_cols].sum(axis=1)
    catchment["accounting_method"] = "sum_of_city_30km_catchments_overlap_retained"
    catchment_monthly["accounting_method"] = "sum_of_city_30km_catchments_overlap_retained"
    catchment = catchment.sort_values(["iso3", "year"], ignore_index=True)
    catchment_monthly = catchment_monthly.sort_values(["iso3", "year_month"], ignore_index=True)
    return catchment, catchment_monthly, monthly_qc, identity_qc


def balanced_month_annual(monthly: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Use the same ten calendar months in every year; no imputation."""
    keep = ~monthly["year_month"].str.slice(4, 6).isin(["05", "10"])
    component_cols = [f"e_{p.lower()}_tonnes" for p in POLLUTANTS]
    out = monthly.loc[keep].groupby(keys + ["year"], as_index=False)[component_cols].sum()
    out["e_total_tonnes"] = out[component_cols].sum(axis=1)
    out["balanced_months"] = "Jan-Apr, Jun-Sep, Nov-Dec (May and October excluded every year)"
    return out


def add_monthly_flags(monthly: pd.DataFrame) -> pd.DataFrame:
    monthly = monthly.copy()
    monthly["year"] = monthly["year_month"].str.slice(0, 4).astype(int)
    monthly["month"] = monthly["year_month"].str.slice(4, 6).astype(int)
    keys = ["outcome", "ship_type", "pollutant", "month"]
    median = monthly.groupby(keys, observed=True)["emission_tonnes"].transform("median")
    monthly["ratio_to_same_calendar_month_median"] = monthly["emission_tonnes"] / median
    monthly["low_total_flag"] = monthly["ratio_to_same_calendar_month_median"] < 0.5
    monthly["high_total_flag"] = monthly["ratio_to_same_calendar_month_median"] > 2.0
    return monthly


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    results_root = workspace / "results_monthly"
    output = workspace / "empirical_trade_emissions" / "data_processed"
    qc_dir = workspace / "empirical_trade_emissions" / "results" / "qc"
    output.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    catchments = workspace / "port2city" / "outputs" / "city_catchments_30km.gpkg"
    node_map = load_node_map(catchments)
    node_map.to_csv(output / "coastal_city_node_country_map.csv", index=False, encoding="utf-8-sig")
    (
        node_map[["iso3"]]
        .drop_duplicates()
        .sort_values("iso3")
        .to_csv(output / "coastal_country_universe_raw.csv", index=False, encoding="utf-8-sig")
    )

    edge, edge_country_monthly, edge_monthly, mapping_qc = build_edges(results_root, node_map)
    catchment, catchment_country_monthly, catchment_monthly, identity_qc = build_catchments(
        results_root, node_map
    )
    edge.to_parquet(output / "emission_edges_country_annual.parquet", index=False)
    catchment.to_parquet(output / "emission_catchment_country_annual.parquet", index=False)
    edge_country_monthly.to_parquet(output / "emission_edges_country_monthly.parquet", index=False)
    catchment_country_monthly.to_parquet(
        output / "emission_catchment_country_monthly.parquet", index=False
    )
    edge_balanced = balanced_month_annual(edge_country_monthly, ["i_iso3", "j_iso3"])
    catchment_balanced = balanced_month_annual(catchment_country_monthly, ["iso3"])
    edge_balanced.to_parquet(
        output / "emission_edges_country_annual_balanced_months.parquet", index=False
    )
    catchment_balanced["accounting_method"] = "sum_of_city_30km_catchments_overlap_retained"
    catchment_balanced.to_parquet(
        output / "emission_catchment_country_annual_balanced_months.parquet", index=False
    )

    monthly = add_monthly_flags(pd.DataFrame(edge_monthly + catchment_monthly))
    monthly.to_csv(qc_dir / "emission_monthly_totals.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(mapping_qc).to_csv(qc_dir / "edge_node_mapping_qc.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(identity_qc).to_csv(qc_dir / "city_total_identity_qc.csv", index=False, encoding="utf-8-sig")

    summary = {
        "years": list(YEARS),
        "pollutants": list(POLLUTANTS),
        "ship_directories": list(SHIP_DIRS),
        "node_count": int(node_map["node_id"].nunique()),
        "raw_coastal_iso3_count": int(node_map["original_iso3"].nunique()),
        "analysis_coastal_iso3_count": int(node_map["iso3"].nunique()),
        "edge_rows": int(len(edge)),
        "edge_positive_dyads": int(edge[["i_iso3", "j_iso3"]].drop_duplicates().shape[0]),
        "catchment_rows": int(len(catchment)),
        "catchment_countries": int(catchment["iso3"].nunique()),
        "low_month_flags": int(monthly["low_total_flag"].sum()),
        "high_month_flags": int(monthly["high_total_flag"].sum()),
        "catchment_accounting_method": "sum_of_city_30km_catchments_overlap_retained",
        "balanced_month_robustness": "May and October excluded in every year; no imputation",
        "catchments_sha256": file_sha256(catchments),
    }
    (qc_dir / "emission_build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
