#!/usr/bin/env python3
"""Construct complete coastal frames and model-ready PPML samples."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


YEARS = tuple(range(2019, 2025))


def safe_log(series: pd.Series) -> pd.Series:
    """Natural log for strictly positive observations without evaluating log(0)."""
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    positive = series.gt(0)
    out.loc[positive] = np.log(series.loc[positive].astype(float))
    return out


def load_inputs(base: Path, balanced: bool = False) -> dict[str, pd.DataFrame]:
    processed = base / "data_processed"
    suffix = "_balanced_months" if balanced else ""
    return {
        "edge": pd.read_parquet(processed / f"emission_edges_country_annual{suffix}.parquet"),
        "catchment": pd.read_parquet(processed / f"emission_catchment_country_annual{suffix}.parquet"),
        "trade": pd.read_parquet(processed / "trade_dyad_annual.parquet"),
        "country_trade": pd.read_parquet(processed / "trade_country_annual.parquet"),
        "geodist": pd.read_parquet(processed / "geodist_contiguity_V202211.parquet"),
        "groups": pd.read_csv(processed / "country_classifications.csv"),
        "coastal": pd.read_csv(processed / "coastal_country_universe_raw.csv"),
    }


def dyad_panel(
    data: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coastal = sorted(data["coastal"]["iso3"].dropna().astype(str).unique())
    frame = pd.DataFrame(
        (
            (i, j, year)
            for i, j in itertools.permutations(coastal, 2)
            for year in YEARS
        ),
        columns=["i_iso3", "j_iso3", "year"],
    )
    trade_cols = [
        "i_iso3",
        "j_iso3",
        "year",
        "q_tonnes",
        "v_kusd",
        "q_cov",
        "flow_count",
        "q_reported_flow_count",
    ]
    panel = frame.merge(data["trade"][trade_cols], on=["i_iso3", "j_iso3", "year"], how="left")
    emission_cols = [
        "i_iso3",
        "j_iso3",
        "year",
        "e_so2_tonnes",
        "e_pm_tonnes",
        "e_nox_tonnes",
        "e_total_tonnes",
    ]
    panel = panel.merge(data["edge"][emission_cols], on=["i_iso3", "j_iso3", "year"], how="left")
    panel = panel.merge(data["geodist"], on=["i_iso3", "j_iso3"], how="left", validate="many_to_one")
    for col in ["q_tonnes", "v_kusd", "flow_count", "q_reported_flow_count", *emission_cols[3:]]:
        panel[col] = panel[col].fillna(0)
    panel["q_noncontig_tonnes"] = np.where(panel["contig"].eq(0), panel["q_tonnes"], 0.0)
    panel["dyad_id"] = panel["i_iso3"] + "->" + panel["j_iso3"]
    panel["it_id"] = panel["i_iso3"] + "_" + panel["year"].astype(str)
    panel["jt_id"] = panel["j_iso3"] + "_" + panel["year"].astype(str)
    panel["ln_Q"] = safe_log(panel["q_tonnes"])

    group_cols = ["iso3", "income_code_2019", "income_group_2019", "region_group_5"]
    groups = data["groups"][group_cols].drop_duplicates("iso3").rename(columns={"iso3": "j_iso3"})
    panel = panel.merge(groups, on="j_iso3", how="left", validate="many_to_one")

    main = panel.loc[panel["q_tonnes"].gt(0)].copy()
    positive_by_dyad = main.groupby("dyad_id")["e_total_tonnes"].transform("sum").gt(0)
    main["separated_all_zero_dyad"] = ~positive_by_dyad
    estimation = main.loc[positive_by_dyad].copy()

    symmetric = panel.copy()
    symmetric["a_iso3"] = symmetric[["i_iso3", "j_iso3"]].min(axis=1)
    symmetric["b_iso3"] = symmetric[["i_iso3", "j_iso3"]].max(axis=1)
    sum_cols = ["q_tonnes", "v_kusd", "e_so2_tonnes", "e_pm_tonnes", "e_nox_tonnes", "e_total_tonnes"]
    symmetric = symmetric.groupby(["a_iso3", "b_iso3", "year"], as_index=False)[sum_cols].sum()
    symmetric = symmetric.loc[symmetric["q_tonnes"].gt(0)].copy()
    symmetric["pair_id"] = symmetric["a_iso3"] + "--" + symmetric["b_iso3"]
    symmetric["at_id"] = symmetric["a_iso3"] + "_" + symmetric["year"].astype(str)
    symmetric["bt_id"] = symmetric["b_iso3"] + "_" + symmetric["year"].astype(str)
    symmetric["ln_Q"] = np.log(symmetric["q_tonnes"])
    keep_pair = symmetric.groupby("pair_id")["e_total_tonnes"].transform("sum").gt(0)
    symmetric = symmetric.loc[keep_pair].copy()

    anomaly = panel.loc[panel["e_total_tonnes"].gt(0) & panel["q_tonnes"].eq(0)].copy()
    return panel, main, estimation, symmetric, anomaly


def country_panel(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    coastal = sorted(data["coastal"]["iso3"].dropna().astype(str).unique())
    frame = pd.MultiIndex.from_product([coastal, YEARS], names=["iso3", "year"]).to_frame(index=False)
    trade_cols = [
        "iso3",
        "year",
        "q_exports_tonnes",
        "q_imports_tonnes",
        "q_total_tonnes",
        "v_total_kusd",
        "q_noncontig_exports_tonnes",
        "q_noncontig_imports_tonnes",
        "q_noncontig_total_tonnes",
    ]
    panel = frame.merge(data["country_trade"][trade_cols], on=["iso3", "year"], how="left")
    emission_cols = [
        "iso3",
        "year",
        "e_so2_tonnes",
        "e_pm_tonnes",
        "e_nox_tonnes",
        "e_total_tonnes",
    ]
    panel = panel.merge(data["catchment"][emission_cols], on=["iso3", "year"], how="left")
    for col in trade_cols[2:] + emission_cols[2:]:
        panel[col] = panel[col].fillna(0)
    group_cols = [
        "iso3",
        "income_code_2019",
        "income_group_2019",
        "region_group_5",
        "seca_exposed_node_share_2019",
        "seca_exposed_admin_area_share_2019",
    ]
    panel = panel.merge(data["groups"][group_cols].drop_duplicates("iso3"), on="iso3", how="left")
    panel["ln_Q"] = safe_log(panel["q_total_tonnes"])
    panel["ln_Q_exports"] = safe_log(panel["q_exports_tonnes"])
    panel["ln_Q_imports"] = safe_log(panel["q_imports_tonnes"])
    panel["ln_Q_noncontig"] = safe_log(panel["q_noncontig_total_tonnes"])
    panel["post2020"] = panel["year"].ge(2020).astype(int)
    panel["seca_proxy_x_post2020"] = panel["seca_exposed_node_share_2019"] * panel["post2020"]
    return panel.sort_values(["iso3", "year"], ignore_index=True)


def sample_attrition(
    data: dict[str, pd.DataFrame], panel: pd.DataFrame, main: pd.DataFrame, estimation: pd.DataFrame
) -> pd.DataFrame:
    trade_countries = sorted(set(data["trade"]["i_iso3"]) | set(data["trade"]["j_iso3"]))
    coastal = sorted(data["coastal"]["iso3"].dropna().astype(str).unique())
    rows = [
        {
            "step": "BACI economy square including self-pairs",
            "observations": len(trade_countries) ** 2 * len(YEARS),
            "dyads_or_pairs": len(trade_countries) ** 2,
        },
        {
            "step": "Coastal study-economy square including self-pairs",
            "observations": len(coastal) ** 2 * len(YEARS),
            "dyads_or_pairs": len(coastal) ** 2,
        },
        {
            "step": "Exclude self-pairs",
            "observations": len(panel),
            "dyads_or_pairs": panel["dyad_id"].nunique(),
        },
        {
            "step": "Require positive BACI physical trade Q",
            "observations": len(main),
            "dyads_or_pairs": main["dyad_id"].nunique(),
        },
        {
            "step": "Drop dyads with emissions zero in every observed year",
            "observations": len(estimation),
            "dyads_or_pairs": estimation["dyad_id"].nunique(),
        },
        {
            "step": "Income heterogeneity complete cases",
            "observations": int(estimation["income_group_2019"].notna().sum()),
            "dyads_or_pairs": estimation.loc[
                estimation["income_group_2019"].notna(), "dyad_id"
            ].nunique(),
        },
        {
            "step": "Region heterogeneity complete cases",
            "observations": int(estimation["region_group_5"].notna().sum()),
            "dyads_or_pairs": estimation.loc[
                estimation["region_group_5"].notna(), "dyad_id"
            ].nunique(),
        },
    ]
    return pd.DataFrame(rows)


def diagnostic_tables(
    base: Path,
    data: dict[str, pd.DataFrame],
    panel: pd.DataFrame,
    main: pd.DataFrame,
    estimation: pd.DataFrame,
    anomaly: pd.DataFrame,
    country: pd.DataFrame,
    label: str,
) -> None:
    qc = base / "results" / "qc"
    qc.mkdir(parents=True, exist_ok=True)
    suffix = f"_{label}" if label else ""
    sample_attrition(data, panel, main, estimation).to_csv(
        qc / f"table_s1_sample_attrition{suffix}.csv", index=False, encoding="utf-8-sig"
    )
    structural = panel.groupby("year", as_index=False).agg(
        observations=("dyad_id", "size"),
        q_zero=("q_tonnes", lambda s: int(s.eq(0).sum())),
        e_zero=("e_total_tonnes", lambda s: int(s.eq(0).sum())),
    )
    both = panel.assign(both_zero=panel["q_tonnes"].eq(0) & panel["e_total_tonnes"].eq(0))
    structural = structural.merge(
        both.groupby("year", as_index=False)["both_zero"].sum(), on="year", validate="one_to_one"
    )
    for col in ["q_zero", "e_zero", "both_zero"]:
        structural[f"{col}_share"] = structural[col] / structural["observations"]
    structural.to_csv(qc / f"structural_zero_qc{suffix}.csv", index=False, encoding="utf-8-sig")

    qcov = data["trade"].loc[data["trade"]["q_tonnes"].gt(0)].groupby("year")["q_cov"].agg(
        median="median", p10=lambda s: s.quantile(0.10), p25=lambda s: s.quantile(0.25), n="count"
    ).reset_index()
    qcov.to_csv(qc / "baci_q_coverage_quantiles.csv", index=False, encoding="utf-8-sig")

    anomaly.to_csv(qc / f"emission_positive_trade_zero_edges{suffix}.csv", index=False, encoding="utf-8-sig")
    endpoints = pd.concat(
        [
            anomaly[["i_iso3", "year"]].rename(columns={"i_iso3": "iso3"}),
            anomaly[["j_iso3", "year"]].rename(columns={"j_iso3": "iso3"}),
        ],
        ignore_index=True,
    )
    anomalous_counts = endpoints.groupby("iso3", as_index=False).size().rename(columns={"size": "anomalous_endpoint_rows"})
    positive_edges = panel.loc[panel["e_total_tonnes"].gt(0)]
    all_endpoints = pd.concat(
        [
            positive_edges[["i_iso3"]].rename(columns={"i_iso3": "iso3"}),
            positive_edges[["j_iso3"]].rename(columns={"j_iso3": "iso3"}),
        ],
        ignore_index=True,
    ).groupby("iso3", as_index=False).size().rename(columns={"size": "positive_emission_endpoint_rows"})
    country_anomaly = all_endpoints.merge(anomalous_counts, on="iso3", how="left").fillna(0)
    country_anomaly["anomalous_share"] = (
        country_anomaly["anomalous_endpoint_rows"] / country_anomaly["positive_emission_endpoint_rows"]
    )
    country_anomaly["over_20pct_flag"] = country_anomaly["anomalous_share"].gt(0.2)
    country_anomaly.to_csv(qc / f"trade_zero_anomaly_by_country{suffix}.csv", index=False, encoding="utf-8-sig")

    coastal = set(data["coastal"]["iso3"].dropna().astype(str))
    baci = set(data["trade"]["i_iso3"]) | set(data["trade"]["j_iso3"])
    coverage = pd.DataFrame({"iso3": sorted(coastal)})
    coverage["appears_in_baci_202601"] = coverage["iso3"].isin(baci)
    coverage["has_income_fy19"] = coverage["iso3"].isin(
        set(data["groups"].loc[data["groups"]["income_group_2019"].notna(), "iso3"])
    )
    coverage["has_m49_region"] = coverage["iso3"].isin(
        set(data["groups"].loc[data["groups"]["region_group_5"].notna(), "iso3"])
    )
    coverage.to_csv(qc / "coastal_economy_coverage.csv", index=False, encoding="utf-8-sig")

    country_zero = country.groupby("iso3", as_index=False).agg(
        all_years_emission_zero=("e_total_tonnes", lambda s: bool(s.eq(0).all())),
        any_year_trade_zero=("q_total_tonnes", lambda s: bool(s.eq(0).any())),
    )
    country_zero.to_csv(qc / f"country_panel_zero_qc{suffix}.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    base = workspace / "empirical_trade_emissions"
    output = base / "data_processed"

    summaries = {}
    for balanced, label in [(False, "main"), (True, "balanced_months")]:
        data = load_inputs(base, balanced=balanced)
        panel, positive_trade, estimation, symmetric, anomaly = dyad_panel(data)
        country = country_panel(data)
        if not balanced:
            panel.to_parquet(output / "ppml_dyad_full_coastal.parquet", index=False)
            positive_trade.to_parquet(output / "ppml_dyad_positive_trade.parquet", index=False)
            estimation.to_parquet(output / "ppml_dyad_sample.parquet", index=False)
            symmetric.to_parquet(output / "ppml_dyad_symmetric_sample.parquet", index=False)
            country.to_parquet(output / "ppml_country_sample.parquet", index=False)
            estimation.to_csv(output / "ppml_dyad_sample.csv.gz", index=False, compression="gzip")
            symmetric.to_csv(output / "ppml_dyad_symmetric_sample.csv.gz", index=False, compression="gzip")
            country.to_csv(output / "ppml_country_sample.csv.gz", index=False, compression="gzip")
        else:
            estimation.to_parquet(output / "ppml_dyad_sample_balanced_months.parquet", index=False)
            country.to_parquet(output / "ppml_country_sample_balanced_months.parquet", index=False)
            estimation.to_csv(
                output / "ppml_dyad_sample_balanced_months.csv.gz", index=False, compression="gzip"
            )
            country.to_csv(
                output / "ppml_country_sample_balanced_months.csv.gz", index=False, compression="gzip"
            )
        diagnostic_tables(base, data, panel, positive_trade, estimation, anomaly, country, label)
        summaries[label] = {
            "coastal_economies": int(data["coastal"]["iso3"].nunique()),
            "full_directed_dyad_years_excluding_self": int(len(panel)),
            "positive_trade_rows": int(len(positive_trade)),
            "pre_estimation_rows_after_all_zero_dyad_drop": int(len(estimation)),
            "pre_estimation_dyads": int(estimation["dyad_id"].nunique()),
            "symmetric_rows": int(len(symmetric)),
            "country_year_rows": int(len(country)),
            "country_positive_trade_rows": int(country["q_total_tonnes"].gt(0).sum()),
            "e_positive_q_zero_rows": int(len(anomaly)),
        }
    summary_path = base / "results" / "qc" / "panel_build_summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
