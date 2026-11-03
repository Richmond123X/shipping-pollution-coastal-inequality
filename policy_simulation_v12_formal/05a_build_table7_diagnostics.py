# -*- coding: utf-8 -*-
"""构造 V12 Table 7 与入选区域七项诊断。

静态字段可在 AIS 情景完成前用 ``--static-only`` 预审；正式运行要求 04/05
步骤的合并与情景输出存在，并在同一张表中加入岸电、现行管制与 S7 效果。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from policy_run_config import (
    BASELINE_MONTHLY_ROOT as RESULTS_MONTHLY,
    CONSOLIDATED_ROOT as CON, ROOT, SCENARIO_ROOT as SCN,
    SELECTION_RISK_YEAR, SIMULATION_YEAR, WORK,
)


TABLES = ROOT / "tables"
MEMBERSHIP = WORK / "policy_clustering_v12_formal" / "tables" / "coastal_regions_geo_membership.csv"
REGIONS = ROOT / "tables" / "selected_regions_50pct.csv"
REGION_NAMES = ROOT / "tables" / "table_s14_region_names.csv"
SPRI = WORK / "results_v10_reconstructed_2019_2027" / "data" / "SPRI_city_annual_2019_2025.parquet"
COV = WORK / "covariates" / "outputs" / "city_covariates_2019_2028.csv"
STRICT_SECA = WORK / "covariates" / "outputs" / "strict_seca_0p1_city_year.csv"
INCOME = WORK / "empirical_trade_emissions" / "data_processed" / "world_bank_income_fy19.csv"
M49 = WORK / "empirical_trade_emissions" / "data_processed" / "un_m49_region_groups.csv"
PORT_MAP = WORK / "port2city" / "outputs" / "port_city_mapping.csv"
SHIP_TYPES = ["Bulk carrier", "General cargo", "Oil tanker", "Container"]
POLS = ["SO2", "PM", "NOx"]
MODES = ["Berth", "Anchored", "Manoeuvring", "Sea"]
WEIGHTS = {"SO2": 2.0, "PM": 5.0, "NOx": 1.0}


def _baseline_city() -> pd.DataFrame:
    """模拟年份城市集水区综合加权基线；保持5:2:1口径一致。"""
    parts = []
    for month in range(1, 13):
        ym = f"{SIMULATION_YEAR}{month:02d}"
        for ship_type in SHIP_TYPES:
            path = RESULTS_MONTHLY / ym / ship_type / f"city_emissions_{ship_type}_{ym}.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            d = pd.read_csv(path)
            value = pd.Series(0.0, index=d.index)
            for pol in POLS:
                cols = [f"{pol}_{mode}" for mode in MODES]
                value += WEIGHTS[pol] * d[cols].apply(
                    pd.to_numeric, errors="coerce"
                ).fillna(0.0).sum(axis=1)
            parts.append(pd.DataFrame({
                "node_id": d["node_id"].astype(str),
                "baseline_weighted_g": value,
            }))
    return pd.concat(parts, ignore_index=True).groupby(
        "node_id", as_index=False
    )["baseline_weighted_g"].sum()


def _static_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    members = pd.read_csv(MEMBERSHIP)
    regions = pd.read_csv(REGIONS)
    selected = set(regions["region_id"].astype(str))
    members = members[members["region_id"].astype(str).isin(selected)].copy()

    risk = pd.read_parquet(SPRI)
    risk = risk[
        (risk["year"] == SELECTION_RISK_YEAR) & (risk["oversized_flag"] == 0)
    ].copy()
    cov = pd.read_csv(COV)
    cov = cov[
        pd.to_numeric(cov["year"], errors="coerce") == SELECTION_RISK_YEAR
    ].copy()
    cov_risk = cov[
        pd.to_numeric(cov["oversized_flag"], errors="coerce").fillna(0) == 0
    ].copy()
    seca = pd.read_csv(STRICT_SECA)
    seca = seca[
        pd.to_numeric(seca["year"], errors="coerce") == SIMULATION_YEAR
    ][
        ["node_id", "strict_seca"]
    ].copy()
    risk = risk.merge(
        cov_risk[["node_id", "S1_pop", "pop_density_km2", "neca"]],
        on="node_id", how="left", validate="1:1",
    ).merge(seca, on="node_id", how="left", validate="1:1")
    return members, regions, risk, cov


def _unit_map(members: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[str, str]]]:
    units = (members.groupby(
        ["region_id", "region_rank", "policy_iso3", "policy_country"],
        as_index=False,
    ).agg(
        geo_iso3s=("geo_iso3", lambda s: ";".join(sorted(set(map(str, s))))),
        geo_countries=("geo_country", lambda s: ";".join(sorted(set(map(str, s))))),
    ))
    units["unit_id"] = units["region_id"].astype(str) + "__" + units["policy_iso3"].astype(str)
    mapping: dict[str, tuple[str, str]] = {}
    for row in members.itertuples(index=False):
        key = str(row.geo_iso3)
        val = (str(row.region_id), str(row.policy_iso3))
        if key in mapping and mapping[key] != val:
            raise RuntimeError(f"同一入选地理单元映射到多个政策单元: {key}")
        mapping[key] = val
    return units, mapping


def _region_label(regions: pd.DataFrame) -> pd.Series:
    def label(row) -> str:
        return f"{row.region_id} {row.region_name_en}"
    return regions.apply(label, axis=1)


def _static_table(members: pd.DataFrame, regions: pd.DataFrame,
                  risk: pd.DataFrame, cov: pd.DataFrame,
                  baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    units, _ = _unit_map(members)
    regions = regions.copy()
    regions["region_label"] = _region_label(regions)
    base = cov[["node_id", "iso3", "pop_density_km2"]].merge(
        baseline, on="node_id", how="left", validate="1:1"
    )
    base["baseline_weighted_g"] = base["baseline_weighted_g"].fillna(0.0)
    base["baseline_pwe"] = base["baseline_weighted_g"] * base["pop_density_km2"].fillna(0.0)
    global_nodes = int(risk["node_id"].nunique())
    # X=50% 的筛选人口口径在聚类阶段保留全部沿海节点人口；oversized 只从
    # SPRI 城市等权均值与节点数中剔除。这里直接沿用正式 membership 的人口，
    # 才能精确复现 54.30% 的已固定覆盖率。
    # 区域 membership 只含 186 个可进入 Max-p 的单元；候选总体另有 11 个
    # 孤立/不可行单元。正式累计曲线的全球分母已经包含全部 197 个单元，
    # 因而从入选区域累计量与最终累计份额反推，避免漏掉孤立单元。
    global_pop = float(
        regions["coastal_population"].sum()
        / regions.iloc[-1]["cumulative_population_share"]
    )
    global_emission = float(base["baseline_weighted_g"].sum())

    rows = []
    for unit in units.itertuples(index=False):
        geo = str(unit.geo_iso3s).split(";")
        d = risk[risk["iso3"].astype(str).isin(geo)].copy()
        b = base[base["iso3"].astype(str).isin(geo)]
        member_rows = members[
            members["region_id"].eq(unit.region_id)
            & members["policy_iso3"].eq(unit.policy_iso3)
        ]
        unit_population = float(member_rows["coastal_population"].sum())
        if d.empty:
            raise RuntimeError(f"政策单元无风险城市: {unit.unit_id}")
        dim = d[["E", "S", "LAC"]].mean()
        dim_sum = float(dim.sum())
        rinfo = regions.loc[regions["region_id"].eq(unit.region_id)].iloc[0]
        rows.append({
            "row_type": "country",
            "unit_id": unit.unit_id,
            "region_id": unit.region_id,
            "region_rank": int(unit.region_rank),
            "region_label": rinfo["region_label"],
            "region_name_en": rinfo["region_name_en"],
            "region_name_zh": rinfo["region_name_zh"],
            "policy_iso3": unit.policy_iso3,
            "policy_country": unit.policy_country,
            "geo_iso3s": unit.geo_iso3s,
            "country_spri_eq": float(d["SPRI"].mean()),
            "n_coastal_nodes": int(d["node_id"].nunique()),
            "single_node_flag": bool(d["node_id"].nunique() == 1),
            "coastal_population": unit_population,
            "coastal_population_global_share": unit_population / global_pop,
            "region_cumulative_population_share": float(rinfo["cumulative_population_share"]),
            "strict_seca_city_share": float(d["strict_seca"].fillna(0).mean()),
            "neca_city_share": float(pd.to_numeric(d["neca"], errors="coerce").fillna(0).mean()),
            "mean_E": float(dim["E"]),
            "mean_S": float(dim["S"]),
            "mean_LAC": float(dim["LAC"]),
            "E_relative_contribution": float(dim["E"] / dim_sum),
            "S_relative_contribution": float(dim["S"] / dim_sum),
            "LAC_relative_contribution": float(dim["LAC"] / dim_sum),
            "dominant_spri_dimension": str(dim.idxmax()),
            "catchment_weighted_emission_g": float(b["baseline_weighted_g"].sum()),
            "catchment_emission_global_share": (
                float(b["baseline_weighted_g"].sum()) / global_emission
                if global_emission else np.nan
            ),
            "emerging_city_count": int(d["quadrant"].eq("emerging").sum()),
            "emerging_city_share": float(d["quadrant"].eq("emerging").mean()),
            "spri_median": float(d["SPRI"].median()),
            "spri_q25": float(d["SPRI"].quantile(0.25)),
            "spri_q75": float(d["SPRI"].quantile(0.75)),
            "spri_iqr": float(d["SPRI"].quantile(0.75) - d["SPRI"].quantile(0.25)),
            "spri_min": float(d["SPRI"].min()),
            "spri_max": float(d["SPRI"].max()),
        })
    table = pd.DataFrame(rows)
    income = pd.read_csv(INCOME)[["iso3", "income_group_2019"]]
    m49 = pd.read_csv(M49)[["iso3", "region_name", "subregion_name", "region_group_5"]]
    table = table.merge(income, left_on="policy_iso3", right_on="iso3", how="left").drop(
        columns="iso3"
    ).merge(m49, left_on="policy_iso3", right_on="iso3", how="left").drop(columns="iso3")
    table = table.rename(columns={
        "region_name": "policy_state_m49_region",
        "subregion_name": "policy_state_m49_subregion",
        "region_group_5": "policy_state_region_group_5",
    })
    table.attrs.update({
        "global_nodes": global_nodes,
        "global_population": global_pop,
        "global_emission": global_emission,
    })
    top_rows = []
    for region_id, group in members.groupby("region_id", sort=True):
        geo = set(group["geo_iso3"].astype(str))
        d = risk[risk["iso3"].astype(str).isin(geo)]
        top = d.nlargest(5, "SPRI")[[
            "node_id", "city", "country", "iso3", "SPRI",
        ]].copy()
        top.insert(0, "rank_within_region", np.arange(1, len(top) + 1))
        top.insert(0, "region_id", region_id)
        top_rows.append(top)
    return table, pd.concat(top_rows, ignore_index=True)


def _map_ports(calls: pd.DataFrame, geo_map: dict[str, tuple[str, str]]) -> pd.DataFrame:
    pmap = pd.read_csv(PORT_MAP, encoding="utf-8-sig")
    pm = pmap[["Port_Name", "iso3", "country"]].drop_duplicates("Port_Name")
    out = calls.merge(pm, on="Port_Name", how="left", validate="m:1")
    mapped = out["iso3"].astype(str).map(geo_map)
    out["region_id"] = mapped.map(lambda x: x[0] if isinstance(x, tuple) else np.nan)
    out["policy_iso3"] = mapped.map(lambda x: x[1] if isinstance(x, tuple) else np.nan)
    out["unit_id"] = out["region_id"].astype(str) + "__" + out["policy_iso3"].astype(str)
    out.loc[out["region_id"].isna(), "unit_id"] = np.nan
    return out


def _dynamic_fields(table: pd.DataFrame, cov: pd.DataFrame,
                    geo_map: dict[str, tuple[str, str]]) -> pd.DataFrame:
    totals = pd.read_parquet(
        CON / f"policy_delta_totals_{SIMULATION_YEAR}.parquet"
    )
    calls = pd.read_parquet(CON / f"ops_calls_container_{SIMULATION_YEAR}.parquet")
    city_panel = pd.read_parquet(SCN / "city_scenario_effect_panel.parquet")
    calls = _map_ports(calls, geo_map)
    calls["eligible"] = (
        (pd.to_numeric(calls["Gross_Tonnage"], errors="coerce") > 5000)
        & (pd.to_numeric(calls["Strict_Berth_Hours"], errors="coerce") >= 2.0)
        & calls["Ship_Type"].astype(str).str.casefold().eq("container")
    )
    calls_sel = calls[calls["unit_id"].notna()].copy()
    ops = calls_sel.groupby("unit_id", as_index=False).agg(
        all_container_calls=("Call_ID", "size"),
        eligible_ops_calls=("eligible", "sum"),
        eligible_ops_energy_kwh=("OPS_Energy_kWh", lambda s: float(s[calls_sel.loc[s.index, "eligible"]].sum())),
    )
    ops["eligible_ops_call_share"] = ops["eligible_ops_calls"] / ops["all_container_calls"].replace(0, np.nan)

    for col in ["B_SO2", "B_PM", "B_NOx", "D_SO2_SECA", "D_PM_SECA", "D_NOx_NECA",
                "NME_SO2_CURRENT", "NME_PM_CURRENT", "NME_NOx_CURRENT"]:
        if col not in totals:
            raise RuntimeError(f"正式 AIS 合并表缺少列: {col}")
    totals["unit_id"] = totals["Zone_ID"].astype(str)
    totals["policy_baseline_weighted_g"] = sum(
        WEIGHTS[p] * totals[f"B_{p}"] for p in POLS
    )
    totals["area_reduction_weighted_g"] = (
        WEIGHTS["SO2"] * totals["D_SO2_SECA"]
        + WEIGHTS["PM"] * totals["D_PM_SECA"]
        + WEIGHTS["NOx"] * totals["D_NOx_NECA"]
    )
    totals["no_margin_current_weighted_g"] = (
        WEIGHTS["SO2"] * totals["NME_SO2_CURRENT"]
        + WEIGHTS["PM"] * totals["NME_PM_CURRENT"]
        + WEIGHTS["NOx"] * totals["NME_NOx_CURRENT"]
    )
    tz = totals.groupby("unit_id", as_index=False).agg(
        policy_baseline_weighted_g=("policy_baseline_weighted_g", "sum"),
        area_reduction_weighted_g=("area_reduction_weighted_g", "sum"),
        no_margin_current_weighted_g=("no_margin_current_weighted_g", "sum"),
    )
    tz["no_margin_current_emission_share"] = (
        tz["no_margin_current_weighted_g"] / tz["policy_baseline_weighted_g"].replace(0, np.nan)
    )

    elig = calls_sel[calls_sel["eligible"]].copy()
    elig["ops_gross_weighted_g"] = sum(
        WEIGHTS[p] * elig[f"Berth_{p}_g"] for p in POLS
    )
    elig["ops_overlap_weighted_g"] = (
        WEIGHTS["SO2"] * elig["D_SO2_SECA"]
        + WEIGHTS["PM"] * elig["D_PM_SECA"]
        + WEIGHTS["NOx"] * elig["D_NOx_NECA"]
    )
    og = elig.groupby("unit_id", as_index=False).agg(
        ops_gross_weighted_g=("ops_gross_weighted_g", "sum"),
        ops_overlap_weighted_g=("ops_overlap_weighted_g", "sum"),
    )
    effect = tz.merge(og, on="unit_id", how="left")
    for col in ["ops_gross_weighted_g", "ops_overlap_weighted_g"]:
        effect[col] = effect[col].fillna(0.0)
    effect["s7_exact_weighted_reduction_g"] = (
        effect["area_reduction_weighted_g"]
        + effect["ops_gross_weighted_g"] - effect["ops_overlap_weighted_g"]
    )
    effect["s7_exact_emission_reduction_rate"] = (
        effect["s7_exact_weighted_reduction_g"]
        / effect["policy_baseline_weighted_g"].replace(0, np.nan)
    )

    cp = city_panel[city_panel["scenario"].eq("S7")].copy()
    cp = cp.drop(columns=["iso3"], errors="ignore").merge(
        cov[["node_id", "iso3"]], on="node_id", how="inner", validate="m:1"
    )
    mapped = cp["iso3"].astype(str).map(geo_map)
    cp["unit_id"] = mapped.map(
        lambda x: f"{x[0]}__{x[1]}" if isinstance(x, tuple) else np.nan
    )
    pwe = cp[cp["unit_id"].notna()].groupby("unit_id", as_index=False).agg(
        country_baseline_pwe=("base_pwe", "sum"),
        s7_exposure_reduction=("reduction_pwe", "sum"),
    )
    pwe["s7_exposure_reduction_rate"] = (
        pwe["s7_exposure_reduction"] / pwe["country_baseline_pwe"].replace(0, np.nan)
    )
    return table.merge(ops, on="unit_id", how="left").merge(
        effect, on="unit_id", how="left"
    ).merge(pwe, on="unit_id", how="left")


def _s6_country_pollutant_mode(geo_map: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """S7 精确点总量的国家×污染物×工况分解（岸电优先去重）。"""
    totals = pd.read_parquet(
        CON / f"policy_delta_totals_{SIMULATION_YEAR}.parquet"
    )
    calls = _map_ports(
        pd.read_parquet(CON / f"ops_calls_container_{SIMULATION_YEAR}.parquet"),
        geo_map,
    )
    calls["eligible"] = (
        (pd.to_numeric(calls["Gross_Tonnage"], errors="coerce") > 5000)
        & (pd.to_numeric(calls["Strict_Berth_Hours"], errors="coerce") >= 2.0)
        & calls["Ship_Type"].astype(str).str.casefold().eq("container")
        & calls["unit_id"].notna()
    )
    mode_map = {
        "Berth_Class": "Berth", "Anchored_Class": "Anchored",
        "Manoeuvring_Class": "Manoeuvring", "Sea_Class": "Sea",
    }
    area_rows = []
    for pol, col in (
        ("SO2", "D_SO2_SECA"), ("PM", "D_PM_SECA"), ("NOx", "D_NOx_NECA"),
    ):
        area_rows.append(pd.DataFrame({
            "unit_id": totals["Zone_ID"].astype(str),
            "Ship_Type": totals["Ship_Type"].astype(str),
            "Mode": totals["Mode_Class"].map(mode_map),
            "Pollutant": pol,
            "reduction_g": pd.to_numeric(totals[col], errors="coerce").fillna(0.0),
        }))
    ops_rows = []
    e = calls[calls["eligible"]].copy()
    for pol, overlap in (
        ("SO2", "D_SO2_SECA"), ("PM", "D_PM_SECA"), ("NOx", "D_NOx_NECA"),
    ):
        ops_rows.append(pd.DataFrame({
            "unit_id": e["unit_id"], "Ship_Type": "Container",
            "Mode": "Berth", "Pollutant": pol,
            "reduction_g": (
                pd.to_numeric(e[f"Berth_{pol}_g"], errors="coerce").fillna(0.0)
                - pd.to_numeric(e[overlap], errors="coerce").fillna(0.0)
            ),
        }))
    out = pd.concat(area_rows + ops_rows, ignore_index=True)
    if (out["reduction_g"] < -1e-6).any():
        raise RuntimeError("Figure S6 数据出现岸电去重后负减排")
    out["reduction_g"] = out["reduction_g"].clip(lower=0.0)
    out[["region_id", "policy_iso3"]] = out["unit_id"].str.extract(
        r"^(R\d+)__(.+)$"
    )
    return out.groupby(
        ["region_id", "policy_iso3", "unit_id", "Ship_Type", "Mode", "Pollutant"],
        as_index=False,
    )["reduction_g"].sum()


def _add_region_subtotals(table: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame:
    out = []
    numeric_sum = [
        "n_coastal_nodes", "coastal_population", "catchment_weighted_emission_g",
        "all_container_calls", "eligible_ops_calls", "eligible_ops_energy_kwh",
        "policy_baseline_weighted_g", "no_margin_current_weighted_g",
        "s7_exact_weighted_reduction_g", "country_baseline_pwe", "s7_exposure_reduction",
    ]
    for region_id, group in table.groupby("region_id", sort=False):
        group = group.sort_values("country_spri_eq", ascending=False)
        out.append(group)
        r = regions.loc[regions["region_id"].eq(region_id)].iloc[0]
        rec = {c: np.nan for c in table.columns}
        rec.update({
            "row_type": "region_subtotal", "region_id": region_id,
            "region_rank": int(r["rank"]), "region_label": r["region_label"],
            "region_name_en": r["region_name_en"],
            "region_name_zh": r["region_name_zh"],
            "policy_iso3": "SUBTOTAL", "policy_country": "Regional subtotal",
            "country_spri_eq": float(r["region_spri"]),
            "region_cumulative_population_share": float(r["cumulative_population_share"]),
            "spri_median": float(r["city_spri_median"]),
            "spri_q25": float(r["city_spri_q25"]),
            "spri_q75": float(r["city_spri_q75"]),
            "spri_iqr": float(r["city_spri_iqr"]),
            "spri_min": float(r["city_spri_min"]),
            "spri_max": float(r["city_spri_max"]),
        })
        for col in numeric_sum:
            if col in group:
                rec[col] = float(group[col].fillna(0).sum())
        for share, num, den in (
            ("eligible_ops_call_share", "eligible_ops_calls", "all_container_calls"),
            ("no_margin_current_emission_share", "no_margin_current_weighted_g", "policy_baseline_weighted_g"),
            ("s7_exact_emission_reduction_rate", "s7_exact_weighted_reduction_g", "policy_baseline_weighted_g"),
            ("s7_exposure_reduction_rate", "s7_exposure_reduction", "country_baseline_pwe"),
        ):
            if num in rec and den in rec:
                rec[share] = rec[num] / rec[den] if rec[den] else np.nan
        rec["coastal_population_global_share"] = float(group["coastal_population_global_share"].sum())
        rec["catchment_emission_global_share"] = float(group["catchment_emission_global_share"].sum())
        out.append(pd.DataFrame([rec]))
    return pd.concat(out, ignore_index=True)


def _diagnostics(table: pd.DataFrame, top5: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = table[table["row_type"].eq("country")].copy()
    rows = []
    for col, diagnostic in (
        ("income_group_2019", "01_income_composition"),
        ("policy_state_m49_region", "01_m49_region_composition"),
        ("dominant_spri_dimension", "03_dominant_spri_dimension"),
    ):
        z = c[col].fillna("Missing").value_counts(dropna=False)
        for category, n in z.items():
            rows.append({
                "diagnostic": diagnostic, "category": category,
                "count": int(n), "share": float(n / len(c)) if len(c) else np.nan,
            })
    summary = pd.DataFrame(rows)
    return summary, top5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-only", action="store_true")
    args = ap.parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    members, regions, risk, cov = _static_inputs()
    regions = regions.merge(
        pd.read_csv(REGION_NAMES), on="region_id", how="left", validate="1:1"
    )
    if regions[["region_name_en", "region_name_zh"]].isna().any().any():
        raise RuntimeError("Table S14 区域名称未覆盖全部入选区域")
    regions["region_label"] = _region_label(regions)
    units, geo_map = _unit_map(members)
    if len(units) != 75:
        raise RuntimeError(f"正式政策单元数应为 75，实际 {len(units)}")
    baseline = _baseline_city()
    table, top5 = _static_table(members, regions, risk, cov, baseline)
    if not args.static_only:
        required = [
            CON / f"policy_delta_totals_{SIMULATION_YEAR}.parquet",
            CON / f"ops_calls_container_{SIMULATION_YEAR}.parquet",
            SCN / "city_scenario_effect_panel.parquet",
        ]
        missing = [p for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Table 7 动态字段未就绪: {missing}")
        table = _dynamic_fields(table, cov, geo_map)
        _s6_country_pollutant_mode(geo_map).to_csv(
            TABLES / "figure_s6_s7_country_pollutant_mode_data.csv",
            index=False, encoding="utf-8-sig",
        )
    table = _add_region_subtotals(table, regions)
    composition, top5 = _diagnostics(table, top5)

    suffix = "_static_staging" if args.static_only else ""
    table.to_csv(TABLES / f"table_7_selected_regions_countries{suffix}.csv",
                 index=False, encoding="utf-8-sig")
    composition.to_csv(TABLES / f"table_7_selection_composition{suffix}.csv",
                       index=False, encoding="utf-8-sig")
    top5.to_csv(TABLES / f"table_7_region_country_top5_cities{suffix}.csv",
                index=False, encoding="utf-8-sig")
    print(
        f"[complete] static_only={args.static_only} countries={len(table[table.row_type.eq('country')])} "
        f"regions={len(table[table.row_type.eq('region_subtotal')])} rows={len(table)}"
    )


if __name__ == "__main__":
    main()
