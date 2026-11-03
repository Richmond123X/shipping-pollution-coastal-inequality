# -*- coding: utf-8 -*-
"""V12 政策模拟正式产物的独立终检。

本脚本不生成或修补结果。它只读取 03--07 的正式产物，验证年度全频分片运行、
S1--S7 完整性、点级/分解守恒、组合情景去重边界、岸电扫描单调
性、Table 6/7 与补充表覆盖，以及 PNG-only 图件要求。任一必需检查失败时
仍会先写出机器可读审计表，再以非零状态退出。
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from policy_run_config import (
    AIS_PREFETCH_BATCH_SIZE, AIS_SAMPLE_INTERVAL_SEC, AIS_SHARD_SIZE,
    CONSOLIDATED_ROOT as CON,
    DIRECT_MANIFEST, DIRECT_ROOT as DIRECT,
    ROOT, SCENARIO_ROOT as SCN, SIMULATION_YEAR,
)


TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
ZONES = ROOT / "inputs" / "formal_policy_zones_50pct.gpkg"

POLS = ["SO2", "PM", "NOx"]
AREA_SCENARIOS = ["S1", "S2", "S4", "S5", "S6", "S7"]
S3_SCENARIOS = [f"S3_{a:03d}" for a in range(10, 101, 10)]
EXPECTED_SCENARIOS = set(AREA_SCENARIOS + S3_SCENARIOS)
REDESIGNED_FIGURES = {
    "fig7a_s7_so2_regional_reduction_map.png",
    "fig7b_s7_pm_regional_reduction_map.png",
    "fig7c_s7_nox_regional_reduction_map.png",
    "fig7d_scenario_pollutant_reduction_rates.png",
    "fig7e_combination_marginal_overlap_net.png",
    "fig7f_ops_diminishing_returns.png",
    "fig8a_s7_regional_exposure_heterogeneity_map.png",
    "fig8b_global_emission_exposure_reduction.png",
    "fig8c_exposure_efficiency.png",
    "fig8d_vulnerable_population_benefit_share.png",
    "fig8e_gdp_quartile_reduction_rate_heatmap.png",
    "fig8f_gdp_quartile_reduction_distribution.png",
    "fig8g_s7_risk_adjusted_policy_benefit_alignment_map.png",
    "fig8h_s7_regional_population_exposure_reduction_map.png",
}
SUPPLEMENTARY_FIGURES = {
    "figS5a_policy_zone_intersection_area_map.png",
    "figS5b_policy_zone_overlap_share_map.png",
    "figS6_s7_policy_unit_pollutant_mode_map.png",
}
EXPECTED_FIGURES = REDESIGNED_FIGURES | SUPPLEMENTARY_FIGURES


class Verifier:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, check: str, passed: bool, evidence: str, severity: str = "required") -> None:
        self.rows.append({
            "check": check,
            "status": "pass" if passed else "fail",
            "severity": severity,
            "evidence": str(evidence),
        })

    def info(self, check: str, evidence: str) -> None:
        self.rows.append({
            "check": check, "status": "info", "severity": "diagnostic",
            "evidence": str(evidence),
        })

    def require_file(self, path: Path, check: str) -> bool:
        ok = path.is_file() and path.stat().st_size > 0
        self.add(check, ok, f"{path}; bytes={path.stat().st_size if path.is_file() else 0}")
        return ok

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def failed(self) -> bool:
        return any(r["status"] == "fail" and r["severity"] == "required" for r in self.rows)


def close(a: float, b: float, atol: float = 1e-4, rtol: float = 1e-10) -> bool:
    return bool(abs(a - b) <= max(atol, rtol * max(abs(a), abs(b))))


def pivot_exact(exact: pd.DataFrame) -> pd.DataFrame:
    return exact.pivot_table(
        index="scenario", columns="Pollutant", values="reduction_g",
        aggfunc="sum", fill_value=0.0,
    ).reindex(columns=POLS, fill_value=0.0)


def check_cache_and_run(v: Verifier) -> None:
    direct_state = DIRECT / "direct_run_state.json"
    if v.require_file(direct_state, "direct annual shard run state exists"):
        state = json.loads(direct_state.read_text(encoding="utf-8"))
        ok = (
            state.get("status") == "complete"
            and state.get("execution_mode") == "direct_annual_shards"
            and set(state.get("ship_types", []))
            == {"Bulk carrier", "General cargo", "Oil tanker", "Container"}
            and int(state.get("config_key", {}).get("sample_interval_sec", -1))
            == AIS_SAMPLE_INTERVAL_SEC
            and int(state.get("config_key", {}).get("prefetch_batch_size", -1))
            == AIS_PREFETCH_BATCH_SIZE
            and int(state.get("config_key", {}).get("shard_size", -1))
            == AIS_SHARD_SIZE
            and state.get("config_key", {}).get("policy_zone_layer") == "policy_units"
            and int(state.get("config_key", {}).get("simulation_year", -1))
            == SIMULATION_YEAR
        )
        v.add("direct annual shard parameters", ok, json.dumps(state, ensure_ascii=False))

    shard_path = DIRECT_MANIFEST
    if v.require_file(shard_path, "direct annual shard manifest exists"):
        shard = pd.read_csv(shard_path)
        status = shard.get("status", pd.Series(dtype=str)).astype(str).str.casefold()
        coverage_ok = len(shard) > 0
        for _, group in shard.groupby("ship_type"):
            group = group.sort_values("start_index")
            starts = group["start_index"].astype(int).tolist()
            sizes = group["n_requested"].astype(int).tolist()
            expected_starts = []
            cursor = 0
            for size in sizes:
                expected_starts.append(cursor)
                cursor += size
            universe = int(group["universe_n"].iloc[0])
            coverage_ok = coverage_ok and starts == expected_starts and cursor == universe
        v.add(
            "all direct annual AIS shards complete",
            coverage_ok
            and set(shard["ship_type"].astype(str))
            == {"Bulk carrier", "General cargo", "Oil tanker", "Container"}
            and status.isin({"complete", "completed", "skipped_complete"}).all(),
            f"rows={len(shard)} status_counts={status.value_counts(dropna=False).to_dict()}",
        )

    for name in ("ais_policy_shard_processing_qc.csv", "ais_policy_consolidation_qc.csv"):
        v.require_file(TABLES / name, f"{name} exists")


def check_scenarios(v: Verifier) -> None:
    exact_path = TABLES / "scenario_exact_reductions_by_pollutant.csv"
    detail_path = TABLES / "scenario_exact_reductions_ship_mode_pollutant.csv"
    totals_path = CON / f"policy_delta_totals_{SIMULATION_YEAR}.parquet"
    if not all(v.require_file(p, f"{p.name} exists") for p in (exact_path, detail_path, totals_path)):
        return

    exact = pd.read_csv(exact_path)
    detail = pd.read_csv(detail_path)
    totals = pd.read_parquet(totals_path)
    actual = set(exact["scenario"].astype(str))
    v.add(
        "S1-S7 and ten S3 alpha levels present",
        actual == EXPECTED_SCENARIOS,
        f"actual={sorted(actual)} missing={sorted(EXPECTED_SCENARIOS-actual)} extra={sorted(actual-EXPECTED_SCENARIOS)}",
    )
    vals = pd.to_numeric(exact["reduction_g"], errors="coerce")
    v.add(
        "scenario exact reductions are finite and nonnegative",
        vals.notna().all() and (vals >= -1e-6).all(),
        f"rows={len(vals)} min={vals.min()} nonfinite={int(vals.isna().sum())}",
    )

    dsum = detail.groupby(["scenario", "Pollutant"], as_index=False)["reduction_g"].sum()
    cmp = dsum.merge(exact, on=["scenario", "Pollutant"], how="outer",
                     suffixes=("_detail", "_exact")).fillna(0.0)
    diff = (cmp["reduction_g_detail"] - cmp["reduction_g_exact"]).abs()
    tol = np.maximum(1e-4, cmp["reduction_g_exact"].abs() * 1e-10)
    v.add(
        "ship-mode-pollutant decomposition conserves exact totals",
        bool((diff <= tol).all()),
        f"max_abs_diff_g={float(diff.max() if len(diff) else 0)}",
    )

    for dcol, bcol in (("D_SO2_SECA", "B_SO2"), ("D_PM_SECA", "B_PM"),
                       ("D_NOx_NECA", "B_NOx")):
        d = pd.to_numeric(totals[dcol], errors="coerce")
        b = pd.to_numeric(totals[bcol], errors="coerce")
        excess = d - b
        v.add(
            f"{dcol} does not exceed policy-zone baseline {bcol}",
            d.notna().all() and b.notna().all()
            and (d >= -1e-6).all() and (b >= -1e-6).all()
            and bool((excess <= np.maximum(1e-4, b.abs() * 1e-10)).all()),
            f"sum_delta={float(d.sum())} sum_base={float(b.sum())} max_excess={float(excess.max())}",
        )

    ep = pivot_exact(exact)
    s3 = ep.loc["S3_100"]
    v.add(
        "S4 equals S1 plus S2",
        all(close(float(ep.loc["S4", p]), float(ep.loc["S1", p] + ep.loc["S2", p])) for p in POLS),
        "per-pollutant exact comparison",
    )
    combo_specs = {
        "S5": ep.loc["S1"],
        "S6": ep.loc["S2"],
        "S7": ep.loc["S4"],
    }
    for sc, area in combo_specs.items():
        got = ep.loc[sc]
        lower = np.maximum(area.to_numpy(float), s3.to_numpy(float))
        upper = area.to_numpy(float) + s3.to_numpy(float)
        ok = bool(np.all(got.to_numpy(float) >= lower - np.maximum(1e-4, lower * 1e-10))
                  and np.all(got.to_numpy(float) <= upper + np.maximum(1e-4, upper * 1e-10)))
        v.add(
            f"{sc} respects shore-power-first overlap bounds", ok,
            f"got={got.to_dict()} lower={dict(zip(POLS, lower))} upper={dict(zip(POLS, upper))}",
        )

    s3p = ep.reindex(S3_SCENARIOS)[POLS]
    v.add(
        "S3 pollutant reductions are monotonic in alpha",
        bool((s3p.diff().iloc[1:] >= -1e-6).all().all()),
        f"rows={len(s3p)}",
    )


def check_ops_and_grid(v: Verifier) -> None:
    scan_path = TABLES / "ops_alpha_scan.csv"
    calls_path = SCN / "ops_calls_with_eligibility.parquet"
    if not all(v.require_file(p, f"{p.name} exists") for p in (scan_path, calls_path)):
        return
    scan = pd.read_csv(scan_path).sort_values("alpha")
    calls = pd.read_parquet(calls_path)
    observed_alpha = scan["alpha"].to_numpy(float)
    v.add(
        "shore-power scan contains exactly 10%-100%",
        len(observed_alpha) == 10
        and np.allclose(observed_alpha, np.arange(0.1, 1.01, 0.1)),
        f"alpha={scan['alpha'].tolist()}",
    )
    mono_cols = ["nominal_calls", "ops_energy_kwh", "gross_weighted_reduction_g",
                 "exposure_reduction"]
    bad = {c: float(pd.to_numeric(scan[c], errors="coerce").diff().min()) for c in mono_cols}
    v.add(
        "shore-power calls, energy, gross reduction and exposure are monotonic",
        all(np.isfinite(x) and x >= -1e-6 for x in bad.values()),
        str(bad),
    )

    eligible = calls.get("eligible", pd.Series(False, index=calls.index)).fillna(False).astype(bool)
    ge2 = pd.to_numeric(calls.get("Strict_Berth_Hours"), errors="coerce") >= 2.0
    gt_missing = pd.to_numeric(calls.get("Gross_Tonnage"), errors="coerce").isna()
    indeterminate = ge2 & gt_missing
    v.info(
        "GT completeness among policy-scope container calls",
        json.dumps({
            "all_calls": int(len(calls)),
            "eligible_calls": int(eligible.sum()),
            "berth_ge2h_calls": int(ge2.sum()),
            "indeterminate_gt_calls_ge2h": int(indeterminate.sum()),
            "indeterminate_share_of_ge2h": (
                float(indeterminate.sum() / ge2.sum()) if ge2.sum() else None
            ),
        }, ensure_ascii=False),
    )

    country_grid_path = TABLES / "country_ops_grid_emissions_by_alpha.csv"
    if eligible.any():
        if v.require_file(country_grid_path, "country grid replacement table exists for eligible calls"):
            cg = pd.read_csv(country_grid_path)
            required = [f"grid_{p}_g" for p in POLS] + [f"net_{p}_g" for p in POLS]
            missing_cols = [c for c in required if c not in cg]
            ok = not missing_cols and cg[required].apply(pd.to_numeric, errors="coerce").notna().all().all()
            v.add(
                "eligible shore-power countries have complete SO2/PM10/NOx grid effects",
                ok,
                f"rows={len(cg)} missing_columns={missing_cols}",
            )
    else:
        v.info("country grid replacement table", "no eligible calls; all shore-power effects are zero")


def check_tables(v: Verifier) -> None:
    required = [
        "policy_scenario_data_feasibility.csv",
        "table_s11_coastal_geo_pairwise_adjacency.csv",
        "table_s12a_maxp_region_summary.csv",
        "table_s12b_maxp_geo_membership.csv",
        "table_s12c_maxp_200_start_results.csv",
        "table_s12d_maxp_solution_stability.csv",
        "table_s13_manual_geographic_review.csv",
        "table_s14_region_names.csv",
        "table_6_policy_scenario_summary.csv",
        "table_7_selected_regions_countries.csv",
        "table_7_selection_composition.csv",
        "table_7_region_country_top5_cities.csv",
        "table_s15_quartile_effects.csv",
        "table_s16_ci_pwe_policy_change.csv",
        "table_s17_country_trade_exposure_intensity.csv",
        "table_s18_ops_effective_subset_by_port.csv",
        "table_s18_ops_rounding_by_port_alpha.csv",
        "table_s18_ops_effective_subset_by_country.csv",
        "table_s18_ops_effective_subset_metrics.csv",
        "table_s19_region_risk_benefit_alignment.csv",
        "figure_s5_policy_zone_pairwise_overlap_data.csv",
        "figure_s6_s7_country_pollutant_mode_data.csv",
    ]
    present = {name: v.require_file(TABLES / name, f"{name} exists") for name in required}
    if present["policy_scenario_data_feasibility.csv"]:
        feasibility = pd.read_csv(TABLES / "policy_scenario_data_feasibility.csv")
        scenarios = set(feasibility["scenario"].astype(str))
        raw_required = feasibility["raw_single_ship_ais_required"].astype(str).str.casefold().isin(
            {"true", "1", "yes"}
        )
        aggregate_sufficient = feasibility["aggregated_network_sufficient"].astype(str).str.casefold().isin(
            {"true", "1", "yes"}
        )
        v.add(
            "scenario feasibility audit covers S1-S7 and records AIS necessity",
            scenarios == {f"S{i}" for i in range(1, 8)}
            and raw_required.all() and not aggregate_sufficient.any(),
            f"scenarios={sorted(scenarios)}; raw_required={int(raw_required.sum())}/7; "
            f"aggregated_sufficient={int(aggregate_sufficient.sum())}/7",
        )
    if present["table_s11_coastal_geo_pairwise_adjacency.csv"]:
        s11 = pd.read_csv(TABLES / "table_s11_coastal_geo_pairwise_adjacency.csv")
        v.add(
            "Table S11 contains every unordered pair of 197 coastal geographic units",
            len(s11) == 197 * 196 // 2
            and not s11[["geo_iso3_a", "geo_iso3_b"]].duplicated().any(),
            f"rows={len(s11)} expected={197*196//2}",
        )
    if present["table_s13_manual_geographic_review.csv"]:
        s13 = pd.read_csv(TABLES / "table_s13_manual_geographic_review.csv")
        v.add(
            "Table S13 records one manual decision for each of 56 regions",
            len(s13) == 56 and s13["region_id"].nunique() == 56
            and s13["manual_decision"].notna().all(),
            f"rows={len(s13)} decisions={s13['manual_decision'].value_counts().to_dict()}",
        )
    if present["table_6_policy_scenario_summary.csv"]:
        t6 = pd.read_csv(TABLES / "table_6_policy_scenario_summary.csv")
        sources = set(t6["source_scenario"].astype(str))
        expected = {"Baseline", "S1", "S2", "S4", "S5", "S6", "S7",
                    "S5-S1", "S6-S2", "S7-S4"}
        has_s3 = t6["source_scenario"].astype(str).str.startswith("S3_").any()
        v.add(
            "Table 6 contains baseline, seven main scenarios and three OPS marginals",
            expected.issubset(sources) and has_s3,
            f"source_scenario={sorted(sources)}",
        )
        if (SCN / "ops_calls_with_eligibility.parquet").is_file():
            calls = pd.read_parquet(SCN / "ops_calls_with_eligibility.parquet")
            elig = calls.get("eligible", pd.Series(False, index=calls.index)).fillna(False).astype(bool)
            net = t6.get("row_type", pd.Series(dtype=str)).astype(str).eq("country_net")
            v.add(
                "Table 6 reports country-level net rows when OPS is effective",
                (not elig.any()) or int(net.sum()) == 4,
                f"eligible_calls={int(elig.sum())} country_net_rows={int(net.sum())}",
            )

    if present["table_7_selected_regions_countries.csv"]:
        t7 = pd.read_csv(TABLES / "table_7_selected_regions_countries.csv")
        rc = t7.get("row_type", pd.Series(dtype=str)).astype(str).value_counts().to_dict()
        v.add(
            "Table 7 has 75 country rows and 22 region subtotal rows",
            rc.get("country", 0) == 75 and rc.get("region_subtotal", 0) == 22,
            f"row_type_counts={rc}",
        )
        dynamic = ["eligible_ops_calls", "no_margin_current_emission_share",
                   "s7_exact_emission_reduction_rate", "s7_exposure_reduction_rate"]
        missing = [c for c in dynamic if c not in t7]
        v.add("Table 7 contains formal AIS diagnostics and S7 effects", not missing,
              f"missing_columns={missing}")

    if present["table_s15_quartile_effects.csv"]:
        s15 = pd.read_csv(TABLES / "table_s15_quartile_effects.csv")
        sums = s15.groupby(["scenario", "grouping"])["distribution_share"].sum(min_count=1)
        finite = sums.dropna()
        v.add(
            "Table S15 nonempty distribution shares sum to one",
            len(finite) > 0 and np.allclose(finite.to_numpy(float), 1.0, atol=1e-10),
            f"groups={len(sums)} max_abs_error={float((finite-1).abs().max() if len(finite) else math.inf)}",
        )

    if present["table_s19_region_risk_benefit_alignment.csv"]:
        s19 = pd.read_csv(TABLES / "table_s19_region_risk_benefit_alignment.csv")
        needed = {
            "region_id", "baseline_vulnerable_pwe", "s7_vulnerable_benefit",
            "vulnerable_risk_share", "vulnerable_benefit_share",
            "alignment_ratio", "log2_alignment",
        }
        columns_ok = needed.issubset(s19.columns)
        v.add(
            "Table S19 contains one finite risk-benefit alignment record per selected region",
            columns_ok and len(s19) == 22 and s19["region_id"].nunique() == 22
            and np.isfinite(s19[list(needed - {"region_id"})].to_numpy(float)).all(),
            f"rows={len(s19)} unique_regions={s19['region_id'].nunique()} "
            f"missing_columns={sorted(needed-set(s19.columns))}",
        )
        if columns_ok:
            share_ok = (
                np.isclose(s19["vulnerable_risk_share"].sum(), 1.0, atol=1e-10)
                and np.isclose(s19["vulnerable_benefit_share"].sum(), 1.0, atol=1e-10)
            )
            formula_ok = np.allclose(
                s19["alignment_ratio"].to_numpy(float),
                (s19["vulnerable_benefit_share"]
                 / s19["vulnerable_risk_share"]).to_numpy(float),
                rtol=1e-10, atol=1e-12,
            ) and np.allclose(
                s19["log2_alignment"].to_numpy(float),
                np.log2(s19["alignment_ratio"].to_numpy(float)),
                rtol=1e-10, atol=1e-12,
            )
            v.add(
                "Table S19 shares and log-ratio reproduce the stated alignment equations",
                share_ok and formula_ok,
                f"risk_share_sum={s19['vulnerable_risk_share'].sum():.12f} "
                f"benefit_share_sum={s19['vulnerable_benefit_share'].sum():.12f} "
                f"formula_ok={formula_ok}",
            )


def check_figures(v: Verifier) -> None:
    actual = {p.name for p in FIGURES.glob("*.png") if not p.name.startswith("_preview")}
    v.add(
        "all required independent PNG figures exist",
        EXPECTED_FIGURES.issubset(actual),
        f"missing={sorted(EXPECTED_FIGURES-actual)} actual={sorted(actual)}",
    )
    bad = []
    dims = {}
    for name in sorted(EXPECTED_FIGURES & actual):
        p = FIGURES / name
        try:
            with Image.open(p) as im:
                dims[name] = [int(im.width), int(im.height)]
                ok = im.format == "PNG" and im.width >= 500 and im.height >= 400
                im.verify()
                if not ok:
                    bad.append(f"{name}:{dims[name]}")
        except Exception as exc:
            bad.append(f"{name}:{exc}")
    v.add("required PNG files decode and have usable dimensions", not bad,
          f"bad={bad}; dimensions={dims}")
    redesigned_dims = {name: tuple(dims[name]) for name in REDESIGNED_FIGURES if name in dims}
    v.add(
        "redesigned Figure 7-8 panels use one standalone canvas size",
        len(redesigned_dims) == len(REDESIGNED_FIGURES)
        and len(set(redesigned_dims.values())) == 1,
        f"dimensions={redesigned_dims}",
    )
    manifest_path = TABLES / "policy_redesigned_figure_manifest.csv"
    if v.require_file(manifest_path, "redesigned figure manifest exists"):
        manifest = pd.read_csv(manifest_path)
        manifest_files = set(manifest.get("file", pd.Series(dtype=str)).astype(str))
        group_counts = manifest.get("group", pd.Series(dtype=str)).value_counts().to_dict()
        v.add(
            "redesigned figure manifest contains all standalone policy panels",
            manifest_files == REDESIGNED_FIGURES
            and group_counts == {"Figure 7": 6, "Figure 8": 8},
            f"files={sorted(manifest_files)} group_counts={group_counts}",
        )
    pdfs = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*.pdf")]
    v.add("formal policy directory contains no PDF outputs", not pdfs, f"pdfs={pdfs}")


def main() -> None:
    v = Verifier()
    for check in (check_cache_and_run, check_scenarios, check_ops_and_grid,
                  check_tables, check_figures):
        try:
            check(v)
        except Exception as exc:
            v.add(f"{check.__name__} completed without exception", False,
                  f"{type(exc).__name__}: {exc}")

    frame = v.frame()
    out_csv = TABLES / "formal_output_verification.csv"
    out_json = ROOT / "formal_output_verification.json"
    frame.to_csv(out_csv, index=False, encoding="utf-8-sig")
    summary = {
        "status": "fail" if v.failed() else "pass",
        "required_pass": int(((frame.status == "pass") & (frame.severity == "required")).sum()),
        "required_fail": int(((frame.status == "fail") & (frame.severity == "required")).sum()),
        "diagnostic_info": int((frame.status == "info").sum()),
        "audit_csv": str(out_csv),
    }
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if v.failed():
        failed = frame[(frame.status == "fail") & (frame.severity == "required")]
        raise SystemExit("\n" + failed[["check", "evidence"]].to_string(index=False))


if __name__ == "__main__":
    main()
