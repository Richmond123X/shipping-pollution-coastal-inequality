#!/usr/bin/env python3
"""Independent final-delivery checks for the empirical research package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    base = workspace / "empirical_trade_emissions"
    processed = base / "data_processed"
    results = base / "results"
    models = results / "models"

    required = [
        processed / "emission_edges_country_annual.parquet",
        processed / "emission_catchment_country_annual.parquet",
        processed / "trade_dyad_annual.parquet",
        processed / "trade_country_annual.parquet",
        processed / "ppml_dyad_sample.parquet",
        processed / "ppml_country_sample.parquet",
        results / "empirical_report_zh.md",
        results / "table_1_main_results.csv",
        results / "table_s1_sample_attrition_final.csv",
        results / "table_s2_robustness.csv",
        results / "completion_audit.csv",
        models / "model_coefficients.csv",
        models / "group_elasticities.csv",
        models / "wald_tests.csv",
        models / "bias_corrected_coefficients.csv",
        models / "bias_correction_author_example_validation.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required deliverables: {missing}")

    source_manifest = pd.read_csv(base / "data_raw" / "source_manifest.csv")
    for row in source_manifest.itertuples(index=False):
        path = Path(row.file)
        if not path.exists() or path.stat().st_size != int(row.bytes) or digest(path) != row.sha256:
            raise RuntimeError(f"Raw-source provenance check failed: {path}")

    artifact_manifest = pd.read_csv(results / "artifact_manifest.csv")
    for row in artifact_manifest.itertuples(index=False):
        path = Path(row.file)
        if not path.exists() or path.stat().st_size != int(row.bytes) or digest(path) != row.sha256:
            raise RuntimeError(f"Final-artifact hash check failed: {path}")

    coefs = pd.read_csv(models / "model_coefficients.csv")
    failures = pd.read_csv(models / "model_failures.csv")
    validation = pd.read_csv(models / "bias_correction_author_example_validation.csv")
    if len(failures):
        raise RuntimeError("Model failures are present")
    if not coefs.converged.astype(str).str.lower().eq("true").all():
        raise RuntimeError("A reported model did not converge")
    if coefs.model_id.nunique() != 20:
        raise RuntimeError(f"Expected 20 fitted models, found {coefs.model_id.nunique()}")
    if validation.absolute_difference.max() >= 5e-6:
        raise RuntimeError("Bias correction does not reproduce the authors' example")

    dyad = pd.read_parquet(processed / "ppml_dyad_sample.parquet")
    country = pd.read_parquet(processed / "ppml_country_sample.parquet")
    if set(dyad.year.unique()) != set(range(2019, 2025)):
        raise RuntimeError("Dyad sample has the wrong year window")
    if set(country.year.unique()) != set(range(2019, 2025)):
        raise RuntimeError("Country sample has the wrong year window")
    d1 = coefs.loc[(coefs.model_id == "D1_base") & (coefs.coefficient_term == "ln_Q")].iloc[0]
    c1 = coefs.loc[(coefs.model_id == "C1_base") & (coefs.coefficient_term == "ln_Q")].iloc[0]
    if len(dyad) != 61002 or int(d1.nobs) != 60963 or int(d1.units) != 10553:
        raise RuntimeError("D1 sample attrition does not match model metadata")
    if int(country.q_total_tonnes.gt(0).sum()) != int(c1.nobs) or int(c1.nobs) != 1068:
        raise RuntimeError("C1 sample does not match model metadata")

    report = (results / "empirical_report_zh.md").read_text(encoding="utf-8")
    table = pd.read_csv(results / "table_1_main_results.csv")
    expected_d = f"{d1.estimate:.4f}"
    expected_c = f"{c1.estimate:.4f}"
    if expected_d not in str(table.iloc[0, 1]) or expected_c not in str(table.iloc[0, 4]):
        raise RuntimeError("Table 1 does not reproduce the model coefficients")
    if expected_d not in report or expected_c not in report:
        raise RuntimeError("Report does not reproduce the main model coefficients")
    if re.search(r"0\.__|TODO|TBD|待填|占位符", report, flags=re.I):
        raise RuntimeError("The final report still contains a placeholder")

    scripts_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (base / "scripts").glob("*")
        if path.is_file() and path.name != Path(__file__).name
    ).lower()
    if "uncomtrade" in scripts_text or "un_comtrade" in scripts_text:
        raise RuntimeError("A processing script refers to a standalone UN Comtrade input")

    edge_files = list((workspace / "results_monthly").glob("20????/*/city_edge_list_*.csv"))
    city_files = list((workspace / "results_monthly").glob("20????/*/city_emissions_*.csv"))
    if len(edge_files) < 864 or len(city_files) < 288:
        raise RuntimeError(
            f"Emission inputs are incomplete: {len(edge_files)} edge files, {len(city_files)} city files"
        )

    audit = pd.read_csv(results / "completion_audit.csv")
    if audit.status.eq("FAIL").any():
        raise RuntimeError("Completion audit contains a failed requirement")

    summary = {
        "status": "PASS",
        "official_source_files_hash_verified": int(len(source_manifest)),
        "final_artifacts_hash_verified": int(len(artifact_manifest)),
        "edge_input_files_found": len(edge_files),
        "city_input_files_found": len(city_files),
        "fitted_models": int(coefs.model_id.nunique()),
        "model_failures": int(len(failures)),
        "d1_nobs": int(d1.nobs),
        "c1_nobs": int(c1.nobs),
        "bias_validation_max_abs_difference": float(validation.absolute_difference.max()),
        "known_warning": (
            "Country catchment uses the plan-authorized sum-of-city fallback; "
            "city overlaps remain because 2019-2024 point-level emissions are unavailable."
        ),
    }
    (results / "final_verification.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
