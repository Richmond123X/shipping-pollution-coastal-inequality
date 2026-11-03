# -*- coding: utf-8 -*-
"""Cross-check the reconstructed v10 delivery and write an auditable report."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageStat


OUT = Path(os.environ.get(
    "V10_OUT", r"E:\transfer justice\results_v10_reconstructed_2019_2027"
))
VALIDATION = OUT / "validation"
VALIDATION.mkdir(parents=True, exist_ok=True)
checks: list[dict] = []


def check(name: str, condition: bool, detail: object) -> None:
    checks.append({"check": name, "passed": bool(condition), "detail": str(detail)})


def main() -> None:
    reconstruction = pd.read_csv(OUT / "reconstruction_validation.csv")
    check("reconstruction validation", reconstruction.passed.all(),
          f"{int(reconstruction.passed.sum())}/{len(reconstruction)} checks passed")

    annual = pd.read_parquet(OUT / "data" / "emission_city_annual_2019_2025.parquet")
    years = sorted(annual.year.unique().tolist())
    check("annual emission years", years == list(range(2019, 2026)), years)
    counts = annual.groupby("year").node_id.nunique()
    check("balanced annual emission panel", counts.eq(1425).all() and len(annual) == 9975,
          counts.to_dict())
    check("unique annual node-year keys", not annual.duplicated(["node_id", "year"]).any(), len(annual))
    composite = 2 * annual.SO2_t + 5 * annual.PM_t + annual.NOx_t
    check("composite pollutant identity", np.allclose(annual.composite_t, composite, rtol=2e-10, atol=2e-7),
          float(np.max(np.abs(annual.composite_t - composite))))
    ship_sum = annual[[
        "composite_ship_container_t", "composite_ship_general_cargo_t",
        "composite_ship_oil_tanker_t", "composite_ship_bulk_carrier_t",
    ]].sum(axis=1)
    check("ship-type decomposition identity", np.allclose(annual.composite_t, ship_sum, rtol=2e-10, atol=2e-7),
          float(np.max(np.abs(annual.composite_t - ship_sum))))
    check("nonnegative annual emissions", annual.filter(regex="_t$").ge(-1e-10).all().all(),
          float(annual.filter(regex="_t$").min().min()))

    edge_years, edge_rows, edge_bad = [], {}, 0
    for year in range(2019, 2026):
        edge = pd.read_parquet(OUT / "data" / f"network_edges_{year}.parquet")
        edge_years.append(year)
        edge_rows[year] = len(edge)
        edge_bad += int(edge.w_t.lt(0).sum() + edge[["From_City", "To_City"]].isna().any(axis=1).sum())
    check("annual edge networks", edge_years == list(range(2019, 2026)) and edge_bad == 0, edge_rows)

    ppml = pd.read_csv(OUT / "tables" / "Table_2_city_PPML.csv")
    strict = pd.read_csv(OUT / "tables" / "Table_2B_strict_SECA_10pct_trade_effects.csv")
    check("PPML convergence", ppml.converged.all(), f"{int(ppml.converged.sum())}/{len(ppml)}")
    ppml_sample = pd.read_parquet(OUT / "data" / "ppml_city_sample.parquet")
    check("PPML formal sample", set(ppml_sample.year.unique()) == set(range(2020, 2025))
          and ppml.loc[ppml.panel.ne("C_GDP_strict"), "specification"].str.contains(
              "main_sample_2020-2024", regex=False).all(),
          {"years": sorted(ppml_sample.year.unique().tolist()),
           "specifications": sorted(ppml.specification.unique().tolist())})
    check("strict SECA reporting contract",
          set(strict.outcome) == {"SO2_t", "PM_t"} and set(strict.group) == {"inside", "outside"}
          and strict.sample_start.eq(2020).all() and strict.sample_end.eq(2024).all(),
          strict.groupby("outcome").group.nunique().to_dict())

    performance = pd.read_csv(OUT / "tables" / "Table_4_forecast_model_performance.csv")
    selected = performance.loc[performance.formal_selected, "model"].tolist()
    check("single formal forecast model", selected == ["gwnet"], selected)
    check("comparison models excluded from formal forecast",
          not performance.loc[performance.model.isin(["HA", "ARIMA_restricted", "LSTM"]), "formal_selected"].any(),
          performance.set_index("model").formal_selected.to_dict())
    check("forecast evaluation includes LSTM", "LSTM" in set(performance.model), performance.model.tolist())
    recursive = pd.read_csv(OUT / "tables" / "Table_4_two_step_recursive_diagnostics.csv")
    check("recursive forecast diagnostic finite",
          np.isfinite(recursive.select_dtypes(include=np.number).to_numpy()).all(), recursive.shape)

    forecast = pd.read_csv(OUT / "data" / "forecast_city_annual_2026_2027.csv")
    fcounts = forecast.groupby("year").node_id.nunique()
    bounds_ok = (forecast.composite_lo_t.ge(0) &
                 forecast.composite_mean_t.ge(forecast.composite_lo_t) &
                 forecast.composite_hi_t.ge(forecast.composite_mean_t)).all()
    check("formal forecast panel", set(forecast.year) == {2026, 2027} and fcounts.eq(1425).all()
          and len(forecast) == 2850 and bounds_ok, fcounts.to_dict())
    check("forecast values finite", np.isfinite(forecast.select_dtypes(include=np.number)).all().all(), forecast.shape)
    manifest = json.loads((OUT / "models" / "forecast_run_manifest.json").read_text(encoding="utf-8"))
    check("forecast run manifest years",
          manifest.get("test_period") == "2025" and manifest.get("forecast_period") == "2026-2027"
          and manifest.get("selected_model") == "gwnet", manifest)

    risk = pd.read_parquet(OUT / "data" / "SPRI_city_annual_2019_2025.parquet")
    rcounts = risk.groupby("year").node_id.nunique()
    check("balanced SPRI panel", set(risk.year) == set(range(2019, 2026)) and rcounts.eq(1425).all()
          and len(risk) == 9975, rcounts.to_dict())
    check("SPRI range and uncertainty", risk.SPRI.between(0, 100).all()
          and risk.SPRI_low.le(risk.SPRI).all() and risk.SPRI_high.ge(risk.SPRI).all(),
          [float(risk.SPRI.min()), float(risk.SPRI.max())])
    weights = pd.read_csv(OUT / "tables" / "Table_5A_CRITIC_weights.csv")
    weight_sums = weights.groupby("dimension").weight.sum()
    check("2025 fixed CRITIC weights", weights.weight_year.eq(2025).all()
          and np.allclose(weight_sums, 1), weight_sums.to_dict())
    ci = pd.read_csv(OUT / "tables" / "Table_5E_city_equal_SPRI_concentration_index.csv")
    check("city-equal SPRI concentration index", set(ci.year) == set(range(2019, 2026))
          and ci.weighting.eq("city_equal").all() and ci.n_cities.eq(1425).all(),
          ci[["year", "spri_concentration_index"]].to_dict("records"))

    ablation = pd.read_csv(OUT / "tables" / "Table_5B_time_information_ablation_summary.csv")
    ranks = pd.read_csv(OUT / "tables" / "Table_S10_city_model_ranks_2019_2025.csv")
    check("ablation summary dimensions", set(ablation.panel) == {"A_overall", "B_region", "C_income"}
          and {"M0", "M1", "M2"} <= set(ablation.model), sorted(ablation.panel.unique()))
    check("city-level ablation rank table", len(ranks) == 9975 and not ranks.duplicated(["node_id", "year"]).any()
          and {"rank_m0", "rank_m1", "rank_m2", "rank_m3"} <= set(ranks.columns), ranks.shape)

    policy_panel = pd.read_csv(OUT / "policy_inputs" / "spri_panel_2019_2027.csv")
    pcounts = policy_panel.groupby("year").node_id.nunique()
    check("policy-ready 2019-2027 panel", set(policy_panel.year) == set(range(2019, 2028))
          and pcounts.eq(1425).all() and not policy_panel.isna().any().any(), pcounts.to_dict())
    pmanifest = json.loads((OUT / "policy_inputs" / "policy_input_manifest.json").read_text(encoding="utf-8"))
    check("policy run deferred by design", pmanifest.get("policy_simulation_executed") is False
          and pmanifest.get("status") == "ready_for_policy_simulation", pmanifest.get("status"))
    city_files = list((OUT / "reconstructed_monthly").glob("2025??/*/city_emissions_*_2025??.csv"))
    check("2025 policy baseline inventory", len(city_files) == 48, len(city_files))

    established = list((OUT / "figures" / "established_panels").rglob("*.png"))
    v10 = list((OUT / "figures" / "v10_new").rglob("*.png"))
    check("figure inventory", len(established) == 40 and len(v10) == 40,
          {"established_panels": len(established), "v10_new": len(v10)})
    pdfs = list(OUT.rglob("*.pdf"))
    check("PNG-only delivery", not pdfs, [str(x) for x in pdfs])
    bad_images = []
    image_rows = []
    for path in established + v10:
        try:
            with Image.open(path) as image:
                width, height = image.size
                thumb = image.convert("L")
                thumb.thumbnail((256, 256))
                std = float(ImageStat.Stat(thumb).stddev[0])
                ok = width >= 900 and height >= 600 and std >= 4.0
                image_rows.append({"file": str(path.relative_to(OUT)), "width": width,
                                   "height": height, "grayscale_std": std, "passed": ok})
                if not ok:
                    bad_images.append(str(path))
        except Exception as exc:
            bad_images.append(f"{path}: {exc}")
    pd.DataFrame(image_rows).to_csv(VALIDATION / "figure_render_audit.csv", index=False)
    check("figure render integrity", not bad_images, bad_images)

    empty_tables = []
    for path in sorted((OUT / "tables").glob("*.csv")):
        try:
            pd.read_csv(path)
        except pd.errors.EmptyDataError:
            empty_tables.append(path.name)
    check("CSV table schemas readable", not empty_tables, empty_tables)

    audit = pd.DataFrame(checks)
    audit.to_csv(VALIDATION / "final_delivery_audit.csv", index=False, encoding="utf-8-sig")
    failed = audit.loc[~audit.passed]
    summary = {
        "status": "PASS" if failed.empty else "FAIL",
        "checks_total": int(len(audit)),
        "checks_passed": int(audit.passed.sum()),
        "checks_failed": int((~audit.passed).sum()),
        "failed_checks": failed.check.tolist(),
        "observation_period": "2019-2025",
        "formal_test_year": 2025,
        "forecast_period": "2026-2027",
        "policy_simulation_executed": False,
    }
    (VALIDATION / "final_delivery_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    if not failed.empty:
        raise SystemExit(f"delivery verification failed: {failed[['check', 'detail']].to_dict('records')}")


if __name__ == "__main__":
    main()
