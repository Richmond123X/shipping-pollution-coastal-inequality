# -*- coding: utf-8 -*-
"""Build comparable 2025 baseline/policy SPRI values from real AIS emissions.

The policy evaluation changes only the exposure dimension of SPRI. Vulnerability
and adaptive-capacity indicators are held fixed. Normalisation limits and CRITIC
weights are estimated once from the baseline panel, then reused unchanged for
every scenario so baseline-scenario differences are directly comparable.

The 2025 forecast exposure in ``spri_panel.csv`` is replaced by the completed
2025 AIS city-emission inventory. A city's 2025 policy reduction ratio is also
applied to its 2026-2027 forecast exposure, representing a persistent policy and
preventing the rolling future term from creating an artificial rebound.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "emission_network"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ANCHORS = list(range(2019, 2026))
GAMMA = np.log(2.0) / 5.0
MU, LAM, EPS = 1.0, 1.33, 1e-6
E_IND = ["E1", "E2", "E3"]
AC_IND = ["AC1", "AC2", "AC3", "AC4", "AC5"]
S_IND = ["S1", "S2"]
TVAR = E_IND + AC_IND
LEVEL_LOG = {"E1", "E2", "AC1"}
S_LOG = {"S1"}
F_RATIO = {"E1", "E2", "AC1", "AC2"}
POLLUTANT_WEIGHTS = {"SO2": 2.0, "PM": 5.0, "NOx": 1.0}
SCENARIO_POLLUTANTS = {
    "SECA": ["SO2", "PM"],
    "NECA": ["NOx"],
    "BOTH": ["SO2", "PM", "NOx"],
}
G_PER_T = 1e6


def _load_cfg() -> dict:
    from config_loader import load_config
    return load_config()


def _critic(columns: list[np.ndarray]) -> np.ndarray:
    matrix = np.column_stack(columns)
    sigma = matrix.std(axis=0, ddof=1)
    corr = np.nan_to_num(np.corrcoef(matrix, rowvar=False), nan=0.0)
    information = sigma * (1.0 - corr).sum(axis=1)
    total = information.sum()
    return information / total if total > 0 else np.ones(matrix.shape[1]) / matrix.shape[1]


def _apply(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    lo, hi = bounds
    clipped = np.clip(np.nan_to_num(values, nan=lo), lo, hi)
    return np.zeros_like(clipped) if hi == lo else (clipped - lo) / (hi - lo)


def _fit_pooled(raw: dict[tuple[str, int], np.ndarray],
                indicators: list[str], winsor: float = 0.01
                ) -> tuple[dict[tuple[str, int], np.ndarray], dict[str, tuple[float, float]]]:
    normalised: dict[tuple[str, int], np.ndarray] = {}
    bounds: dict[str, tuple[float, float]] = {}
    for indicator in indicators:
        pooled = np.concatenate([raw[(indicator, year)] for year in ANCHORS])
        lo, hi = np.nanquantile(pooled, winsor), np.nanquantile(pooled, 1.0 - winsor)
        bounds[indicator] = (float(lo), float(hi))
        for year in ANCHORS:
            normalised[(indicator, year)] = _apply(raw[(indicator, year)], bounds[indicator])
    return normalised, bounds


def _apply_pooled(raw: dict[tuple[str, int], np.ndarray],
                  indicators: list[str], bounds: dict[str, tuple[float, float]]
                  ) -> dict[tuple[str, int], np.ndarray]:
    return {(indicator, year): _apply(raw[(indicator, year)], bounds[indicator])
            for indicator in indicators for year in ANCHORS}


def _terms(panel: pd.DataFrame, nodes: list[str]) -> tuple[
        dict[tuple[str, int], np.ndarray],
        dict[tuple[str, int], np.ndarray],
        dict[tuple[str, int], np.ndarray],
        dict[tuple[str, int], np.ndarray]]:
    wide = {column: panel.pivot(index="node_id", columns="year", values=column)
            for column in TVAR + S_IND}

    def column(indicator: str, year: int) -> np.ndarray:
        if year not in wide[indicator].columns:
            return np.zeros(len(nodes))
        return (pd.to_numeric(wide[indicator][year], errors="coerce")
                .reindex(nodes).fillna(0.0).to_numpy(dtype=float))

    def level(indicator: str, values: np.ndarray) -> np.ndarray:
        return np.log1p(np.clip(values, 0, None)) if indicator in (LEVEL_LOG | S_LOG) else values

    xnow: dict[tuple[str, int], np.ndarray] = {}
    history: dict[tuple[str, int], np.ndarray] = {}
    future: dict[tuple[str, int], np.ndarray] = {}
    for indicator in TVAR:
        for anchor in ANCHORS:
            xnow[(indicator, anchor)] = level(indicator, column(indicator, anchor))
            h = np.zeros(len(nodes))
            for year in range(2019, anchor):
                h += np.exp(-GAMMA * (anchor - 1 - year)) * level(
                    indicator, column(indicator, year))
            history[(indicator, anchor)] = h
            if indicator in F_RATIO:
                current = column(indicator, anchor)
                ahead = 0.667 * column(indicator, anchor + 1) + 0.333 * column(indicator, anchor + 2)
                ok = (ahead > 0) & (current > 0)
                ratio = np.ones(len(nodes))
                ratio[ok] = (ahead[ok] + EPS) / (current[ok] + EPS)
                future[(indicator, anchor)] = np.log(ratio)
            else:
                future[(indicator, anchor)] = np.zeros(len(nodes))
    sensitivity = {(indicator, anchor): level(indicator, column(indicator, anchor))
                   for indicator in S_IND for anchor in ANCHORS}
    return xnow, history, future, sensitivity


def _city_emission_inventory(results_root: str, year: int,
                             expected_ship_types: int) -> pd.DataFrame:
    pattern = os.path.join(results_root, f"{year}??", "*", f"city_emissions_*_{year}??.csv")
    files = sorted(glob.glob(pattern))
    by_month: dict[str, list[str]] = {f"{year}{month:02d}": [] for month in range(1, 13)}
    for path in files:
        match = re.search(rf"{year}(0[1-9]|1[0-2])", path)
        if match:
            by_month[f"{year}{match.group(1)}"].append(path)
    missing = [month for month, paths in by_month.items() if len(paths) < expected_ship_types]
    if missing:
        detail = ", ".join(f"{month}:{len(by_month[month])}/{expected_ship_types}" for month in missing)
        raise SystemExit(f"2025 baseline is incomplete; city-emission files by month: {detail}")

    parts = []
    for path in files:
        data = pd.read_csv(path)
        required = {"node_id"} | {
            f"{pollutant}_{suffix}"
            for pollutant in POLLUTANT_WEIGHTS
            for suffix in ("Total", "Berth", "Anchored")
        }
        absent = required - set(data.columns)
        if absent:
            raise SystemExit(f"Missing city-emission columns {sorted(absent)}: {path}")
        total = sum(weight * pd.to_numeric(data[f"{pollutant}_Total"], errors="coerce").fillna(0)
                    for pollutant, weight in POLLUTANT_WEIGHTS.items())
        stationary = sum(
            weight * (pd.to_numeric(data[f"{pollutant}_Berth"], errors="coerce").fillna(0)
                      + pd.to_numeric(data[f"{pollutant}_Anchored"], errors="coerce").fillna(0))
            for pollutant, weight in POLLUTANT_WEIGHTS.items())
        parts.append(pd.DataFrame({"node_id": data["node_id"], "E1": total,
                                   "stationary_weighted": stationary}))
    annual = pd.concat(parts, ignore_index=True).groupby("node_id", as_index=False).sum()
    annual["E3"] = np.where(annual["E1"] > 0,
                              annual["stationary_weighted"] / annual["E1"], 0.0)
    return annual.drop(columns="stationary_weighted")


def _replace_actual_2025(panel: pd.DataFrame, inventory: pd.DataFrame,
                         covariates: pd.DataFrame, year: int) -> pd.DataFrame:
    updated = panel.copy()
    density = (covariates.loc[covariates["year"].eq(year), ["node_id", "pop_density_km2"]]
               .drop_duplicates("node_id").set_index("node_id")["pop_density_km2"])
    actual = inventory.set_index("node_id")
    mask = updated["year"].eq(year)
    node_index = updated.loc[mask, "node_id"]
    e1 = node_index.map(actual["E1"]).fillna(0.0).to_numpy()
    e3 = node_index.map(actual["E3"]).fillna(0.0).to_numpy()
    pop_density = pd.to_numeric(node_index.map(density), errors="coerce").fillna(0.0).to_numpy()
    updated.loc[mask, "E1"] = e1
    updated.loc[mask, "E2"] = e1 * pop_density
    updated.loc[mask, "E3"] = e3
    return updated


def _scenario_reduction_fractions(city: pd.DataFrame, baseline_e1: pd.Series,
                                  scenarios: list[str]) -> dict[str, pd.Series]:
    required = {"node_id", "Pollutant", "Delta_t"}
    absent = required - set(city.columns)
    if absent:
        raise SystemExit(f"Policy city table is missing columns: {sorted(absent)}")
    data = city.copy()
    data["Delta_t"] = pd.to_numeric(data["Delta_t"], errors="coerce").fillna(0.0).clip(lower=0)
    data["weighted_delta_g"] = data["Delta_t"] * data["Pollutant"].map(POLLUTANT_WEIGHTS).fillna(0) * G_PER_T
    fractions: dict[str, pd.Series] = {}
    for scenario in scenarios:
        pollutants = SCENARIO_POLLUTANTS[scenario]
        reduction = (data.loc[data["Pollutant"].isin(pollutants)]
                     .groupby("node_id")["weighted_delta_g"].sum())
        aligned = reduction.reindex(baseline_e1.index, fill_value=0.0)
        ratio = pd.Series(np.where(baseline_e1 > 0, aligned / baseline_e1, 0.0),
                          index=baseline_e1.index)
        fractions[scenario] = ratio.clip(0.0, 1.0)
    return fractions


def _scenario_panel(baseline: pd.DataFrame, fraction: pd.Series,
                    policy_year: int) -> pd.DataFrame:
    scenario = baseline.copy()
    factors = (1.0 - fraction).clip(0.0, 1.0)
    mask = scenario["year"].between(policy_year, policy_year + 2)
    scale = scenario.loc[mask, "node_id"].map(factors).fillna(1.0).to_numpy()
    scenario.loc[mask, "E1"] = pd.to_numeric(scenario.loc[mask, "E1"], errors="coerce").fillna(0).to_numpy() * scale
    scenario.loc[mask, "E2"] = pd.to_numeric(scenario.loc[mask, "E2"], errors="coerce").fillna(0).to_numpy() * scale
    return scenario


def _spri_pair(baseline: pd.DataFrame, scenario: pd.DataFrame,
               nodes: list[str], year: int) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    bx, bh, bf, bs = _terms(baseline, nodes)
    sx, sh, sf, _ = _terms(scenario, nodes)
    bxn, x_bounds = _fit_pooled(bx, TVAR)
    bhn, h_bounds = _fit_pooled(bh, TVAR)
    bfn, f_bounds = _fit_pooled(bf, TVAR)
    sxn = _apply_pooled(sx, TVAR, x_bounds)
    shn = _apply_pooled(sh, TVAR, h_bounds)
    sfn = _apply_pooled(sf, TVAR, f_bounds)
    beval = {(indicator, anchor): bxn[(indicator, anchor)]
             + MU * bhn[(indicator, anchor)] + LAM * bfn[(indicator, anchor)]
             for indicator in TVAR for anchor in ANCHORS}
    seval = {(indicator, anchor): sxn[(indicator, anchor)]
             + MU * shn[(indicator, anchor)] + LAM * sfn[(indicator, anchor)]
             for indicator in TVAR for anchor in ANCHORS}
    bevaln, eval_bounds = _fit_pooled(beval, TVAR)
    sevaln = _apply_pooled(seval, TVAR, eval_bounds)
    bsn, _ = _fit_pooled(bs, S_IND)

    w_e = _critic([bevaln[(indicator, year)] for indicator in E_IND])
    w_ac = _critic([bevaln[(indicator, year)] for indicator in AC_IND])
    w_s = _critic([bsn[(indicator, year)] for indicator in S_IND])

    def dimension(source: dict[tuple[str, int], np.ndarray],
                  indicators: list[str], weights: np.ndarray) -> np.ndarray:
        return np.column_stack([source[(indicator, year)] for indicator in indicators]) @ weights

    e_base = dimension(bevaln, E_IND, w_e)
    e_scenario = dimension(sevaln, E_IND, w_e)
    ac = dimension(bevaln, AC_IND, w_ac)
    sensitivity = dimension(bsn, S_IND, w_s)
    lac = np.clip(1.0 - ac, 0, None)
    spri_base = np.cbrt(np.clip(e_base, 0, None) * np.clip(sensitivity, 0, None) * lac)
    spri_scenario = np.cbrt(np.clip(e_scenario, 0, None) * np.clip(sensitivity, 0, None) * lac)
    weights = {"wE": w_e, "wAC": w_ac, "wS": w_s}
    return spri_base, spri_scenario, weights


def main() -> None:
    cfg = _load_cfg()
    paths = cfg["paths"]
    policy = cfg.get("policy") or {}
    emission = cfg["emission_network"]
    default_study = os.path.join(ROOT, "study_run_2019_2023")
    default_eval = str(policy.get("eval_output_dir", os.path.join(HERE, "outputs", "evaluation")))

    parser = argparse.ArgumentParser(description="Build fixed-baseline policy SPRI scenarios")
    parser.add_argument("--year", type=int, default=int(policy.get("eval_year", 2025)))
    parser.add_argument("--results-root", default=str(paths["results_root"]))
    parser.add_argument("--spri-panel", default=os.path.join(default_study, "spri_panel.csv"))
    parser.add_argument("--policy-city-csv", default=os.path.join(
        default_eval, f"policy_eval_city_{int(policy.get('eval_year', 2025))}.csv"))
    parser.add_argument("--covariates-csv", default=os.path.join(
        paths["covariates_dir"], "city_covariates_2019_2028.csv"))
    parser.add_argument("--out-csv", default=os.path.join(
        default_eval, f"policy_spri_scenarios_{int(policy.get('eval_year', 2025))}.csv"))
    parser.add_argument("--ship-types", default=",".join(emission["ship_types"]))
    parser.add_argument("--scenarios", default="SECA,NECA,BOTH")
    args = parser.parse_args()
    if args.year != 2025:
        raise SystemExit("The rolling policy-SPRI implementation currently requires policy year 2025")

    for path in (args.spri_panel, args.policy_city_csv, args.covariates_csv):
        if not os.path.isfile(path):
            raise SystemExit(f"Required input does not exist: {path}")
    scenarios = [value.strip().upper() for value in args.scenarios.split(",") if value.strip()]
    unknown = sorted(set(scenarios) - set(SCENARIO_POLLUTANTS))
    if unknown:
        raise SystemExit(f"Unknown scenarios: {unknown}")
    ship_types = [value.strip() for value in args.ship_types.split(",") if value.strip()]

    panel = pd.read_csv(args.spri_panel)
    required_panel = {"node_id", "year", *TVAR, *S_IND}
    absent = required_panel - set(panel.columns)
    if absent:
        raise SystemExit(f"SPRI panel is missing columns: {sorted(absent)}")
    if not set(range(2019, 2028)) <= set(pd.to_numeric(panel["year"], errors="coerce").dropna().astype(int)):
        raise SystemExit("SPRI panel must contain 2019-2027 for the 2025 rolling evaluation")
    panel["year"] = pd.to_numeric(panel["year"], errors="raise").astype(int)
    nodes = sorted(panel["node_id"].dropna().astype(str).unique())

    print("[1/4] Auditing and aggregating the completed 2025 AIS city inventory")
    inventory = _city_emission_inventory(args.results_root, args.year, len(ship_types))
    covariates = pd.read_csv(args.covariates_csv,
                             usecols=["node_id", "year", "pop_density_km2"])
    baseline = _replace_actual_2025(panel, inventory, covariates, args.year)
    baseline_e1 = (baseline.loc[baseline["year"].eq(args.year), ["node_id", "E1"]]
                   .drop_duplicates("node_id").set_index("node_id")["E1"].reindex(nodes).fillna(0.0))

    print("[2/4] Deriving city-specific health-weighted E1 reductions")
    city = pd.read_csv(args.policy_city_csv)
    fractions = _scenario_reduction_fractions(city, baseline_e1, scenarios)

    print("[3/4] Recomputing SPRI with fixed baseline normalisation and weights")
    outputs = []
    for scenario_name in scenarios:
        scenario_panel = _scenario_panel(baseline, fractions[scenario_name], args.year)
        spri_base, spri_scenario, weights = _spri_pair(
            baseline, scenario_panel, nodes, args.year)
        fraction = fractions[scenario_name].reindex(nodes).fillna(0.0).to_numpy()
        outputs.append(pd.DataFrame({
            "node_id": nodes,
            "scenario": scenario_name,
            "SPRI_baseline": spri_base,
            "SPRI_scenario": spri_scenario,
            "delta_spri": spri_scenario - spri_base,
            "E1_reduction_fraction": fraction,
            "affected": fraction > 0,
        }))
        print(f"  {scenario_name}: affected={int((fraction > 0).sum())}, "
              f"median ΔSPRI among affected="
              f"{np.median((spri_scenario-spri_base)[fraction > 0]) if (fraction > 0).any() else np.nan:.5f}; "
              f"wE={np.round(weights['wE'], 3)}")

    output = pd.concat(outputs, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    output.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"[4/4] Wrote {args.out_csv} ({len(output):,} rows)")


if __name__ == "__main__":
    main()
