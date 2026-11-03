# -*- coding: utf-8 -*-
"""由官方 EDGAR v8.1 与 Ember 数据构造 2022 年分国电网污染强度。

分子：EDGAR v8.1，IPCC 2006 代码 1.A.1.a（Main Activity Electricity and
Heat Production），单位 Gg/年；分母：Ember Yearly Electricity Data 的 Total
Generation，单位 TWh/年。因为 1 Gg / 1 TWh = 1 g/kWh，二者可直接相除。

船侧模型使用 Fourth IMO GHG Study 的 PM10 排放因子公式，因此正式 PM 净效应
使用 EDGAR PM10，而不是 PM2.5；这使船侧与电网侧粒径口径一致。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\transfer justice\policy_simulation_v12_formal")
SOURCE = ROOT / "inputs" / "official_grid_sources"
SELECTED = ROOT / "tables" / "selected_policy_states_50pct.csv"
OUT = ROOT / "inputs" / "grid_pollution_intensity_2022.csv"
MANIFEST = ROOT / "inputs" / "grid_pollution_intensity_2022_manifest.json"

EMBER = SOURCE / "ember_yearly_full_release_long_format.csv"
EDGAR = {
    "SO2": SOURCE / "SO2_country" / "EDGAR_SO2_1970_2022.xlsx",
    "NOx": SOURCE / "NOx_country" / "EDGAR_NOx_1970_2022.xlsx",
    "PM": SOURCE / "PM10_country" / "EDGAR_PM10_1970_2022.xlsx",
}

EDGAR_URL = "https://edgar.jrc.ec.europa.eu/index.php/dataset_ap81"
EMBER_URL = "https://ember-energy.org/data/yearly-electricity-data/"
YEAR = 2022
POWER_CODE = "1.A.1.a"
DOWNLOAD_DATE = "2026-08-23"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_edgar(path: Path, pollutant: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_excel(path, sheet_name="IPCC 2006", header=9)
    required = {
        "Country_code_A3", "Name", "ipcc_code_2006_for_standard_report",
        "ipcc_code_2006_for_standard_report_name", "fossil_bio", f"Y_{YEAR}",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} 缺少列: {missing}")
    x = df[
        df["ipcc_code_2006_for_standard_report"].astype(str).eq(POWER_CODE)
    ].copy()
    x["iso3"] = x["Country_code_A3"].astype(str).str.strip().str.upper()
    x[f"{pollutant}_Gg"] = pd.to_numeric(x[f"Y_{YEAR}"], errors="coerce")
    # fossil 与 bio 分项相加；若某国所有分项均缺失，保留为 NaN 而非误记为 0。
    out = (x.groupby("iso3", as_index=False)[f"{pollutant}_Gg"]
             .sum(min_count=1))
    if out["iso3"].duplicated().any():
        raise AssertionError(f"{pollutant}: EDGAR 聚合后 iso3 不唯一")
    if (out[f"{pollutant}_Gg"].dropna() < 0).any():
        raise ValueError(f"{pollutant}: EDGAR 出现负排放")
    return out


def load_ember() -> pd.DataFrame:
    if not EMBER.is_file():
        raise FileNotFoundError(EMBER)
    use = ["Area", "ISO 3 code", "Year", "Area type", "Category",
           "Subcategory", "Variable", "Unit", "Value"]
    df = pd.read_csv(EMBER, usecols=use)
    x = df[
        (df["Year"] == YEAR)
        & df["Area type"].eq("Country or economy")
        & df["Category"].eq("Electricity generation")
        & df["Subcategory"].eq("Total")
        & df["Variable"].eq("Total Generation")
        & df["Unit"].eq("TWh")
    ].copy()
    x["iso3"] = x["ISO 3 code"].astype(str).str.strip().str.upper()
    x["electricity_generation_TWh"] = pd.to_numeric(x["Value"], errors="coerce")
    x = x[["iso3", "Area", "electricity_generation_TWh"]]
    if x["iso3"].duplicated().any():
        dup = x.loc[x["iso3"].duplicated(False), "iso3"].unique().tolist()
        raise ValueError(f"Ember 2022 Total Generation iso3 重复: {dup}")
    if (x["electricity_generation_TWh"].dropna() < 0).any():
        bad = x.loc[x["electricity_generation_TWh"].lt(0), "iso3"].tolist()
        raise ValueError(f"Ember 2022 总发电量为负: {bad}")
    return x


def main() -> None:
    selected = pd.read_csv(SELECTED)
    selected["iso3"] = selected["policy_iso3"].astype(str).str.upper()
    if selected["iso3"].duplicated().any() or len(selected) != 75:
        raise ValueError("正式 X=50% 政策责任国清单应为 75 个互异 iso3")

    out = selected[["iso3", "policy_country"]].copy()
    out = out.merge(load_ember(), on="iso3", how="left", validate="1:1")
    for pollutant, path in EDGAR.items():
        out = out.merge(load_edgar(path, pollutant), on="iso3", how="left",
                        validate="1:1")

    for pollutant in ("SO2", "NOx", "PM"):
        out[f"{pollutant}_g_per_kWh"] = (
            out[f"{pollutant}_Gg"] / out["electricity_generation_TWh"]
        ).replace([np.inf, -np.inf], np.nan)

    required_numeric = [
        "electricity_generation_TWh", "SO2_Gg", "NOx_Gg", "PM_Gg",
        "SO2_g_per_kWh", "NOx_g_per_kWh", "PM_g_per_kWh",
    ]
    complete = (
        np.isfinite(out[required_numeric]).all(axis=1)
        & out["electricity_generation_TWh"].gt(0)
    )
    out["coverage_status"] = np.select(
        [out["electricity_generation_TWh"].isna(),
         out["electricity_generation_TWh"].le(0),
         out[["SO2_Gg", "NOx_Gg", "PM_Gg"]].isna().any(axis=1)],
        ["missing_ember_generation", "nonpositive_ember_generation",
         "missing_edgar_power_emission"],
        default="complete",
    )
    if not complete.equals(out["coverage_status"].eq("complete")):
        raise AssertionError("电网强度覆盖状态与数值完整性不一致")

    out["PM_comparable"] = complete
    out["PM_basis"] = "EDGAR PM10; matches ship-side Fourth IMO GHG Study PM10 EF"
    out["year"] = YEAR
    out["edgar_sector"] = POWER_CODE
    out["edgar_source_url"] = EDGAR_URL
    out["ember_source_url"] = EMBER_URL

    cols = [
        "iso3", "policy_country", "SO2_g_per_kWh", "NOx_g_per_kWh",
        "PM_g_per_kWh", "PM_comparable", "coverage_status",
        "electricity_generation_TWh", "SO2_Gg", "NOx_Gg", "PM_Gg",
        "PM_basis", "year", "edgar_sector", "edgar_source_url",
        "ember_source_url",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out[cols].to_csv(OUT, index=False, encoding="utf-8-sig")

    missing = out.loc[~complete, ["iso3", "policy_country", "coverage_status"]]
    manifest = {
        "status": "complete_with_documented_missing" if len(missing) else "complete",
        "year": YEAR,
        "download_date": DOWNLOAD_DATE,
        "edgar_release": "EDGAR v8.1 (1970-2022), August 2024 release",
        "ember_release_note": "Official Yearly Electricity Data file downloaded on 2026-08-23",
        "n_selected_policy_states": int(len(out)),
        "n_complete": int(complete.sum()),
        "n_missing": int((~complete).sum()),
        "missing": missing.to_dict("records"),
        "formula": "EDGAR power-sector Gg / Ember total generation TWh = g/kWh",
        "pm_comparability": (
            "Ship-side PM is PM10 because the implemented 0.23 + SFC*7*0.02247*"
            "(S-0.0024) factor is the Fourth IMO GHG Study PM10 equation; EDGAR PM10 "
            "is therefore used for like-for-like net effects."
        ),
        "source_files": {
            str(EMBER.relative_to(ROOT)): sha256(EMBER),
            **{str(p.relative_to(ROOT)): sha256(p) for p in EDGAR.values()},
        },
        "sources": {"EDGAR_v8.1": EDGAR_URL, "Ember": EMBER_URL},
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(
        f"[complete] rows={len(out)} complete={int(complete.sum())} "
        f"missing={len(missing)} output={OUT}", flush=True
    )
    if len(missing):
        print(missing.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
