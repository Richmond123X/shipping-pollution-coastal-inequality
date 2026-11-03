# -*- coding: utf-8 -*-
"""AC2 医疗资源：每万人医生数 + 每万人床位数（方案 3.9.1）。

数据：data/医生数/doctor.xlsx (HWF_0001, 1990-2024)、data/床位数/bed.xlsx (WHS6_102, 2000-2023)。
表头第 3 行（skiprows=2），Location type=="Country"，数值列 FactValueNumeric，iso3 列 SpatialDimValueCode。

处理：逐国内部线性插值 → 尾部门槛式 ARIMA（观测≥10 且最新年≥2015，预测非负截断）/
否则末值延展 / 单观测取常数；台湾从 taiwan_medical.json 补录；
医生、床位各自 min-max 标准化后等权平均 → AC2。

输出:
  outputs/ac2_medical.csv: iso3, year, doctor_per_10k, bed_per_10k,
                           doctor_method, bed_method, doctor_last_obs, bed_last_obs, AC2
  outputs/ac2_stale_countries.csv: 床位最新观测早于 2019 的国家清单（SI 敏感性用）
"""
from __future__ import annotations

import json
import os

import pandas as pd

from common import HERE, OUT, YEARS, load_config, city_iso3_set, interp_and_extend


def read_gho(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, skiprows=2)
    df = df[df["Location type"] == "Country"]
    out = df[["SpatialDimValueCode", "Period", "FactValueNumeric"]].rename(
        columns={"SpatialDimValueCode": "iso3", "Period": "year", "FactValueNumeric": "value"})
    out["year"] = out["year"].astype(int)
    out = out.dropna(subset=["value"])
    # 同国同年多条（不同 Dim 维度）取均值
    return out.groupby(["iso3", "year"], as_index=False)["value"].mean()


def build_indicator(raw: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for iso, g in raw.groupby("iso3"):
        s = g.set_index("year")["value"]
        ext, method = interp_and_extend(s, YEARS, use_arima=True)
        for y in YEARS:
            rows.append({"iso3": iso, "year": y, f"{label}_per_10k": ext[y],
                         f"{label}_method": method, f"{label}_last_obs": int(s.index.max())})
    return pd.DataFrame(rows)


def add_taiwan(doctor: pd.DataFrame, bed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tw_path = os.path.join(HERE, "taiwan_medical.json")
    if not os.path.exists(tw_path):
        print("警告: taiwan_medical.json 不存在，台湾 AC2 缺失")
        return doctor, bed
    with open(tw_path, encoding="utf-8") as f:
        tw = json.load(f)

    def tw_rows(series: dict, label: str) -> pd.DataFrame:
        s = pd.Series({int(k): float(v) for k, v in series.items()}).sort_index()
        ext, method = interp_and_extend(s, YEARS, use_arima=True)
        return pd.DataFrame([{"iso3": "TWN", "year": y, f"{label}_per_10k": ext[y],
                              f"{label}_method": f"taiwan_mohw_{method}",
                              f"{label}_last_obs": int(s.index.max())} for y in YEARS])

    doctor = pd.concat([doctor[doctor.iso3 != "TWN"], tw_rows(tw["doctor_per_10k"], "doctor")])
    bed = pd.concat([bed[bed.iso3 != "TWN"], tw_rows(tw["bed_per_10k"], "bed")])
    print("台湾 AC2 已补录 (来源: taiwan_medical.json)")
    return doctor, bed


def main() -> None:
    cfg = load_config()
    doctor_raw = read_gho(cfg["paths"]["doctor_xlsx"])
    bed_raw = read_gho(cfg["paths"]["bed_xlsx"])
    print(f"doctor: {doctor_raw.iso3.nunique()} 国; bed: {bed_raw.iso3.nunique()} 国")

    doctor = build_indicator(doctor_raw, "doctor")
    bed = build_indicator(bed_raw, "bed")
    doctor, bed = add_taiwan(doctor, bed)

    panel = doctor.merge(bed, on=["iso3", "year"], how="outer")

    # min-max 标准化（全样本逐列）后等权平均 → AC2；一列缺失时退化为另一列
    for col in ["doctor_per_10k", "bed_per_10k"]:
        lo, hi = panel[col].min(), panel[col].max()
        panel[col + "_norm"] = (panel[col] - lo) / (hi - lo)
    panel["AC2"] = panel[["doctor_per_10k_norm", "bed_per_10k_norm"]].mean(axis=1)

    os.makedirs(OUT, exist_ok=True)
    panel.drop(columns=["doctor_per_10k_norm", "bed_per_10k_norm"]).to_csv(
        os.path.join(OUT, "ac2_medical.csv"), index=False, encoding="utf-8-sig")

    # 陈旧度清单（床位最新观测早于 2019）
    stale = (panel[panel["bed_last_obs"] < 2019][["iso3", "bed_last_obs"]]
             .drop_duplicates().sort_values("bed_last_obs"))
    stale.to_csv(os.path.join(OUT, "ac2_stale_countries.csv"), index=False)

    iso_keep = city_iso3_set(cfg)
    missing = iso_keep - set(panel["iso3"])
    print(f"AC2 完成: {panel.iso3.nunique()} 国; 床位陈旧国 {len(stale)}; "
          f"城市国家缺失: {sorted(missing)}")
    print("外推方式分布(doctor):", doctor.drop_duplicates('iso3')['doctor_method'].value_counts().to_dict())


if __name__ == "__main__":
    main()
