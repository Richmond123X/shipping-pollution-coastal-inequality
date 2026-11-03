# -*- coding: utf-8 -*-
"""AC4 治理质量：WGI 五维（VA/GE/RQ/RL/CC，剔除 PV）等权平均（方案 3.9.3）。

数据：data/治理质量/wgidataset.dta —— WGI 2024 Update 官方完整长表
（datacatalog DR0046190，1996-2023，含 Taiwan TWN，estimate 标准刻度约 -2.5~+2.5）。
五维同量纲直接等权平均；2024+ 末值延展（有界慢变量，不做 ARIMA）。

输出 outputs/ac4_wgi.csv: iso3, year, AC4, va, ge, rq, rl, cc, last_obs_year
"""
from __future__ import annotations

import os

import pandas as pd

from common import OUT, YEARS, load_config, city_iso3_set, interp_and_extend

DIMS = ["va", "ge", "rq", "rl", "cc"]  # 剔除 pv（已单列为 AC5）


# 世行旧国家码 → ISO3（GADM GID_0 口径）
LEGACY_CODES = {"ZAR": "COD", "ROM": "ROU", "TMP": "TLS", "ADO": "AND", "KSV": "XKO"}


def load_wgi_long(cfg: dict) -> pd.DataFrame:
    path = os.path.join(os.path.dirname(cfg["paths"]["wgi_file"]), "wgidataset.dta")
    df = pd.read_stata(path)
    df = df[["code", "year", "indicator", "estimate"]].dropna(subset=["estimate"])
    df["year"] = df["year"].astype(int)
    df["code"] = df["code"].replace(LEGACY_CODES)
    return df.rename(columns={"code": "iso3"})


def main() -> None:
    cfg = load_config()
    wgi = load_wgi_long(cfg)
    iso_keep = city_iso3_set(cfg)
    wgi = wgi[wgi.iso3.isin(iso_keep)]
    print(f"WGI dta: {wgi.iso3.nunique()} 经济体, {wgi.year.min()}-{wgi.year.max()}, "
          f"TWN={'TWN' in set(wgi.iso3)}")

    rows = []
    for iso, g in wgi.groupby("iso3"):
        per_dim, last_obs = {}, 0
        for dim in DIMS:
            s = g[g.indicator == dim].set_index("year")["estimate"]
            if s.empty:
                continue
            last_obs = max(last_obs, int(s.index.max()))
            per_dim[dim], _ = interp_and_extend(s, YEARS, use_arima=False)
        if not per_dim:
            continue
        df = pd.DataFrame(per_dim)
        df["AC4"] = df.mean(axis=1)
        for y in YEARS:
            rows.append({"iso3": iso, "year": y, "AC4": df.loc[y, "AC4"],
                         **{d: (df.loc[y, d] if d in df else None) for d in DIMS},
                         "last_obs_year": last_obs})

    out = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(os.path.join(OUT, "ac4_wgi.csv"), index=False, encoding="utf-8-sig")
    missing = iso_keep - set(out["iso3"])
    print(f"AC4 完成: {out.iso3.nunique()} 国; 值域 {out.AC4.min():.2f}~{out.AC4.max():.2f}; "
          f"缺失: {sorted(missing)}")


if __name__ == "__main__":
    main()
