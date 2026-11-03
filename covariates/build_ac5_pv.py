# -*- coding: utf-8 -*-
"""AC5 政治稳定性：WGI PV.EST（方案 3.9.2）。

数据：data/治理质量/wgidataset.dta（WGI 2024 Update 官方长表，1996-2023，含 Taiwan）。
与原方案差异说明：方案原计划用 WDI 宽表 CSV + govindicators 台湾补录；
官方 .dta 长表与 WDI 数值一致（已抽查 NOR/SYR/CHN/USA/SGP）且自带台湾行，
故统一用 .dta；用户原下载的 API_PV.EST_DS2_*.csv 保留作交叉校验。

规则：仅保留城市 iso3 全集；2019-2023 直取；2024+ 末值延展；正向（越高越稳定）。
输出 outputs/ac5_pv.csv: iso3, year, AC5, last_obs_year
"""
from __future__ import annotations

import os

import pandas as pd

from common import OUT, YEARS, load_config, city_iso3_set, interp_and_extend
from build_ac4_wgi import load_wgi_long


def main() -> None:
    cfg = load_config()
    wgi = load_wgi_long(cfg)
    pv = wgi[wgi.indicator == "pv"]
    iso_keep = city_iso3_set(cfg)
    pv = pv[pv.iso3.isin(iso_keep)]
    print(f"PV: {pv.iso3.nunique()} 经济体（含 TWN={'TWN' in set(pv.iso3)}）")

    rows = []
    for iso, g in pv.groupby("iso3"):
        s = g.set_index("year")["estimate"]
        ext, _ = interp_and_extend(s, YEARS, use_arima=False)
        for y in YEARS:
            rows.append({"iso3": iso, "year": y, "AC5": ext[y],
                         "last_obs_year": int(s.index.max())})

    out = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(os.path.join(OUT, "ac5_pv.csv"), index=False, encoding="utf-8-sig")
    missing = iso_keep - set(out["iso3"])
    print(f"AC5 完成: {out.iso3.nunique()} 国; 值域 {out.AC5.min():.2f}~{out.AC5.max():.2f}; "
          f"缺失: {sorted(missing)}")


if __name__ == "__main__":
    main()
