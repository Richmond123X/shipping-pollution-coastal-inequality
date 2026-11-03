# -*- coding: utf-8 -*-
"""周度港口输出(绿色韧性研究)冒烟测试。

场景：沪甬往返 + 在上海港界外 ~17km 处锚泊 6h(Speed 2.0 → Anchored 工况,
不落入 10km 港界 visit 区间,应由锚地扩展(≤25km)归属到 Shanghai)。

断言：
  1. port_visits_* 含 Arrive_UTC/Depart_UTC 且到港早于离港;
  2. port_emissions_weekly_* 产出; Attr 同时含 visit 与 anchor;
     anchor 行的 Port_Name 为 Shanghai 且工况为 Anchored_Class;
  3. ISO 周正确: 2024-01-31 与 2024-02-01 同属 2024-W05,
     该周同时出现在 202401 与 202402 两个月文件中(跨月周可下游合并);
  4. 周度(归属口径)排放 ≤ 真实总量且 > 0。

用法： cd emission_network && python tests/test_weekly_smoke.py
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)

import emission_network_main as M  # noqa: E402
from city_aggregation import load_city_data  # noqa: E402
from config_loader import load_config  # noqa: E402
from make_synthetic_ais import SHIP, SH, NB  # noqa: E402

MMSI = SHIP["MMSI"]
ANCH = (121.85, 30.90)  # 上海港南 ~17km: 港界(10km)外, 锚地扩展(25km)内


def build_tracks() -> pd.DataFrame:
    t0 = pd.Timestamp("2024-01-31 06:00:00", tz="UTC").timestamp()
    rows = []

    def add_leg(p_from, p_to, t_start, hours, speed_kn):
        n = max(int(hours * 6), 2)
        for k in range(n):
            f = k / (n - 1)
            rows.append({
                "MMSI": MMSI,
                "UTCtime": t_start + f * hours * 3600,
                "Longitude": p_from[0] + f * (p_to[0] - p_from[0]),
                "Latitude": p_from[1] + f * (p_to[1] - p_from[1]),
                "Speed": speed_kn,
            })
        return t_start + hours * 3600

    t = add_leg(SH, SH, t0, 10, 0.3)       # 上海靠泊(1月, W05)
    t = add_leg(ANCH, ANCH, t, 6, 2.0)     # 港外锚泊 6h(Anchored, 锚地扩展目标)
    t = add_leg(SH, NB, t, 10, 9.0)        # 沪→甬, 跨 1/31→2/1 月界(同为 W05)
    t = add_leg(NB, NB, t, 12, 0.3)        # 宁波靠泊(2月, W05)
    t = add_leg(NB, SH, t, 10, 9.0)
    t = add_leg(SH, SH, t, 8, 0.3)
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    tmp = tempfile.mkdtemp(prefix="emis_weekly_smoke_")
    ais = build_tracks()
    ais["ym"] = pd.to_datetime(ais["UTCtime"], unit="s", utc=True).dt.strftime("%Y%m")
    db_paths = []
    for ym in ["202401", "202402"]:
        sub = ais[ais["ym"] == ym].drop(columns="ym")
        p = os.path.join(tmp, f"ais_{ym}.db")
        conn = sqlite3.connect(p)
        pd.DataFrame([SHIP]).to_sql("ship_info", conn, index=False)
        sub.to_sql("ais_tracks", conn, index=False)
        conn.close()
        db_paths.append(p)

    port_rows = []
    for name, (lon, lat) in [("Shanghai", SH), ("Ningbo-Zhoushan", NB)]:
        for dx in (0.0, 0.01):
            port_rows.append({"Port.Name": name, "Lon_port": lon, "Lat_port": lat,
                              "Lon_berth": lon + dx, "Lat_berth": lat + dx})
    port_coor = pd.DataFrame(port_rows)
    city_catch, mapping = load_city_data(cfg["paths"])
    keep = {"China__Shanghai", "China__Ningbo-Zhoushan"}
    catch_small = city_catch[city_catch.node_id.isin(keep)].reset_index(drop=True)

    assert M.WEEKLY_CFG["enabled"], "config emission_network.weekly_port_outputs 应为 true"
    M.run_emission_for_year_ship_type(
        "Container", pd.DataFrame([SHIP]), ["202401", "202402"], db_paths, "2024",
        port_coor, pd.DataFrame(), pd.DataFrame(), {}, tmp,
        M.POLLUTANTS, n_cores=1,
        city_catchments=catch_small, port_city_mapping=mapping,
    )

    wk_all, true_so2 = [], 0.0
    for ym in ["202401", "202402"]:
        d = os.path.join(tmp, ym, "Container")
        pv = pd.read_csv(os.path.join(d, f"port_visits_Container_{ym}.csv"))
        assert {"Arrive_UTC", "Depart_UTC", "Waiting_Hours", "DWT"} <= set(pv.columns)
        if len(pv):
            assert (pv["Arrive_UTC"] < pv["Depart_UTC"]).all(), "到港时刻应早于离港"
        wk = pd.read_csv(os.path.join(d, f"port_emissions_weekly_Container_{ym}.csv"))
        assert len(wk) > 0, f"{ym} 周度排放为空"
        assert (wk["YearMonth"].astype(str) == ym).all()
        wk_all.append(wk)
        tt = pd.read_csv(os.path.join(d, f"true_totals_Container_{ym}.csv"))
        true_so2 += tt["SO2"].iloc[0]
    wk = pd.concat(wk_all, ignore_index=True)

    # 锚地扩展: anchor 行存在、归 Shanghai、工况 Anchored
    anc = wk[wk["Attr"] == "anchor"]
    assert len(anc) > 0, "锚地扩展未产出 anchor 行"
    assert set(anc["Port_Name"]) == {"Shanghai"}, f"锚泊应归上海: {set(anc['Port_Name'])}"
    assert set(anc["Mode_Class"]) == {"Anchored_Class"}
    assert (wk["Attr"] == "visit").any(), "缺 visit 归属行"

    # 跨月同周: 2024-01-31 与 2024-02-01 同属 2024-W05, 两个月文件都应有 W05
    w05 = wk[wk["ISO_Week"] == "2024-W05"]
    assert set(w05["YearMonth"].astype(str)) == {"202401", "202402"}, \
        f"W05 应跨两个月文件: {set(w05['YearMonth'])}"

    # 守恒: 归属口径(visit+anchor) ≤ 真实总量
    assert 0 < wk["SO2"].sum() <= true_so2 + 1e-6, \
        f"周度归属口径 SO2 {wk['SO2'].sum():,.0f} 应 ≤ 真实总量 {true_so2:,.0f}"

    sh_anch_so2 = anc["SO2"].sum()
    print(f"锚地扩展归属: Shanghai Anchored SO2 = {sh_anch_so2:,.0f} g "
          f"(共 {len(anc)} 行)")
    print(f"周度/真实总量 SO2: {wk['SO2'].sum():,.0f} / {true_so2:,.0f} "
          f"({wk['SO2'].sum()/true_so2:.0%} 归属到港口)")
    print("\n✓ 周度港口输出冒烟测试通过 "
          "(visit明细含时刻 / 锚地扩展归属 / ISO周跨月一致 / 守恒上界)")
    print(f"  输出目录: {tmp}")


if __name__ == "__main__":
    main()
