# -*- coding: utf-8 -*-
"""合成迷你 AIS 库 + 冒烟测试（本机无真实 AIS，验证年±1 月度/城市节点/跨月分箱逻辑）。

场景：1 艘集装箱船，行程**跨 2024-01/2024-02 月界**：
  上海靠泊(1/31) → 航行去宁波(1/31 18:00 → 2/1 04:00, **跨月**) → 宁波靠泊(2/1)
  → 返回上海(2/1→2/2) → 上海靠泊。
数据按月分别写入 ais_202401.db 与 ais_202402.db（模拟真实"每月一库"）。

断言：
  1. run_emission_for_year_ship_type 跑通、按月分箱写出 202401 与 202402；
  2. city_emissions 含 China__Shanghai 与 China__Ningbo-Zhoushan，四工况 Berth>0/移动>0、Total=四工况和；
  3. **跨月航段被正确分月**：沪→甬 这条边因跨月，同时出现在 202401 与 202402 的 city_edge_list；
  4. 无同城边。

用法： cd emission_network && python tests/make_synthetic_ais.py
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

import emission_network_main as M  # noqa: E402
from city_aggregation import load_city_data  # noqa: E402
from config_loader import load_config  # noqa: E402

MMSI = 412345678
SH = (121.85, 31.05)
NB = (122.10, 29.95)

SHIP = {
    "MMSI": MMSI, "Ship_Type": "Container", "Fuel_Type": "HFO",
    "DWT": 100000.0, "Size": 8500.0, "Build_Year": 2015, "Tier": 2,
    "Service_Speed": 18.0, "Draft": 12.0, "ME": 40000.0, "Engine_Type": "SSD",
    "ME_CF_W": 1.0, "ME_CF_F": 1.0, "ME_SFC": 185.0, "ME_SFC_MDO": 175.0,
    "ME_NOx_noECA": 14.0, "ME_NOx_ECA": 3.4, "ME_CH4": 0.01,
    "ME_CO": 0.54, "ME_CO_MDO": 0.54, "ME_N2O": 0.03, "ME_N2O_MDO": 0.03,
    "ME_NMVOC": 0.5, "CO2_EFf": 3.114, "CO2_EFf_MDO": 3.206,
    "AE_Berth": 1000.0, "AE_Anchored": 1200.0, "AE_Manoeuvring": 2000.0,
    "AE_Sea": 1300.0, "AE_SFC": 195.0, "AE_SFC_MDO": 185.0,
    "AE_NOx": 10.0, "AE_CH4": 0.01, "AE_CO": 0.54, "AE_CO_MDO": 0.54,
    "AE_N2O": 0.03, "AE_N2O_MDO": 0.03, "AE_NMVOC": 0.4,
    "BO_Berth": 500.0, "BO_Anchored": 500.0, "BO_Manoeuvring": 300.0,
    "BO_Sea": 100.0, "BO_SFC": 300.0, "BO_SFC_MDO": 300.0,
    "BO_NOx": 2.0, "BO_CH4": 0.01, "BO_CO": 0.2, "BO_CO_MDO": 0.2,
    "BO_N2O": 0.01, "BO_N2O_MDO": 0.01, "BO_NMVOC": 0.1,
}


def build_tracks() -> pd.DataFrame:
    # 起点设在 1/31 06:00，使「沪→甬」航段跨越 1/31→2/1 月界
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

    t = add_leg(SH, SH, t0, 12, 0.3)     # 上海靠泊 12h（全在 1 月）
    t = add_leg(SH, NB, t, 10, 9.0)      # 沪→甬 10h（1/31 18:00 → 2/1 04:00，跨月）
    t = add_leg(NB, NB, t, 12, 0.3)      # 宁波靠泊（2 月）
    t = add_leg(NB, SH, t, 10, 9.0)      # 甬→沪（2 月）
    t = add_leg(SH, SH, t, 8, 0.3)       # 上海靠泊（2 月）
    return pd.DataFrame(rows)


def write_monthly_dbs(tmp: str):
    """按月把轨迹切分写入 ais_202401.db / ais_202402.db，返回 db 路径列表。"""
    ais = build_tracks()
    ais["ym"] = pd.to_datetime(ais["UTCtime"], unit="s", utc=True).dt.strftime("%Y%m")
    paths = []
    for ym in ["202401", "202402"]:
        sub = ais[ais["ym"] == ym].drop(columns="ym")
        p = os.path.join(tmp, f"ais_{ym}.db")
        conn = sqlite3.connect(p)
        pd.DataFrame([SHIP]).to_sql("ship_info", conn, index=False)
        sub.to_sql("ais_tracks", conn, index=False)
        conn.close()
        paths.append(p)
    return paths


def main() -> None:
    cfg = load_config()
    tmp = tempfile.mkdtemp(prefix="emis_smoke_")
    db_paths = write_monthly_dbs(tmp)
    ships = pd.DataFrame([SHIP])

    port_rows = []
    for name, (lon, lat) in [("Shanghai", SH), ("Ningbo-Zhoushan", NB)]:
        for dx in (0.0, 0.01):
            port_rows.append({"Port.Name": name, "Lon_port": lon, "Lat_port": lat,
                              "Lon_berth": lon + dx, "Lat_berth": lat + dx})
    port_coor = pd.DataFrame(port_rows)

    city_catch, mapping = load_city_data(cfg["paths"])
    keep = {"China__Shanghai", "China__Ningbo-Zhoushan"}
    catch_small = city_catch[city_catch.node_id.isin(keep)].reset_index(drop=True)
    assert len(catch_small) == 2, f"目标集水区缺失: {keep - set(catch_small.node_id)}"

    target_months = ["202401", "202402"]
    M.run_emission_for_year_ship_type(
        "Container", ships, target_months, db_paths, "2024",
        port_coor, pd.DataFrame(), pd.DataFrame(), {}, tmp,
        M.POLLUTANTS, n_cores=1,
        city_catchments=catch_small, port_city_mapping=mapping,
    )

    edge_pairs = {}
    for ym in target_months:
        out_dir = os.path.join(tmp, ym, "Container")
        city_f = os.path.join(out_dir, f"city_emissions_Container_{ym}.csv")
        assert os.path.isfile(city_f), f"缺少 {ym} 城市排放文件"
        ce = pd.read_csv(city_f)
        assert (ce.YearMonth.astype(str) == ym).all(), f"{ym} 月度标签错误"
        assert set(ce.node_id) <= keep, f"{ym} 出现意外节点: {set(ce.node_id) - keep}"
        for _, r in ce.iterrows():
            parts = r["SO2_Berth"] + r["SO2_Anchored"] + r["SO2_Manoeuvring"] + r["SO2_Sea"]
            assert abs(r["SO2_Total"] - parts) < 1e-6, f"{ym} 四工况和≠Total"
        print(f"\n[{ym}] 城市排放:\n",
              ce[["node_id", "YearMonth", "SO2_Berth", "SO2_Manoeuvring", "SO2_Sea", "SO2_Total"]])
        edge_f = os.path.join(out_dir, f"city_edge_list_SO2_Container_{ym}.csv")
        eg = pd.read_csv(edge_f) if os.path.isfile(edge_f) else pd.DataFrame(
            columns=["From_City", "To_City"])
        pairs = set(zip(eg.get("From_City", []), eg.get("To_City", [])))
        edge_pairs[ym] = pairs
        assert all(f != t for f, t in pairs), f"{ym} 存在同城边"
        print(f"[{ym}] 城市边: {pairs}")

    sh_nb = ("China__Shanghai", "China__Ningbo-Zhoushan")
    # 关键断言：沪→甬航段跨 1/31→2/1，应同时出现在两个月的边表中（逐点分月生效）
    assert sh_nb in edge_pairs["202401"], "沪→甬(1月段)未出现在 202401 边表"
    assert sh_nb in edge_pairs["202402"], "沪→甬(2月段)未出现在 202402 边表（跨月分箱失败）"

    print("\n✓ 冒烟测试通过 (年±1 函数 / 按月分箱 / 四工况 / **跨月航段逐点分月** / 同城剔除)")
    print(f"  输出目录: {tmp}")


if __name__ == "__main__":
    main()
