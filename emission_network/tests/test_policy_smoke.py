# -*- coding: utf-8 -*-
"""政策情景(研究 4)冒烟测试：复用 make_synthetic_ais 的沪甬合成航线。

场景：矩形政策区 TEST_CN 只罩住上海及沪甬航段北半段(不含宁波)。
HFO 船、2024 年(post-2020)、ECA_list 为空 → 基线硫率 0.5%、SFC 恒为 MDO 档。

断言：
  1. 老船(Build_Year 2015 < 2016)：D_NOx_NECA 全为 0；
     区内 D_SO2_SECA / B_SO2 == 0.8 精确(0.001/0.005 → 5 倍削减)；
     0 < D_PM_SECA < 0.6×B_PM(PM 是仿射项,削减率必然低于 SO2)。
  2. 新船(Build_Year 2018 ≥ 2016)：D_NOx_NECA > 0 且 < B_NOx(NECA 只动主机)。
  3. Zone_ID 全为 TEST_CN；city_policy_delta 含 China__Shanghai;
     delta 文件与 totals 文件按月产出。

用法： cd emission_network && python tests/test_policy_smoke.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import pandas as pd
import geopandas as gpd
from shapely.geometry import box

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)

import emission_network_main as M  # noqa: E402
from city_aggregation import load_city_data, POLICY_DELTA_COLS  # noqa: E402
from config_loader import load_config  # noqa: E402
from make_synthetic_ais import SHIP, write_monthly_dbs, SH, NB  # noqa: E402

# 政策区：罩住上海(121.85,31.05)与沪甬航段北半段；宁波(122.10,29.95)在区外
ZONE = gpd.GeoDataFrame(
    {"zone_id": ["TEST_CN"]},
    geometry=[box(121.3, 30.7, 122.3, 31.4)], crs="EPSG:4326")

POLICY_CFG = {"seca_sulfur": 0.001, "neca_build_year": 2016}
TARGET_MONTHS = ["202401", "202402"]


def _run_once(ship_dict: dict, cfg) -> str:
    """跑一遍合成流水线(单船型/单核/政策区启用)，返回输出根目录。"""
    tmp = tempfile.mkdtemp(prefix="emis_policy_smoke_")
    import make_synthetic_ais as S
    old_ship = dict(S.SHIP)
    S.SHIP.clear()
    S.SHIP.update(ship_dict)
    try:
        db_paths = write_monthly_dbs(tmp)
    finally:
        S.SHIP.clear()
        S.SHIP.update(old_ship)
    ships = pd.DataFrame([ship_dict])

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

    M.run_emission_for_year_ship_type(
        "Container", ships, TARGET_MONTHS, db_paths, "2024",
        port_coor, pd.DataFrame(), pd.DataFrame(), {}, tmp,
        M.POLLUTANTS, n_cores=1,
        city_catchments=catch_small, port_city_mapping=mapping,
        policy_zones=ZONE, policy_cfg=POLICY_CFG,
    )
    return tmp


def _load_all(root: str, base: str) -> pd.DataFrame:
    frames = []
    for ym in TARGET_MONTHS:
        f = os.path.join(root, ym, "Container", f"{base}_Container_{ym}.csv")
        assert os.path.isfile(f), f"缺少政策产物: {f}"
        d = pd.read_csv(f)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    cfg = load_config()

    # ---------- 老船(2015 建造)：SECA 生效、NECA 不生效 ----------
    root_old = _run_once(dict(SHIP), cfg)
    tot = _load_all(root_old, "policy_delta_totals")
    tot = tot[tot["Zone_ID"].notna()]
    assert len(tot) > 0, "政策区 totals 为空(轨迹未落入 TEST_CN?)"
    assert set(tot["Zone_ID"]) == {"TEST_CN"}, f"Zone_ID 异常: {set(tot['Zone_ID'])}"

    d_so2, b_so2 = tot["D_SO2_SECA"].sum(), tot["B_SO2"].sum()
    d_pm, b_pm = tot["D_PM_SECA"].sum(), tot["B_PM"].sum()
    d_nox, b_nox = tot["D_NOx_NECA"].sum(), tot["B_NOx"].sum()
    assert b_so2 > 0 and b_pm > 0 and b_nox > 0, "区内基线为 0?"
    # HFO post-2020 非ECA: SFC 基线/情景同为 MDO 档 → D/B = 1 - 0.001/0.005 = 0.8 精确
    assert abs(d_so2 / b_so2 - 0.8) < 1e-9, f"SO2 削减率应为 0.8, 实得 {d_so2/b_so2:.6f}"
    assert 0 < d_pm < 0.6 * b_pm, f"PM 削减率异常: {d_pm/b_pm:.4f}"
    assert d_nox == 0, f"老船(2015)不应有 NECA 减排, 实得 {d_nox}"

    city = _load_all(root_old, "city_policy_delta")
    assert len(city) > 0, "城市 delta 为空"
    assert set(city["Zone_ID"]) == {"TEST_CN"}
    assert "China__Shanghai" in set(city["node_id"]), "上海节点缺失于城市 delta"
    assert (city[POLICY_DELTA_COLS].to_numpy() >= 0).all(), "存在负 delta"
    print(f"[老船] SO2 削减率 {d_so2/b_so2:.4f} (精确 0.8)  "
          f"PM 削减率 {d_pm/b_pm:.4f}  NOx delta = 0 ✓")

    # ---------- 新船(2018 建造)：NECA 生效(仅主机) ----------
    ship_new = dict(SHIP)
    ship_new["Build_Year"] = 2018
    root_new = _run_once(ship_new, cfg)
    tot2 = _load_all(root_new, "policy_delta_totals")
    d_nox2, b_nox2 = tot2["D_NOx_NECA"].sum(), tot2["B_NOx"].sum()
    assert d_nox2 > 0, "新船(2018)应有 NECA 减排"
    assert d_nox2 < b_nox2, "NECA 只动主机, delta 必须小于含辅机/锅炉的基线总量"
    # SECA 部分与建造年无关,应与老船完全一致
    assert abs(tot2["D_SO2_SECA"].sum() / tot2["B_SO2"].sum() - 0.8) < 1e-9
    print(f"[新船] NOx delta {d_nox2:.1f} g / 基线 {b_nox2:.1f} g "
          f"({d_nox2/b_nox2:.2%}, 仅主机份额) ✓")

    print("\n✓ 政策情景冒烟测试通过 "
          "(区内点识别 / SECA 0.8 精确削减 / PM 仿射比 / NECA 建造年门槛 / 城市聚合)")
    print(f"  输出目录: {root_old} , {root_new}")


if __name__ == "__main__":
    main()
