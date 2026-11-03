# -*- coding: utf-8 -*-
"""城市节点聚合（研究方案 3.6.5 + 5.2）。

两条独立路径：
1. 节点排放 = AIS 点级排放 sjoin 城市集水区（30km），predicate=within，
   一个点可命中多个重叠集水区 → 重复计入（空间溢出语义）；
   按四工况 Berth(靠泊)/Anchored(锚泊)/Manoeuvring(机动)/Sea(航行,含NA) 各自拆分；
   E3 停泊排放占比 = (Berth+Anchored)/Total 由这些列下游合成。
2. OD 边 = berth-first 挂靠的 From/To 港口经 port_city_mapping.csv 映射为城市节点，
   同城边剔除；映射不到的港口计数告警（QC）。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import geopandas as gpd

# 四种工况（见 emission_network_functions.py，按 Speed/ME_load 判定）各自单列：
#   Berth(Speed<1) / Anchored(1≤Speed<3) / Manoeuvring(load<0.2&Speed≥3) / Sea(其余)
# Mode='NA'（速度/负荷缺失、四条件均不满足的少数点）按口径并入 Sea_Class（航行）。
_MODE_TO_CLASS = {
    "berth": "Berth_Class",
    "anchored": "Anchored_Class",
    "manoeuvring": "Manoeuvring_Class",
    "sea": "Sea_Class",
}


def load_city_data(paths: dict) -> tuple[gpd.GeoDataFrame, dict[str, str]]:
    """加载集水区面与 港口→城市 映射表。"""
    catch = gpd.read_file(paths["city_catchments_gpkg"])[["node_id", "geometry"]]
    if catch.crs is None:
        catch = catch.set_crs("EPSG:4326")
    mp = pd.read_csv(paths["port_city_mapping_csv"], encoding="utf-8-sig")
    mapping = dict(zip(mp["Port_Name"], mp["node_id"]))
    # 排放代码在加载港口表时已把 Ningbo/Zhoushan 改名为组合港名，补登组合名别名
    for combo, members in {"Ningbo-Zhoushan": ["Ningbo", "Zhoushan"]}.items():
        if combo not in mapping:
            for m in members:
                if m in mapping:
                    mapping[combo] = mapping[m]
                    break
    return catch, mapping


def _mode_class(mode_series: pd.Series) -> pd.Series:
    s = mode_series.astype(str).str.strip().str.casefold()
    # 未知/NA 默认归 Sea_Class（航行）
    return s.map(_MODE_TO_CLASS).fillna("Sea_Class").to_numpy()


def aggregate_points_to_cities(
    ship_ais: pd.DataFrame,
    city_catchments: gpd.GeoDataFrame,
    pollutants: list[str],
    mmsi,
) -> pd.DataFrame:
    """单船 AIS 点级排放 → 城市集水区聚合（允许重叠重复计入）。

    返回列: node_id, YearMonth, Mode_Class, MMSI, <每个污染物总量(克)>
    """
    cols = [f"Total_{p}" for p in pollutants]
    need = ["Longitude", "Latitude", "UTCtime", "Mode"] + cols
    miss = [c for c in need if c not in ship_ais.columns]
    if miss:
        raise ValueError(f"aggregate_points_to_cities 缺少列: {miss}")
    df = ship_ais[need].copy()
    df["Mode_Class"] = _mode_class(df["Mode"])
    df["YearMonth"] = pd.to_datetime(df["UTCtime"], unit="s", utc=True).dt.strftime("%Y%m")

    pts = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs="EPSG:4326",
    )
    # inner sjoin: 点 × 命中集水区 一行一对 → 重叠区自然重复计入
    joined = gpd.sjoin(pts, city_catchments, predicate="within", how="inner")
    if len(joined) == 0:
        return pd.DataFrame()
    agg = (joined.groupby(["node_id", "YearMonth", "Mode_Class"], as_index=False)[cols]
           .sum())
    agg = agg.rename(columns={f"Total_{p}": p for p in pollutants})
    agg["MMSI"] = mmsi
    return agg


POLICY_DELTA_COLS = ["D_SO2_SECA", "D_PM_SECA", "D_NOx_NECA"]
POLICY_NO_MARGIN_COLS = [
    "NME_SO2_CURRENT", "NME_PM_CURRENT", "NME_NOx_CURRENT",
]
_POLICY_BASE_MAP = {"Total_SO2": "B_SO2", "Total_PM": "B_PM", "Total_NOx": "B_NOx"}


def aggregate_policy_deltas_to_cities(
    ship_ais: pd.DataFrame,
    city_catchments: gpd.GeoDataFrame,
    mmsi,
) -> pd.DataFrame:
    """政策情景点级 delta → 城市集水区聚合（与 aggregate_points_to_cities 同口径:
    within、重叠重复计入）。仅保留有减排的点(政策区内且 delta>0)。

    返回列: node_id, YearMonth, Mode_Class, Zone_ID, D_SO2_SECA, D_PM_SECA,
    D_NOx_NECA, MMSI
    """
    if "PZ_Zone" not in ship_ais.columns:
        return pd.DataFrame()
    df = ship_ais[ship_ais["PZ_Zone"] != ""]
    if len(df) == 0:
        return pd.DataFrame()
    df = df[(df[POLICY_DELTA_COLS].to_numpy() > 0).any(axis=1)]
    if len(df) == 0:
        return pd.DataFrame()
    need = ["Longitude", "Latitude", "UTCtime", "Mode", "PZ_Zone"] + POLICY_DELTA_COLS
    df = df[need].copy()
    df["Mode_Class"] = _mode_class(df["Mode"])
    df["YearMonth"] = pd.to_datetime(df["UTCtime"], unit="s", utc=True).dt.strftime("%Y%m")
    pts = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, city_catchments, predicate="within", how="inner")
    if len(joined) == 0:
        return pd.DataFrame()
    agg = (joined.groupby(["node_id", "YearMonth", "Mode_Class", "PZ_Zone"],
                          as_index=False)[POLICY_DELTA_COLS].sum())
    agg = agg.rename(columns={"PZ_Zone": "Zone_ID"})
    agg["MMSI"] = mmsi
    return agg


OPS_ENERGY_COLS = ["OPS_AE_kWh", "OPS_BO_kWh"]
OPS_BASE_COLS = ["Total_SO2", "Total_PM", "Total_NOx"]


def aggregate_ops_calls_to_cities(
    ship_ais: pd.DataFrame,
    city_catchments: gpd.GeoDataFrame,
    mmsi,
) -> pd.DataFrame:
    """严格靠泊点 -> 挂靠 × 城市集水区的岸电基础量。

    ``OPS_Call_ID`` 由挂靠识别程序只赋给 ``Mode=Berth`` 且内外港口标签
    一致的 AIS 点。这里仍使用与城市排放完全相同的 ``within`` 空间连接；
    一个靠泊点命中多个重叠集水区时重复计入，以保持暴露口径一致。

    返回的基线排放和政策 delta 均为克，能量为 kWh。组合政策可按
    ``岸电优先`` 用 ``靠泊基线 - 靠泊范围内面状政策 delta`` 精确去重。
    """
    if "OPS_Call_ID" not in ship_ais.columns:
        return pd.DataFrame()
    df = ship_ais[ship_ais["OPS_Call_ID"].notna()].copy()
    if len(df) == 0:
        return pd.DataFrame()

    need = [
        "Longitude", "Latitude", "UTCtime", "OPS_Call_ID", "OPS_Port_Name",
        *OPS_ENERGY_COLS, *OPS_BASE_COLS, *POLICY_DELTA_COLS,
    ]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"aggregate_ops_calls_to_cities 缺少列: {miss}")
    df = df[need].copy()
    df["YearMonth"] = pd.to_datetime(
        df["UTCtime"], unit="s", utc=True
    ).dt.strftime("%Y%m")
    pts = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, city_catchments, predicate="within", how="inner")
    if len(joined) == 0:
        return pd.DataFrame()
    val_cols = OPS_ENERGY_COLS + OPS_BASE_COLS + POLICY_DELTA_COLS
    agg = (
        joined.groupby(
            ["OPS_Call_ID", "OPS_Port_Name", "node_id", "YearMonth"],
            as_index=False,
        )[val_cols]
        .sum()
        .rename(columns={
            "OPS_Call_ID": "Call_ID",
            "OPS_Port_Name": "Port_Name",
            "Total_SO2": "Berth_SO2_g",
            "Total_PM": "Berth_PM_g",
            "Total_NOx": "Berth_NOx_g",
        })
    )
    agg["MMSI"] = mmsi
    return agg


def policy_delta_zone_totals(ship_ais: pd.DataFrame, mmsi) -> pd.DataFrame:
    """政策区内精确合计（不经集水区,守恒口径）: delta + 区内基线(SO2/PM/NOx)。

    返回列: YearMonth, Mode_Class, Zone_ID, D_*, B_SO2, B_PM, B_NOx, MMSI。
    B_* 为区内基线总量,供 zone 级减排率(D/B)使用。
    """
    if "PZ_Zone" not in ship_ais.columns:
        return pd.DataFrame()
    df = ship_ais[ship_ais["PZ_Zone"] != ""]
    if len(df) == 0:
        return pd.DataFrame()
    base_cols = list(_POLICY_BASE_MAP)
    out = df[["UTCtime", "Mode", "PZ_Zone"] + POLICY_DELTA_COLS
             + POLICY_NO_MARGIN_COLS + base_cols].copy()
    out["Mode_Class"] = _mode_class(out["Mode"])
    out["YearMonth"] = pd.to_datetime(out["UTCtime"], unit="s", utc=True).dt.strftime("%Y%m")
    agg = (out.groupby(["YearMonth", "Mode_Class", "PZ_Zone"], as_index=False)
           [POLICY_DELTA_COLS + POLICY_NO_MARGIN_COLS + base_cols].sum())
    agg = agg.rename(columns={"PZ_Zone": "Zone_ID", **_POLICY_BASE_MAP})
    agg["MMSI"] = mmsi
    return agg


def map_edges_to_cities(
    edge_df: pd.DataFrame,
    mapping: dict[str, str],
    pollutants: list[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """港口级 OD 边 → 城市级 OD 边。返回 (城市边表, 未映射港口计数)。"""
    if edge_df is None or len(edge_df) == 0:
        return pd.DataFrame(), {}
    df = edge_df.copy()
    df["From_City"] = df["From_Port"].map(mapping)
    df["To_City"] = df["To_Port"].map(mapping)
    unmapped: dict[str, int] = {}
    for port_col, city_col in (("From_Port", "From_City"), ("To_Port", "To_City")):
        bad = df.loc[df[city_col].isna(), port_col]
        for name, cnt in bad.value_counts().items():
            unmapped[name] = unmapped.get(name, 0) + int(cnt)
    df = df.dropna(subset=["From_City", "To_City"])
    df = df[df["From_City"] != df["To_City"]]  # 同城跳过
    if len(df) == 0:
        return pd.DataFrame(), unmapped
    pol_cols = [p for p in pollutants if p in df.columns]
    out = df.groupby(["From_City", "To_City"], as_index=False)[pol_cols].sum()
    return out, unmapped
