"""
航运网络构建（无排放计算）——sjoin 预计算版
核心优化：在 process_ship_shipping_network 入口处，对所有 AIS 点做一次性 sjoin，
预先标注每个点是否在「港口缓冲区」「泊位缓冲区」「运河走廊缓冲区」「运河等待区缓冲区」内，
后续所有判定只查布尔列，不再计算距离。

逻辑：
- 普通港口：靠泊确认(Mode=Berth 且 in_berth=True) → 倒推进入6nm缓冲区(in_port=True)时刻 →
  前推离开缓冲区时刻 → 在港时间=离开-进入；靠泊时长允许短暂中断；等待=在港-靠泊
- 运河：确认通过(in_canal_corridor=True 的连续段≥300s) → 提取走廊段全过程 →
  航行时间=走廊段内Mode=Sea的时段；等待时间=从走廊段起始向前搜索 in_canal_waiting=True 的累计时长
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPoint, Point
from shapely.ops import voronoi_diagram
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 非港口节点过滤：锚地/海峡/SPM/Oil Terminal/Lightering 等不参与港口构建
#（运河单独在 CANAL_PORT_NAMES 中处理，不受此过滤影响）
# ---------------------------------------------------------------------------
NON_PORT_NAME_PATTERN = re.compile(
    r"\b("
    r"ANCHORAGE|"
    r"STRAITS?|"        # Strait / Straits 都剔除（如 Kanmon Kaikyo (Straits)）
    r"LIGHTERING|"
    r"SPM|"
    r"OIL\s+TERMINAL"
    r")\b",
    re.IGNORECASE,
)


def is_non_port_name(name) -> bool:
    """根据 Port.Name 判断该条目是否为非港口节点（锚地/海峡/SPM/Oil Terminal/Lightering）。"""
    if not isinstance(name, str):
        return False
    return bool(NON_PORT_NAME_PATTERN.search(name))

# ---------------------------------------------------------------------------
# 运河常量
# ---------------------------------------------------------------------------
CANAL_PORT_NAMES = frozenset({"Suez Canal", "Panama Canal"})

CANAL_WAITING_SUBPORTS = frozenset(
    {
        "Suez Canal-N",
        "Suez Canal-S",
        "Panama Canal-S1",
        "Panama Canal-S2",
        "Panama Canal-N1",
        "Panama Canal-N2",
        "Panama Canal-N3",
    }
)

CANAL_WAITING_PORTS_BY_MAIN: Dict[str, Tuple[str, ...]] = {
    "Suez Canal": ("Suez Canal-N", "Suez Canal-S"),
    "Panama Canal": (
        "Panama Canal-N1",
        "Panama Canal-N2",
        "Panama Canal-N3",
        "Panama Canal-S1",
        "Panama Canal-S2",
    ),
}

CANAL_WAITING_BUFFER_NM = 10.0

CANAL_TAG_FIELDS = {
    "Suez Canal": {
        "port_lon": 32.28333,
        "port_lat": 30.583333,
        "berth_lon": 32.28333,
        "berth_lat": 30.583333,
        "port_number": "PO1512",
        "country": "Egypt",
    },
    "Panama Canal": {
        "port_lon": -79.71667,
        "port_lat": 9.116667,
        "berth_lon": -79.71667,
        "berth_lat": 9.116667,
        "port_number": "PO3434",
        "country": "Panama",
    },
}

CANAL_MIN_PASSAGE_SEC = 300.0

# 运河两端端口坐标（用于构建端口缓冲区，判定船舶进入/离开运河）
CANAL_ENDPOINTS: Dict[str, List[Tuple[str, float, float]]] = {
    "Suez Canal": [
        ("Suez_N", 32.31, 31.26),    # 北端 Port Said
        ("Suez_S", 32.55, 29.96),    # 南端 Suez
    ],
    "Panama Canal": [
        ("Panama_Atlantic", -79.92, 9.36),   # 大西洋端 Colón
        ("Panama_Pacific", -79.56, 8.95),    # 太平洋端 Balboa
    ],
}

# 运河端口缓冲区半径（海里）
CANAL_ENDPOINT_BUFFER_NM = 5.0


# ===========================================================================
# 基础工具
# ===========================================================================

def haversine_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    lon1 = np.asarray(lon1, dtype=float)
    lat1 = np.asarray(lat1, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    rlon1, rlat1, rlon2, rlat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = rlon2 - rlon1
    dlat = rlat2 - rlat1
    a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1 - a)))
    return 6371.0 * c


def _nm_to_km(nm: float) -> float:
    return float(nm) * 1.852


def _nm_to_m(nm: float) -> float:
    return float(nm) * 1852.0


def _km_to_m(km: float) -> float:
    return float(km) * 1000.0


def _normalize_waiting_port_name(name: str) -> str:
    s = str(name).strip().casefold()
    for ch in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"):
        s = s.replace(ch, "-")
    return " ".join(s.split())


def _waiting_port_lookup_keys(canonical_name: str) -> List[str]:
    n = _normalize_waiting_port_name(canonical_name)
    sp = n.replace("-", " ")
    hy = "-".join(n.split())
    keys = list(dict.fromkeys([n, sp, hy]))
    return keys


# ===========================================================================
# AIS 插值
# ===========================================================================

def _interp_1d_maxgap(t: np.ndarray, y: np.ndarray, maxgap_sec: float) -> np.ndarray:
    n = len(t)
    out = np.asarray(y, dtype=float).copy()
    valid = np.isfinite(out)
    idx = np.flatnonzero(valid)
    for k in range(len(idx) - 1):
        a, b = int(idx[k]), int(idx[k + 1])
        if t[b] - t[a] > maxgap_sec:
            continue
        sl = slice(a, b + 1)
        out[sl] = np.interp(t[sl], [t[a], t[b]], [out[a], out[b]])
    return out


def interpolate_ais_r_style(ais: pd.DataFrame) -> pd.DataFrame:
    df = ais.sort_values("UTCtime").reset_index(drop=True)
    if "Interp" not in df.columns:
        df["Interp"] = 0
    t = df["UTCtime"].astype(float).values
    new_rows: List[dict] = []
    mmsi_val = df["MMSI"].iloc[0] if "MMSI" in df.columns else None
    for i in range(len(t) - 1):
        dt = float(t[i + 1] - t[i])
        if 1800 < dt < 3600 * 24:
            n = int(np.floor(dt / 1800))
            for k in range(1, n + 1):
                nt = t[i] + 1800 * k
                if nt >= t[i + 1]:
                    break
                row = {"UTCtime": nt, "Speed": np.nan, "Interp": 1, "Longitude": np.nan, "Latitude": np.nan}
                if mmsi_val is not None:
                    row["MMSI"] = mmsi_val
                new_rows.append(row)
    if new_rows:
        extra = pd.DataFrame(new_rows)
        df = pd.concat([df, extra], ignore_index=True)
        df = df.sort_values("UTCtime").reset_index(drop=True)
        df = df.drop_duplicates(subset=["UTCtime"], keep="first")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    tt = df["UTCtime"].astype(float).values
    df["Longitude"] = _interp_1d_maxgap(tt, df["Longitude"].values, 86400.0)
    df["Latitude"] = _interp_1d_maxgap(tt, df["Latitude"].values, 86400.0)
    df["Speed"] = pd.to_numeric(df["Speed"], errors="coerce").ffill().bfill()
    return df


# ===========================================================================
# 船舶信息 / Mode
# ===========================================================================

def _ship_dwt(ship_info: Any) -> float:
    if hasattr(ship_info, "get"):
        v = ship_info.get("DWT", np.nan)
    elif isinstance(ship_info, dict):
        v = ship_info.get("DWT", np.nan)
    else:
        v = np.nan
    if pd.notna(v) and v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    return float(np.nan)


def _mode_from_speed_me(speed: np.ndarray, me_load: np.ndarray, design_speed: float) -> np.ndarray:
    mode = np.full(len(speed), "NA", dtype=object)
    mode[speed < 1] = "Berth"
    mode[(speed >= 1) & (speed < 3)] = "Anchored"
    mode[(me_load < 0.2) & (speed >= 3)] = "Manoeuvring"
    mode[(me_load >= 0.2) & (speed >= 3)] = "Sea"
    return mode


def _me_load_proxy(
    speed: np.ndarray,
    design_speed: float,
    me_power: float,
    ship_type: str | None = None,
    size: float | None = None,
    me_cf_w: float = 1.0,
    me_cf_f: float = 1.0,
) -> np.ndarray:
    """
    主机负载（与 emission_network_functions 对齐）：
        MEPower = ME * (Speed/DesignSpeed)^3 * ME_CF_S / (ME_CF_W * ME_CF_F)
    其中 ME_CF_S 取决于船型：
        - Container 且 Size >= 14500: 0.75
        - Cruise: 0.70
        - 其它: 1.0
    然后截断至 0.98*ME，最后 round(power/ME, 2) 作为离散化的 ME_load。
    """
    ds = max(float(design_speed), 1e-3)
    mp = max(float(me_power), 1e-3)
    if ship_type == "Container" and size is not None and float(size) >= 14500:
        me_cf_s = 0.75
    elif ship_type == "Cruise":
        me_cf_s = 0.70
    else:
        me_cf_s = 1.0
    cf_wf = max(float(me_cf_w) * float(me_cf_f), 1e-6)
    power = mp * (np.clip(speed, 0, None) / ds) ** 3 * me_cf_s / cf_wf
    power = np.minimum(power, 0.98 * mp)
    return np.round(power / mp, 2)


# ===========================================================================
# 缓冲区构建（主进程中只执行一次）
# ===========================================================================

def build_port_area_buffers(
    port_coor: pd.DataFrame,
    berth_radius_km: float,
) -> gpd.GeoDataFrame:
    """
    港口区域缓冲区。

    构建逻辑：
      1) 每个泊位的 berth_radius_km 圆 union，按 Port.Name dissolve
         → 得到「该港的覆盖范围」（raw polygon），保证 8 km 触达。
      2) 邻港互斥（基于「最近泊位归属」的 Voronoi 切分）：
         对所有泊位点做 Voronoi 图，每个 Voronoi cell 归属其包含的泊位的港；
         把 cells 按 Port.Name dissolve 得到「该港的归属域」。
      3) 该港最终多边形 = raw polygon ∩ 归属域。
         直观含义：一个 AIS 点要既位于该港某泊位的 8 km 圆内（"够近"），
         又比其他任何港的所有泊位都更接近该港（"归我管"），才被判进该港。
         结果：邻港之间用泊位连线的中垂线划界，互不重叠，
         不会因 8 km 圆相互覆盖而把某一港"整个吞掉"。

    返回列：Port.Name, Lon_port, Lat_port, geometry
        Lon_port/Lat_port = 该港多边形质心，仅用于元信息显示；
        下游引用泊位坐标都直接查 port_coor，不依赖此处的 Lon_port/Lat_port。
    """
    if port_coor is None or len(port_coor) == 0:
        return gpd.GeoDataFrame(
            columns=["Port.Name", "Lon_port", "Lat_port", "geometry"], crs="EPSG:4326"
        )

    pc = port_coor.copy()
    pc["__lo"] = pd.to_numeric(pc.get("Lon_berth"), errors="coerce")
    pc["__la"] = pd.to_numeric(pc.get("Lat_berth"), errors="coerce")
    pc = pc.dropna(subset=["__lo", "__la", "Port.Name"]).reset_index(drop=True)
    if len(pc) == 0:
        return gpd.GeoDataFrame(
            columns=["Port.Name", "Lon_port", "Lat_port", "geometry"], crs="EPSG:4326"
        )

    radius_m = _km_to_m(berth_radius_km)

    # 同坐标的泊位会让 Voronoi 退化（生成空 cell），先按 (lo, la, Port.Name) 去重
    pc = pc.drop_duplicates(subset=["__lo", "__la", "Port.Name"]).reset_index(drop=True)

    # 1) 每个泊位的点（3857）和圆形 buffer
    berth_pts_3857 = gpd.GeoSeries(
        gpd.points_from_xy(pc["__lo"], pc["__la"]),
        index=pc.index,
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")
    berth_circles_3857 = berth_pts_3857.buffer(radius_m)

    # 2) 按 Port.Name dissolve 得到每港"raw polygon"（8 km 触达域）
    raw_polys: dict[str, Any] = {}
    for pname, idxs in pc.groupby("Port.Name").groups.items():
        circles = berth_circles_3857.loc[list(idxs)]
        if len(circles) == 0:
            continue
        raw_polys[str(pname)] = circles.unary_union

    # 3) 全球泊位 Voronoi → 每 cell 归属其包含的泊位的港 → 按港 dissolve
    #    voronoi_diagram 输出顺序与输入不一致，需用 sjoin 把 cell 关联回 berth
    pts_unique = list(berth_pts_3857.values)
    mp = MultiPoint(pts_unique)
    # envelope 取所有泊位外包矩形再外扩，确保所有 cell 边界落在 envelope 内
    env = mp.envelope.buffer(max(radius_m * 50, 1e6))
    vor = voronoi_diagram(mp, envelope=env, edges=False)
    cells_gdf = gpd.GeoDataFrame(
        geometry=list(vor.geoms),
        crs="EPSG:3857",
    )
    pts_gdf = gpd.GeoDataFrame(
        {"_berth_port": pc["Port.Name"].astype(str).values},
        geometry=berth_pts_3857.values,
        crs="EPSG:3857",
    )
    # 每个 cell 必含且仅含一个 berth point；用 contains 把 cell 关联到 port
    cells_joined = gpd.sjoin(cells_gdf, pts_gdf, predicate="contains", how="left")
    cells_joined = cells_joined.dropna(subset=["_berth_port"])
    voronoi_by_port = (
        cells_joined.dissolve(by="_berth_port")["geometry"].to_dict()
    )

    # 4) raw ∩ Voronoi 归属域 → 最终多边形
    rows: list[dict] = []
    for pname, raw in raw_polys.items():
        own_voronoi = voronoi_by_port.get(pname)
        if own_voronoi is None or own_voronoi.is_empty:
            continue
        final = raw.intersection(own_voronoi)
        if final.is_empty:
            continue
        poly_4326 = (
            gpd.GeoSeries([final], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]
        )
        c = poly_4326.centroid
        rows.append({
            "Port.Name": pname,
            "Lon_port": float(c.x),
            "Lat_port": float(c.y),
            "geometry": poly_4326,
        })

    if not rows:
        return gpd.GeoDataFrame(
            columns=["Port.Name", "Lon_port", "Lat_port", "geometry"], crs="EPSG:4326"
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def build_berth_buffers(
    port_coor: pd.DataFrame,
    berth_km: float,
    canal_coor: pd.DataFrame | None = None,
) -> gpd.GeoDataFrame:
    """
    泊位缓冲区：以每个泊位坐标为圆心、berth_km 为半径的圆。
    用于 sjoin 判定 AIS 点是否在某港口的泊位附近。
    包含普通港口泊位和运河泊位。
    """
    rows = []
    for _, r in port_coor.iterrows():
        lo = pd.to_numeric(r.get("Lon_berth"), errors="coerce")
        la = pd.to_numeric(r.get("Lat_berth"), errors="coerce")
        pn = r.get("Port.Name")
        if pd.notna(lo) and pd.notna(la) and pd.notna(pn):
            rows.append({"berth_port_name": str(pn), "geometry": Point(float(lo), float(la))})
    # 运河泊位
    if canal_coor is not None and len(canal_coor) > 0:
        for _, r in canal_coor.iterrows():
            lo = pd.to_numeric(r.get("Lon_berth"), errors="coerce")
            la = pd.to_numeric(r.get("Lat_berth"), errors="coerce")
            pn = r.get("Port.Name")
            if pd.notna(lo) and pd.notna(la) and pd.notna(pn):
                rows.append({"berth_port_name": str(pn), "geometry": Point(float(lo), float(la))})
    # 运河兜底（R 固定坐标）
    for cname in CANAL_PORT_NAMES:
        meta = CANAL_TAG_FIELDS[cname]
        rows.append({"berth_port_name": cname, "geometry": Point(float(meta["berth_lon"]), float(meta["berth_lat"]))})

    if not rows:
        return gpd.GeoDataFrame(columns=["berth_port_name", "geometry"], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf = gdf.to_crs("EPSG:3857")
    gdf["geometry"] = gdf.geometry.buffer(_km_to_m(berth_km))
    gdf = gdf.to_crs("EPSG:4326")
    # 按港口名合并（dissolve），同一港口多个泊位合并成一个多边形
    gdf = gdf.dissolve(by="berth_port_name", as_index=False)
    return gdf


def build_canal_corridor_buffers(
    canal_coor: pd.DataFrame | None,
    canal_km: float,
) -> gpd.GeoDataFrame:
    """运河走廊缓冲区：以运河参考泊位为圆心、canal_km 为半径。"""
    rows = []
    for cname in CANAL_PORT_NAMES:
        rlon, rlat = _canal_reference_lonlat(cname, canal_coor)
        rows.append({"canal_name": cname, "geometry": Point(rlon, rlat)})
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf = gdf.to_crs("EPSG:3857")
    gdf["geometry"] = gdf.geometry.buffer(_km_to_m(canal_km))
    return gdf.to_crs("EPSG:4326")


def build_canal_waiting_buffers(
    port_coor: pd.DataFrame,
    canal_coor: pd.DataFrame | None,
) -> gpd.GeoDataFrame:
    """运河等待区缓冲区：以等待子港坐标为圆心、CANAL_WAITING_BUFFER_NM 为半径。"""
    rows = []
    # 合并 port_coor 和 canal_coor 来搜索等待子港
    all_coor = pd.concat([port_coor, canal_coor], ignore_index=True) if canal_coor is not None and len(canal_coor) > 0 else port_coor.copy()
    pn_norm = all_coor["Port.Name"].map(_normalize_waiting_port_name)
    for cname in CANAL_PORT_NAMES:
        subs = CANAL_WAITING_PORTS_BY_MAIN.get(cname, ())
        for name in subs:
            for key in _waiting_port_lookup_keys(name):
                m = all_coor.loc[pn_norm == key]
                if len(m) > 0:
                    r = m.iloc[0]
                    lo = pd.to_numeric(r.get("Lon_port"), errors="coerce")
                    la = pd.to_numeric(r.get("Lat_port"), errors="coerce")
                    if pd.isna(lo) or pd.isna(la):
                        lo = pd.to_numeric(r.get("Lon_berth"), errors="coerce")
                        la = pd.to_numeric(r.get("Lat_berth"), errors="coerce")
                    if pd.notna(lo) and pd.notna(la):
                        rows.append({"canal_name": cname, "waiting_port": name, "geometry": Point(float(lo), float(la))})
                    break
    if not rows:
        return gpd.GeoDataFrame(columns=["canal_name", "waiting_port", "geometry"], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf = gdf.to_crs("EPSG:3857")
    gdf["geometry"] = gdf.geometry.buffer(_nm_to_m(CANAL_WAITING_BUFFER_NM))
    gdf = gdf.to_crs("EPSG:4326")
    # 按运河名合并
    gdf = gdf.dissolve(by="canal_name", as_index=False)
    return gdf


def build_canal_endpoint_buffers(
    endpoint_buffer_nm: float = CANAL_ENDPOINT_BUFFER_NM,
) -> gpd.GeoDataFrame:
    """
    运河端口缓冲区：以运河两端坐标为圆心、endpoint_buffer_nm 为半径。
    每个端口单独保留（不 dissolve），用于判定船舶第一次和第二次进入端口。
    列：canal_name, endpoint_name, geometry
    """
    rows = []
    for cname, endpoints in CANAL_ENDPOINTS.items():
        for (ep_name, ep_lon, ep_lat) in endpoints:
            rows.append({
                "canal_name": cname,
                "endpoint_name": ep_name,
                "geometry": Point(ep_lon, ep_lat),
            })
    if not rows:
        return gpd.GeoDataFrame(columns=["canal_name", "endpoint_name", "geometry"], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf = gdf.to_crs("EPSG:3857")
    gdf["geometry"] = gdf.geometry.buffer(_nm_to_m(endpoint_buffer_nm))
    return gdf.to_crs("EPSG:4326")


def _canal_reference_lonlat(port_name: str, canal_coor: pd.DataFrame | None) -> Tuple[float, float]:
    if canal_coor is not None and len(canal_coor) > 0:
        sub = canal_coor[canal_coor["Port.Name"] == port_name]
        if len(sub) > 0:
            r = sub.iloc[0]
            return float(r["Lon_berth"]), float(r["Lat_berth"])
    meta = CANAL_TAG_FIELDS[port_name]
    return float(meta["berth_lon"]), float(meta["berth_lat"])


# ===========================================================================
# 一次性 sjoin 预计算所有空间关系
# ===========================================================================

def _sjoin_one(ship_gdf: gpd.GeoDataFrame, buffer_gdf: gpd.GeoDataFrame, key_col: str, result_col: str) -> pd.Series:
    """
    对 ship_gdf 做 sjoin(buffer_gdf)，返回一个 Series，
    对每个 AIS 点给出它落入的 buffer 的 key_col 值（首次匹配），未匹配则 NaN。
    """
    if buffer_gdf is None or len(buffer_gdf) == 0:
        return pd.Series(np.nan, index=ship_gdf.index, name=result_col)
    cols = [key_col, "geometry"]
    buf = buffer_gdf[[c for c in cols if c in buffer_gdf.columns]].copy()
    if buf.crs is None:
        buf = buf.set_crs("EPSG:4326")
    joined = gpd.sjoin(ship_gdf, buf, predicate="within", how="left")
    # 去重：同一 AIS 点可能匹配多个缓冲区，保留第一个
    joined = joined.reset_index().drop_duplicates(subset="index", keep="first").set_index("index")
    joined = joined.reindex(ship_gdf.index)
    return joined[key_col].rename(result_col) if key_col in joined.columns else pd.Series(np.nan, index=ship_gdf.index, name=result_col)


def precompute_spatial_tags(
    ais: pd.DataFrame,
    port_buffers: gpd.GeoDataFrame,
    berth_buffers: gpd.GeoDataFrame,
    canal_corridor_buffers: gpd.GeoDataFrame,
    canal_waiting_buffers: gpd.GeoDataFrame,
    canal_endpoint_buffers: gpd.GeoDataFrame | None = None,
) -> pd.DataFrame:
    """
    一次性 sjoin：对所有 AIS 点标注空间归属。
    新增列：
      - Port_Name: 该点所在的港口缓冲区名（或 NaN）
      - Port_Event: 'In_Port' 或 'At_Sea'
      - berth_port_name: 该点所在的泊位缓冲区对应的港口名（或 NaN）
      - canal_corridor: 该点所在的运河走廊名（或 NaN）
      - canal_waiting: 该点所在的运河等待区名（或 NaN）
      - canal_endpoint: 该点所在的运河端口名（或 NaN），如 'Suez_N', 'Panama_Pacific'
      - canal_endpoint_canal: 该端口属于哪个运河（或 NaN）
    """
    ship_gdf = gpd.GeoDataFrame(
        ais.copy(),
        geometry=gpd.points_from_xy(ais["Longitude"], ais["Latitude"]),
        crs="EPSG:4326",
    )

    # 1. 港口缓冲区
    port_cols = [c for c in ["Port.Name", "Lon_port", "Lat_port", "geometry"] if c in port_buffers.columns]
    pbuf = port_buffers[port_cols].copy()
    if pbuf.crs is None:
        pbuf = pbuf.set_crs("EPSG:4326")
    joined_port = gpd.sjoin(ship_gdf, pbuf, predicate="within", how="left")
    joined_port = joined_port.reset_index().drop_duplicates(subset="index", keep="first").set_index("index")
    joined_port = joined_port.reindex(ship_gdf.index)

    out = ais.copy()
    out["Port_Name"] = joined_port["Port.Name"].values if "Port.Name" in joined_port.columns else np.nan
    out["port_lon"] = joined_port["Lon_port"].values if "Lon_port" in joined_port.columns else np.nan
    out["port_lat"] = joined_port["Lat_port"].values if "Lat_port" in joined_port.columns else np.nan
    out["nearest_port_name"] = out["Port_Name"].copy()
    out["Port_Event"] = np.where(out["Port_Name"].notna(), "In_Port", "At_Sea")

    # 2. 泊位缓冲区
    berth_result = _sjoin_one(ship_gdf, berth_buffers, "berth_port_name", "berth_port_name")
    out["berth_port_name"] = berth_result.values

    # 3. 运河走廊缓冲区
    corridor_result = _sjoin_one(ship_gdf, canal_corridor_buffers, "canal_name", "canal_corridor")
    out["canal_corridor"] = corridor_result.values

    # 4. 运河等待区缓冲区
    waiting_result = _sjoin_one(ship_gdf, canal_waiting_buffers, "canal_name", "canal_waiting")
    out["canal_waiting"] = waiting_result.values

    # 5. 运河端口缓冲区
    if canal_endpoint_buffers is not None and len(canal_endpoint_buffers) > 0:
        ep_cols = [c for c in ["canal_name", "endpoint_name", "geometry"] if c in canal_endpoint_buffers.columns]
        ep_buf = canal_endpoint_buffers[ep_cols].copy()
        if ep_buf.crs is None:
            ep_buf = ep_buf.set_crs("EPSG:4326")
        joined_ep = gpd.sjoin(ship_gdf, ep_buf, predicate="within", how="left")
        joined_ep = joined_ep.reset_index().drop_duplicates(subset="index", keep="first").set_index("index")
        joined_ep = joined_ep.reindex(ship_gdf.index)
        out["canal_endpoint"] = joined_ep["endpoint_name"].values if "endpoint_name" in joined_ep.columns else np.nan
        out["canal_endpoint_canal"] = joined_ep["canal_name"].values if "canal_name" in joined_ep.columns else np.nan
    else:
        out["canal_endpoint"] = np.nan
        out["canal_endpoint_canal"] = np.nan

    return out


# ===========================================================================
# 运河标记
# ===========================================================================

def apply_canal_tags_r_style(
    ais: pd.DataFrame,
    canal_coor: pd.DataFrame | None,
    threshold_km: float = 10.0,
) -> pd.DataFrame:
    """与 R 一致：对每个运河参考点找全轨迹最近 AIS 点，若距离 < threshold_km 则标记。"""
    out = ais.copy()
    if len(out) == 0:
        return out
    lon = out["Longitude"].astype(float).values
    lat = out["Latitude"].astype(float).values
    for cname in CANAL_PORT_NAMES:
        rlon, rlat = _canal_reference_lonlat(cname, canal_coor)
        d_km = haversine_km(rlon, rlat, lon, lat)
        j = int(np.argmin(d_km))
        if float(d_km[j]) < threshold_km:
            meta = CANAL_TAG_FIELDS[cname]
            out.loc[out.index[j], "nearest_port_name"] = cname
            out.loc[out.index[j], "Port_Name"] = cname
            out.loc[out.index[j], "Port_Event"] = "In_Port"
            out.loc[out.index[j], "port_lon"] = meta["port_lon"]
            out.loc[out.index[j], "port_lat"] = meta["port_lat"]
    return out


def strip_canal_waiting_subports(ais: pd.DataFrame) -> pd.DataFrame:
    out = ais.copy()
    if "Port_Name" not in out.columns:
        return out
    pn = out["Port_Name"]
    m = pn.notna() & pn.isin(list(CANAL_WAITING_SUBPORTS))
    if not m.any():
        return out
    out.loc[m, "Port_Name"] = np.nan
    out.loc[m, "nearest_port_name"] = np.nan
    out.loc[m, "Port_Event"] = "At_Sea"
    return out


# ===========================================================================
# 辅助
# ===========================================================================

def _port_reference_lonlat(
    port_coor: pd.DataFrame,
    port_name: str,
) -> Optional[Tuple[float, float]]:
    pname = str(port_name)
    if pname in CANAL_PORT_NAMES:
        meta = CANAL_TAG_FIELDS[pname]
        return float(meta["port_lon"]), float(meta["port_lat"])
    sub = port_coor[port_coor["Port.Name"] == pname]
    for _, r in sub.iterrows():
        lo = pd.to_numeric(r.get("Lon_port"), errors="coerce")
        la = pd.to_numeric(r.get("Lat_port"), errors="coerce")
        if pd.notna(lo) and pd.notna(la):
            return float(lo), float(la)
    return None


def _bool_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """True 连续段，返回 [start, end) 下标。"""
    runs: List[Tuple[int, int]] = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        runs.append((i, j))
        i = j
    return runs


# ===========================================================================
# 普通港口：靠泊确认 → 倒推（使用预计算的布尔列）
# ===========================================================================

def _find_all_berth_events_from_tags(
    ais: pd.DataFrame,
    port_name: str,
) -> List[Tuple[int, float]]:
    """
    使用预计算列找有效靠泊点：
    Port_Name == port_name 且 Mode == 'Berth' 且 berth_port_name == port_name。
    使用 pandas 向量化比较，避免逐元素 str() 循环。
    """
    pn_match = (ais["Port_Name"].astype(str) == port_name) & ais["Port_Name"].notna()
    mode_match = (ais["Mode"] == "Berth")
    berth_match = (ais["berth_port_name"].astype(str) == port_name) & ais["berth_port_name"].notna()

    mask = (pn_match & mode_match & berth_match).values
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return []
    times = ais["UTCtime"].astype(float).values
    return [(int(idx), float(times[idx])) for idx in indices]


def _expand_buffer_backward(
    port_name_arr: np.ndarray,
    berth_idx: int,
    port_name: str,
    gap_limit: int = 10,
) -> int:
    i = berth_idx
    while i > 0:
        prev_pn = port_name_arr[i - 1]
        if pd.notna(prev_pn) and str(prev_pn) == port_name:
            i -= 1
        elif pd.isna(prev_pn) or str(prev_pn) == "":
            found_port_again = False
            j = i - 1
            gap_count = 0
            while j >= 0 and gap_count < gap_limit:
                pn_j = port_name_arr[j]
                if pd.notna(pn_j) and str(pn_j) == port_name:
                    found_port_again = True
                    break
                elif pd.notna(pn_j) and str(pn_j) != port_name:
                    break
                gap_count += 1
                j -= 1
            if found_port_again:
                i = j
            else:
                break
        else:
            break
    return i


def _expand_buffer_forward(
    port_name_arr: np.ndarray,
    berth_idx: int,
    port_name: str,
    n: int,
    gap_limit: int = 10,
) -> int:
    i = berth_idx
    while i < n - 1:
        next_pn = port_name_arr[i + 1]
        if pd.notna(next_pn) and str(next_pn) == port_name:
            i += 1
        elif pd.isna(next_pn) or str(next_pn) == "":
            found_port_again = False
            j = i + 1
            gap_count = 0
            while j < n and gap_count < gap_limit:
                pn_j = port_name_arr[j]
                if pd.notna(pn_j) and str(pn_j) == port_name:
                    found_port_again = True
                    break
                elif pd.notna(pn_j) and str(pn_j) != port_name:
                    break
                gap_count += 1
                j += 1
            if found_port_again:
                i = j
            else:
                break
        else:
            break
    return i


def _compute_berth_hours_from_tags(
    times: np.ndarray,
    modes: np.ndarray,
    berth_pn_arr: np.ndarray,
    port_name: str,
    berth_gap_tolerance_sec: float,
) -> float:
    """
    使用预计算列计算靠泊时长。
    at_berth = Mode=='Berth' 且 berth_port_name==port_name。
    """
    n = len(times)
    if n < 2:
        return 0.0

    # 向量化比较，避免逐元素 str() 循环
    berth_pn_series = pd.Series(berth_pn_arr)
    berth_match = (berth_pn_series.astype(str) == port_name) & berth_pn_series.notna()
    at_berth = (modes == "Berth") & berth_match.values

    first_berth_indices = np.flatnonzero(at_berth)
    if len(first_berth_indices) == 0:
        return 0.0
    first_berth_idx = int(first_berth_indices[0])

    total_berth_sec = 0.0
    in_gap = False
    gap_start_time = 0.0

    for idx in range(first_berth_idx, n - 1):
        dt = times[idx + 1] - times[idx]
        if dt <= 0 or dt > 86400 * 2:
            continue
        if at_berth[idx]:
            in_gap = False
            total_berth_sec += dt
        else:
            if not in_gap:
                in_gap = True
                gap_start_time = times[idx]
            gap_duration = times[idx + 1] - gap_start_time
            if gap_duration > berth_gap_tolerance_sec:
                break

    return max(0.0, total_berth_sec / 3600.0)


def _first_anchor_before_berth_from_tags(
    times: np.ndarray,
    modes: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    t_berth0: float,
    port_lon: float,
    port_lat: float,
    max_nm_from_port: float,
) -> Optional[float]:
    """在首次靠泊之前找 Mode=Anchored 且距港口参考点 ≤ max_nm 的最早 AIS 点。"""
    max_km = _nm_to_km(max_nm_from_port)
    before_mask = times < t_berth0
    anchor_mask = (modes == "Anchored")
    candidate_mask = before_mask & anchor_mask
    candidate_indices = np.flatnonzero(candidate_mask)
    if len(candidate_indices) == 0:
        return None
    dists = haversine_km(lons[candidate_indices], lats[candidate_indices], port_lon, port_lat)
    valid = dists <= max_km
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        return None
    return float(times[candidate_indices[valid_indices[0]]])


def _build_port_visits_berth_first(
    ais: pd.DataFrame,
    port_coor: pd.DataFrame,
    berth_gap_tolerance_sec: float,
    anchor_max_km: float = 20.0,
) -> List[Dict[str, Any]]:
    """
    berth-first 港口访问识别。

    参数：
      anchor_max_km — 远锚泊倒推的最大距离阈值（km，默认 20）。
                      用途：当 port_buffer 覆盖不到的远锚泊（仅 Mode=Anchored）
                      在首次靠泊前 ≤ anchor_max_km 内出现时，把 InPort_Hours
                      起点拉到该锚泊点。专门用于 shipping pipeline 计算等待时间。
                      emission pipeline 不消费 t_anchor0，可不关心此参数。
    """
    port_names_in_ais = set()
    for pn in ais["Port_Name"].dropna().unique():
        pn_str = str(pn)
        if pn_str not in CANAL_PORT_NAMES:
            port_names_in_ais.add(pn_str)

    all_visits: List[Dict[str, Any]] = []

    # 预提取数组
    port_name_arr = ais["Port_Name"].values
    ais_lons = ais["Longitude"].astype(float).values
    ais_lats = ais["Latitude"].astype(float).values
    ais_times = ais["UTCtime"].astype(float).values
    ais_modes = ais["Mode"].values
    ais_berth_pn = ais["berth_port_name"].values
    n_ais = len(ais)

    for port_name in port_names_in_ais:
        berth_events = _find_all_berth_events_from_tags(ais, port_name)
        if not berth_events:
            continue

        # ---------------------------------------------------------------
        # 优化：不对每个靠泊点单独扩展，而是先将靠泊点按连续性分组，
        # 每组只用第一个和最后一个靠泊点的索引做扩展。
        # 「连续」定义：两个相邻靠泊点的索引差 ≤ gap_index_limit
        # （中间可能有少量 At_Sea/Anchored 点）
        # ---------------------------------------------------------------
        berth_indices = [idx for (idx, _) in berth_events]
        # 将靠泊点按索引间隔分组
        gap_index_limit = 100  # 中间隔 100 个以上非靠泊点才算不同次停靠
        berth_groups: List[List[int]] = []
        current_group = [berth_indices[0]]
        for k in range(1, len(berth_indices)):
            if berth_indices[k] - berth_indices[k - 1] <= gap_index_limit:
                current_group.append(berth_indices[k])
            else:
                berth_groups.append(current_group)
                current_group = [berth_indices[k]]
        berth_groups.append(current_group)

        # 每组只从首尾靠泊点扩展
        raw_ranges: List[Tuple[int, int]] = []
        for group in berth_groups:
            first_berth_idx = group[0]
            last_berth_idx = group[-1]
            start_idx = _expand_buffer_backward(port_name_arr, first_berth_idx, port_name)
            end_idx = _expand_buffer_forward(port_name_arr, last_berth_idx, port_name, n_ais)
            raw_ranges.append((start_idx, end_idx))

        # 合并重叠区间
        raw_ranges.sort(key=lambda x: x[0])
        merged_ranges: List[Tuple[int, int]] = []
        for s, e in raw_ranges:
            if merged_ranges and s <= merged_ranges[-1][1] + 1:
                merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], e))
            else:
                merged_ranges.append((s, e))

        for (start_idx, end_idx) in merged_ranges:
            if end_idx < start_idx:
                continue
            sl = slice(start_idx, end_idx + 1)
            sl_times = ais_times[sl]
            sl_modes = ais_modes[sl]
            sl_berth_pn = ais_berth_pn[sl]
            sl_lons = ais_lons[sl]
            sl_lats = ais_lats[sl]

            if len(sl_times) == 0:
                continue

            t_enter = float(sl_times[0])
            t_leave = float(sl_times[-1])

            # 找首次靠泊时间
            sl_berth_pn_series = pd.Series(sl_berth_pn)
            berth_match_sl = (sl_berth_pn_series.astype(str) == port_name) & sl_berth_pn_series.notna()
            at_berth_sl = (sl_modes == "Berth") & berth_match_sl.values
            berth_idx_arr = np.flatnonzero(at_berth_sl)
            first_berth_t = float(sl_times[berth_idx_arr[0]]) if len(berth_idx_arr) > 0 else None

            # 锚泊倒推：在首次靠泊前找 Mode=Anchored 且 ≤ anchor_max_km 的最早点
            # km → nm 换算（_first_anchor_before_berth_from_tags 内部用 nm 接口）
            pref = _port_reference_lonlat(port_coor, port_name)
            t_anchor0: Optional[float] = None
            if pref is not None and first_berth_t is not None and anchor_max_km > 0:
                anchor_max_nm = anchor_max_km / 1.852  # km → nm
                t_anchor0 = _first_anchor_before_berth_from_tags(
                    sl_times, sl_modes, sl_lons, sl_lats,
                    first_berth_t, pref[0], pref[1], anchor_max_nm,
                )

            t_port_start = t_anchor0 if t_anchor0 is not None else t_enter
            stay_h = (t_leave - t_port_start) / 3600.0

            berth_h = _compute_berth_hours_from_tags(
                sl_times, sl_modes, sl_berth_pn, port_name, berth_gap_tolerance_sec
            )
            wait_h = max(0.0, stay_h - berth_h)

            dep_dt = pd.to_datetime(t_leave, unit="s", utc=True)
            dep_yyyymm = int(dep_dt.strftime("%Y%m"))

            all_visits.append({
                "Port_Name": port_name,
                "Visit_Count": 1,
                "InPort_Hours": stay_h,
                "Waiting_Hours": wait_h,
                "Berth_Hours": berth_h,
                "Canal_Sailing_Hours": 0.0,
                "Depart_UTC": t_leave,
                "Arrive_UTC": t_port_start,
                "Depart_YearMonth": dep_yyyymm,
                "_start_idx": start_idx,
                "_end_idx": end_idx,
            })

    all_visits.sort(key=lambda x: x["Arrive_UTC"])
    return all_visits


# ===========================================================================
# 运河（使用预计算列）
# ===========================================================================

def _build_canal_visits_from_tags(
    ais: pd.DataFrame,
    port_coor: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    运河挂靠计算（端口缓冲区版）：
    1. canal_corridor 确认船舶通过了运河（是/否）
    2. 找船舶第一次进入端口缓冲区的时刻 t_enter_1（航行/靠泊开始）
    3. 找船舶第二次进入另一端端口缓冲区的时刻 t_enter_2（航行/靠泊结束）
    4. 航行时间（靠泊时间）= t_enter_2 - t_enter_1
    5. 等待时间 = 从 t_enter_1 向前搜索在等待子港缓冲区内的累计时长
    6. 在港时间 = 等待时间 + 航行时间
    """
    canal_visits: List[Dict[str, Any]] = []

    times = ais["UTCtime"].astype(float).values
    modes = ais["Mode"].values
    canal_corridor_arr = ais["canal_corridor"].values
    canal_waiting_arr = ais["canal_waiting"].values
    canal_ep_arr = ais["canal_endpoint"].values          # endpoint_name，如 'Suez_N'
    canal_ep_canal_arr = ais["canal_endpoint_canal"].values  # canal_name，如 'Suez Canal'

    for cname in CANAL_PORT_NAMES:
        # 1. 确认通过运河
        cc_series = pd.Series(canal_corridor_arr)
        corridor_mask = ((cc_series.astype(str) == cname) & cc_series.notna()).values
        if not corridor_mask.any():
            continue

        runs = _bool_runs(corridor_mask)
        has_valid_run = False
        for a, b in runs:
            if times[b - 1] - times[a] >= CANAL_MIN_PASSAGE_SEC:
                has_valid_run = True
                break
        if not has_valid_run:
            continue

        # 2. 找端口缓冲区进入事件
        # 该运河的端口名列表
        ep_names = [ep[0] for ep in CANAL_ENDPOINTS.get(cname, [])]
        if len(ep_names) < 2:
            continue

        # 找所有在该运河端口缓冲区内的点
        ep_canal_series = pd.Series(canal_ep_canal_arr)
        ep_in_canal = ((ep_canal_series.astype(str) == cname) & ep_canal_series.notna()).values
        ep_indices = np.flatnonzero(ep_in_canal)

        if len(ep_indices) == 0:
            continue

        # 找第一次进入端口缓冲区的端口名和时刻
        first_ep_idx = ep_indices[0]
        first_ep_name = str(canal_ep_arr[first_ep_idx])
        t_enter_1 = float(times[first_ep_idx])

        # 找第二次进入另一端端口缓冲区的时刻
        # 即在 ep_indices 中找第一个端口名 != first_ep_name 的点
        t_enter_2 = None
        second_ep_idx = None
        for k in range(len(ep_indices)):
            idx = ep_indices[k]
            ep_name = str(canal_ep_arr[idx])
            if ep_name != first_ep_name:
                t_enter_2 = float(times[idx])
                second_ep_idx = idx
                break

        if t_enter_2 is None:
            # 没找到另一端，可能数据不完整，跳过
            continue

        # 3. 航行时间（靠泊时间）= t_enter_2 - t_enter_1
        transit_h = (t_enter_2 - t_enter_1) / 3600.0

        # 4. 航行时间细分：transit 段内 Mode=Sea 的时段
        transit_start = min(first_ep_idx, second_ep_idx)
        transit_end = max(first_ep_idx, second_ep_idx) + 1
        seg_modes = modes[transit_start:transit_end]
        seg_times = times[transit_start:transit_end]
        if len(seg_times) >= 2:
            dt = np.diff(seg_times)
            mode_prev = seg_modes[:-1]
            valid = (dt > 0) & (dt <= 86400 * 2)
            is_sea = (mode_prev == "Sea")
            sail_h = float(np.sum(dt[valid & is_sea])) / 3600.0
        else:
            sail_h = 0.0

        # 5. 等待时间：从 t_enter_1 向前搜索在等待子港缓冲区内的累计时长
        cw_series = pd.Series(canal_waiting_arr)
        waiting_mask = ((cw_series.astype(str) == cname) & cw_series.notna()).values
        before_mask = (times < t_enter_1)
        before_indices = np.flatnonzero(before_mask)[::-1]  # 时间倒序

        total_wait_sec = 0.0
        earliest_wait_time = t_enter_1
        consecutive_outside = 0
        gap_limit = 10

        for k_idx in range(len(before_indices)):
            idx = before_indices[k_idx]
            if waiting_mask[idx]:
                consecutive_outside = 0
                t_k = float(times[idx])
                earliest_wait_time = t_k
                if k_idx == 0:
                    dt_val = t_enter_1 - t_k
                else:
                    dt_val = float(times[before_indices[k_idx - 1]]) - t_k
                if 0 < dt_val <= 86400 * 2:
                    total_wait_sec += dt_val
            else:
                consecutive_outside += 1
                if consecutive_outside >= gap_limit:
                    break

        wait_h = total_wait_sec / 3600.0

        # 6. 汇总
        inport_h = wait_h + transit_h        # 在港时间 = 等待 + 航行
        berth_h = transit_h                   # 靠泊时间 = 航行时间（通过运河的过程）

        dep_dt = pd.to_datetime(t_enter_2, unit="s", utc=True)
        dep_yyyymm = int(dep_dt.strftime("%Y%m"))

        canal_visits.append({
            "Port_Name": cname,
            "Visit_Count": 1,
            "InPort_Hours": inport_h,
            "Waiting_Hours": wait_h,
            "Berth_Hours": berth_h,
            "Canal_Sailing_Hours": sail_h,
            "Depart_UTC": t_enter_2,
            "Arrive_UTC": earliest_wait_time,
            "Depart_YearMonth": dep_yyyymm,
            "_start_idx": int(min(first_ep_idx, second_ep_idx)),
            "_end_idx": int(max(first_ep_idx, second_ep_idx)),
        })

    return canal_visits


# ===========================================================================
# 边（航段）构建
# ===========================================================================

def _build_edges_from_visits(
    visits: List[Dict[str, Any]],
    mmsi: int,
    dwt: float,
) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    if len(visits) < 2:
        return edges
    for k in range(len(visits) - 1):
        from_port = visits[k]["Port_Name"]
        to_port = visits[k + 1]["Port_Name"]
        if str(from_port) != str(to_port):
            edges.append({
                "From_Port": str(from_port),
                "To_Port": str(to_port),
                "MMSI": mmsi,
                "Total_Tonnage_DWT": dwt,
            })
    return edges


# ===========================================================================
# 主处理函数
# ===========================================================================

def process_ship_shipping_network(
    ship_ais_data: pd.DataFrame,
    ship_info: Any,
    port_coor: pd.DataFrame,
    port_buffers: gpd.GeoDataFrame,
    berth_distance_threshold_km: float,
    canal_coor: pd.DataFrame | None = None,
    canal_proximity_km: float = 10.0,
    berth_gap_tolerance_sec: float = 600.0,
    berth_buffers: gpd.GeoDataFrame | None = None,
    canal_corridor_buffers: gpd.GeoDataFrame | None = None,
    canal_waiting_buffers: gpd.GeoDataFrame | None = None,
    canal_endpoint_buffers: gpd.GeoDataFrame | None = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "success": False,
        "mmsi": None,
        "visits": None,
        "edges": None,
        "error_message": None,
    }

    try:
        if ship_info is None:
            result["error_message"] = "无船舶静态信息"
            return result

        if isinstance(ship_info, pd.DataFrame):
            ship_info = ship_info.iloc[0]

        mmsi = ship_info.get("MMSI")
        result["mmsi"] = mmsi
        dwt = _ship_dwt(ship_info)
        design_speed = float(ship_info.get("Service_Speed", 15.0))
        me_power = float(ship_info.get("ME", 5000.0))
        # 与 emission 对齐的 MEPower 修正参数
        ship_type_val = ship_info.get("Ship_Type", None)
        size_val = ship_info.get("Size", None)
        me_cf_w = float(ship_info.get("ME_CF_W", 1.0) or 1.0)
        me_cf_f = float(ship_info.get("ME_CF_F", 1.0) or 1.0)

        ais = ship_ais_data.copy()
        ais = ais.drop_duplicates(subset=["UTCtime"])
        ais = ais.sort_values("UTCtime").reset_index(drop=True)
        ais = ais.dropna(subset=["Latitude", "Longitude", "UTCtime"])
        ais = ais[
            (ais["Latitude"].between(-90, 90)) & (ais["Longitude"].between(-180, 180))
        ].reset_index(drop=True)

        if len(ais) <= 2:
            result["error_message"] = "AIS 记录过少"
            return result

        if "Speed" not in ais.columns:
            result["error_message"] = "AIS 缺少 Speed 列"
            return result

        ais["Speed"] = pd.to_numeric(ais["Speed"], errors="coerce")
        ais = interpolate_ais_r_style(ais)

        ais.loc[ais["Speed"] > 1.5 * design_speed, "Speed"] = design_speed
        ais["TimeInv"] = ais["UTCtime"].diff().fillna(0)
        ais = ais[ais["TimeInv"] < 3600 * 24].reset_index(drop=True)
        if len(ais) <= 2:
            result["error_message"] = "插值/过滤时间间隔后 AIS 过少"
            return result

        me_load = _me_load_proxy(
            ais["Speed"].values, design_speed, me_power,
            ship_type=ship_type_val, size=size_val,
            me_cf_w=me_cf_w, me_cf_f=me_cf_f,
        )
        ais["ME_load"] = me_load
        ais["Mode"] = _mode_from_speed_me(ais["Speed"].values, me_load, design_speed)

        # =============================================================
        # 一次性 sjoin 预计算所有空间关系
        # =============================================================
        ais = precompute_spatial_tags(
            ais, port_buffers, berth_buffers, canal_corridor_buffers, canal_waiting_buffers,
            canal_endpoint_buffers
        )

        # 运河标记（R 兼容：全轨迹最近点标记）
        ais = strip_canal_waiting_subports(ais)
        ais = apply_canal_tags_r_style(ais, canal_coor, threshold_km=canal_proximity_km)
        ais["Port_Event"] = np.where(ais["Port_Name"].notna(), "In_Port", "At_Sea")

        # 普通港口挂靠
        port_visits = _build_port_visits_berth_first(
            ais, port_coor, berth_gap_tolerance_sec
        )

        # 运河挂靠
        canal_visits = _build_canal_visits_from_tags(ais, port_coor)

        # 合并
        all_visits = port_visits + canal_visits
        all_visits.sort(key=lambda x: x["Arrive_UTC"])

        for v in all_visits:
            v["MMSI"] = mmsi
            v["DWT"] = dwt

        edges_list = _build_edges_from_visits(all_visits, mmsi, dwt)

        for v in all_visits:
            v.pop("_start_idx", None)
            v.pop("_end_idx", None)
            v.pop("Arrive_UTC", None)

        visits = pd.DataFrame(all_visits) if all_visits else pd.DataFrame()
        edges = pd.DataFrame(edges_list) if edges_list else pd.DataFrame()

        result["success"] = True
        result["visits"] = visits
        result["edges"] = edges
        return result

    except Exception as e:
        result["error_message"] = str(e)
        return result


# ===========================================================================
# 聚合
# ===========================================================================

def aggregate_port_node_features(visits: pd.DataFrame, num_months: int) -> pd.DataFrame:
    if visits is None or len(visits) == 0:
        return pd.DataFrame(
            columns=[
                "Port_Name", "Avg_Calls_Per_Month", "Avg_DWT", "Avg_InPort_Hours",
                "Avg_Waiting_Hours", "Avg_Berth_Hours", "Avg_Canal_Sailing_Hours",
                "Total_Visits", "Total_DWT",
            ]
        )
    visits = visits.copy()
    if "Berth_Hours" not in visits.columns:
        visits["Berth_Hours"] = 0.0
    if "Canal_Sailing_Hours" not in visits.columns:
        visits["Canal_Sailing_Hours"] = 0.0
    nm = max(int(num_months), 1)
    g = visits.groupby("Port_Name", as_index=False)
    out = g.agg(
        Total_Visits=("Visit_Count", "sum"),
        Total_DWT=("DWT", "sum"),
        Avg_DWT=("DWT", "mean"),
        Avg_InPort_Hours=("InPort_Hours", "mean"),
        Avg_Waiting_Hours=("Waiting_Hours", "mean"),
        Avg_Berth_Hours=("Berth_Hours", "mean"),
        Avg_Canal_Sailing_Hours=("Canal_Sailing_Hours", "mean"),
    )
    out["Avg_Calls_Per_Month"] = out["Total_Visits"] / nm
    return out[[
        "Port_Name", "Avg_Calls_Per_Month", "Avg_DWT", "Total_DWT",
        "Avg_InPort_Hours", "Avg_Waiting_Hours", "Avg_Berth_Hours",
        "Avg_Canal_Sailing_Hours", "Total_Visits",
    ]]


def aggregate_edge_tonnage(edges: pd.DataFrame) -> pd.DataFrame:
    if edges is None or len(edges) == 0:
        return pd.DataFrame(columns=["From_Port", "To_Port", "Total_Tonnage_DWT", "Voyage_Count"])
    g = edges.groupby(["From_Port", "To_Port"], as_index=False).agg(
        Total_Tonnage_DWT=("Total_Tonnage_DWT", "sum"),
        Voyage_Count=("MMSI", "count"),
    )
    return g