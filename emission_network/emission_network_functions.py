"""
################################################################################
# 项目: 船舶排放网络分析
# 子项目2: 排放网络计算  
# 文件: emission_network_functions_v3_final.py
# 版本: 最终完全修复版（与R代码100%一致）
# 日期: 2025-01-15
#
# 修复内容（相比v2_fixed）：
# 1. 添加：主机负载<7%时排放归0（R代码第792-800行）
# 2. 确认：所有排放计算与R代码完全一致
# 3. 优化：代码结构和注释
################################################################################
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import geopandas as gpd
from shapely.geometry import Point, Polygon
from matplotlib.path import Path
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# berth-first 双 buffer 港口判定（与 shipping pipeline 对齐）
# - 输入要求：ais 含 Port_Name（来自 port_buffer sjoin）+ berth_port_name（来自 berth_buffer sjoin）
#            + Mode + UTCtime + Longitude + Latitude；port_coor 含 Port.Name + Lon_port + Lat_port。
# - 返回：visits 列表，每条含 _start_idx / _end_idx / Port_Name / InPort_Hours / Berth_Hours / ...
# 这里直接复用 shipping_network_functions 内部实现，避免重复维护两套靠泊推导逻辑。
from shipping_network_functions import (
    _build_canal_visits_from_tags,
    _build_port_visits_berth_first as _shipping_visits_berth_first,
    apply_canal_tags_r_style,
    interpolate_ais_r_style,
    precompute_spatial_tags,
    strip_canal_waiting_subports,
)


def _norm_mode(mode) -> str:
    """AIS Mode 规整字符串。"""
    if mode is None:
        return ""
    try:
        if isinstance(mode, (float, np.floating)) and np.isnan(mode):
            return ""
    except Exception:
        pass
    return str(mode).strip()


def _mode_is_at_sea_for_waiting(mode) -> bool:
    """在港挂靠段内需扣减的『海上航行』AIS 状态（本流水线推断为 Sea，原始数据可能出现的 At Sea 同等处理）。"""
    m = _norm_mode(mode).lower().replace("_", " ")
    return m in ("sea", "at sea")


def is_clean_fuel_ship(fuel) -> bool:
    """
    清洁燃料：LNG、MeOH、氨（NH3 / Ammonia 等）。
    与 ship_info.Fuel_Type 字符串大小写/空格无关。
    """
    if fuel is None:
        return False
    try:
        if pd.isna(fuel):
            return False
    except Exception:
        pass
    fs = str(fuel).strip()
    if fs == "氨":
        return True
    f = "".join(fs.upper().split())
    if f in ("LNG",):
        return True
    if f in ("MEOH", "METHANOL"):
        return True
    if f in ("NH3", "AMMONIA"):
        return True
    return False


def _exclude_berth_anchored_for_speed(mode) -> bool:
    """港内统计船速：排除靠泊(Berth)与锚泊(Anchored)。"""
    m = _norm_mode(mode).lower()
    return m in ("berth", "anchored")


def _policy_zone_labels(points: pd.DataFrame, policy_zones) -> pd.Series:
    """为 AIS 点赋正式政策区标签；相邻区重叠时稳定取首个 zone_id。"""
    labels = pd.Series("", index=points.index, dtype=object)
    if policy_zones is None or len(policy_zones) == 0 or len(points) == 0:
        return labels
    pz = policy_zones[["zone_id", "geometry"]]
    if pz.crs is None:
        pz = pz.set_crs("EPSG:4326")
    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(points["Longitude"], points["Latitude"]),
        index=points.index,
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, pz, predicate="within", how="inner")
    if len(joined):
        joined = joined[~joined.index.duplicated(keep="first")]
        labels.loc[joined.index] = joined["zone_id"].astype(str).values
    return labels


# ============================================================================
# 辅助函数1：计算两点间Haversine距离
# ============================================================================
def haversine_dist(lon1, lat1, lon2, lat2):
    """
    计算两点间的Haversine距离（km）
    """
    lon1_rad = np.radians(lon1)
    lat1_rad = np.radians(lat1)
    lon2_rad = np.radians(lon2)
    lat2_rad = np.radians(lat2)
    
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = 6371 * c
    
    return distance


# ============================================================================
# 辅助函数2：计算平均速度
# ============================================================================
def calculate_avg_speed(start_time, end_time, ship_ais):
    """计算指定时间段内的平均速度"""
    segment = ship_ais[(ship_ais['UTCtime'] >= start_time) & 
                        (ship_ais['UTCtime'] <= end_time)]
    if len(segment) > 0:
        return segment['Speed'].mean()
    else:
        return np.nan


# ============================================================================
# 辅助函数3：计算航程距离
# ============================================================================
def calculate_distance(start_time, end_time, ship_ais):
    """计算指定时间段内的航程距离"""
    segment = ship_ais[(ship_ais['UTCtime'] >= start_time) & 
                        (ship_ais['UTCtime'] <= end_time)]
    if len(segment) > 1:
        distances = []
        for i in range(len(segment) - 1):
            dist = haversine_dist(
                segment.iloc[i]['Longitude'],
                segment.iloc[i]['Latitude'],
                segment.iloc[i+1]['Longitude'],
                segment.iloc[i+1]['Latitude']
            )
            distances.append(dist)
        return sum(distances)
    else:
        return np.nan


def _track_distance_km_from_ais(ais: pd.DataFrame, i0: int, i1: int) -> float:
    """沿 AIS 折线累计 Haversine 距离（km），索引闭区间 [i0, i1]。"""
    if i1 < i0 or i0 < 0 or i1 >= len(ais):
        return 0.0
    seg = ais.iloc[i0 : i1 + 1]
    if len(seg) < 2:
        return 0.0
    lons = seg["Longitude"].astype(float).values
    lats = seg["Latitude"].astype(float).values
    total = 0.0
    for j in range(len(seg) - 1):
        total += haversine_dist(lons[j], lats[j], lons[j + 1], lats[j + 1])
    return float(total)


def _build_emission_network_from_visits(
    ais: pd.DataFrame,
    visits: list,
    pollutants: list,
    mmsi,
    dwt: float,
    shiptype: str,
    fuel,
    gross_tonnage=np.nan,
    build_year=np.nan,
    imo=np.nan,
    gt_source="missing",
    gt_match_method="",
) -> tuple:
    """
    由 shipping 流水线 visit 列表汇总节点排放、边排放与航段距离/时间。
    港口/运河在港、等待、周转时间与 shipping_network_functions 一致。
    """
    port_emissions_list = []
    edge_emissions_list = []
    port_operational_visits_list = []
    edge_voyage_list = []
    ops_calls_list = []

    if not visits:
        return (port_emissions_list, edge_emissions_list,
                port_operational_visits_list, edge_voyage_list, ops_calls_list)

    # 只在严格靠泊点打挂靠标签；随后由城市集水区函数一次性做空间归属。
    # object 列用 None 初始化，避免空字符串被误认为有效 Call_ID。
    ais["OPS_Call_ID"] = None
    ais["OPS_Port_Name"] = None
    ais["OPS_AE_kWh"] = 0.0
    ais["OPS_BO_kWh"] = 0.0

    visits = sorted(visits, key=lambda v: float(v["Arrive_UTC"]))
    n_ais = len(ais)
    times = ais["UTCtime"].astype(float).values

    for v in visits:
        s = int(v["_start_idx"])
        e = int(v["_end_idx"])
        if e < s or s < 0 or e >= n_ais:
            continue
        port_data = ais.iloc[s : e + 1]
        if len(port_data) == 0:
            continue

        pname = v["Port_Name"]
        in_port_h = float(v.get("InPort_Hours", 0.0))
        wait_h = float(v.get("Waiting_Hours", 0.0))
        berth_h = float(v.get("Berth_Hours", 0.0))
        canal_sail_h = float(v.get("Canal_Sailing_Hours", 0.0))

        at_sea_mask = port_data["Mode"].map(_mode_is_at_sea_for_waiting)
        at_sea_time = port_data.loc[at_sea_mask, "TimeInv"].sum() / 3600.0
        mov_mask = ~port_data["Mode"].map(_exclude_berth_anchored_for_speed)
        spd_series = port_data.loc[mov_mask, "Speed"]
        avg_spd_mov = float(spd_series.mean()) if len(spd_series) > 0 else np.nan

        if "Depart_YearMonth" in v and v["Depart_YearMonth"] is not None:
            ym = str(int(v["Depart_YearMonth"]))
        else:
            ym = pd.to_datetime(
                port_data["UTCtime"].iloc[0], unit="s", utc=True
            ).strftime("%Y%m")

        # 岸电正式口径：仅 Mode=Berth 且 berth/port 标签为同一港口的点。
        # visit 自带的 Berth_Hours 只作既有运营统计；正式岸电门槛与用能均由
        # 下列逐点严格掩码重新计算，避免把锚泊/机动时间混入靠泊。
        if "berth_port_name" in port_data.columns:
            berth_mask = (
                (port_data["Mode"] == "Berth")
                & port_data["berth_port_name"].notna()
                & (port_data["berth_port_name"].astype(str) == str(pname))
            )
            if "Port_Name" in port_data.columns:
                berth_mask &= (
                    port_data["Port_Name"].notna()
                    & (port_data["Port_Name"].astype(str) == str(pname))
                )
        else:
            berth_mask = port_data["Mode"] == "Berth"
        berth_idx = port_data.index[berth_mask]
        strict_berth_h = float(
            port_data.loc[berth_mask, "TimeInv"].sum() / 3600.0
        )
        ae_kwh = float(
            (port_data.loc[berth_mask, "AE_Power"]
             * port_data.loc[berth_mask, "TimeInv"] / 3600.0).sum()
        )
        bo_kwh = float(
            (port_data.loc[berth_mask, "BO_Power"]
             * port_data.loc[berth_mask, "TimeInv"] / 3600.0).sum()
        )
        arrive_utc = float(v.get("Arrive_UTC", times[s]))
        depart_utc = float(v.get("Depart_UTC", times[e]))
        safe_port = "".join(
            ch if ch.isalnum() else "-" for ch in str(pname)
        ).strip("-") or "PORT"
        call_id = f"{mmsi}_{int(arrive_utc)}_{int(depart_utc)}_{safe_port}"
        if len(berth_idx) > 0:
            ais.loc[berth_idx, "OPS_Call_ID"] = call_id
            ais.loc[berth_idx, "OPS_Port_Name"] = str(pname)
            ais.loc[berth_idx, "OPS_AE_kWh"] = (
                ais.loc[berth_idx, "AE_Power"]
                * ais.loc[berth_idx, "TimeInv"] / 3600.0
            )
            ais.loc[berth_idx, "OPS_BO_kWh"] = (
                ais.loc[berth_idx, "BO_Power"]
                * ais.loc[berth_idx, "TimeInv"] / 3600.0
            )

        ops_row = {
            "Call_ID": call_id,
            "YearMonth": ym,
            "Port_Name": pname,
            "MMSI": mmsi,
            "IMO": imo,
            "Ship_Type": shiptype,
            "Fuel_Type": fuel,
            "Gross_Tonnage": gross_tonnage,
            "GT_Missing": bool(pd.isna(gross_tonnage)),
            "GT_Source": gt_source,
            "GT_Match_Method": gt_match_method,
            "DWT": dwt,
            "Build_Year": build_year,
            "Arrive_UTC": arrive_utc,
            "Depart_UTC": depart_utc,
            "Strict_Berth_Hours": strict_berth_h,
            "OPS_AE_kWh": ae_kwh,
            "OPS_BO_kWh": bo_kwh,
            "OPS_Energy_kWh": ae_kwh + bo_kwh,
            "Eligible_GT5000_Berth2h": bool(
                str(shiptype).strip().casefold() == "container"
                and pd.notna(gross_tonnage)
                and float(gross_tonnage) > 5000.0
                and strict_berth_h >= 2.0
            ),
        }
        for pol in ("SO2", "PM", "NOx"):
            ops_row[f"Berth_{pol}_g"] = float(
                port_data.loc[berth_mask, f"Total_{pol}"].sum()
            )
        for col in ("D_SO2_SECA", "D_PM_SECA", "D_NOx_NECA"):
            ops_row[col] = float(
                port_data.loc[berth_mask, col].sum()
            ) if col in port_data.columns else 0.0
        zones = sorted({
            str(z) for z in port_data.loc[berth_mask, "PZ_Zone"].tolist()
            if pd.notna(z) and str(z) != ""
        }) if "PZ_Zone" in port_data.columns else []
        ops_row["Policy_Zone_IDs"] = ";".join(zones)
        ops_calls_list.append(ops_row)

        # 港口在港停留较短，跨月罕见：整次挂靠的排放与运营统计归到该次的月份 ym
        row = {
            "Port_Name": pname,
            "MMSI": mmsi,
            "YearMonth": ym,
            "Turnaround_Hours": in_port_h,
            "Berth_Hours": berth_h,
            "DWT": dwt,
            "Visit_Count": 1,
        }
        for pol in pollutants:
            row[pol] = port_data[f"Total_{pol}"].sum()
        port_emissions_list.append(row)

        port_operational_visits_list.append(
            {
                "YearMonth": ym,
                "Port_Name": pname,
                "MMSI": mmsi,
                "Ship_Type": shiptype,
                "Fuel_Type": fuel,
                "DWT": dwt,
                # 逐次挂靠时刻(周度功能序列用;绿色韧性研究)
                "Arrive_UTC": float(v.get("Arrive_UTC", times[s])),
                "Depart_UTC": float(v.get("Depart_UTC", times[e])),
                "InPort_Hours": in_port_h,
                "Berth_Hours": berth_h,
                "AtSea_Hours": canal_sail_h if canal_sail_h > 0 else at_sea_time,
                "Waiting_Hours": wait_h,
                "Canal_Sailing_Hours": canal_sail_h,
                "Avg_Speed_InPort_Moving_kn": avg_spd_mov,
            }
        )

    for k in range(len(visits) - 1):
        v0, v1 = visits[k], visits[k + 1]
        fp = str(v0["Port_Name"])
        tp = str(v1["Port_Name"])
        if fp == tp:
            continue
        i_depart = int(v0["_end_idx"])
        i_arrive = int(v1["_start_idx"])
        if i_depart < 0 or i_arrive >= n_ais:
            continue
        seg = ais.iloc[i_depart : i_arrive + 1]
        if len(seg) == 0:
            continue
        sailing_h = max(0.0, (float(times[i_arrive]) - float(times[i_depart])) / 3600.0)
        dist_km = _track_distance_km_from_ais(ais, i_depart, i_arrive)

        # 航段排放按逐点时间戳分月：跨月航段分别计入各月，月度守恒。
        seg_ym = pd.to_datetime(seg["UTCtime"], unit="s", utc=True).dt.strftime("%Y%m")
        for ym_e, g in seg.groupby(seg_ym.values):
            edge_row = {"From_Port": fp, "To_Port": tp, "MMSI": mmsi, "YearMonth": ym_e}
            for pol in pollutants:
                edge_row[pol] = g[f"Total_{pol}"].sum()
            edge_emissions_list.append(edge_row)
        # 航次级统计（航次数/航时/航距）为每航次整体属性，不可拆分：整次归到「到港月」。
        arr_ym = pd.to_datetime(
            float(times[i_arrive]), unit="s", utc=True
        ).strftime("%Y%m")
        edge_voyage_list.append(
            {
                "From_Port": fp,
                "To_Port": tp,
                "MMSI": mmsi,
                "YearMonth": arr_ym,
                "Sailing_Hours": sailing_h,
                "Sailing_Distance_km": dist_km,
            }
        )

    return (port_emissions_list, edge_emissions_list,
            port_operational_visits_list, edge_voyage_list, ops_calls_list)


# ============================================================================
# 辅助函数3.5：周度港口排放（绿色韧性研究；港口级 × ISO周 × 工况）
# ============================================================================
_WEEKLY_POLS = ("SO2", "PM", "NOx", "CO2")


def build_weekly_port_emissions(ship_ais, visits, port_coor, anchor_km):
    """点级港口归属 → 港口 × ISO周 × 工况 排放聚合。

    归属规则(两层):
      1. visit 区间内的点 → 该挂靠港(Attr='visit');
      2. 其余 Anchored 工况点、距最近港 ≤ anchor_km → 该港(Attr='anchor')。
         这是锚地拥堵归属扩展:排放清单的港口节点严格按 10km 港界,
         拥堵排队锚地(如 2021 洛杉矶 30km+ 外)会漏掉,周度口径必须补上。
    要求 ship_ais 已 reset_index(位置=标签)且 Total_* 列已计算。
    返回列: Port_Name, YearMonth, ISO_Week, Mode_Class, Attr, SO2/PM/NOx/CO2(克)
    """
    from city_aggregation import _mode_class
    n = len(ship_ais)
    attr = np.full(n, None, dtype=object)
    kind = np.full(n, "", dtype=object)
    for v in visits or []:
        s, e = int(v.get("_start_idx", -1)), int(v.get("_end_idx", -2))
        if 0 <= s <= e < n:
            attr[s:e + 1] = str(v["Port_Name"])
            kind[s:e + 1] = "visit"

    mode_cls = _mode_class(ship_ais["Mode"])

    if anchor_km and anchor_km > 0 and port_coor is not None and len(port_coor) > 0:
        un = np.array([a is None for a in attr]) & (mode_cls == "Anchored_Class")
        if un.any():
            pc = port_coor.drop_duplicates("Port.Name")
            plon = pc["Lon_port"].astype(float).to_numpy()
            plat = pc["Lat_port"].astype(float).to_numpy()
            names = pc["Port.Name"].astype(str).to_numpy()
            qlon = ship_ais["Longitude"].astype(float).to_numpy()[un]
            qlat = ship_ais["Latitude"].astype(float).to_numpy()[un]
            k = min(5, len(names))
            tree = cKDTree(np.column_stack([plon, plat]))
            _, cand = tree.query(np.column_stack([qlon, qlat]), k=k)
            cand = cand.reshape(len(qlon), k)
            # 向量化 haversine: 逐候选列精确复核(度距查询在高纬有偏,k近邻+复核纠正)
            R = 6371.0088
            la1 = np.radians(qlat)[:, None]
            lo1 = np.radians(qlon)[:, None]
            la2 = np.radians(plat[cand])
            lo2 = np.radians(plon[cand])
            h = (np.sin((la2 - la1) / 2) ** 2
                 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
            dkm = 2 * R * np.arcsin(np.sqrt(h))
            best = dkm.argmin(axis=1)
            rows = np.arange(len(qlon))
            ok = dkm[rows, best] <= float(anchor_km)
            un_idx = np.where(un)[0]
            attr[un_idx[ok]] = names[cand[rows, best]][ok]
            kind[un_idx[ok]] = "anchor"

    has = np.array([a is not None for a in attr])
    if not has.any():
        return pd.DataFrame()
    cols = [f"Total_{p}" for p in _WEEKLY_POLS]
    sub = ship_ais.iloc[np.where(has)[0]][["UTCtime"] + cols].copy()
    ts = pd.to_datetime(sub["UTCtime"], unit="s", utc=True)
    iso = ts.dt.isocalendar()
    sub["ISO_Week"] = (iso["year"].astype(int).astype(str) + "-W"
                       + iso["week"].astype(int).map("{:02d}".format))
    sub["YearMonth"] = ts.dt.strftime("%Y%m")
    sub["Port_Name"] = attr[has]
    sub["Attr"] = kind[has]
    sub["Mode_Class"] = mode_cls[has]
    agg = (sub.groupby(["Port_Name", "YearMonth", "ISO_Week", "Mode_Class", "Attr"],
                       as_index=False)[cols].sum())
    return agg.rename(columns={f"Total_{p}": p for p in _WEEKLY_POLS})


# ============================================================================
# 辅助函数4：众数（移除NA）
# ============================================================================
def mode_rmNA(x):
    """计算众数，移除NA值"""
    x_clean = pd.Series(x).dropna()
    if len(x_clean) == 0:
        return None
    mode_result = x_clean.mode()
    return mode_result[0] if len(mode_result) > 0 else None


# ============================================================================
# 核心函数：单船排放网络计算（最终完全修复版）
# ============================================================================
def calculate_ship_emission_network(ship_ais_data, ship_info, port_coor, 
                                    port_buffers_3nm, LAF, 
                                    ECA_list, month_year,
                                    berth_distance_threshold=2,
                                    berth_buffers=None,
                                    berth_gap_tolerance_sec=3600.0,
                                    canal_coor=None,
                                    canal_corridor_buffers=None,
                                    canal_waiting_buffers=None,
                                    canal_endpoint_buffers=None,
                                    canal_proximity_km=10.0,
                                    city_catchments=None,
                                    policy_zones=None,
                                    policy_cfg=None,
                                    weekly_cfg=None,
                                    policy_only=False):
    """
    计算单艘船舶的排放网络（最终完全修复版，与R代码100%一致）

    与R代码的差异：
    1. R代码输出：空间排放矩阵（0.1度网格） + 单船汇总排放
    2. Python代码输出：港口节点排放 + 航线连边排放

    但排放计算方法完全一致！

    港口停靠识别（Section 14）：
    - 当 berth_buffers 提供时使用 berth-first 双 buffer 判定（与 shipping pipeline 对齐）：
      AIS 点须同时落在 port_buffer 和某港 berth_buffer 内 + Mode=Berth 才算靠泊确认；
      从靠泊段倒推/前推到 port_buffer 边界，得到完整在港时段。
    - 当 berth_buffers=None 时退化为单 buffer sjoin（向后兼容）。
    """
    # 初始化返回结果结构
    result = {
        'success': False,
        'mmsi': None,
        'port_emissions': None,
        'edge_emissions': None,
        'port_operational_visits': None,
        'edge_voyage_stats': None,
        'city_emissions': None,
        'total_emissions': None,
        'policy_delta_city': None,
        'policy_delta_total': None,
        'ops_calls': None,
        'ops_call_city': None,
        'port_weekly': None,
        'error_message': None
    }
    
    # 定义污染物列表
    pollutants = ['SO2', 'PM', 'CO2', 'NOx', 'CO', 'CH4', 'N2O', 'NMVOC']
    
    try:
        # =====================================================================
        # 1. 数据准备
        # =====================================================================
        Ship_1_ais = ship_ais_data.copy()
        Ship_1_ais = Ship_1_ais.drop_duplicates(subset=['UTCtime'])
        
        if len(Ship_1_ais) <= 2:
            result['error_message'] = "AIS记录数<=2"
            return result
        
        # 确保ship_info是Series
        if isinstance(ship_info, pd.DataFrame):
            ship_info = ship_info.iloc[0]
        
        Ship_1_MMSI = ship_info['MMSI']
        result['mmsi'] = Ship_1_MMSI
        
        # 提取静态信息（完整版）
        DesignSpeed_1 = ship_info.get('Service_Speed', 15.0)
        Draught_1 = ship_info.get('Draft', 10.0)
        ME_1 = ship_info.get('ME', 5000.0)
        DWT_1 = ship_info.get('DWT', 50000.0)
        Build_Year_1 = ship_info.get('Build_Year', 2000)
        Gross_Tonnage_1 = pd.to_numeric(
            pd.Series([ship_info.get('Gross_Tonnage', np.nan)]),
            errors='coerce').iloc[0]
        IMO_1 = ship_info.get('IMO_external', ship_info.get('IMO', np.nan))
        GT_Source_1 = ship_info.get('GT_Source', 'missing')
        GT_Match_Method_1 = ship_info.get('GT_Match_Method', '')
        Size_1 = ship_info.get('Size', 50000.0)
        ME_CF_W_1 = ship_info.get('ME_CF_W', 1.0)
        ME_CF_F_1 = ship_info.get('ME_CF_F', 1.0)
        Shiptype_1 = ship_info.get('Ship_Type', 'Container')
        FuelType_1 = ship_info.get('Fuel_Type', 'HFO')
        EngineType_1 = ship_info.get('Engine_Type', 'MSD')
        Tier_1 = ship_info.get('Tier', 2)
        
        # 辅机功率（不同模式下）
        AE_Berth_1 = ship_info.get('AE_Berth', 500.0)
        AE_Anchored_1 = ship_info.get('AE_Anchored', 250.0)
        AE_Manoeuvring_1 = ship_info.get('AE_Manoeuvring', 750.0)
        AE_Sea_1 = ship_info.get('AE_Sea', 1000.0)
        
        # 锅炉功率（不同模式下）
        BO_Berth_1 = ship_info.get('BO_Berth', 100.0)
        BO_Anchored_1 = ship_info.get('BO_Anchored', 50.0)
        BO_Manoeuvring_1 = ship_info.get('BO_Manoeuvring', 150.0)
        BO_Sea_1 = ship_info.get('BO_Sea', 200.0)
        
        # 燃料消耗率 SFC (g/kWh)
        ME_SFC_1 = ship_info.get('ME_SFC', 180.0)
        ME_SFC_MDO_1 = ship_info.get('ME_SFC_MDO', 185.0)
        AE_SFC_1 = ship_info.get('AE_SFC', 200.0)
        AE_SFC_MDO_1 = ship_info.get('AE_SFC_MDO', 205.0)
        BO_SFC_1 = ship_info.get('BO_SFC', 220.0)
        BO_SFC_MDO_1 = ship_info.get('BO_SFC_MDO', 225.0)
        
        # CO2排放因子
        CO2_EFf_1 = ship_info.get('CO2_EFf', 3.206)
        CO2_EFf_MDO_1 = ship_info.get('CO2_EFf_MDO', 3.206)
        
        # NOx排放因子 (g/kWh)
        ME_NOx_1 = ship_info.get('ME_NOx_noECA', 17.0)
        ME_NOx_3_1 = ship_info.get('ME_NOx_ECA', 3.4)
        AE_NOx_1 = ship_info.get('AE_NOx', 14.0)
        BO_NOx_1 = ship_info.get('BO_NOx', 2.0)
        
        # 其他排放因子 (g/kWh)
        ME_CH4_1 = ship_info.get('ME_CH4', 0.04)
        AE_CH4_1 = ship_info.get('AE_CH4', 0.04)
        BO_CH4_1 = ship_info.get('BO_CH4', 0.04)
        
        ME_CO_1 = ship_info.get('ME_CO', 0.5)
        ME_CO_MDO_1 = ship_info.get('ME_CO_MDO', 0.5)
        AE_CO_1 = ship_info.get('AE_CO', 0.5)
        AE_CO_MDO_1 = ship_info.get('AE_CO_MDO', 0.5)
        BO_CO_1 = ship_info.get('BO_CO', 0.5)
        BO_CO_MDO_1 = ship_info.get('BO_CO_MDO', 0.5)
        
        ME_N2O_1 = ship_info.get('ME_N2O', 0.08)
        ME_N2O_MDO_1 = ship_info.get('ME_N2O_MDO', 0.08)
        AE_N2O_1 = ship_info.get('AE_N2O', 0.08)
        AE_N2O_MDO_1 = ship_info.get('AE_N2O_MDO', 0.08)
        BO_N2O_1 = ship_info.get('BO_N2O', 0.08)
        BO_N2O_MDO_1 = ship_info.get('BO_N2O_MDO', 0.08)
        
        ME_NMVOC_1 = ship_info.get('ME_NMVOC', 0.4)
        AE_NMVOC_1 = ship_info.get('AE_NMVOC', 0.4)
        BO_NMVOC_1 = ship_info.get('BO_NMVOC', 0.4)
        
        # =====================================================================
        # 2. 预处理AIS数据
        # =====================================================================
        Ship_1_ais = Ship_1_ais.sort_values('UTCtime').reset_index(drop=True)
        
        # 坐标检查
        Ship_1_ais = Ship_1_ais[
            (Ship_1_ais['Latitude'] <= 90) & (Ship_1_ais['Latitude'] >= -90) &
            (Ship_1_ais['Longitude'] <= 180) & (Ship_1_ais['Longitude'] >= -180)
        ].reset_index(drop=True)
        
        Ship_1_ais = Ship_1_ais.dropna(subset=['Latitude', 'Longitude', 'UTCtime'])
        
        # =====================================================================
        # 3. AIS 插值（与 shipping pipeline 完全一致）
        # ---------------------------------------------------------------------
        # 改用 shipping_network_functions.interpolate_ais_r_style：
        #   - 触发窗口 1800 s < gap < 24 h（emission 旧版本是 3600 s）
        #   - 步长 30 min（旧版本是 60 min）
        #   - Speed 用 ffill/bfill；Lon/Lat 线性插值（同一函数对两条流水线产出相同 AIS 序列）
        # 之后再对插值带来的 NaN 静态列做 ffill 兜底（如 ShipType 等）。
        # =====================================================================
        if 'ShipType' not in Ship_1_ais.columns:
            Ship_1_ais['ShipType'] = Shiptype_1
        Ship_1_ais = interpolate_ais_r_style(Ship_1_ais)
        # 插值新行的静态列（ShipType / MMSI / 其它静态字段）由 ffill 兜底
        Ship_1_ais = Ship_1_ais.ffill()

        # =====================================================================
        # 4. 计算时间间隔 (Re-calculate after interpolation)
        # =====================================================================
        Ship_1_ais['TimeInv'] = Ship_1_ais['UTCtime'].diff().fillna(0)
        
        # 清洗异常时间间隔（>24小时的记录）
        Ship_1_ais = Ship_1_ais[Ship_1_ais['TimeInv'] < 3600*24].reset_index(drop=True)

        # 正式政策专用模式先做一次空间筛选。非集装箱船只保留政策区点；
        # 集装箱船保留“政策区点 ∪ 入选国研究港口的泊位缓冲点”，后者供逐次
        # 岸电核算。TimeInv 已在全轨迹上计算，裁剪不会改变保留点的原始权重。
        if policy_only and policy_zones is not None and len(policy_zones) > 0:
            Ship_1_ais['PZ_Zone'] = _policy_zone_labels(Ship_1_ais, policy_zones)
            relevant = Ship_1_ais['PZ_Zone'] != ''
            if (str(Shiptype_1).strip().casefold() == 'container'
                    and berth_buffers is not None and len(berth_buffers) > 0):
                pts = gpd.GeoDataFrame(
                    geometry=gpd.points_from_xy(
                        Ship_1_ais['Longitude'], Ship_1_ais['Latitude']),
                    index=Ship_1_ais.index, crs='EPSG:4326')
                bbuf = berth_buffers[['berth_port_name', 'geometry']].copy()
                if bbuf.crs is None:
                    bbuf = bbuf.set_crs('EPSG:4326')
                jb = gpd.sjoin(pts, bbuf, predicate='within', how='left')
                jb = (jb.reset_index().drop_duplicates('index', keep='first')
                      .set_index('index').reindex(Ship_1_ais.index))
                Ship_1_ais['berth_port_name'] = jb['berth_port_name'].values
                relevant |= Ship_1_ais['berth_port_name'].notna()
            Ship_1_ais = Ship_1_ais.loc[relevant].reset_index(drop=True)
            # 上述泊位标签只用于预筛；正式 visit 流水线会用同一 berth buffer
            # 重新赋值。先删除可避免第二次 sjoin 产生 _left/_right 重名列。
            Ship_1_ais = Ship_1_ais.drop(
                columns=['berth_port_name'], errors='ignore')
            if len(Ship_1_ais) == 0:
                for key in (
                    'port_emissions', 'edge_emissions', 'port_operational_visits',
                    'edge_voyage_stats', 'city_emissions', 'policy_delta_city',
                    'policy_delta_total', 'ops_calls', 'ops_call_city', 'port_weekly',
                ):
                    result[key] = pd.DataFrame()
                result['total_emissions'] = {}
                result['total_emissions_by_month'] = {}
                result['success'] = True
                return result
        
        # 提取年份（用于判断是否使用MDO）
        # 必须在过滤后重新计算，确保长度一致！
        Ship_1_ais['Year'] = pd.to_datetime(Ship_1_ais['UTCtime'], unit='s', utc=True).dt.year
        # year_vec 已移除，直接使用 Ship_1_ais['Year']
        
        # 提取月份和季节 (用于后续计算，例如辅机载荷)
        Ship_1_ais['Month'] = pd.to_datetime(Ship_1_ais['UTCtime'], unit='s', utc=True).dt.month
        # month_vec 已移除，直接使用 Ship_1_ais['Month']
        
        # =====================================================================
        # 5. 速度处理 (Wait, logic sequence?)
        # R: 1. Interp -> 2. TimeInv -> 3. Speed clean -> 4. Year -> 5. Power
        # =====================================================================
        # Ship_1_ais['Speed'] already ffilled if interp happened.
        # But we need to handle the speed cap.
        
        Ship_1_ais['Speed'] = Ship_1_ais['Speed'].ffill()
        Ship_1_ais.loc[Ship_1_ais['Speed'] > 1.5 * DesignSpeed_1, 'Speed'] = DesignSpeed_1
        
        # =====================================================================
        # 6. 计算主机功率和负载
        # =====================================================================
        if Shiptype_1 == 'Container' and Size_1 >= 14500:
            ME_CF_S_1 = 0.75
        elif Shiptype_1 == 'Cruise':
            ME_CF_S_1 = 0.70
        else:
            ME_CF_S_1 = 1.0
        
        Ship_1_ais['MEPower'] = (ME_1 * (Ship_1_ais['Speed'] / DesignSpeed_1)**3 * ME_CF_S_1) / (ME_CF_W_1 * ME_CF_F_1)
        Ship_1_ais.loc[Ship_1_ais['MEPower'] > ME_1, 'MEPower'] = 0.98 * ME_1
        Ship_1_ais['ME_load'] = np.round(Ship_1_ais['MEPower'] / ME_1, 2)
        
        # 负载修正系数（多项式拟合）- 作为 DataFrame 列
        # 这样可以确保索引始终对齐，避免 "Length of values does not match length of index" 错误
        # R code: load_correction <- 0.455 * load^2 - 0.710 * load + 1.280
        # 重要：.values 是必须的，否则在后续 .loc[idx] 操作中会发生索引对齐，导致长度不匹配错误
        # ⚠️ CRITICAL FIX: Explicitly handle potential NaN or index issues
        # Use numpy array assignment directly to avoid any index alignment
        Ship_1_ais['load_correction'] = (
            0.455 * Ship_1_ais['ME_load'].values**2 - 
            0.710 * Ship_1_ais['ME_load'].values + 
            1.280
        )
        
        # =====================================================================
        # 6. 判断操作模式
        # =====================================================================
        Ship_1_ais['Mode'] = 'NA'
        Ship_1_ais.loc[Ship_1_ais['Speed'] < 1, 'Mode'] = 'Berth'
        Ship_1_ais.loc[(Ship_1_ais['Speed'] >= 1) & (Ship_1_ais['Speed'] < 3), 'Mode'] = 'Anchored'
        Ship_1_ais.loc[(Ship_1_ais['ME_load'] < 0.2) & (Ship_1_ais['Speed'] >= 3), 'Mode'] = 'Manoeuvring'
        Ship_1_ais.loc[(Ship_1_ais['ME_load'] >= 0.2) & (Ship_1_ais['Speed'] >= 3), 'Mode'] = 'Sea'
        
        # =====================================================================
        # 7. 负载调整因子（LAF）
        # =====================================================================
        Ship_1_ais['PM_LAF'] = 1.0
        Ship_1_ais['NOx_LAF'] = 1.0
        Ship_1_ais['CH4_LAF'] = 1.0
        Ship_1_ais['CO_LAF'] = 1.0
        Ship_1_ais['N2O_LAF'] = 1.0
        Ship_1_ais['NMVOC_LAF'] = 1.0
        
        # Hardcoded mapping from R code (ship_emi_fun_chunk_new.R)
        # Format: "load": [CH4, N2O, NMVOC, NOx, PM, CO]
        laf_mapping = {
            "0.07": [2.18, 2.18, 3.52, 1.45, 1.79, 2.79],
            "0.08": [2.18, 2.18, 2.95, 1.35, 1.61, 2.45],
            "0.09": [2.18, 2.18, 2.52, 1.27, 1.48, 2.18],
            "0.10": [2.18, 2.18, 2.18, 1.22, 1.38, 1.96],
            "0.11": [1.00, 1.00, 1.96, 1.17, 1.30, 1.79],
            "0.12": [1.00, 1.00, 1.76, 1.14, 1.24, 1.64],
            "0.13": [1.00, 1.00, 1.60, 1.11, 1.19, 1.52],
            "0.14": [1.00, 1.00, 1.47, 1.08, 1.15, 1.41],
            "0.15": [1.00, 1.00, 1.36, 1.06, 1.11, 1.32],
            "0.16": [1.00, 1.00, 1.26, 1.05, 1.08, 1.24],
            "0.17": [1.00, 1.00, 1.18, 1.03, 1.06, 1.17],
            "0.18": [1.00, 1.00, 1.11, 1.02, 1.04, 1.11],
            "0.19": [1.00, 1.00, 1.05, 1.01, 1.02, 1.05]
        }
        
        # Apply mapping
        # Convert ME_load to string with 2 decimals for matching
        # Note: We iterate or use map. Iteration is safer for exact string match.
        
        # Optimize using map
        # Create mapping dicts for each pollutant
        ch4_map = {float(k): v[0] for k, v in laf_mapping.items()}
        n2o_map = {float(k): v[1] for k, v in laf_mapping.items()}
        nmvoc_map = {float(k): v[2] for k, v in laf_mapping.items()}
        nox_map = {float(k): v[3] for k, v in laf_mapping.items()}
        pm_map = {float(k): v[4] for k, v in laf_mapping.items()}
        co_map = {float(k): v[5] for k, v in laf_mapping.items()}
        
        # Update values where ME_load matches keys
        # We need to ensure ME_load is rounded to 2 decimals (already done in Step 5)
        
        # Use map with fillna(1.0) - but only for keys that exist? 
        # No, existing values are 1.0. We update only where match found.
        
        # To avoid overwriting non-matches with NaN, we can use combine_first or update
        # But map returns NaN for missing keys.
        
        # Let's iterate over keys, it's safer and matches the R loop logic
        for load_str, factors in laf_mapping.items():
            load_val = float(load_str)
            # Find rows with this load (allow small float tolerance if needed, but we rounded to 2)
            mask = (np.abs(Ship_1_ais['ME_load'] - load_val) < 1e-5)
            
            if mask.any():
                Ship_1_ais.loc[mask, 'CH4_LAF'] = factors[0]
                Ship_1_ais.loc[mask, 'N2O_LAF'] = factors[1]
                Ship_1_ais.loc[mask, 'NMVOC_LAF'] = factors[2]
                Ship_1_ais.loc[mask, 'NOx_LAF'] = factors[3]
                Ship_1_ais.loc[mask, 'PM_LAF'] = factors[4]
                Ship_1_ais.loc[mask, 'CO_LAF'] = factors[5]
        
        # =====================================================================
        # 8. 判断ECA区域
        # =====================================================================
        Ship_1_ais['ECA'] = 0
        
        eca_mapping = {
            'EU': 1, 'Europe': 1,
            'CN': 2, 'China': 2,
            'US': 3, 'USA': 3,
            'KR': 4, 'Korea': 4,
            'Med': 5, 'Mediterranean': 5
        }
        
        if ECA_list:
            # Prepare points for vectorized check
            points = Ship_1_ais[['Longitude', 'Latitude']].values
            
            for eca_name, eca_coords in ECA_list.items():
                if eca_coords and 'lon' in eca_coords and 'lat' in eca_coords:
                    eca_code = 1
                    # 前缀匹配（区划名是 load_port_laf_eca_data 里的固定键）。
                    # 不能用子串匹配：'Med_Piraeus' 含 'eu'、'Korea_Busan' 含 'us'，
                    # 会误入 EU/US 分支，被错误套用 Tier III NOx 和生效时间
                    lower_name = eca_name.lower()
                    if lower_name.startswith('med_'):
                        eca_code = 4  # 地中海港口级 0.1%，随 code 4 的 2022 时间门
                    elif 'mediterranean' in lower_name:
                        eca_code = 5  # 地中海海区 SECA, 2025-05-01 起
                    elif lower_name.startswith(('korea', 'kr')):
                        eca_code = 4
                    elif lower_name.startswith(('us', 'usa')):
                        eca_code = 3
                    elif lower_name.startswith(('china', 'cn')):
                        eca_code = 2
                    elif lower_name.startswith(('europe', 'eu')):
                        eca_code = 1
                    
                    # Create Path from vertices
                    poly_path = Path(list(zip(eca_coords['lon'], eca_coords['lat'])))
                    
                    # Vectorized contains check
                    mask = poly_path.contains_points(points)
                    
                    if mask.any():
                        Ship_1_ais.loc[mask, 'ECA'] = eca_code
        
        # =====================================================================
        # 8.1 Apply Time Constraints for ECAs
        # =====================================================================
        # China (Code 2): Starts 2019-01-01
        Ship_1_ais.loc[(Ship_1_ais['ECA'] == 2) & (Ship_1_ais['Year'] < 2019), 'ECA'] = 0
        
        # Korea (Code 4): Starts 2022-01-01
        Ship_1_ais.loc[(Ship_1_ais['ECA'] == 4) & (Ship_1_ais['Year'] < 2022), 'ECA'] = 0
        
        # Mediterranean (Code 5): Starts 2025-05-01
        # Use UTCtime for precise check
        med_start_ts = 1746057600 # 2025-05-01 00:00:00 UTC
        Ship_1_ais.loc[(Ship_1_ais['ECA'] == 5) & (Ship_1_ais['UTCtime'] < med_start_ts), 'ECA'] = 0
        
        # load_correction 已在前面作为 DataFrame 列计算，此处不再重复

        
        # =====================================================================
        # 10. 计算主机排放（完全修复版）
        # =====================================================================
        
        # 10.1 主机CO2排放（单位：克）
        if FuelType_1 == 'HFO':
            idx_eca = Ship_1_ais['ECA'] != 0
            idx_non_pre = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] < 2020)
            idx_non_post = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] >= 2020)
            
            Ship_1_ais['ME_CO2'] = 0.0
            Ship_1_ais.loc[idx_eca, 'ME_CO2'] = (
                Ship_1_ais.loc[idx_eca, 'MEPower'] * 
                ME_SFC_MDO_1 * 
                Ship_1_ais.loc[idx_eca, 'load_correction'] * 
                Ship_1_ais.loc[idx_eca, 'TimeInv'] / 3600 * 
                3.206
            )
            Ship_1_ais.loc[idx_non_pre, 'ME_CO2'] = (
                Ship_1_ais.loc[idx_non_pre, 'MEPower'] * 
                ME_SFC_1 * 
                Ship_1_ais.loc[idx_non_pre, 'load_correction'] * 
                Ship_1_ais.loc[idx_non_pre, 'TimeInv'] / 3600 * 
                3.206
            )
            Ship_1_ais.loc[idx_non_post, 'ME_CO2'] = (
                Ship_1_ais.loc[idx_non_post, 'MEPower'] * 
                ME_SFC_MDO_1 * 
                Ship_1_ais.loc[idx_non_post, 'load_correction'] * 
                Ship_1_ais.loc[idx_non_post, 'TimeInv'] / 3600 * 
                3.206
            )
        elif FuelType_1 == 'LNG':
            Ship_1_ais['ME_CO2'] = (
                Ship_1_ais['MEPower'] * 
                ME_SFC_1 * 
                Ship_1_ais['load_correction'] * 
                Ship_1_ais['TimeInv'] / 3600 * 
                2.750
            )
        elif FuelType_1 == 'MeOH':
            Ship_1_ais['ME_CO2'] = (
                Ship_1_ais['MEPower'] * 
                ME_SFC_1 * 
                Ship_1_ais['load_correction'] * 
                Ship_1_ais['TimeInv'] / 3600 * 
                1.375
            )
        else:  # MDO
            Ship_1_ais['ME_CO2'] = (
                Ship_1_ais['MEPower'] * 
                ME_SFC_1 * 
                Ship_1_ais['load_correction'] * 
                Ship_1_ais['TimeInv'] / 3600 * 
                3.206
            )
        
        # 非 ECA 区 pre-2020 燃油含硫率: IMO 2020 限硫前全球均值≈2.7%;
        # 2020-01-01 起 0.5%(eca_sulfur[0]/ECA_factors[0]=0.005)。修复原 pre/post 都用 0.5% 的疏漏
        S_NONECA_PRE2020 = 0.027

        # 10.2 主机SO2排放（单位：克）
        SO2_factors = {
            'HFO': 2 * 0.97753,
            'MDO': 2 * 0.97753,
            'LNG': 0.0000317,
            'MeOH': 0.00264
        }
        
        ECA_factors = {
            0: 0.005,  # 非ECA
            1: 0.001,  # EU
            2: 0.005,  # CN
            3: 0.001,  # US
            4: 0.001,  # KR
            5: 0.001   # Med
        }
        
        Ship_1_ais['ME_SO2'] = 0.0
        
        # Mediterranean ECA start time: 2025-05-01 00:00:00 UTC
        med_start_ts = 1746057600
        
        if FuelType_1 == 'MDO':
            for eca_code in [0, 1, 2, 3, 4, 5]:
                idx = Ship_1_ais['ECA'] == eca_code
                if idx.any():
                    Ship_1_ais.loc[idx, 'ME_SO2'] = (
                        Ship_1_ais.loc[idx, 'MEPower'] * 
                        ME_SFC_1 * 
                        Ship_1_ais.loc[idx, 'load_correction'] * 
                        Ship_1_ais.loc[idx, 'TimeInv'] / 3600 * 
                        SO2_factors[FuelType_1] * 
                        ECA_factors[eca_code]
                    )
        
        elif FuelType_1 == 'HFO':
            for eca_code in [1, 2, 3, 4, 5]:
                idx = Ship_1_ais['ECA'] == eca_code
                if idx.any():
                    Ship_1_ais.loc[idx, 'ME_SO2'] = (
                        Ship_1_ais.loc[idx, 'MEPower'] * 
                        ME_SFC_MDO_1 * 
                        Ship_1_ais.loc[idx, 'load_correction'] * 
                        Ship_1_ais.loc[idx, 'TimeInv'] / 3600 * 
                        SO2_factors[FuelType_1] * 
                        ECA_factors[eca_code]
                    )
            
            idx0_pre = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] < 2020)
            idx0_post = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] >= 2020)
            if idx0_pre.any():
                Ship_1_ais.loc[idx0_pre, 'ME_SO2'] = (
                    Ship_1_ais.loc[idx0_pre, 'MEPower'] * 
                    ME_SFC_1 * 
                    Ship_1_ais.loc[idx0_pre, 'load_correction'] * 
                    Ship_1_ais.loc[idx0_pre, 'TimeInv'] / 3600 * 
                    SO2_factors[FuelType_1] * 
                    S_NONECA_PRE2020
                )
            if idx0_post.any():
                Ship_1_ais.loc[idx0_post, 'ME_SO2'] = (
                    Ship_1_ais.loc[idx0_post, 'MEPower'] * 
                    ME_SFC_MDO_1 * 
                    Ship_1_ais.loc[idx0_post, 'load_correction'] * 
                    Ship_1_ais.loc[idx0_post, 'TimeInv'] / 3600 * 
                    SO2_factors[FuelType_1] * 
                    ECA_factors[0]
                )
        
        elif FuelType_1 in ['LNG', 'MeOH']:
            Ship_1_ais['ME_SO2'] = (
                Ship_1_ais['MEPower'] * 
                ME_SFC_1 * 
                Ship_1_ais['load_correction'] * 
                Ship_1_ais['TimeInv'] / 3600 * 
                SO2_factors[FuelType_1]
            )
        
        # 10.3 主机PM排放（单位：克）
        Ship_1_ais['ME_PM'] = 0.0
        
        eca_sulfur = {
            0: 0.005, 1: 0.001, 2: 0.005, 
            3: 0.001, 4: 0.001, 5: 0.001
        }
        
        if FuelType_1 == 'MDO':
            for eca_code in [0, 1, 2, 3, 4, 5]:
                idx = Ship_1_ais['ECA'] == eca_code
                if idx.any():
                    Ship_1_ais.loc[idx, 'ME_PM'] = (
                        Ship_1_ais.loc[idx, 'MEPower'] * 
                        Ship_1_ais.loc[idx, 'TimeInv'] / 3600 * 
                        Ship_1_ais.loc[idx, 'PM_LAF'] * 
                        (0.23 + ME_SFC_1 * Ship_1_ais.loc[idx, 'load_correction'] * 7 * 0.02247 * 
                         (eca_sulfur[eca_code] - 0.0024))
                    )
        
        elif FuelType_1 == 'HFO':
            for eca_code in [1, 2, 3, 4, 5]:
                idx = Ship_1_ais['ECA'] == eca_code
                if idx.any():
                    Ship_1_ais.loc[idx, 'ME_PM'] = (
                        Ship_1_ais.loc[idx, 'MEPower'] * 
                        Ship_1_ais.loc[idx, 'TimeInv'] / 3600 * 
                        Ship_1_ais.loc[idx, 'PM_LAF'] * 
                        (0.23 + ME_SFC_MDO_1 * Ship_1_ais.loc[idx, 'load_correction'] * 7 * 0.02247 * 
                         (eca_sulfur[eca_code] - 0.0024))
                    )
            
            idx0_pre = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] < 2020)
            idx0_post = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] >= 2020)
            if idx0_pre.any():
                Ship_1_ais.loc[idx0_pre, 'ME_PM'] = (
                    Ship_1_ais.loc[idx0_pre, 'MEPower'] * 
                    Ship_1_ais.loc[idx0_pre, 'TimeInv'] / 3600 * 
                    Ship_1_ais.loc[idx0_pre, 'PM_LAF'] * 
                    (0.23 + ME_SFC_1 * Ship_1_ais.loc[idx0_pre, 'load_correction'] * 7 * 0.02247 * 
                     (S_NONECA_PRE2020 - 0.0024))
                )
            if idx0_post.any():
                Ship_1_ais.loc[idx0_post, 'ME_PM'] = (
                    Ship_1_ais.loc[idx0_post, 'MEPower'] * 
                    Ship_1_ais.loc[idx0_post, 'TimeInv'] / 3600 * 
                    Ship_1_ais.loc[idx0_post, 'PM_LAF'] * 
                    (0.23 + ME_SFC_MDO_1 * Ship_1_ais.loc[idx0_post, 'load_correction'] * 7 * 0.02247 * 
                     (eca_sulfur[0] - 0.0024))
                )
        
        elif FuelType_1 == 'MeOH':
            for eca_code in [0, 1, 2, 3, 4, 5]:
                idx = Ship_1_ais['ECA'] == eca_code
                if idx.any():
                    Ship_1_ais.loc[idx, 'ME_PM'] = (
                        0.1 * 
                        Ship_1_ais.loc[idx, 'MEPower'] * 
                        Ship_1_ais.loc[idx, 'TimeInv'] / 3600 * 
                        Ship_1_ais.loc[idx, 'PM_LAF'] * 
                        (0.23 + ME_SFC_MDO_1 * Ship_1_ais.loc[idx, 'load_correction'] * 7 * 0.02247 * 
                         (eca_sulfur[eca_code] - 0.0024))
                    )
        
        elif EngineType_1 == 'LNG-Diesel':
            Ship_1_ais['ME_PM'] = (
                Ship_1_ais['MEPower'] * 
                Ship_1_ais['TimeInv'] / 3600 * 
                0.01 * 
                Ship_1_ais['PM_LAF']
            )
        
        # 10.4 主机NOx排放（单位：克）
        Ship_1_ais['ME_NOx'] = (
            Ship_1_ais['MEPower'] * 
            Ship_1_ais['TimeInv'] / 3600 * 
            ME_NOx_1 * 
            Ship_1_ais['NOx_LAF']
        )
        
        # Tier III 仅约束 NECA 生效后建造的船(EU 北海/波罗的海: 2021 年起建造;
        # 美国: 2016 年起建造),存量老船维持 Tier I/II。建造年隐含日历年下限,
        # 无需再加 Year 条件。Build_Year 缺失默认 2000, 即按老船处理(不套 Tier III)
        idx_eu_t3 = (Ship_1_ais['ECA'] == 1) & (Build_Year_1 >= 2021)
        idx_us_t3 = (Ship_1_ais['ECA'] == 3) & (Build_Year_1 >= 2016)
        if idx_eu_t3.any():
            Ship_1_ais.loc[idx_eu_t3, 'ME_NOx'] = (
                Ship_1_ais.loc[idx_eu_t3, 'MEPower'] * 
                Ship_1_ais.loc[idx_eu_t3, 'TimeInv'] / 3600 * 
                ME_NOx_3_1 * 
                Ship_1_ais.loc[idx_eu_t3, 'NOx_LAF']
            )
        if idx_us_t3.any():
            Ship_1_ais.loc[idx_us_t3, 'ME_NOx'] = (
                Ship_1_ais.loc[idx_us_t3, 'MEPower'] * 
                Ship_1_ais.loc[idx_us_t3, 'TimeInv'] / 3600 * 
                ME_NOx_3_1 * 
                Ship_1_ais.loc[idx_us_t3, 'NOx_LAF']
            )
        
        # 10.5 主机CH4排放（单位：克）
        Ship_1_ais['ME_CH4'] = (
            Ship_1_ais['MEPower'] * 
            Ship_1_ais['TimeInv'] / 3600 * 
            ME_CH4_1 * 
            Ship_1_ais['CH4_LAF']
        )
        
        # 10.6 主机CO排放（单位：克）
        if FuelType_1 == 'HFO':
            idx_eca = Ship_1_ais['ECA'] != 0
            idx_non_pre = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] < 2020)
            idx_non_post = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] >= 2020)
            
            Ship_1_ais['ME_CO'] = 0.0
            if idx_eca.any():
                Ship_1_ais.loc[idx_eca, 'ME_CO'] = (
                    Ship_1_ais.loc[idx_eca, 'MEPower'] * 
                    Ship_1_ais.loc[idx_eca, 'TimeInv'] / 3600 * 
                    ME_CO_MDO_1 * 
                    Ship_1_ais.loc[idx_eca, 'CO_LAF']
                )
            if idx_non_pre.any():
                Ship_1_ais.loc[idx_non_pre, 'ME_CO'] = (
                    Ship_1_ais.loc[idx_non_pre, 'MEPower'] * 
                    Ship_1_ais.loc[idx_non_pre, 'TimeInv'] / 3600 * 
                    ME_CO_1 * 
                    Ship_1_ais.loc[idx_non_pre, 'CO_LAF']
                )
            if idx_non_post.any():
                Ship_1_ais.loc[idx_non_post, 'ME_CO'] = (
                    Ship_1_ais.loc[idx_non_post, 'MEPower'] * 
                    Ship_1_ais.loc[idx_non_post, 'TimeInv'] / 3600 * 
                    ME_CO_MDO_1 * 
                    Ship_1_ais.loc[idx_non_post, 'CO_LAF']
                )
        else:
            Ship_1_ais['ME_CO'] = (
                Ship_1_ais['MEPower'] * 
                Ship_1_ais['TimeInv'] / 3600 * 
                ME_CO_1 * 
                Ship_1_ais['CO_LAF']
            )
        
        # 10.7 主机N2O排放（单位：克）
        if FuelType_1 == 'HFO':
            idx_eca = Ship_1_ais['ECA'] != 0
            idx_non_pre = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] < 2020)
            idx_non_post = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] >= 2020)
            
            Ship_1_ais['ME_N2O'] = 0.0
            if idx_eca.any():
                Ship_1_ais.loc[idx_eca, 'ME_N2O'] = (
                    Ship_1_ais.loc[idx_eca, 'MEPower'] * 
                    Ship_1_ais.loc[idx_eca, 'TimeInv'] / 3600 * 
                    ME_N2O_MDO_1 * 
                    Ship_1_ais.loc[idx_eca, 'N2O_LAF']
                )
            if idx_non_pre.any():
                Ship_1_ais.loc[idx_non_pre, 'ME_N2O'] = (
                    Ship_1_ais.loc[idx_non_pre, 'MEPower'] * 
                    Ship_1_ais.loc[idx_non_pre, 'TimeInv'] / 3600 * 
                    ME_N2O_1 * 
                    Ship_1_ais.loc[idx_non_pre, 'N2O_LAF']
                )
            if idx_non_post.any():
                Ship_1_ais.loc[idx_non_post, 'ME_N2O'] = (
                    Ship_1_ais.loc[idx_non_post, 'MEPower'] * 
                    Ship_1_ais.loc[idx_non_post, 'TimeInv'] / 3600 * 
                    ME_N2O_MDO_1 * 
                    Ship_1_ais.loc[idx_non_post, 'N2O_LAF']
                )
        else:
            Ship_1_ais['ME_N2O'] = (
                Ship_1_ais['MEPower'] * 
                Ship_1_ais['TimeInv'] / 3600 * 
                ME_N2O_1 * 
                Ship_1_ais['N2O_LAF']
            )
        
        # 10.8 主机NMVOC排放（单位：克）
        Ship_1_ais['ME_NMVOC'] = (
            Ship_1_ais['MEPower'] * 
            Ship_1_ais['TimeInv'] / 3600 * 
            ME_NMVOC_1 * 
            Ship_1_ais['NMVOC_LAF']
        )
        
        # =====================================================================
        # ⚠️ 【关键】：主机负载<7%时排放归0（R代码第792-800行）
        # =====================================================================
        idx_low_load = Ship_1_ais['ME_load'] < 0.07
        Ship_1_ais.loc[idx_low_load, 'ME_SO2'] = 0
        Ship_1_ais.loc[idx_low_load, 'ME_NOx'] = 0
        Ship_1_ais.loc[idx_low_load, 'ME_CO2'] = 0
        Ship_1_ais.loc[idx_low_load, 'ME_CO'] = 0
        Ship_1_ais.loc[idx_low_load, 'ME_CH4'] = 0
        Ship_1_ais.loc[idx_low_load, 'ME_N2O'] = 0
        Ship_1_ais.loc[idx_low_load, 'ME_NMVOC'] = 0
        Ship_1_ais.loc[idx_low_load, 'ME_PM'] = 0
        
        # =====================================================================
        # 11. 计算辅机排放（完全修复版）
        # =====================================================================
        Ship_1_ais['AE_Power'] = 0.0
        Ship_1_ais.loc[Ship_1_ais['Mode'] == 'Berth', 'AE_Power'] = AE_Berth_1
        Ship_1_ais.loc[Ship_1_ais['Mode'] == 'Anchored', 'AE_Power'] = AE_Anchored_1
        Ship_1_ais.loc[Ship_1_ais['Mode'] == 'Manoeuvring', 'AE_Power'] = AE_Manoeuvring_1
        Ship_1_ais.loc[Ship_1_ais['Mode'] == 'Sea', 'AE_Power'] = AE_Sea_1
        
        # 辅机CO2排放（单位：克）
        if FuelType_1 == 'HFO':
            idx_eca = Ship_1_ais['ECA'] != 0
            idx_non_pre = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] < 2020)
            idx_non_post = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] >= 2020)
            
            Ship_1_ais['AE_CO2'] = 0.0
            if idx_eca.any():
                Ship_1_ais.loc[idx_eca, 'AE_CO2'] = (
                    Ship_1_ais.loc[idx_eca, 'AE_Power'] * 
                    Ship_1_ais.loc[idx_eca, 'TimeInv'] / 3600 * 
                    AE_SFC_MDO_1 * 
                    3.206
                )
            if idx_non_pre.any():
                Ship_1_ais.loc[idx_non_pre, 'AE_CO2'] = (
                    Ship_1_ais.loc[idx_non_pre, 'AE_Power'] * 
                    Ship_1_ais.loc[idx_non_pre, 'TimeInv'] / 3600 * 
                    AE_SFC_1 * 
                    3.206
                )
            if idx_non_post.any():
                Ship_1_ais.loc[idx_non_post, 'AE_CO2'] = (
                    Ship_1_ais.loc[idx_non_post, 'AE_Power'] * 
                    Ship_1_ais.loc[idx_non_post, 'TimeInv'] / 3600 * 
                    AE_SFC_MDO_1 * 
                    3.206
                )
        elif FuelType_1 == 'LNG':
            Ship_1_ais['AE_CO2'] = (
                Ship_1_ais['AE_Power'] * 
                Ship_1_ais['TimeInv'] / 3600 * 
                AE_SFC_1 * 
                2.750
            )
        elif FuelType_1 == 'MeOH':
            Ship_1_ais['AE_CO2'] = (
                Ship_1_ais['AE_Power'] * 
                Ship_1_ais['TimeInv'] / 3600 * 
                AE_SFC_1 * 
                1.375
            )
        else:
            Ship_1_ais['AE_CO2'] = (
                Ship_1_ais['AE_Power'] * 
                Ship_1_ais['TimeInv'] / 3600 * 
                AE_SFC_1 * 
                3.206
            )
        
        # 辅机SO2排放（单位：克）
        Ship_1_ais['AE_SO2'] = 0.0
        
        if FuelType_1 == 'MDO':
            for eca_code in [0, 1, 2, 3, 4, 5]:
                idx = Ship_1_ais['ECA'] == eca_code
                if idx.any():
                    Ship_1_ais.loc[idx, 'AE_SO2'] = (
                        Ship_1_ais.loc[idx, 'AE_Power'] * 
                        AE_SFC_1 * 
                        Ship_1_ais.loc[idx, 'TimeInv'] / 3600 * 
                        SO2_factors[FuelType_1] * 
                        ECA_factors[eca_code]
                    )
        
        elif FuelType_1 == 'HFO':
            for eca_code in [1, 2, 3, 4, 5]:
                idx = Ship_1_ais['ECA'] == eca_code
                if idx.any():
                    Ship_1_ais.loc[idx, 'AE_SO2'] = (
                        Ship_1_ais.loc[idx, 'AE_Power'] * 
                        AE_SFC_MDO_1 * 
                        Ship_1_ais.loc[idx, 'TimeInv'] / 3600 * 
                        SO2_factors[FuelType_1] * 
                        ECA_factors[eca_code]
                    )
            
            idx0_pre = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] < 2020)
            idx0_post = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] >= 2020)
            if idx0_pre.any():
                Ship_1_ais.loc[idx0_pre, 'AE_SO2'] = (
                    Ship_1_ais.loc[idx0_pre, 'AE_Power'] * 
                    AE_SFC_1 * 
                    Ship_1_ais.loc[idx0_pre, 'TimeInv'] / 3600 * 
                    SO2_factors[FuelType_1] * 
                    S_NONECA_PRE2020
                )
            if idx0_post.any():
                Ship_1_ais.loc[idx0_post, 'AE_SO2'] = (
                    Ship_1_ais.loc[idx0_post, 'AE_Power'] * 
                    AE_SFC_MDO_1 * 
                    Ship_1_ais.loc[idx0_post, 'TimeInv'] / 3600 * 
                    SO2_factors[FuelType_1] * 
                    ECA_factors[0]
                )
        
        elif FuelType_1 in ['LNG', 'MeOH']:
            Ship_1_ais['AE_SO2'] = (
                Ship_1_ais['AE_Power'] * 
                AE_SFC_1 * 
                Ship_1_ais['TimeInv'] / 3600 * 
                SO2_factors[FuelType_1]
            )
        
        # 辅机PM排放（单位：克）
        Ship_1_ais['AE_PM'] = 0.0
        
        if FuelType_1 == 'MDO':
            for eca_code in [0, 1, 2, 3, 4, 5]:
                idx = Ship_1_ais['ECA'] == eca_code
                if idx.any():
                    Ship_1_ais.loc[idx, 'AE_PM'] = (
                        Ship_1_ais.loc[idx, 'AE_Power'] * 
                        Ship_1_ais.loc[idx, 'TimeInv'] / 3600 * 
                        Ship_1_ais.loc[idx, 'PM_LAF'] * 
                        (0.23 + AE_SFC_1 * 7 * 0.02247 * (eca_sulfur[eca_code] - 0.0024))
                    )
        
        elif FuelType_1 == 'HFO':
            for eca_code in [1, 2, 3, 4, 5]:
                idx = Ship_1_ais['ECA'] == eca_code
                if idx.any():
                    Ship_1_ais.loc[idx, 'AE_PM'] = (
                        Ship_1_ais.loc[idx, 'AE_Power'] * 
                        Ship_1_ais.loc[idx, 'TimeInv'] / 3600 * 
                        Ship_1_ais.loc[idx, 'PM_LAF'] * 
                        (0.23 + AE_SFC_MDO_1 * 7 * 0.02247 * (eca_sulfur[eca_code] - 0.0024))
                    )
            
            idx0_pre = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] < 2020)
            idx0_post = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] >= 2020)
            if idx0_pre.any():
                Ship_1_ais.loc[idx0_pre, 'AE_PM'] = (
                    Ship_1_ais.loc[idx0_pre, 'AE_Power'] * 
                    Ship_1_ais.loc[idx0_pre, 'TimeInv'] / 3600 * 
                    (0.23 + AE_SFC_1 * 7 * 0.02247 * (S_NONECA_PRE2020 - 0.0024))
                )
            if idx0_post.any():
                Ship_1_ais.loc[idx0_post, 'AE_PM'] = (
                    Ship_1_ais.loc[idx0_post, 'AE_Power'] * 
                    Ship_1_ais.loc[idx0_post, 'TimeInv'] / 3600 * 
                    (0.23 + AE_SFC_MDO_1 * 7 * 0.02247 * (eca_sulfur[0] - 0.0024))
                )
        
        elif FuelType_1 == 'LNG':
            Ship_1_ais['AE_PM'] = (
                Ship_1_ais['AE_Power'] * 
                Ship_1_ais['TimeInv'] / 3600 * 
                0.02
            )
        
        elif FuelType_1 == 'MeOH':
            for eca_code in [0, 1, 2, 3, 4, 5]:
                idx = Ship_1_ais['ECA'] == eca_code
                if idx.any():
                    Ship_1_ais.loc[idx, 'AE_PM'] = (
                        0.1 * 
                        Ship_1_ais.loc[idx, 'AE_Power'] * 
                        Ship_1_ais.loc[idx, 'TimeInv'] / 3600 * 
                        Ship_1_ais.loc[idx, 'PM_LAF'] * 
                        (0.23 + AE_SFC_MDO_1 * 7 * 0.02247 * (eca_sulfur[eca_code] - 0.0024))
                    )
        
        # 辅机NOx排放（单位：克）
        Ship_1_ais['AE_NOx'] = (
            Ship_1_ais['AE_Power'] * 
            Ship_1_ais['TimeInv'] / 3600 * 
            AE_NOx_1
        )
        
        # 辅机CO排放（单位：克）
        if FuelType_1 == 'HFO':
            idx_eca = Ship_1_ais['ECA'] != 0
            idx_non_pre = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] < 2020)
            idx_non_post = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] >= 2020)
            
            Ship_1_ais['AE_CO'] = 0.0
            if idx_eca.any():
                Ship_1_ais.loc[idx_eca, 'AE_CO'] = (
                    Ship_1_ais.loc[idx_eca, 'AE_Power'] * 
                    Ship_1_ais.loc[idx_eca, 'TimeInv'] / 3600 * 
                    AE_CO_MDO_1
                )
            if idx_non_pre.any():
                Ship_1_ais.loc[idx_non_pre, 'AE_CO'] = (
                    Ship_1_ais.loc[idx_non_pre, 'AE_Power'] * 
                    Ship_1_ais.loc[idx_non_pre, 'TimeInv'] / 3600 * 
                    AE_CO_1
                )
            if idx_non_post.any():
                Ship_1_ais.loc[idx_non_post, 'AE_CO'] = (
                    Ship_1_ais.loc[idx_non_post, 'AE_Power'] * 
                    Ship_1_ais.loc[idx_non_post, 'TimeInv'] / 3600 * 
                    AE_CO_MDO_1
                )
        else:
            Ship_1_ais['AE_CO'] = (
                Ship_1_ais['AE_Power'] * 
                Ship_1_ais['TimeInv'] / 3600 * 
                AE_CO_1
            )
        
        # 辅机CH4排放（单位：克）
        Ship_1_ais['AE_CH4'] = (
            Ship_1_ais['AE_Power'] * 
            Ship_1_ais['TimeInv'] / 3600 * 
            AE_CH4_1
        )
        
        # 辅机N2O排放（单位：克）
        if FuelType_1 == 'HFO':
            idx_eca = Ship_1_ais['ECA'] != 0
            idx_non_pre = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] < 2020)
            idx_non_post = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] >= 2020)
            
            Ship_1_ais['AE_N2O'] = 0.0
            if idx_eca.any():
                Ship_1_ais.loc[idx_eca, 'AE_N2O'] = (
                    Ship_1_ais.loc[idx_eca, 'AE_Power'] * 
                    Ship_1_ais.loc[idx_eca, 'TimeInv'] / 3600 * 
                    AE_N2O_MDO_1
                )
            if idx_non_pre.any():
                Ship_1_ais.loc[idx_non_pre, 'AE_N2O'] = (
                    Ship_1_ais.loc[idx_non_pre, 'AE_Power'] * 
                    Ship_1_ais.loc[idx_non_pre, 'TimeInv'] / 3600 * 
                    AE_N2O_1
                )
            if idx_non_post.any():
                Ship_1_ais.loc[idx_non_post, 'AE_N2O'] = (
                    Ship_1_ais.loc[idx_non_post, 'AE_Power'] * 
                    Ship_1_ais.loc[idx_non_post, 'TimeInv'] / 3600 * 
                    AE_N2O_MDO_1
                )
        else:
            Ship_1_ais['AE_N2O'] = (
                Ship_1_ais['AE_Power'] * 
                Ship_1_ais['TimeInv'] / 3600 * 
                AE_N2O_1
            )
        
        # 辅机NMVOC排放（单位：克）
        Ship_1_ais['AE_NMVOC'] = (
            Ship_1_ais['AE_Power'] * 
            Ship_1_ais['TimeInv'] / 3600 * 
            AE_NMVOC_1
        )
        
        # =====================================================================
        # 12. 计算锅炉排放（完全修复版）
        # =====================================================================
        Ship_1_ais['BO_Power'] = 0.0
        Ship_1_ais.loc[Ship_1_ais['Mode'] == 'Berth', 'BO_Power'] = BO_Berth_1
        Ship_1_ais.loc[Ship_1_ais['Mode'] == 'Anchored', 'BO_Power'] = BO_Anchored_1
        Ship_1_ais.loc[Ship_1_ais['Mode'] == 'Manoeuvring', 'BO_Power'] = BO_Manoeuvring_1
        Ship_1_ais.loc[Ship_1_ais['Mode'] == 'Sea', 'BO_Power'] = BO_Sea_1
        
        # 锅炉CO2排放（单位：克）
        if FuelType_1 == 'HFO':
            idx_eca = Ship_1_ais['ECA'] != 0
            idx_non_pre = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] < 2020)
            idx_non_post = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] >= 2020)
            
            Ship_1_ais['BO_CO2'] = 0.0
            if idx_eca.any():
                Ship_1_ais.loc[idx_eca, 'BO_CO2'] = (
                    Ship_1_ais.loc[idx_eca, 'BO_Power'] * 
                    Ship_1_ais.loc[idx_eca, 'TimeInv'] / 3600 * 
                    BO_SFC_MDO_1 * 
                    3.206
                )
            if idx_non_pre.any():
                Ship_1_ais.loc[idx_non_pre, 'BO_CO2'] = (
                    Ship_1_ais.loc[idx_non_pre, 'BO_Power'] * 
                    Ship_1_ais.loc[idx_non_pre, 'TimeInv'] / 3600 * 
                    BO_SFC_1 * 
                    3.206
                )
            if idx_non_post.any():
                Ship_1_ais.loc[idx_non_post, 'BO_CO2'] = (
                    Ship_1_ais.loc[idx_non_post, 'BO_Power'] * 
                    Ship_1_ais.loc[idx_non_post, 'TimeInv'] / 3600 * 
                    BO_SFC_MDO_1 * 
                    3.206
                )
        elif FuelType_1 == 'LNG':
            Ship_1_ais['BO_CO2'] = (
                Ship_1_ais['BO_Power'] * 
                Ship_1_ais['TimeInv'] / 3600 * 
                BO_SFC_1 * 
                2.750
            )
        elif FuelType_1 == 'MeOH':
            Ship_1_ais['BO_CO2'] = (
                Ship_1_ais['BO_Power'] * 
                Ship_1_ais['TimeInv'] / 3600 * 
                BO_SFC_1 * 
                1.375
            )
        else:
            Ship_1_ais['BO_CO2'] = (
                Ship_1_ais['BO_Power'] * 
                Ship_1_ais['TimeInv'] / 3600 * 
                BO_SFC_1 * 
                3.206
            )
        
        # 锅炉SO2排放（单位：克）
        Ship_1_ais['BO_SO2'] = 0.0
        
        if FuelType_1 == 'MDO':
            for eca_code in [0, 1, 2, 3, 4, 5]:
                idx = Ship_1_ais['ECA'] == eca_code
                if idx.any():
                    Ship_1_ais.loc[idx, 'BO_SO2'] = (
                        Ship_1_ais.loc[idx, 'BO_Power'] * 
                        BO_SFC_1 * 
                        Ship_1_ais.loc[idx, 'TimeInv'] / 3600 * 
                        SO2_factors[FuelType_1] * 
                        ECA_factors[eca_code]
                    )
        
        elif FuelType_1 == 'HFO':
            for eca_code in [1, 2, 3, 4, 5]:
                idx = Ship_1_ais['ECA'] == eca_code
                if idx.any():
                    Ship_1_ais.loc[idx, 'BO_SO2'] = (
                        Ship_1_ais.loc[idx, 'BO_Power'] * 
                        BO_SFC_MDO_1 * 
                        Ship_1_ais.loc[idx, 'TimeInv'] / 3600 * 
                        SO2_factors[FuelType_1] * 
                        ECA_factors[eca_code]
                    )
            
            idx0_pre = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] < 2020)
            idx0_post = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] >= 2020)
            if idx0_pre.any():
                Ship_1_ais.loc[idx0_pre, 'BO_SO2'] = (
                    Ship_1_ais.loc[idx0_pre, 'BO_Power'] * 
                    BO_SFC_1 * 
                    Ship_1_ais.loc[idx0_pre, 'TimeInv'] / 3600 * 
                    SO2_factors[FuelType_1] * 
                    S_NONECA_PRE2020
                )
            if idx0_post.any():
                Ship_1_ais.loc[idx0_post, 'BO_SO2'] = (
                    Ship_1_ais.loc[idx0_post, 'BO_Power'] * 
                    BO_SFC_MDO_1 * 
                    Ship_1_ais.loc[idx0_post, 'TimeInv'] / 3600 * 
                    SO2_factors[FuelType_1] * 
                    ECA_factors[0]
                )
        
        elif FuelType_1 in ['LNG', 'MeOH']:
            Ship_1_ais['BO_SO2'] = (
                Ship_1_ais['BO_Power'] * 
                BO_SFC_1 * 
                Ship_1_ais['TimeInv'] / 3600 * 
                SO2_factors[FuelType_1]
            )
        
        # 锅炉PM排放（单位：克）
        Ship_1_ais['BO_PM'] = 0.0
        
        if FuelType_1 == 'MDO':
            for eca_code in [0, 1, 2, 3, 4, 5]:
                idx = Ship_1_ais['ECA'] == eca_code
                if idx.any():
                    Ship_1_ais.loc[idx, 'BO_PM'] = (
                        Ship_1_ais.loc[idx, 'BO_Power'] * 
                        Ship_1_ais.loc[idx, 'TimeInv'] / 3600 * 
                        (0.23 + BO_SFC_1 * 7 * 0.02247 * (eca_sulfur[eca_code] - 0.0024))
                    )
        
        elif FuelType_1 == 'HFO':
            for eca_code in [1, 2, 3, 4, 5]:
                idx = Ship_1_ais['ECA'] == eca_code
                if idx.any():
                    Ship_1_ais.loc[idx, 'BO_PM'] = (
                        Ship_1_ais.loc[idx, 'BO_Power'] * 
                        Ship_1_ais.loc[idx, 'TimeInv'] / 3600 * 
                        (0.23 + BO_SFC_MDO_1 * 7 * 0.02247 * (eca_sulfur[eca_code] - 0.0024))
                    )
            
            idx0_pre = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] < 2020)
            idx0_post = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] >= 2020)
            if idx0_pre.any():
                Ship_1_ais.loc[idx0_pre, 'BO_PM'] = (
                    Ship_1_ais.loc[idx0_pre, 'BO_Power'] * 
                    Ship_1_ais.loc[idx0_pre, 'TimeInv'] / 3600 * 
                    (0.23 + BO_SFC_1 * 7 * 0.02247 * (S_NONECA_PRE2020 - 0.0024))
                )
            if idx0_post.any():
                Ship_1_ais.loc[idx0_post, 'BO_PM'] = (
                    Ship_1_ais.loc[idx0_post, 'BO_Power'] * 
                    Ship_1_ais.loc[idx0_post, 'TimeInv'] / 3600 * 
                    (0.23 + BO_SFC_MDO_1 * 7 * 0.02247 * (eca_sulfur[0] - 0.0024))
                )
        
        elif FuelType_1 == 'LNG':
            Ship_1_ais['BO_PM'] = (
                Ship_1_ais['BO_Power'] * 
                Ship_1_ais['TimeInv'] / 3600 * 
                0.03
            )
        
        elif FuelType_1 == 'MeOH':
            for eca_code in [0, 1, 2, 3, 4, 5]:
                idx = Ship_1_ais['ECA'] == eca_code
                if idx.any():
                    Ship_1_ais.loc[idx, 'BO_PM'] = (
                        0.1 * 
                        Ship_1_ais.loc[idx, 'BO_Power'] * 
                        Ship_1_ais.loc[idx, 'TimeInv'] / 3600 * 
                        (0.23 + BO_SFC_MDO_1 * 7 * 0.02247 * (eca_sulfur[eca_code] - 0.0024))
                    )
        
        # 锅炉NOx排放（单位：克）
        Ship_1_ais['BO_NOx'] = (
            Ship_1_ais['BO_Power'] * 
            Ship_1_ais['TimeInv'] / 3600 * 
            BO_NOx_1
        )
        
        # 锅炉CO排放（单位：克）
        if FuelType_1 == 'HFO':
            idx_eca = Ship_1_ais['ECA'] != 0
            idx_non_pre = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] < 2020)
            idx_non_post = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] >= 2020)
            
            Ship_1_ais['BO_CO'] = 0.0
            if idx_eca.any():
                Ship_1_ais.loc[idx_eca, 'BO_CO'] = (
                    Ship_1_ais.loc[idx_eca, 'BO_Power'] * 
                    Ship_1_ais.loc[idx_eca, 'TimeInv'] / 3600 * 
                    BO_CO_MDO_1
                )
            if idx_non_pre.any():
                Ship_1_ais.loc[idx_non_pre, 'BO_CO'] = (
                    Ship_1_ais.loc[idx_non_pre, 'BO_Power'] * 
                    Ship_1_ais.loc[idx_non_pre, 'TimeInv'] / 3600 * 
                    BO_CO_1
                )
            if idx_non_post.any():
                Ship_1_ais.loc[idx_non_post, 'BO_CO'] = (
                    Ship_1_ais.loc[idx_non_post, 'BO_Power'] * 
                    Ship_1_ais.loc[idx_non_post, 'TimeInv'] / 3600 * 
                    BO_CO_MDO_1
                )
        else:
            Ship_1_ais['BO_CO'] = (
                Ship_1_ais['BO_Power'] * 
                Ship_1_ais['TimeInv'] / 3600 * 
                BO_CO_1
            )
        
        # 锅炉CH4排放（单位：克）
        Ship_1_ais['BO_CH4'] = (
            Ship_1_ais['BO_Power'] * 
            Ship_1_ais['TimeInv'] / 3600 * 
            BO_CH4_1
        )
        
        # 锅炉N2O排放（单位：克）
        if FuelType_1 == 'HFO':
            idx_eca = Ship_1_ais['ECA'] != 0
            idx_non_pre = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] < 2020)
            idx_non_post = (Ship_1_ais['ECA'] == 0) & (Ship_1_ais['Year'] >= 2020)
            
            Ship_1_ais['BO_N2O'] = 0.0
            if idx_eca.any():
                Ship_1_ais.loc[idx_eca, 'BO_N2O'] = (
                    Ship_1_ais.loc[idx_eca, 'BO_Power'] * 
                    Ship_1_ais.loc[idx_eca, 'TimeInv'] / 3600 * 
                    BO_N2O_MDO_1
                )
            if idx_non_pre.any():
                Ship_1_ais.loc[idx_non_pre, 'BO_N2O'] = (
                    Ship_1_ais.loc[idx_non_pre, 'BO_Power'] * 
                    Ship_1_ais.loc[idx_non_pre, 'TimeInv'] / 3600 * 
                    BO_N2O_1
                )
            if idx_non_post.any():
                Ship_1_ais.loc[idx_non_post, 'BO_N2O'] = (
                    Ship_1_ais.loc[idx_non_post, 'BO_Power'] * 
                    Ship_1_ais.loc[idx_non_post, 'TimeInv'] / 3600 * 
                    BO_N2O_MDO_1
                )
        else:
            Ship_1_ais['BO_N2O'] = (
                Ship_1_ais['BO_Power'] * 
                Ship_1_ais['TimeInv'] / 3600 * 
                BO_N2O_1
            )
        
        # 锅炉NMVOC排放（单位：克）
        Ship_1_ais['BO_NMVOC'] = (
            Ship_1_ais['BO_Power'] * 
            Ship_1_ais['TimeInv'] / 3600 * 
            BO_NMVOC_1
        )
        
        # =====================================================================
        # 13. 计算总排放（ME + AE + BO）
        # =====================================================================
        for pollutant in pollutants:
            Ship_1_ais[f'Total_{pollutant}'] = (
                Ship_1_ais[f'ME_{pollutant}'] +
                Ship_1_ais[f'AE_{pollutant}'] +
                Ship_1_ais[f'BO_{pollutant}']
            )
        
        # =====================================================================
        # 13.5 政策情景点级 delta（研究方案 v6 第七章：SECA 0.1% / NECA Tier III）
        # ---------------------------------------------------------------------
        # 单遍多情景：不复制基线公式，对基线列做精确比例变换求减排量：
        #   SO2 ∝ SFC×S；PM ∝ (0.23 + SFC(×load)×7×0.02247×(S−0.0024)) 仿射项；
        #   NOx 用 TierIII/基线 EF 之比。HFO 区内换 MDO 油(SFC 同步切换)；
        #   MDO 只降硫；MeOH 仅 PM 依赖硫率；LNG 无硫依赖。前置系数在比值中消去，
        #   因此与基线各分支(含 pre-2020 AE/BO 缺 PM_LAF 的写法)逐点严格一致。
        # 与现行 ECA 重叠取更严标准：S 已≤情景硫率、或已按现行 NECA TierIII 的点 delta=0。
        # 组合情景无需单算：SECA 只动 SO2/PM，NECA 只动 NOx，无交互。
        # =====================================================================
        if policy_zones is not None and len(policy_zones) > 0:
            _cfg_pol = policy_cfg or {}
            _seca_s = float(_cfg_pol.get('seca_sulfur', 0.001))
            _neca_by = int(_cfg_pol.get('neca_build_year', 2016))
            _n = len(Ship_1_ais)

            if 'PZ_Zone' not in Ship_1_ais.columns:
                Ship_1_ais['PZ_Zone'] = _policy_zone_labels(
                    Ship_1_ais, policy_zones)
            _d_so2 = np.zeros(_n)
            _d_pm = np.zeros(_n)
            _d_nox = np.zeros(_n)
            # 政策区内已经受到同等或更严现行管制、因而在反事实中不再产生
            # 新增减排的基线排放。该量只用于报告“无边际效果占比”，不参与
            # S1--S7 的减排计算。
            _nme_so2 = np.zeros(_n)
            _nme_pm = np.zeros(_n)
            _nme_nox = np.zeros(_n)

            _inz = (Ship_1_ais['PZ_Zone'] != '').to_numpy()
            if _inz.any():
                _eca_arr = Ship_1_ais['ECA'].to_numpy()
                _pre2020 = (Ship_1_ais['Year'].to_numpy() < 2020) & (_eca_arr == 0)
                _s_base = np.array([eca_sulfur[c] for c in range(6)])[_eca_arr]
                if FuelType_1 == 'HFO':
                    _s_base = np.where(_pre2020, S_NONECA_PRE2020, _s_base)
                _s_scn = np.where(_inz, np.minimum(_s_base, _seca_s), _s_base)
                _chg = _inz & (_s_scn < _s_base)
                _already_seca = _inz & (_s_base <= _seca_s + 1e-12)

                # 只把硫含量约束会作用到的燃料计入“现行管制无边际效果”。
                # LNG 的硫氧化物本来即近零，不能归因为现行 SECA。
                if FuelType_1 in ('HFO', 'MDO'):
                    _nme_so2 = np.where(
                        _already_seca, Ship_1_ais['Total_SO2'].to_numpy(), 0.0)
                if FuelType_1 in ('HFO', 'MDO', 'MeOH'):
                    _nme_pm = np.where(
                        _already_seca, Ship_1_ais['Total_PM'].to_numpy(), 0.0)

                # --- SECA delta: SO2(HFO/MDO) 与 PM(HFO/MDO/MeOH), ME+AE+BO ---
                if _chg.any() and FuelType_1 in ('HFO', 'MDO', 'MeOH'):
                    _lc = Ship_1_ais['load_correction'].to_numpy()
                    for _eng, _sfc_std, _sfc_mdo, _has_load in (
                            ('ME', ME_SFC_1, ME_SFC_MDO_1, True),
                            ('AE', AE_SFC_1, AE_SFC_MDO_1, False),
                            ('BO', BO_SFC_1, BO_SFC_MDO_1, False)):
                        if FuelType_1 == 'HFO':
                            _sfc_b = np.where(_pre2020, _sfc_std, _sfc_mdo)
                            _sfc_s = np.where(_chg, _sfc_mdo, _sfc_b)
                        elif FuelType_1 == 'MDO':
                            _sfc_b = np.full(_n, float(_sfc_std))
                            _sfc_s = _sfc_b
                        else:  # MeOH: 基线各分支恒用 MDO SFC，无换油切换
                            _sfc_b = np.full(_n, float(_sfc_mdo))
                            _sfc_s = _sfc_b
                        if FuelType_1 in ('HFO', 'MDO'):
                            _b_so2 = Ship_1_ais[f'{_eng}_SO2'].to_numpy()
                            _r = (_sfc_s * _s_scn) / (_sfc_b * _s_base)
                            _d_so2 += np.where(_chg, _b_so2 * (1.0 - _r), 0.0)
                        _k = 7 * 0.02247 * (_lc if _has_load else 1.0)
                        _par_b = 0.23 + _sfc_b * _k * (_s_base - 0.0024)
                        _par_s = 0.23 + _sfc_s * _k * (_s_scn - 0.0024)
                        _b_pm = Ship_1_ais[f'{_eng}_PM'].to_numpy()
                        _pm_r = np.divide(_par_s, _par_b,
                                          out=np.ones(_n), where=_par_b != 0)
                        _d_pm += np.where(_chg, _b_pm * (1.0 - _pm_r), 0.0)

                # --- NECA delta: 仅主机 NOx, 仅 Build_Year ≥ 生效建造年的船 ---
                if Build_Year_1 >= _neca_by and ME_NOx_1 > ME_NOx_3_1:
                    _already_t3 = (((_eca_arr == 1) & (Build_Year_1 >= 2021)) |
                                   ((_eca_arr == 3) & (Build_Year_1 >= 2016)))
                    _nme_nox = np.where(
                        _inz & _already_t3,
                        Ship_1_ais['ME_NOx'].to_numpy(),
                        0.0,
                    )
                    _elig = _inz & (~_already_t3)
                    if _elig.any():
                        _d_nox = np.where(
                            _elig,
                            Ship_1_ais['ME_NOx'].to_numpy()
                            * (1.0 - ME_NOx_3_1 / ME_NOx_1),
                            0.0)

            Ship_1_ais['D_SO2_SECA'] = _d_so2
            Ship_1_ais['D_PM_SECA'] = _d_pm
            Ship_1_ais['D_NOx_NECA'] = _d_nox
            Ship_1_ais['NME_SO2_CURRENT'] = _nme_so2
            Ship_1_ais['NME_PM_CURRENT'] = _nme_pm
            Ship_1_ais['NME_NOx_CURRENT'] = _nme_nox

        # =====================================================================
        # 14. 港口停靠识别（berth-first 双 buffer 判定，与 shipping pipeline 对齐）
        # ---------------------------------------------------------------------
        # 输入：
        #   port_buffers_3nm — 港口区域 buffer（外层，泊位 union + Voronoi 互斥）
        #   berth_buffers    — 泊位 buffer（内层，按港 dissolve）
        # 步骤：
        #   1) 双 sjoin 给每个 AIS 点打两个标签：Port_Name（外层归属）、berth_port_name（内层归属）
        #   2) berth-first 算法：找 Mode=Berth ∧ 内外标签同港 的连续靠泊段，
        #      按索引间隔 ≤100 分组（同一次停靠 vs 不同次），
        #      从首/末靠泊点向 port_buffer 边界倒推/前推，得到完整在港时段
        #   3) 用 visits 区间重新设值 Port_Event / Port_Name —— Section 15 状态机的输入
        # 注意：未在任何 visit 区间内的点（哪怕落在 port_buffer 内但没有 berth 确认）
        #       一律按 At_Sea 处理，等价于「过路、未真正靠泊」。
        # =====================================================================
        Ship_1_ais = Ship_1_ais.reset_index(drop=True)

        need_visit_processing = (
            (not policy_only)
            or str(Shiptype_1).strip().casefold() == "container"
        )
        use_visit_pipeline = (
            need_visit_processing
            and
            port_buffers_3nm is not None and len(port_buffers_3nm) > 0
            and berth_buffers is not None and len(berth_buffers) > 0
        )
        all_visits = []

        if use_visit_pipeline:
            has_canal = (
                canal_corridor_buffers is not None
                and len(canal_corridor_buffers) > 0
                and canal_waiting_buffers is not None
                and len(canal_waiting_buffers) > 0
            )
            if has_canal:
                Ship_1_ais = precompute_spatial_tags(
                    Ship_1_ais,
                    port_buffers_3nm,
                    berth_buffers,
                    canal_corridor_buffers,
                    canal_waiting_buffers,
                    canal_endpoint_buffers,
                )
            else:
                ship_gdf = gpd.GeoDataFrame(
                    Ship_1_ais.copy(),
                    geometry=gpd.points_from_xy(
                        Ship_1_ais["Longitude"], Ship_1_ais["Latitude"]
                    ),
                    crs="EPSG:4326",
                )
                pbuf = port_buffers_3nm[
                    ["Port.Name", "geometry"]
                ].copy()
                if pbuf.crs is None:
                    pbuf = pbuf.set_crs("EPSG:4326")
                j_p = gpd.sjoin(ship_gdf, pbuf, predicate="within", how="left")
                j_p = j_p.reset_index().drop_duplicates(
                    subset="index", keep="first"
                ).set_index("index")
                j_p = j_p.reindex(ship_gdf.index)
                Ship_1_ais["Port_Name"] = j_p["Port.Name"].values

                bbuf = berth_buffers[["berth_port_name", "geometry"]].copy()
                if bbuf.crs is None:
                    bbuf = bbuf.set_crs("EPSG:4326")
                j_b = gpd.sjoin(ship_gdf, bbuf, predicate="within", how="left")
                j_b = j_b.reset_index().drop_duplicates(
                    subset="index", keep="first"
                ).set_index("index")
                j_b = j_b.reindex(ship_gdf.index)
                Ship_1_ais["berth_port_name"] = j_b["berth_port_name"].values

            Ship_1_ais = strip_canal_waiting_subports(Ship_1_ais)
            if canal_coor is not None and len(canal_coor) > 0:
                Ship_1_ais = apply_canal_tags_r_style(
                    Ship_1_ais, canal_coor, threshold_km=canal_proximity_km
                )

            port_visits = _shipping_visits_berth_first(
                Ship_1_ais,
                port_coor,
                berth_gap_tolerance_sec=berth_gap_tolerance_sec,
            )
            canal_visits = []
            if has_canal:
                canal_visits = _build_canal_visits_from_tags(Ship_1_ais, port_coor)
            all_visits = port_visits + canal_visits
            all_visits.sort(key=lambda x: float(x["Arrive_UTC"]))

            # 周度港口排放(绿色韧性研究): visit 归属 + 锚地扩展, ISO 周聚合
            if weekly_cfg and weekly_cfg.get("enabled"):
                result['port_weekly'] = build_weekly_port_emissions(
                    Ship_1_ais, all_visits, port_coor,
                    float(weekly_cfg.get("anchor_km", 25.0)))
        elif need_visit_processing:
            # 未提供 berth_buffers：保留旧的单 buffer / cKDTree 兜底逻辑（向后兼容）
            if port_buffers_3nm is not None and len(port_buffers_3nm) > 0:
                ship_gdf = gpd.GeoDataFrame(
                    Ship_1_ais.copy(),
                    geometry=gpd.points_from_xy(Ship_1_ais['Longitude'], Ship_1_ais['Latitude']),
                    crs='EPSG:4326'
                )
                pbuff = port_buffers_3nm.copy()
                if pbuff.crs is None:
                    pbuff.set_crs('EPSG:4326', inplace=True)
                join_cols = [c for c in ['Port.Name', 'Lon_port', 'Lat_port', 'geometry'] if c in pbuff.columns]
                joined = gpd.sjoin(ship_gdf, pbuff[join_cols], predicate='within', how='left')
                joined = joined.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')
                joined = joined.reindex(Ship_1_ais.index)
                Ship_1_ais['Port_Name'] = joined['Port.Name'].values
                Ship_1_ais['Port_Event'] = np.where(Ship_1_ais['Port_Name'].notna(), 'In_Port', 'At_Sea')
            elif len(port_coor) > 0:
                Ship_1_ais_berth = Ship_1_ais[Ship_1_ais['Mode'] == 'Berth'].copy()
                Ship_1_ais_notberth = Ship_1_ais[Ship_1_ais['Mode'] != 'Berth'].copy()
                if len(Ship_1_ais_berth) > 0:
                    port_coords = np.array(list(zip(port_coor['Lon_berth'], port_coor['Lat_berth'])))
                    ship_coords = np.array(list(zip(Ship_1_ais_berth['Longitude'], Ship_1_ais_berth['Latitude'])))
                    tree = cKDTree(port_coords)
                    _, nearest_indices = tree.query(ship_coords)
                    distances_km = [
                        haversine_dist(
                            ship_coords[i, 0], ship_coords[i, 1],
                            port_coords[nearest_indices[i], 0],
                            port_coords[nearest_indices[i], 1],
                        )
                        for i in range(len(ship_coords))
                    ]
                    Ship_1_ais_berth['distances'] = distances_km
                    is_distant = Ship_1_ais_berth['distances'] > berth_distance_threshold
                    Ship_1_ais_berth['Port_Name'] = np.where(
                        is_distant, None, port_coor.iloc[nearest_indices]['Port.Name'].values
                    )
                    Ship_1_ais_notberth['Port_Name'] = None
                    Ship_1_ais = pd.concat([Ship_1_ais_berth, Ship_1_ais_notberth], ignore_index=True)
                    Ship_1_ais = Ship_1_ais.sort_values('UTCtime').reset_index(drop=True)
                    Ship_1_ais['Port_Event'] = np.where(Ship_1_ais['Port_Name'].notna(), 'In_Port', 'At_Sea')
                else:
                    Ship_1_ais['Port_Name'] = None
                    Ship_1_ais['Port_Event'] = 'At_Sea'
            else:
                Ship_1_ais['Port_Name'] = None
                Ship_1_ais['Port_Event'] = 'At_Sea'
        
        # =====================================================================
        # 15. 构建港口/运河节点排放、连边排放、航段距离/时间（与 shipping 挂靠一致）
        # =====================================================================
        if not need_visit_processing:
            port_emissions_list = []
            edge_emissions_list = []
            port_operational_visits_list = []
            edge_voyage_list = []
            ops_calls_list = []
        elif use_visit_pipeline:
            if all_visits:
                (
                    port_emissions_list,
                    edge_emissions_list,
                    port_operational_visits_list,
                    edge_voyage_list,
                    ops_calls_list,
                ) = _build_emission_network_from_visits(
                    Ship_1_ais,
                    all_visits,
                    pollutants,
                    Ship_1_MMSI,
                    DWT_1,
                    Shiptype_1,
                    FuelType_1,
                    Gross_Tonnage_1,
                    Build_Year_1,
                    IMO_1,
                    GT_Source_1,
                    GT_Match_Method_1,
                )
            else:
                port_emissions_list = []
                edge_emissions_list = []
                port_operational_visits_list = []
                edge_voyage_list = []
                ops_calls_list = []
        else:
            port_emissions_list = []
            edge_emissions_list = []
            port_operational_visits_list = []
            edge_voyage_list = []
            ops_calls_list = []

            pe_arr = Ship_1_ais["Port_Event"].values
            pn_arr = Ship_1_ais["Port_Name"].values
            n_rows = len(Ship_1_ais)

            def _flush_port(s_idx, e_idx, pname):
                port_data = Ship_1_ais.iloc[s_idx:e_idx]
                if len(port_data) == 0:
                    return
                turnaround_time = (
                    port_data["UTCtime"].iloc[-1] - port_data["UTCtime"].iloc[0]
                ) / 3600
                if "berth_port_name" in port_data.columns:
                    berth_mask = (
                        (port_data["Mode"] == "Berth")
                        & port_data["berth_port_name"].notna()
                        & (port_data["berth_port_name"] == port_data["Port_Name"])
                    )
                else:
                    berth_mask = port_data["Mode"] == "Berth"
                berth_time = port_data.loc[berth_mask, "TimeInv"].sum() / 3600
                at_sea_mask = port_data["Mode"].map(_mode_is_at_sea_for_waiting)
                at_sea_time = port_data.loc[at_sea_mask, "TimeInv"].sum() / 3600
                in_port_h = turnaround_time
                waiting_h = max(0.0, in_port_h - berth_time - at_sea_time)
                mov_mask = ~port_data["Mode"].map(_exclude_berth_anchored_for_speed)
                spd_series = port_data.loc[mov_mask, "Speed"]
                avg_spd_mov = (
                    float(spd_series.mean())
                    if len(spd_series) > 0
                    else np.nan
                )
                ym = pd.to_datetime(
                    port_data["UTCtime"].iloc[0], unit="s", utc=True
                ).strftime("%Y%m")
                row = {
                    "Port_Name": pname,
                    "MMSI": Ship_1_MMSI,
                    "Turnaround_Hours": turnaround_time,
                    "Berth_Hours": berth_time,
                    "DWT": DWT_1,
                    "Visit_Count": 1,
                }
                for pol in pollutants:
                    row[pol] = port_data[f"Total_{pol}"].sum()
                port_emissions_list.append(row)
                port_operational_visits_list.append(
                    {
                        "YearMonth": ym,
                        "Port_Name": pname,
                        "MMSI": Ship_1_MMSI,
                        "Ship_Type": Shiptype_1,
                        "Fuel_Type": FuelType_1,
                        "DWT": DWT_1,
                        "Arrive_UTC": float(port_data["UTCtime"].iloc[0]),
                        "Depart_UTC": float(port_data["UTCtime"].iloc[-1]),
                        "InPort_Hours": in_port_h,
                        "Berth_Hours": berth_time,
                        "AtSea_Hours": at_sea_time,
                        "Waiting_Hours": waiting_h,
                        "Canal_Sailing_Hours": 0.0,
                        "Avg_Speed_InPort_Moving_kn": avg_spd_mov,
                    }
                )

            def _flush_edge(s_idx, e_idx, fp, tp):
                if s_idx >= e_idx:
                    return
                seg = Ship_1_ais.iloc[s_idx:e_idx]
                row = {"From_Port": fp, "To_Port": tp, "MMSI": Ship_1_MMSI}
                for pol in pollutants:
                    row[pol] = seg[f"Total_{pol}"].sum()
                edge_emissions_list.append(row)
                if e_idx > s_idx:
                    edge_voyage_list.append(
                        {
                            "From_Port": fp,
                            "To_Port": tp,
                            "MMSI": Ship_1_MMSI,
                            "Sailing_Hours": max(
                                0.0,
                                (
                                    float(seg["UTCtime"].iloc[-1])
                                    - float(seg["UTCtime"].iloc[0])
                                )
                                / 3600.0,
                            ),
                            "Sailing_Distance_km": _track_distance_km_from_ais(
                                Ship_1_ais, s_idx, e_idx - 1
                            ),
                        }
                    )

            current_port = None
            from_port = None
            segment_start_idx = 0

            for i in range(n_rows):
                ev = pe_arr[i]
                if ev == "In_Port":
                    if current_port is None:
                        current_port = pn_arr[i]
                        if from_port is not None:
                            _flush_edge(segment_start_idx, i, from_port, current_port)
                        segment_start_idx = i
                elif ev == "At_Sea":
                    if current_port is not None:
                        _flush_port(segment_start_idx, i, current_port)
                        from_port = current_port
                        current_port = None
                        segment_start_idx = i

            if current_port is not None:
                _flush_port(segment_start_idx, n_rows, current_port)

        port_emissions = (
            pd.DataFrame(port_emissions_list) if port_emissions_list else pd.DataFrame()
        )
        edge_emissions = (
            pd.DataFrame(edge_emissions_list) if edge_emissions_list else pd.DataFrame()
        )
        result["port_operational_visits"] = (
            pd.DataFrame(port_operational_visits_list)
            if port_operational_visits_list
            else pd.DataFrame()
        )
        result["edge_voyage_stats"] = (
            pd.DataFrame(edge_voyage_list) if edge_voyage_list else pd.DataFrame()
        )
        result["ops_calls"] = (
            pd.DataFrame(ops_calls_list) if ops_calls_list else pd.DataFrame()
        )

        # =====================================================================
        # 16. 城市集水区点级排放聚合（研究方案 3.6.5：30km 集水区、重叠重复计入）
        # =====================================================================
        if (not policy_only and city_catchments is not None
                and len(city_catchments) > 0):
            from city_aggregation import aggregate_points_to_cities
            result['city_emissions'] = aggregate_points_to_cities(
                Ship_1_ais, city_catchments, pollutants, Ship_1_MMSI
            )

        # =====================================================================
        # 16.5 政策情景 delta 聚合（城市集水区分布 + 政策区精确合计）
        # =====================================================================
        if policy_zones is not None and len(policy_zones) > 0 \
                and 'PZ_Zone' in Ship_1_ais.columns:
            from city_aggregation import (
                aggregate_ops_calls_to_cities,
                aggregate_policy_deltas_to_cities, policy_delta_zone_totals)
            if city_catchments is not None and len(city_catchments) > 0:
                result['policy_delta_city'] = aggregate_policy_deltas_to_cities(
                    Ship_1_ais, city_catchments, Ship_1_MMSI)
            result['policy_delta_total'] = policy_delta_zone_totals(
                Ship_1_ais, Ship_1_MMSI)
            if (str(Shiptype_1).strip().casefold() == 'container'
                    and city_catchments is not None and len(city_catchments) > 0):
                result['ops_call_city'] = aggregate_ops_calls_to_cities(
                    Ship_1_ais, city_catchments, Ship_1_MMSI)

        # =====================================================================
        # 17. 返回结果
        # =====================================================================
        # 全量逐点排放（不做港口/航段归属、不做集水区过滤）= 守恒的真实总排放。
        # 对比口径：port+edge 只含"被归属"的点（漏掉首个挂靠前/末个挂靠后的航段、
        # 同港往返段）；city_emissions 只含 30km 集水区内的点且重叠重复计入。
        # 要报告全球绝对总量须用 total_emissions。
        result['total_emissions'] = {} if policy_only else {
            pol: float(Ship_1_ais[f'Total_{pol}'].sum()) for pol in pollutants
        }
        # 逐月真实总量(守恒)：用于核对各月 港口+航段/集水区 的捕获率
        _ym_all = pd.to_datetime(
            Ship_1_ais['UTCtime'], unit='s', utc=True
        ).dt.strftime('%Y%m')
        result['total_emissions_by_month'] = {} if policy_only else {
            ym: {pol: float(g[f'Total_{pol}'].sum()) for pol in pollutants}
            for ym, g in Ship_1_ais.groupby(_ym_all.values)
        }

        result['success'] = True
        result['port_emissions'] = port_emissions
        result['edge_emissions'] = edge_emissions

        return result
        
    except Exception as e:
        result['error_message'] = str(e)
        return result
