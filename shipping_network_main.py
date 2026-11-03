"""
航运网络主程序：从 AIS 数据库构建港口网络（无排放），输出节点统计与边总 DWT。
sjoin 预计算版：所有空间缓冲区在主进程中预构建，worker 中只做 sjoin 查询。
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from typing import List, Sequence, Tuple

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

from shipping_network_functions import (
    CANAL_MIN_PASSAGE_SEC,
    CANAL_PORT_NAMES,
    aggregate_edge_tonnage,
    aggregate_port_node_features,
    build_port_area_buffers,
    build_berth_buffers,
    build_canal_corridor_buffers,
    build_canal_waiting_buffers,
    build_canal_endpoint_buffers,
    is_non_port_name,
    process_ship_shipping_network,
)


# ---------------------------------------------------------------------------
# 港口筛选
# ---------------------------------------------------------------------------
OFFSHORE_NAME_KEYWORDS = [
    "FSO", "FPSO", "OFFSHORE", "PLATFORM", "FLNG", "FSRU",
    "SBM", "CBM", "MOORING", "OIL RIG", "RIG",
]

OFFSHORE_FACILITY_SUBSTR = (
    "offshore", "off shore", "marine construction", "oil rig", "海工",
)


def _name_is_offshore(port_name: str) -> bool:
    u = str(port_name).upper()
    return any(kw.upper() in u for kw in OFFSHORE_NAME_KEYWORDS)


def _row_is_offshore(row: pd.Series) -> bool:
    if _name_is_offshore(row.get("Port.Name", "")):
        return True
    ft = row.get("Facility.Type")
    if pd.isna(ft):
        return False
    s = str(ft).strip().casefold()
    return any(x in s for x in OFFSHORE_FACILITY_SUBSTR)


def load_port_coordinates(
    path: str,
    merge_dict: dict | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    port_coor = pd.read_excel(path)

    if "Port_Type" in port_coor.columns:
        before = len(port_coor)
        sea_mask = port_coor["Port_Type"].astype(str).str.strip().str.casefold() == "sea"
        port_coor = port_coor[sea_mask].copy()
        print(f"  仅保留 Port_Type==Sea（沿海）: {len(port_coor)}/{before} 行")
    else:
        print("  警告: 港口表无 Port_Type 列，无法按 Sea 剔除内河，将使用全部行")

    before_off = len(port_coor)
    port_coor = port_coor[~port_coor.apply(_row_is_offshore, axis=1)].copy()
    print(f"  剔除离岸/海工设施: {before_off - len(port_coor)} 行 → 剩余 {len(port_coor)} 行")

    # 在合并 / 拆分运河之前，剔除非港口节点（锚地/海峡/SPM/Oil Terminal/Lightering）
    # 运河条目（Suez/Panama 等）名称不含这些关键词，不会被误剔
    before_np = len(port_coor)
    non_port_mask = port_coor["Port.Name"].apply(is_non_port_name)
    port_coor = port_coor[~non_port_mask].copy()
    print(f"  剔除非港口节点（Anchorage/Strait/SPM/Oil Terminal/Lightering）: "
          f"{before_np - len(port_coor)} 行 → 剩余 {len(port_coor)} 行")

    merge_dict = merge_dict or {}
    for old, new in merge_dict.items():
        port_coor.loc[port_coor["Port.Name"] == old, "Port.Name"] = new

    canal_mask = port_coor["Port.Name"].isin(CANAL_PORT_NAMES)
    canal_coor = port_coor[canal_mask].copy()
    port_coor = port_coor[~canal_mask].copy()

    req = ["Port.Name", "Lon_port", "Lat_port", "Lon_berth", "Lat_berth"]
    miss = set(req) - set(port_coor.columns)
    if miss:
        raise ValueError(f"港口表缺少列: {miss}")

    for col in ["Lon_berth", "Lat_berth", "Lon_port", "Lat_port"]:
        port_coor[col] = pd.to_numeric(port_coor[col], errors="coerce")
    port_coor = port_coor.dropna(subset=req).reset_index(drop=True)
    port_coor = port_coor[
        (port_coor["Lon_berth"].between(-180, 180))
        & (port_coor["Lat_berth"].between(-90, 90))
        & (port_coor["Lon_port"].between(-180, 180))
        & (port_coor["Lat_port"].between(-90, 90))
    ].reset_index(drop=True)

    port_coor_unique = port_coor.drop_duplicates(subset=["Lon_port", "Lat_port"], keep="first").reset_index(drop=True)
    return port_coor, canal_coor, port_coor_unique


def extract_all_mmsi(db_paths: List[str], ship_type_filter: str | None = None) -> pd.DataFrame:
    all_ships_info = []
    for db_path in db_paths:
        if not os.path.exists(db_path):
            print(f"  跳过（不存在）: {db_path}")
            continue
        conn = sqlite3.connect(db_path, timeout=30)
        try:
            if ship_type_filter:
                q = "SELECT DISTINCT * FROM ship_info WHERE Ship_Type = ?"
                ship_info = pd.read_sql_query(q, conn, params=(ship_type_filter,))
            else:
                ship_info = pd.read_sql_query("SELECT DISTINCT * FROM ship_info", conn)
            all_ships_info.append(ship_info)
        finally:
            conn.close()
    if not all_ships_info:
        raise ValueError("未能从数据库读取 ship_info")
    combined = pd.concat(all_ships_info, ignore_index=True)
    return combined.drop_duplicates(subset="MMSI", keep="first")


def extract_ship_trajectory_multi_db(mmsi: int, db_paths: List[str]) -> pd.DataFrame:
    parts = []
    for db_path in db_paths:
        if not os.path.exists(db_path):
            continue
        conn = sqlite3.connect(db_path, timeout=30)
        try:
            ship_ais = pd.read_sql_query(
                f"SELECT * FROM ais_tracks WHERE MMSI = {int(mmsi)} ORDER BY UTCtime", conn
            )
            if len(ship_ais) > 0:
                parts.append(ship_ais)
        finally:
            conn.close()
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset="UTCtime", keep="first")
    return combined.sort_values("UTCtime").reset_index(drop=True)


def filter_ais_by_months(ship_ais: pd.DataFrame, yyyymm_list: Sequence[str]) -> pd.DataFrame:
    if not yyyymm_list:
        return ship_ais
    want = {int(x) for x in yyyymm_list}
    dt = pd.to_datetime(ship_ais["UTCtime"], unit="s", utc=True)
    ym = dt.dt.year * 100 + dt.dt.month
    return ship_ais[ym.isin(want)].reset_index(drop=True)


def process_single_ship_worker(args: tuple) -> dict:
    (
        mmsi,
        ship_info_row,
        db_paths,
        yyyymm_list,
        port_coor,
        canal_coor,
        port_buffers,
        berth_km,
        canal_km,
        berth_gap_tolerance_sec,
        berth_buffers,
        canal_corridor_buffers,
        canal_waiting_buffers,
        canal_endpoint_buffers,
    ) = args

    try:
        ship_ais = extract_ship_trajectory_multi_db(mmsi, db_paths)
        ship_ais = filter_ais_by_months(ship_ais, yyyymm_list)
        if len(ship_ais) == 0:
            return {"success": False, "mmsi": mmsi, "error_message": "筛选月份后无 AIS"}

        return process_ship_shipping_network(
            ship_ais_data=ship_ais,
            ship_info=ship_info_row,
            port_coor=port_coor,
            port_buffers=port_buffers,
            berth_distance_threshold_km=berth_km,
            canal_coor=canal_coor if len(canal_coor) > 0 else None,
            canal_proximity_km=canal_km,
            berth_gap_tolerance_sec=berth_gap_tolerance_sec,
            berth_buffers=berth_buffers,
            canal_corridor_buffers=canal_corridor_buffers,
            canal_waiting_buffers=canal_waiting_buffers,
            canal_endpoint_buffers=canal_endpoint_buffers,
        )
    except Exception as e:
        return {"success": False, "mmsi": mmsi, "error_message": str(e)}


def parse_months(s: str) -> List[str]:
    s = s.replace(";", ",")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) == 1 and " " in parts[0]:
        parts = [p for p in parts[0].split() if p.strip()]
    out = []
    for p in parts:
        if len(p) != 6 or not p.isdigit():
            raise ValueError(f"月份格式应为 YYYYMM: {p!r}")
        out.append(p)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="航运网络（无排放）：节点挂靠统计 + 边总 DWT；sjoin 预计算版"
    )
    parser.add_argument("--months", type=str, required=True,
                        help="处理月份 YYYYMM，逗号分隔")
    parser.add_argument("--berth-radius-km", type=float, default=10.0,
                        help="构建港口区域时，每个泊位的缓冲半径（km）。"
                             "同一港口所有泊位的圆做 union 形成该港多边形（互斥扣除邻港泊位）。"
                             "默认 10 km：对齐'港界本体'语义（码头+港内航道+近港锚地）。"
                             "远锚泊（≤20 km）通过 anchor 倒推单独补救，用于计算等待时间。")
    parser.add_argument("--berth-km", type=float, default=5.0,
                        help="靠泊判定距离（km）")
    parser.add_argument("--canal-km", type=float, default=10.0,
                        help="运河走廊距离（km）")
    parser.add_argument("--berth-gap-tolerance-sec", type=float, default=600.0,
                        help="靠泊中断容忍（秒），默认 600 = 10 分钟")
    parser.add_argument("--db-dir", type=str, default=r"E:\SQL_AIS_SingleShip")
    parser.add_argument("--db-file", type=str, default="")
    parser.add_argument("--port-xlsx", type=str,
                        default=r"C:\Users\HP\Desktop\Port Data\port_data_clean.xlsx")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ship-type", type=str, default="")
    args = parser.parse_args()

    months = parse_months(args.months)
    label = args.label.strip() or "_".join(months)
    out_dir = args.output_dir.strip() or os.path.join(os.getcwd(), "shipping_network_out", label)
    os.makedirs(out_dir, exist_ok=True)

    db_file = (args.db_file or "").strip()
    if db_file:
        db_file = os.path.abspath(db_file)
        if not os.path.isfile(db_file):
            raise SystemExit(f"--db-file 不是有效文件: {db_file}")
        db_paths = [db_file]
        print(f"使用单库: {db_paths[0]}")
    else:
        db_paths = []
        for m in months:
            p = os.path.join(args.db_dir, f"ais_{m}.db")
            if os.path.exists(p):
                db_paths.append(p)
            else:
                print(f"警告: 未找到数据库 {p}")
        if not db_paths:
            raise SystemExit("没有可用的数据库文件。")

    print(f"月份: {months}")
    print(f"港口区域泊位缓冲半径: {args.berth_radius_km} km")
    print(f"靠泊距离阈值: {args.berth_km} km")
    print(f"运河距离阈值: {args.canal_km} km")
    print(f"靠泊中断容忍: {args.berth_gap_tolerance_sec} s")
    print(f"输出目录: {out_dir}")
    sys.stdout.flush()

    port_coor, canal_coor, port_coor_unique = load_port_coordinates(
        args.port_xlsx,
        merge_dict={"Ningbo": "Ningbo-Zhoushan", "Zhoushan": "Ningbo-Zhoushan"},
    )
    print(f"主港口（泊位行，不含运河）: {len(port_coor)}，运河记录行: {len(canal_coor)}，唯一港址: {len(port_coor_unique)}")
    sys.stdout.flush()

    # =====================================================================
    # 在主进程中预构建所有空间缓冲区（只做一次）
    # =====================================================================
    print("构建空间缓冲区...")
    sys.stdout.flush()
    # 港口区域 = 该港全部泊位的 berth_radius_km 圆 union（按 Port.Name dissolve），
    # 并对邻港做互斥扣除（A 港多边形里若包含 B 港泊位，则把这些 B 泊位的圆从 A 中减掉）。
    # 这种构建方式天然解决了 Ningbo-Zhoushan / Rotterdam / Houston 这类
    # 地理跨度大的港群被单一圆形 buffer 漏算远端码头的问题。
    port_buffers = build_port_area_buffers(
        port_coor,
        berth_radius_km=args.berth_radius_km,
    )
    print(f"  港口缓冲区: {len(port_buffers)} 个（每个 = 该港全部泊位 {args.berth_radius_km} km 圆 union，邻港互斥扣除）")

    berth_buffers = build_berth_buffers(port_coor, args.berth_km, canal_coor)
    print(f"  泊位缓冲区: {len(berth_buffers)} 个（按港口名合并）")

    canal_corridor_buffers = build_canal_corridor_buffers(canal_coor, args.canal_km)
    print(f"  运河走廊缓冲区: {len(canal_corridor_buffers)} 个")

    canal_waiting_buffers = build_canal_waiting_buffers(port_coor, canal_coor)
    print(f"  运河等待区缓冲区: {len(canal_waiting_buffers)} 个")

    canal_endpoint_buffers = build_canal_endpoint_buffers()
    print(f"  运河端口缓冲区: {len(canal_endpoint_buffers)} 个")
    sys.stdout.flush()

    ship_type_filter = args.ship_type.strip() or None
    ships = extract_all_mmsi(db_paths, ship_type_filter)
    mmsi_list = ships["MMSI"].tolist()
    print(f"船舶数: {len(mmsi_list):,}，开始处理...")
    sys.stdout.flush()

    arg_list = []
    for mmsi in mmsi_list:
        row = ships[ships["MMSI"] == mmsi].iloc[0]
        arg_list.append((
            mmsi,
            row,
            db_paths,
            months,
            port_coor,
            canal_coor,
            port_buffers,
            float(args.berth_km),
            float(args.canal_km),
            float(args.berth_gap_tolerance_sec),
            berth_buffers,
            canal_corridor_buffers,
            canal_waiting_buffers,
            canal_endpoint_buffers,
        ))

    t0 = time.time()
    results = []
    with Pool(processes=max(1, int(args.workers)), maxtasksperchild=20) as pool:
        n = len(arg_list)
        for i, r in enumerate(pool.imap_unordered(process_single_ship_worker, arg_list)):
            results.append(r)
            if (i + 1) % 10 == 0 or (i + 1) == n:
                elapsed = (time.time() - t0) / 60.0
                print(f"\r进度 {i + 1}/{n}  已用 {elapsed:.1f} 分钟", end="", flush=True)
    print(f"\n并行完成，耗时 {(time.time() - t0) / 60:.2f} 分钟")

    visits_parts = []
    edges_parts = []
    ok = 0
    for r in results:
        if not r.get("success"):
            continue
        ok += 1
        v = r.get("visits")
        e = r.get("edges")
        if v is not None and len(v) > 0:
            visits_parts.append(v)
        if e is not None and len(e) > 0:
            edges_parts.append(e)

    visits_all = pd.concat(visits_parts, ignore_index=True) if visits_parts else pd.DataFrame()
    edges_all = pd.concat(edges_parts, ignore_index=True) if edges_parts else pd.DataFrame()

    # 剔除在港时间超过 3 周（504 小时）的异常挂靠记录
    max_inport_hours = 3 * 7 * 24  # 504 小时
    if len(visits_all) > 0 and "InPort_Hours" in visits_all.columns:
        before_filter = len(visits_all)
        visits_all = visits_all[visits_all["InPort_Hours"] <= max_inport_hours].reset_index(drop=True)
        filtered_count = before_filter - len(visits_all)
        if filtered_count > 0:
            print(f"已剔除在港时间超过 {max_inport_hours} 小时（3周）的异常挂靠: {filtered_count} 条")
    else:
        filtered_count = 0

    num_months = len(months)
    nodes_df = aggregate_port_node_features(visits_all, num_months=num_months)
    edges_df = aggregate_edge_tonnage(edges_all)

    nodes_path = os.path.join(out_dir, f"port_node_features_{label}.csv")
    edges_path = os.path.join(out_dir, f"network_edge_DWT_{label}.csv")
    nodes_df.to_csv(nodes_path, index=False, encoding="utf-8-sig")
    edges_df.to_csv(edges_path, index=False, encoding="utf-8-sig")

    meta_path = os.path.join(out_dir, f"run_meta_{label}.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"generated: {datetime.now()}\n")
        f.write(f"months: {months}\n")
        f.write(f"note: sjoin预计算版；所有空间判定通过R-tree索引一次完成\n")
        f.write(f"berth_radius_km: {args.berth_radius_km}\n")
        f.write(f"berth_km: {args.berth_km}\n")
        f.write(f"berth_gap_tolerance_sec: {args.berth_gap_tolerance_sec}\n")
        f.write(f"canal_km: {args.canal_km}\n")
        f.write(f"普通港口: 靠泊确认(Mode=Berth+in_berth_buffer) → 倒推进入6nm缓冲区 → 前推离开\n")
        f.write(f"运河: 确认通过(走廊段≥{CANAL_MIN_PASSAGE_SEC}s) → 走廊段全过程 → 航行+等待分离\n")
        f.write(f"max_inport_hours: {max_inport_hours} (超过此时长的挂靠记录被剔除)\n")
        f.write(f"filtered_visits: {filtered_count}\n")
        f.write(f"ships_ok: {ok}/{len(results)}\n")
        f.write(f"visits_rows: {len(visits_all)}\n")
        f.write(f"edge_rows_raw: {len(edges_all)}\n")

    print(f"节点特征: {nodes_path}")
    print(f"边总 DWT: {edges_path}")


if __name__ == "__main__":
    main()