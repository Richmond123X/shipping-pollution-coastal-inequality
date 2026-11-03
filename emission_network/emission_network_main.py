"""
################################################################################
# 船舶排放网络 — 主执行文件（月度 / 城市节点版）
#
# 流程（与数据一致）:
#   1) 按 --year 组装 14 个月窗口（上年12月 + 本年12个月 + 下年1月）的
#      ais_YYYYMM.db 路径；用目录 scandir 匹配库文件
#   2) extract_all_mmsi: 从各库 ship_info 读静态信息（含 Ship_Type）, MMSI 去重
#      （默认 4 类商船白名单, --all-ship-types 关闭）
#   3) 按船型分组，对每艘船 process_single_ship_multi_month:
#        extract_ship_trajectory_multi_db 从多库 ais_tracks 拼「跨年连续单船 AIS」
#        -> calculate_ship_emission_network（跨月航段逐点分月, 仅输出本年 12 个月）
#   4) 输出到 results_monthly/<YYYYMM>/<船型>/：城市节点排放 city_emissions_*.csv
#      + 城市边表 city_edge_list_*.csv + port_operational_monthly_<船型>_<YYYYMM>.csv
#      + edge_voyage_stats_<船型>_<YYYYMM>.xlsx（港对平均航行距离/时间）
#      挂靠识别与 shipping 流水线对齐（含苏伊士/巴拿马运河专用算法）
#
# 运行: python emission_network_main.py --year 2024
#      python emission_network_main.py --year 2024 --only-months 1,2
#      路径/结果根目录默认取根目录 config.yaml（--db-dir、--results-root 等可覆盖）
################################################################################
"""

import argparse
import os
import re
import sys
# 强制 stdout/stderr 使用 UTF-8 编码，解决 Windows 下重定向输出时的编码问题
# stderr 同样 reconfigure：tqdm 进度条默认写 stderr，不设则在 UTF-8 控制台下中文乱码
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import sqlite3
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime
import warnings
from multiprocessing import Pool, cpu_count
import networkx as nx
from pathlib import Path
import time
from typing import List, Optional
from tqdm import tqdm
warnings.filterwarnings('ignore')

# 导入自定义函数
from emission_network_functions import (
    calculate_ship_emission_network,
    is_clean_fuel_ship,
)
from shipping_network_functions import (
    build_berth_buffers as _build_berth_buffers,
    build_canal_corridor_buffers as _build_canal_corridor_buffers,
    build_canal_endpoint_buffers as _build_canal_endpoint_buffers,
    build_canal_waiting_buffers as _build_canal_waiting_buffers,
    build_port_area_buffers as _build_port_area_buffers,
    is_non_port_name as _is_non_port_name,
)
from config_loader import load_config
from city_aggregation import (
    load_city_data, map_edges_to_cities, POLICY_DELTA_COLS,
    POLICY_NO_MARGIN_COLS,
)

# 全局配置（config.yaml；命令行参数可覆盖其中路径）
_CFG = load_config()
_CFG_PATHS = _CFG["paths"]
_CFG_EMIS = _CFG["emission_network"]


# 港口区域（外层 buffer）= 该港全部泊位 BERTH_RADIUS_KM 圆 union（按 Port.Name dissolve），
# 邻港互斥扣除。10 km 对齐"港界本体"语义：覆盖码头群、港内航道、近港锚地（≤10 km）。
# 多泊位港群（Ningbo-Zhoushan/Rotterdam/Houston 等）通过 union 自然连成完整连通区。
# 邻港 Voronoi 互斥保证不重叠。
#
# 注意：emission 严格按 buffer 几何归因。远锚泊（>10 km）的排放归到航线连边，不算港口节点。
# 这与 shipping pipeline 的设计有意分离——shipping 通过 anchor 倒推（≤20 km）补救
# 远锚泊以计算等待时间；emission 不需要等待时间，因此不接 anchor 补救。
BERTH_RADIUS_KM = 10.0

# 泊位 buffer（内层 buffer）= 每个泊位 BERTH_DETECT_KM 圆，按 Port.Name dissolve。
# 用于「靠泊确认」：AIS 点必须 Mode=Berth 且落在某港 berth_buffer 内，才算真的停在码头。
# 5 km 与 shipping pipeline 默认值一致；比外层严格得多，专门用来判定靠泊事件。
BERTH_DETECT_KM = 5.0

# berth-first 算法允许的「靠泊中断」容忍秒数。停泊期间 GPS 漂移、临时驶离再回都计在内。
# 单位：秒。600 s（10 min）与 shipping pipeline 默认对齐。
BERTH_GAP_TOLERANCE_SEC = 600.0
DEFAULT_CANAL_KM = 10.0

# 与 calculate_ship_emission_network 中污染物集合一致
POLLUTANTS = ['SO2', 'PM', 'CO2', 'NOx', 'CO', 'CH4', 'N2O', 'NMVOC']

# Pool worker：每子进程只设一次，避免每船 pickle port_coor / LAF / ECA 与重复建 buffer
_em_w_port_coor: Optional[pd.DataFrame] = None
_em_w_canal_coor: Optional[pd.DataFrame] = None
_em_w_port_buf = None
_em_w_berth_buf = None
_em_w_canal_corridor_buffers = None
_em_w_canal_waiting_buffers = None
_em_w_canal_endpoint_buffers = None
_em_w_canal_km: float = DEFAULT_CANAL_KM
_em_w_laf: Optional[pd.DataFrame] = None
_em_w_eca = None
_em_w_city_catch = None  # 城市集水区 GeoDataFrame（node_id + geometry）
_em_w_policy_zones = None  # 政策情景区 GeoDataFrame（zone_id + geometry）；None=关闭
_em_w_policy_cfg = None    # 政策情景参数 dict（seca_sulfur / neca_build_year）
_em_w_weekly_cfg = None    # 周度港口输出参数 dict（enabled / anchor_km；绿色韧性研究）
_em_w_policy_only = False  # 正式政策模拟：跳过航线网络与重复基线输出


def _init_emission_worker(
    port_coor: pd.DataFrame,
    canal_coor: pd.DataFrame,
    laf: pd.DataFrame,
    eca_list: dict,
    canal_km: float = DEFAULT_CANAL_KM,
    city_catchments=None,
    policy_zones=None,
    policy_cfg=None,
    weekly_cfg=None,
    need_visit_buffers=True,
    policy_only=False,
):
    global _em_w_port_coor, _em_w_canal_coor, _em_w_port_buf, _em_w_berth_buf
    global _em_w_canal_corridor_buffers, _em_w_canal_waiting_buffers
    global _em_w_canal_endpoint_buffers, _em_w_canal_km, _em_w_laf, _em_w_eca
    global _em_w_city_catch, _em_w_policy_zones, _em_w_policy_cfg, _em_w_weekly_cfg
    global _em_w_policy_only
    _em_w_city_catch = city_catchments
    _em_w_policy_zones = policy_zones
    _em_w_policy_cfg = policy_cfg
    _em_w_weekly_cfg = weekly_cfg
    _em_w_policy_only = bool(policy_only)
    import os as _os
    # 默认不在 worker 内 print：子进程 stdout 会与主进程 tqdm 进度条(stderr)交错，
    # 导致进度条不断换行。需要调试时设环境变量 EMIS_VERBOSE_WORKER=1 打开。
    _verbose = _os.environ.get('EMIS_VERBOSE_WORKER') == '1'
    if _verbose:
        print(
            f'  [worker {_os.getpid()}] 开始初始化 buffer (Voronoi+berth+canal)…',
            flush=True,
        )
    _em_w_port_coor = port_coor
    _em_w_canal_coor = (
        canal_coor if canal_coor is not None and len(canal_coor) > 0 else pd.DataFrame()
    )
    _em_w_laf = laf
    _em_w_eca = eca_list
    _em_w_canal_km = float(canal_km)
    if need_visit_buffers:
        _em_w_port_buf = _build_port_area_buffers(
            port_coor, berth_radius_km=BERTH_RADIUS_KM
        )
        canal_for_berth = _em_w_canal_coor if len(_em_w_canal_coor) > 0 else None
        _em_w_berth_buf = _build_berth_buffers(
            port_coor, berth_km=BERTH_DETECT_KM, canal_coor=canal_for_berth
        )
        _em_w_canal_corridor_buffers = _build_canal_corridor_buffers(
            _em_w_canal_coor, _em_w_canal_km
        )
        _em_w_canal_waiting_buffers = _build_canal_waiting_buffers(
            port_coor, _em_w_canal_coor
        )
        _em_w_canal_endpoint_buffers = _build_canal_endpoint_buffers()
    else:
        _em_w_port_buf = None
        _em_w_berth_buf = None
        _em_w_canal_corridor_buffers = None
        _em_w_canal_waiting_buffers = None
        _em_w_canal_endpoint_buffers = None
    if _verbose:
        ncity = 0 if _em_w_city_catch is None else len(_em_w_city_catch)
        print(
            f'  [worker {_os.getpid()}] buffer 初始化完成 '
            f'(港口 Voronoi/berth/canal + 城市集水区 {ncity} 个已就绪)',
            flush=True,
        )


def _sqlite_connect_readonly(db_path: str):
    """只读连接，多进程并发读同一库时减少锁争用。"""
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=120)


BERTH_DISTANCE_THRESHOLD_KM = 5.0
# 路径全部来自 config.yaml（3.5 节"路径配置外部化"）；命令行参数可覆盖
DEFAULT_WORK_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_DIR = _CFG_PATHS["ais_db_dir"]
DEFAULT_PORT_XLSX = _CFG_PATHS["port_xlsx"]
DEFAULT_LAF_XLSX = _CFG_PATHS["laf_xlsx"]
DEFAULT_ECA_BASE = _CFG_PATHS["eca_base"]
DEFAULT_RESULTS_ROOT = _CFG_PATHS["results_root"]
DEFAULT_SHIP_ATTRIBUTES_CSV = (
    r"C:\Users\86176\Desktop\Ship Emission\Ship_File\Ship_file_new.csv"
)

# 4 类主流商船白名单（3.2 节船舶范围；变体归一匹配，见 _ship_type_allowed）
SHIP_TYPE_WHITELIST = list(_CFG_EMIS["ship_types"])
KEEP_PORT_LEVEL_OUTPUTS = bool(_CFG_EMIS.get("keep_port_level_outputs", True))
# 周度港口输出(绿色韧性研究): port_visits_* + port_emissions_weekly_*
WEEKLY_CFG = {
    "enabled": bool(_CFG_EMIS.get("weekly_port_outputs", True)),
    "anchor_km": float(_CFG_EMIS.get("anchor_attribution_km", 25.0)),
}


def _norm_ship_type(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).casefold())


def _ship_type_allowed(ship_type: str) -> bool:
    """库内 Ship_Type 与白名单的容错匹配（'Oil Tanker'~'oil_tanker'~'Oil Tankers'）。"""
    n = _norm_ship_type(ship_type)
    for w in SHIP_TYPE_WHITELIST:
        wn = _norm_ship_type(w)
        if n == wn or n == wn + "s" or n == wn + "ship" or n == wn + "ships":
            return True
    return False


# ============================================================================
# 月度调度（3.5 节"输出尺度改造"：季度 → 月度）/ 数据库目录探测
# ============================================================================
def _parse_only_months(s: str):
    """'1,3,12' / 'M1,M3' → {1,3,12}；空 = 全年 12 个月。"""
    if not s or not str(s).strip():
        return set(range(1, 13))
    out = set()
    for part in str(s).replace(';', ',').split(','):
        p = part.strip().upper().lstrip('M')
        if not p:
            continue
        if p.isdigit() and 1 <= int(p) <= 12:
            out.add(int(p))
        else:
            raise SystemExit(
                f'--only-months 项无法解析: {part!r}（用 1-12 或 M1-M12）'
            )
    if not out:
        raise SystemExit('--only-months 解析后为空')
    return out


def _safe_folder_name(name: str) -> str:
    s = re.sub(r'[\\/:*?"<>|]', '_', str(name).strip())
    s = re.sub(r'\s+', ' ', s).strip()
    return s if s else 'unknown_ship_type'


def _find_dbs_in_dir(months, db_directory: str):
    """
    用一次 os.scandir 按文件名匹配 ais_YYYYMM.db，避免对单个超大库反复 exists() 卡死。
    """
    print(f'  列目录（一次）: {db_directory!r} ...', flush=True)
    t0 = time.time()
    try:
        entries = list(os.scandir(db_directory))
    except OSError as e:
        print(f'  ✗ 无法打开目录: {e}', flush=True)
        return [], []
    by_name = {e.name.lower(): e for e in entries}
    print(
        f'  目录项: {len(entries)}  列目录耗时 '
        f'{(time.time() - t0) * 1000:.0f} ms', flush=True
    )
    paths, found = [], []
    for m in months:
        fn = f'ais_{m}.db'
        p = os.path.join(db_directory, fn)
        if fn.lower() in by_name:
            paths.append(p)
            found.append(m)
            print(f'  ✓ {m}: {fn}  在目录中', flush=True)
        else:
            print(f'  ✗ {m}: {fn}  未在目录中', flush=True)
    return paths, found


def _month_outputs_done(out_dir, ship_type: str, label: str,
                        policy_active: bool = False,
                        policy_only: bool = False) -> bool:
    """断点续跑判定：城市级输出齐才算完成（港口级仅当开关打开时检查）。"""
    if not policy_only:
        city_f = os.path.join(out_dir, f"city_emissions_{ship_type}_{label}.csv")
        if not (os.path.isfile(city_f) and os.path.getsize(city_f) > 50):
            return False
    if policy_active:
        # 政策情景启用时,老产出(无 delta 文件)不算完成,需重算该月
        for base in ("city_policy_delta", "policy_delta_totals"):
            if not os.path.isfile(
                    os.path.join(out_dir, f"{base}_{ship_type}_{label}.csv")):
                return False
        if _norm_ship_type(ship_type) == "container":
            # 正式岸电情景必须同时具备挂靠级与挂靠×城市级基础量。
            for base in ("ops_calls", "ops_call_city"):
                if not os.path.isfile(
                        os.path.join(out_dir, f"{base}_{ship_type}_{label}.parquet")):
                    return False
    if policy_only:
        return True
    if WEEKLY_CFG["enabled"]:
        # 周度输出启用时,缺 port_visits/周度排放的旧月份自动重算
        for base in ("port_visits", "port_emissions_weekly"):
            if not os.path.isfile(
                    os.path.join(out_dir, f"{base}_{ship_type}_{label}.csv")):
                return False
    for pol in POLLUTANTS:
        f = os.path.join(
            out_dir, f'city_edge_list_{pol}_{ship_type}_{label}.csv'
        )
        if not os.path.isfile(f):
            return False
    if KEEP_PORT_LEVEL_OUTPUTS:
        for pol in POLLUTANTS:
            f = os.path.join(
                out_dir, f'network_{pol}_{ship_type}_{label}_edge_list_simple.csv'
            )
            if not (os.path.isfile(f) and os.path.getsize(f) > 100):
                return False
        op_f = os.path.join(
            out_dir, f"port_operational_monthly_{ship_type}_{label}.csv"
        )
        if not (os.path.isfile(op_f) and os.path.getsize(op_f) > 50):
            return False
    return True


def _write_monthly_port_operational_stats(
    visit_parts: List[pd.DataFrame],
    ship_type: str,
    label: str,
    output_dir: str,
) -> None:
    """按 YearMonth + Port_Name 汇总挂靠级指标（与排放网络并行输出，单独 CSV）。"""
    path = os.path.join(
        output_dir, f"port_operational_monthly_{ship_type}_{label}.csv"
    )
    out_cols = [
        "Ship_Type",
        "YearMonth",
        "Port_Name",
        "Avg_Waiting_Hours",
        "Avg_InPort_Hours",
        "Avg_Berth_Hours",
        "Avg_Canal_Sailing_Hours",
        "Avg_AtSea_Hours",
        "Avg_Speed_InPort_Moving_kn",
        "Visit_Count",
        "Avg_DWT",
        "Total_DWT_CleanFuel",
    ]
    if not visit_parts:
        pd.DataFrame(columns=out_cols).to_csv(
            path, index=False, encoding="utf-8-sig"
        )
        print(
            f"  港口运营(月度): {os.path.basename(path)} (无挂靠记录，仅表头)",
            flush=True,
        )
        return
    ops_all = pd.concat(visit_parts, ignore_index=True)
    ops_all["CleanFuel_DWT"] = np.where(
        ops_all["Fuel_Type"].map(is_clean_fuel_ship),
        ops_all["DWT"].astype(float),
        0.0,
    )
    agg_kw = dict(
        Avg_Waiting_Hours=("Waiting_Hours", "mean"),
        Avg_InPort_Hours=("InPort_Hours", "mean"),
        Avg_Berth_Hours=("Berth_Hours", "mean"),
        Avg_AtSea_Hours=("AtSea_Hours", "mean"),
        Avg_Speed_InPort_Moving_kn=("Avg_Speed_InPort_Moving_kn", "mean"),
        Visit_Count=("MMSI", "count"),
        Avg_DWT=("DWT", "mean"),
        Total_DWT_CleanFuel=("CleanFuel_DWT", "sum"),
    )
    if "Canal_Sailing_Hours" in ops_all.columns:
        agg_kw["Avg_Canal_Sailing_Hours"] = ("Canal_Sailing_Hours", "mean")
    agg = ops_all.groupby(["YearMonth", "Port_Name"], as_index=False).agg(**agg_kw)
    if "Avg_Canal_Sailing_Hours" not in agg.columns:
        agg["Avg_Canal_Sailing_Hours"] = 0.0
    agg.insert(0, "Ship_Type", ship_type)
    agg = agg[out_cols].sort_values(["YearMonth", "Port_Name"]).reset_index(drop=True)
    agg.to_csv(path, index=False, encoding="utf-8-sig")
    print(
        f"  港口运营(月度): {os.path.basename(path)}  行数 {len(agg)}",
        flush=True,
    )


def _parse_main_args():
    p = argparse.ArgumentParser(
        description='船舶排放网络(月度/城市节点): 从各月 ais_*.db 的 ship_info 读 '
        'MMSI(4类商船白名单), 逐月计算排放网络。输出在 results_monthly/<YYYYMM>/<船型>/'
    )
    p.add_argument('--year', type=int, required=True, help='自然年, 如 2024')
    p.add_argument(
        '--only-months', type=str, default='',
        help='只跑部分月: 1,3,12 或 M1,M3; 留空=全年 12 个月',
    )
    p.add_argument('--db-dir', type=str, default=DEFAULT_DB_DIR, help='ais_YYYYMM.db 所在目录')
    p.add_argument(
        '--port-xlsx', type=str, default=DEFAULT_PORT_XLSX,
        help='港口表 port_data_clean.xlsx',
    )
    p.add_argument(
        '--laf-xlsx', type=str, default=DEFAULT_LAF_XLSX,
        help='低负荷修正因子 Excel',
    )
    p.add_argument(
        '--eca-base', type=str, default=DEFAULT_ECA_BASE,
        help='Ship Emission 根目录（含 ECA/、Port-Top_100/）',
    )
    p.add_argument(
        '--results-root', type=str, default='',
        help='默认取 config.yaml 的 paths.results_root, 可自定义',
    )
    p.add_argument(
        '--city-catchments', type=str, default=_CFG_PATHS["city_catchments_gpkg"],
        help='城市集水区 gpkg（Port2City Agent 输出）',
    )
    p.add_argument(
        '--port-city-mapping', type=str, default=_CFG_PATHS["port_city_mapping_csv"],
        help='港口→城市映射表 csv（Port2City Agent 输出）',
    )
    p.add_argument(
        '--all-ship-types', action='store_true',
        help='关闭 4 类商船白名单, 处理全部船型',
    )
    p.add_argument('--n-cores', type=int, default=4, help='每船型并行进程数')
    p.add_argument(
        '--max-ships-per-type', type=int, default=0,
        help='每船型最多处理 N 艘船（0=不限）；与 ship-start-index 组合支持正式可恢复分片',
    )
    p.add_argument(
        '--ship-start-index', type=int, default=0,
        help='按 MMSI 升序从第 N 艘开始；与 max-ships-per-type 组合可做可恢复分片',
    )
    p.add_argument(
        '--only-ship-types', type=str, default='',
        help='只跑指定船型(逗号分隔, 容错匹配), 如 "Container"; 留空=全部白名单',
    )
    p.add_argument(
        '--mmsi-list-csv', type=str, default='',
        help='可选审计白名单 CSV（须含 MMSI）；留空=不限制。正式分片通常留空',
    )
    p.add_argument(
        '--canal-km',
        type=float,
        default=DEFAULT_CANAL_KM,
        help='运河走廊/邻近标记半径 (km)，与 shipping 流水线一致',
    )
    p.add_argument(
        '--no-skip-done', action='store_true',
        help='关闭断点续传（已存在的 8 种污染物边表仍重算）',
    )
    p.add_argument(
        '--policy-zones', type=str, default='',
        help='政策情景区 gpkg(zone_id+geometry, 研究 4)。留空时读 config.yaml '
             'policy.enabled/zones_gpkg; 提供则强制启用',
    )
    p.add_argument(
        '--policy-zone-layer', type=str, default='policy_regions',
        help='政策区 GPKG 图层名；v12 正式文件使用 policy_regions',
    )
    p.add_argument(
        '--ship-attributes-csv', type=str, default=DEFAULT_SHIP_ATTRIBUTES_CSV,
        help='含 MMSI/GT 的船舶属性表；岸电 >5000 GT 筛选使用，GT 不以 DWT 替代',
    )
    p.add_argument(
        '--no-policy', action='store_true',
        help='强制关闭政策情景(覆盖 config.yaml policy.enabled)',
    )
    p.add_argument(
        '--policy-only', action='store_true',
        help='正式政策模拟加速模式：只写政策 delta；Container 另写逐次岸电表，跳过航线网络',
    )
    p.add_argument(
        '--prefetch-batch-size', type=int, default=0,
        help='主进程按 N 艘批量读取月库轨迹，再并行计算；0=沿用逐船随机读取。'
             '正式大库政策模拟建议 10--50',
    )
    p.add_argument(
        '--prefetch-sample-interval-sec', type=int, default=0,
        help='批量预取时按 MMSI×UTC 固定时间桶保留首个真实点；0=全量。'
             '只有通过全量小样本误差审计后才能用于正式运行',
    )
    p.add_argument(
        '--trajectory-cache-dir', type=str, default='',
        help='顺序扫描预生成的逐船 Parquet 目录；提供后不再逐船查询 ais_tracks',
    )
    p.add_argument(
        '--strict-db-read', action='store_true',
        help='数据库或单船计算出现任何错误时立即失败。正式政策模拟必须启用，'
             '避免损坏页或掉盘被静默当作无数据。',
    )
    return p.parse_args()


def _write_edge_voyage_stats_excel(
    voyage_parts: List[pd.DataFrame],
    ship_type: str,
    label: str,
    output_dir: str,
) -> None:
    """港对平均航行距离/时间（按船型-月份单独 Excel）。"""
    path = os.path.join(
        output_dir,
        f"edge_voyage_stats_{ship_type}_{label}.xlsx",
    )
    cols = [
        "Ship_Type",
        "Label",
        "From_Port",
        "To_Port",
        "Avg_Sailing_Hours",
        "Avg_Sailing_Distance_km",
        "Voyage_Count",
    ]
    if not voyage_parts:
        pd.DataFrame(columns=cols).to_excel(path, index=False, engine="openpyxl")
        print(
            f"  航段统计: {os.path.basename(path)} (无航段记录，仅表头)",
            flush=True,
        )
        return
    raw = pd.concat(voyage_parts, ignore_index=True)
    agg = raw.groupby(["From_Port", "To_Port"], as_index=False).agg(
        Avg_Sailing_Hours=("Sailing_Hours", "mean"),
        Avg_Sailing_Distance_km=("Sailing_Distance_km", "mean"),
        Voyage_Count=("MMSI", "count"),
    )
    agg.insert(0, "Label", label)
    agg.insert(0, "Ship_Type", ship_type)
    agg = agg[cols].sort_values(
        ["Voyage_Count", "Avg_Sailing_Distance_km"],
        ascending=[False, False],
    ).reset_index(drop=True)
    agg.to_excel(path, index=False, engine="openpyxl")
    print(
        f"  航段统计: {os.path.basename(path)}  港对 {len(agg)}  原始航次 {len(raw)}",
        flush=True,
    )


# ============================================================================
# 辅助函数1：从多个数据库中提取所有唯一的MMSI
# ============================================================================
def extract_all_mmsi(
    db_paths: List[str], ship_type_filter: str = None, strict: bool = False,
) -> pd.DataFrame:
    """
    从多个数据库中提取所有唯一的MMSI及其静态信息
    
    Parameters:
    -----------
    db_paths : List[str]
        数据库文件路径列表
    ship_type_filter : str, optional
        船型筛选条件，None表示不筛选
    
    Returns:
    --------
    pd.DataFrame : 包含唯一MMSI及其静态信息的DataFrame
    """
    print('\n提取所有MMSI...', flush=True)

    all_ships_info = []

    for db_path in db_paths:
        # 优先从独立小库 ship_info_YYYYMM.db 读（远比扫 64GB 大库快）
        # 命名约定: ais_202401.db -> ship_info_202401.db (同目录)
        bn = os.path.basename(db_path)
        m = bn[4:-3] if bn.startswith('ais_') and bn.endswith('.db') else None
        small = os.path.join(os.path.dirname(db_path), f'ship_info_{m}.db') if m else None

        used_small = False
        if small and os.path.exists(small):
            try:
                t0 = time.time()
                # readonly + immutable, 跳过锁开销
                uri = f'file:///{small.replace(os.sep, "/")}?mode=ro&immutable=1'
                conn = sqlite3.connect(uri, uri=True, timeout=30)
                if ship_type_filter:
                    query = f"SELECT * FROM ship_info WHERE Ship_Type = '{ship_type_filter}'"
                else:
                    query = "SELECT * FROM ship_info"
                ship_info = pd.read_sql_query(query, conn)
                conn.close()
                dt = time.time() - t0
                print(f'  ✓ 从小库 {os.path.basename(small)} 读: {len(ship_info):,} 行  耗时 {dt:.2f}s', flush=True)
                all_ships_info.append(ship_info)
                used_small = True
            except Exception as e:
                print(f'  ⚠️  小库读失败 ({e}), fallback 到大库 ...', flush=True)
                used_small = False

        if used_small:
            continue

        try:
            t0 = time.time()
            conn = sqlite3.connect(db_path, timeout=30)
            if ship_type_filter:
                query = f"SELECT DISTINCT * FROM ship_info WHERE Ship_Type = '{ship_type_filter}'"
            else:
                query = "SELECT DISTINCT * FROM ship_info"
            ship_info = pd.read_sql_query(query, conn)
            conn.close()
            dt = time.time() - t0
            print(f'    找到 {len(ship_info)} 条船舶记录, 耗时 {dt:.1f}s', flush=True)
            all_ships_info.append(ship_info)
        except Exception as e:
            print(f'    ⚠️ 读取失败: {str(e)}', flush=True)
            if strict:
                raise RuntimeError(f"ship_info 严格读取失败: {db_path}: {e}") from e
            continue
    
    if not all_ships_info:
        raise ValueError("未能从任何数据库中读取到船舶信息！")
    
    # 合并所有数据库的船舶信息
    combined_info = pd.concat(all_ships_info, ignore_index=True)
    
    # 去重：保留第一次出现的MMSI记录
    unique_mmsi_info = combined_info.drop_duplicates(subset='MMSI', keep='first')
    
    print(f'\n汇总结果:')
    print(f'  总船舶记录数: {len(combined_info)}')
    print(f'  唯一MMSI数量: {len(unique_mmsi_info)}')
    print(f'  去重移除记录: {len(combined_info) - len(unique_mmsi_info)}')
    
    return unique_mmsi_info


def enrich_ship_attributes(
    ships: pd.DataFrame,
    attributes_csv: str,
) -> pd.DataFrame:
    """把外部船舶表的 GT/IMO 补到数据库 ``ship_info``。

    排放参数仍以数据库记录为准；这里只补岸电法规筛选必需的 GT，并在数据库
    建造年缺失时才用外部 BuildYear 补值。MMSI 无法匹配者保持缺失，绝不以
    DWT 推算 GT。
    """
    out = ships.copy()
    out["Gross_Tonnage"] = np.nan
    out["IMO_external"] = np.nan
    out["GT_Source"] = "missing"
    if not attributes_csv:
        print("  ⚠ 未提供 --ship-attributes-csv，GT 全部保持缺失", flush=True)
        return out
    p = os.path.abspath(attributes_csv)
    if not os.path.isfile(p):
        print(f"  ⚠ 船舶属性表不存在，GT 全部保持缺失: {p}", flush=True)
        return out

    hdr = pd.read_csv(p, nrows=0, encoding_errors="replace").columns.tolist()
    required = {"MMSI", "GT"}
    if not required.issubset(hdr):
        raise ValueError(f"船舶属性表缺少 {sorted(required - set(hdr))}: {p}")
    usecols = [c for c in (
        "MMSI", "GT", "IMO", "BuildYear", "GT_Source", "GT_Match_Method"
    ) if c in hdr]
    ext = pd.read_csv(p, usecols=usecols, low_memory=False, encoding_errors="replace")
    ext["_mmsi_key"] = pd.to_numeric(ext["MMSI"], errors="coerce").astype("Int64")
    ext["Gross_Tonnage"] = pd.to_numeric(ext["GT"], errors="coerce")
    ext = ext.dropna(subset=["_mmsi_key"])
    # 同 MMSI 多条时优先保留有 GT 的记录，再保留首条；不做任何吨位推算。
    ext["_has_gt"] = ext["Gross_Tonnage"].notna()
    ext = ext.sort_values("_has_gt").drop_duplicates("_mmsi_key", keep="last")
    keep = ["_mmsi_key", "Gross_Tonnage"]
    ren = {}
    if "IMO" in ext.columns:
        keep.append("IMO")
        ren["IMO"] = "IMO_external"
    if "BuildYear" in ext.columns:
        keep.append("BuildYear")
        ren["BuildYear"] = "Build_Year_external"
    if "GT_Source" in ext.columns:
        keep.append("GT_Source")
        ren["GT_Source"] = "GT_Source_external"
    if "GT_Match_Method" in ext.columns:
        keep.append("GT_Match_Method")
    ext = ext[keep].rename(columns=ren)

    out["_mmsi_key"] = pd.to_numeric(out["MMSI"], errors="coerce").astype("Int64")
    out = out.drop(columns=["Gross_Tonnage", "IMO_external", "GT_Source"]).merge(
        ext, on="_mmsi_key", how="left", validate="m:1"
    )
    if "IMO_external" not in out.columns:
        out["IMO_external"] = np.nan
    if "GT_Source_external" in out.columns:
        out["GT_Source"] = out["GT_Source_external"].where(
            out["Gross_Tonnage"].notna(), "missing"
        )
    else:
        out["GT_Source"] = np.where(out["Gross_Tonnage"].notna(), p, "missing")
    if "Build_Year_external" in out.columns:
        by = pd.to_numeric(out.get("Build_Year"), errors="coerce")
        by_ext = pd.to_numeric(out["Build_Year_external"], errors="coerce")
        out["Build_Year"] = by.fillna(by_ext)
    matched = int(out["Gross_Tonnage"].notna().sum())
    print(
        f"  GT 补充: {matched:,}/{len(out):,} 艘 "
        f"({matched / len(out):.2%}，未匹配 {len(out)-matched:,})  来源: {p}",
        flush=True,
    )
    return out.drop(columns=["_mmsi_key"], errors="ignore")


# ============================================================================
# 辅助函数2：从多个数据库中提取单艘船的完整轨迹
# ============================================================================
def extract_ship_trajectory_multi_db(mmsi: int, db_paths: List[str]) -> pd.DataFrame:
    """
    从多个数据库中提取单艘船的完整轨迹数据
    
    Parameters:
    -----------
    mmsi : int
        船舶MMSI号
    db_paths : List[str]
        数据库文件路径列表
    
    Returns:
    --------
    pd.DataFrame : 合并后的轨迹数据，按时间排序
    """
    all_trajectories = []
    
    for db_path in db_paths:
        try:
            conn = _sqlite_connect_readonly(db_path)
            
            # 从当前数据库读取该船的轨迹
            query = f"SELECT * FROM ais_tracks WHERE MMSI = {mmsi} ORDER BY UTCtime"
            ship_ais = pd.read_sql_query(query, conn)
            conn.close()
            
            if len(ship_ais) > 0:
                all_trajectories.append(ship_ais)
                
        except Exception as e:
            # 忽略单个数据库的读取错误
            continue
    
    if not all_trajectories:
        return pd.DataFrame()
    
    # 合并所有轨迹数据
    combined_trajectory = pd.concat(all_trajectories, ignore_index=True)
    
    # 按UTCtime去重（保留第一次出现的记录）
    combined_trajectory = combined_trajectory.drop_duplicates(subset='UTCtime', keep='first')
    
    # 按时间排序
    combined_trajectory = combined_trajectory.sort_values('UTCtime').reset_index(drop=True)
    
    return combined_trajectory


def extract_ship_trajectories_batch(
    mmsi_values: List[int], db_paths: List[str], sample_interval_sec: int = 0,
    strict: bool = False,
) -> dict[int, pd.DataFrame]:
    """按月库批量提取一组 MMSI，避免多进程对大库进行逐船随机读取。

    SQLite 的 ``(MMSI, UTCtime)`` 索引适合一次查询一小组船。正式政策模拟
    先由主进程顺序读取磁盘，再把单船 DataFrame 交给计算进程；这不改变任何
    轨迹清洗、插值或排放公式，只改变 I/O 调度方式。
    """
    ids = [int(x) for x in mmsi_values]
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    interval = max(0, int(sample_interval_sec))
    base = (
        "SELECT MMSI, UTCtime, Speed, Longitude, Latitude "
        f"FROM ais_tracks WHERE MMSI IN ({placeholders})"
    )
    if interval:
        # 每个 MMSI 的固定 UTC 时间桶保留首个真实 AIS 点。TimeInv 随后仍按
        # 实际 UTCtime 计算；不使用坐标均值，避免跨 180° 经线产生伪点。
        query = (
            "WITH sampled AS (SELECT MMSI, UTCtime, Speed, Longitude, Latitude, "
            "ROW_NUMBER() OVER (PARTITION BY MMSI, "
            f"CAST(UTCtime/{interval} AS INTEGER) ORDER BY UTCtime) AS rn "
            f"FROM ais_tracks WHERE MMSI IN ({placeholders})) "
            "SELECT MMSI, UTCtime, Speed, Longitude, Latitude "
            "FROM sampled WHERE rn=1"
        )
    else:
        query = base
    parts = []
    for db_path in db_paths:
        try:
            conn = _sqlite_connect_readonly(db_path)
            d = pd.read_sql_query(query, conn, params=ids)
            conn.close()
            if len(d):
                parts.append(d)
        except Exception as exc:
            try:
                conn.close()
            except Exception:
                pass
            if strict:
                raise RuntimeError(
                    f"ais_tracks 严格批量读取失败: {db_path}: {exc}"
                ) from exc
            continue
    if not parts:
        return {m: pd.DataFrame() for m in ids}
    all_tracks = pd.concat(parts, ignore_index=True)
    all_tracks = all_tracks.drop_duplicates(
        subset=["MMSI", "UTCtime"], keep="first"
    ).sort_values(["MMSI", "UTCtime"])
    out = {
        int(m): g.reset_index(drop=True).copy()
        for m, g in all_tracks.groupby("MMSI", sort=False)
    }
    for m in ids:
        out.setdefault(m, pd.DataFrame())
    return out


def extract_ship_trajectories_cache(
    mmsi_values: List[int], cache_dir: str
) -> dict[int, pd.DataFrame]:
    """从顺序扫描生成的逐船 Parquet 缓存读取政策候选轨迹。"""
    root = Path(cache_dir)
    out = {}
    for value in mmsi_values:
        mmsi = int(value)
        path = root / f"{mmsi:09d}.parquet"
        if path.is_file():
            out[mmsi] = pd.read_parquet(path)
        else:
            out[mmsi] = pd.DataFrame()
    return out


# ============================================================================
# Worker函数：处理单艘船舶（多月份版本）
# ============================================================================
def process_single_ship_multi_month(args):
    """
    处理单艘船舶的排放网络计算（多月份版本）
    
    Parameters:
    -----------
    args : tuple
        包含所有必需参数的元组
    
    Returns:
    --------
    dict : 计算结果
    """
    if len(args) == 7:
        (mmsi, ship_info, ship_ais, month_range_str,
         berth_distance_threshold, pollutants, _prefetched) = args
        db_paths = None
    else:
        (mmsi, ship_info, db_paths, month_range_str,
         berth_distance_threshold, pollutants) = args
        ship_ais = None
    
    try:
        need_visits = (not _em_w_policy_only) or (_norm_ship_type(
            ship_info.get("Ship_Type", "")) == "container")
        if _em_w_port_coor is None or (need_visits and _em_w_port_buf is None):
            return {
                "success": False,
                "mmsi": mmsi,
                "error_message": "worker 未初始化缓冲区（Pool initializer 未执行）",
            }
        # 从多个数据库中提取该船的完整轨迹
        if ship_ais is None:
            ship_ais = extract_ship_trajectory_multi_db(mmsi, db_paths)
        
        # 检查数据完整性
        if len(ship_ais) == 0:
            # 注册表(ship_info)中存在但当月 ais_tracks 无轨迹：属"当月未活动"，
            # 不是计算错误。单独标记 no_data，避免计入"失败"。
            return {
                'success': False,
                'no_data': True,
                'mmsi': mmsi,
                'error_message': "无AIS数据"
            }
        
        # 港口 buffer / LAF / ECA 在子进程 initializer 中只建一次（见 _init_emission_worker）
        result = calculate_ship_emission_network(
            ship_ais_data=ship_ais,
            ship_info=ship_info,
            port_coor=_em_w_port_coor,
            port_buffers_3nm=_em_w_port_buf,
            berth_buffers=_em_w_berth_buf,
            berth_gap_tolerance_sec=BERTH_GAP_TOLERANCE_SEC,
            LAF=_em_w_laf,
            ECA_list=_em_w_eca,
            month_year=month_range_str,
            berth_distance_threshold=berth_distance_threshold,
            canal_coor=_em_w_canal_coor,
            canal_corridor_buffers=_em_w_canal_corridor_buffers,
            canal_waiting_buffers=_em_w_canal_waiting_buffers,
            canal_endpoint_buffers=_em_w_canal_endpoint_buffers,
            canal_proximity_km=_em_w_canal_km,
            city_catchments=_em_w_city_catch,
            policy_zones=_em_w_policy_zones,
            policy_cfg=_em_w_policy_cfg,
            weekly_cfg=_em_w_weekly_cfg,
            policy_only=_em_w_policy_only,
        )

        # 轨迹存在但去重/过滤后不足以形成至少两个时间段时，属于“无可计算
        # 航段”，不是程序或数据库故障。旧逻辑把这类返回计入 failed，导致
        # strict 模式在全年结束后误报并丢弃已完成聚合。
        if (
            result
            and not result.get("success")
            and result.get("error_message") in {
                "AIS记录数<=2",
                "AIS 记录过少",
                "插值/过滤时间间隔后 AIS 过少",
            }
        ):
            result["no_data"] = True
            result["mmsi"] = mmsi

        return result
        
    except Exception as e:
        import traceback
        return {
            'success': False,
            'mmsi': mmsi,
            'error_message': f"{str(e)}\n{traceback.format_exc()}"
        }



# ============================================================================
# 配置加载：港口 / LAF / ECA（每轮年度任务只调一次即可）
# ============================================================================
def load_port_laf_eca_data(
    port_data_path: str = None,
    laf_path: str = None,
    eca_base: str = None,
):
    port_data_path = port_data_path or DEFAULT_PORT_XLSX
    laf_path = laf_path or DEFAULT_LAF_XLSX
    eca_base = eca_base or DEFAULT_ECA_BASE

    print("\n" + "=" * 80)
    print("步骤: 加载港口 / LAF / ECA")
    print("=" * 80)

    print("\n读取港口数据...")
    if not os.path.exists(port_data_path):
        raise FileNotFoundError(f"港口数据文件不存在: {port_data_path}")
    port_coor = pd.read_excel(port_data_path)
    print(f"  原始记录: {len(port_coor)}")

    port_coor = port_coor[port_coor["Port_Type"] == "Sea"].copy()
    print(f"  Port_Type=Sea: {len(port_coor)}")

    print("\n排除非港口节点（Anchorage/Strait/SPM/Oil Terminal/Lightering）...")
    before_np = len(port_coor)
    non_port_mask = port_coor["Port.Name"].apply(_is_non_port_name)
    port_coor = port_coor[~non_port_mask].copy()
    print(f"  剔除 {before_np - len(port_coor)} 行 -> 剩余 {len(port_coor)}")

    print("\n合并宁波-舟山, 拆分运河主港记录（与 shipping 一致）...")
    port_coor.loc[port_coor["Port.Name"] == "Ningbo", "Port.Name"] = "Ningbo-Zhoushan"
    port_coor.loc[port_coor["Port.Name"] == "Zhoushan", "Port.Name"] = "Ningbo-Zhoushan"
    canal_mask = port_coor["Port.Name"].isin(["Suez Canal", "Panama Canal"])
    canal_coor = port_coor[canal_mask].copy()
    port_coor = port_coor[~canal_mask].copy()
    print(f"  主港口行: {len(port_coor)}  运河参考行: {len(canal_coor)}")
    if len(port_coor) == 0:
        raise ValueError("没有有效的港口坐标数据")

    required_cols = ["Port.Name", "Lon_port", "Lat_port", "Lon_berth", "Lat_berth"]
    miss = set(required_cols) - set(port_coor.columns)
    if miss:
        raise ValueError(f"港口数据缺少列: {miss}")

    for col in ["Lon_berth", "Lat_berth", "Lon_port", "Lat_port"]:
        port_coor[col] = pd.to_numeric(port_coor[col], errors="coerce")
    port_coor = port_coor.dropna(
        subset=["Lon_berth", "Lat_berth", "Lon_port", "Lat_port"]
    ).reset_index(drop=True)
    port_coor = port_coor[
        (port_coor["Lon_berth"].between(-180, 180))
        & (port_coor["Lat_berth"].between(-90, 90))
        & (port_coor["Lon_port"].between(-180, 180))
        & (port_coor["Lat_port"].between(-90, 90))
    ].reset_index(drop=True)

    port_coor_unique = port_coor.drop_duplicates(
        subset=["Lon_port", "Lat_port"], keep="first"
    ).reset_index(drop=True)
    print(f"  port_coor: {len(port_coor)} | 唯一港 port_coor_unique: {len(port_coor_unique)}")

    _ = gpd.GeoDataFrame(
        port_coor_unique,
        geometry=gpd.points_from_xy(
            port_coor_unique["Lon_port"], port_coor_unique["Lat_port"]
        ),
        crs="EPSG:4326",
    )
    print("  港口点对象准备完成 (worker 内会重建 buffer)")

    if not os.path.exists(laf_path):
        raise FileNotFoundError(f"LAF: {laf_path}")
    LAF = pd.read_excel(laf_path)
    print(f"  LAF 记录: {len(LAF)}")

    eca_files = {
        "Europe_ECA": os.path.join(eca_base, "ECA", "Europe.csv"),
        "China_ECA": os.path.join(eca_base, "ECA", "China.csv"),
        "US_Hawaii_ECA": os.path.join(eca_base, "ECA", "Hawaii.csv"),
        "US_Caribbean_ECA": os.path.join(eca_base, "ECA", "Caribbean.csv"),
        "US_WestCoast_ECA": os.path.join(eca_base, "ECA", "West_coast.csv"),
        "US_EastCoast_ECA": os.path.join(eca_base, "ECA", "East_coast.csv"),
        "Mediterranean_ECA": os.path.join(eca_base, "ECA", "Mediterranean.csv"),
        "Korea_Busan_ECA": os.path.join(eca_base, "Port-Top_100", "6Busan.csv"),
        "Korea_Incheon_ECA": os.path.join(eca_base, "Port-Top_100", "57Incheon.csv"),
        "Korea_Yeosu_ECA": os.path.join(eca_base, "Port-Top_100", "82Yeosu Gwangyang.csv"),
        "Med_Piraeus_ECA": os.path.join(eca_base, "Port-Top_100", "26Piraeus.csv"),
        "Med_Valencia_ECA": os.path.join(eca_base, "Port-Top_100", "28Valencia.csv"),
        "Med_Algeciras_ECA": os.path.join(eca_base, "Port-Top_100", "32Algeciras.csv"),
        "Med_Barcelona_ECA": os.path.join(eca_base, "Port-Top_100", "53Barcelona.csv"),
        "Med_Marsaxlokk_ECA": os.path.join(eca_base, "Port-Top_100", "73Marsaxlokk.csv"),
        "Med_Genoa_ECA": os.path.join(eca_base, "Port-Top_100", "74Genoa.csv"),
        "Med_Gioia_Tauro_ECA": os.path.join(eca_base, "Port-Top_100", "78Gioia Tauro.csv"),
        "China_Kaohsiung_ECA": os.path.join(eca_base, "Port-Top_100", "15Kaohsiung.csv"),
        "China_Taichung_ECA": os.path.join(eca_base, "Port-Top_100", "94Taichung.csv"),
        "China_Taipei_ECA": os.path.join(eca_base, "Port-Top_100", "100Taipei.csv"),
    }
    ECA_list = {}
    for name, path in eca_files.items():
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            df.columns = [c.lower().strip() for c in df.columns]
            if "longitude" in df.columns and "latitude" in df.columns:
                ECA_list[name] = {
                    "lon": df["longitude"].tolist(),
                    "lat": df["latitude"].tolist(),
                }
        except Exception:
            pass
    print(f"  已加载 ECA 区划: {len(ECA_list)} 个")
    return port_coor, canal_coor, port_coor_unique, LAF, ECA_list


# ============================================================================
# 年±1 边界月处理（每船整段算一次、按月分箱），消除跨月航段低估
# ============================================================================
def _year_window(year: int):
    """返回 (target_months[12], window_codes[14])；window 含上年12月与下年1月作边界。"""
    target = [f"{year}{m:02d}" for m in range(1, 13)]
    window = [f"{year - 1}12"] + target + [f"{year + 1}01"]
    return target, window


def _target_month_window(target_months: List[str]) -> List[str]:
    """返回目标月份及其前后各一月，供独立月度任务夹住跨月轨迹。"""
    out = set()
    for ym in target_months:
        year = int(ym[:4])
        month = int(ym[4:])
        for delta in (-1, 0, 1):
            ordinal = year * 12 + (month - 1) + delta
            yy, mm0 = divmod(ordinal, 12)
            out.add(f"{yy:04d}{mm0 + 1:02d}")
    return sorted(out)


def _aggregate_city_nodes(city_m: pd.DataFrame, pollutants: List[str], ship_type: str) -> pd.DataFrame:
    """单月 city_emissions(逐点聚合) -> 四工况宽表。"""
    node_agg = (city_m.groupby(["node_id", "YearMonth", "Mode_Class"],
                               as_index=False)[pollutants].sum())
    wide = node_agg.pivot_table(index=["node_id", "YearMonth"], columns="Mode_Class",
                                values=pollutants, aggfunc="sum", fill_value=0.0)
    rows = []
    for (nid, ym), r in wide.iterrows():
        rec = {"node_id": nid, "YearMonth": ym, "Ship_Type": ship_type}
        for pol in pollutants:
            b = float(r.get((pol, "Berth_Class"), 0.0))
            a = float(r.get((pol, "Anchored_Class"), 0.0))
            mv = float(r.get((pol, "Manoeuvring_Class"), 0.0))
            s = float(r.get((pol, "Sea_Class"), 0.0))
            rec[f"{pol}_Berth"] = b
            rec[f"{pol}_Anchored"] = a
            rec[f"{pol}_Manoeuvring"] = mv
            rec[f"{pol}_Sea"] = s
            rec[f"{pol}_Total"] = b + a + mv + s
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["node_id", "YearMonth"])


def _write_month_outputs(ym, out_dir, ship_type, city_m, port_m, edge_m, op_m, voy_m,
                         pollutants, port_city_mapping, counts,
                         pol_city_m=None, pol_tot_m=None, policy_active=False,
                         true_tot_row=None, weekly_m=None,
                         ops_calls_m=None, ops_call_city_m=None,
                         policy_only=False):
    """写出单个目标月份的全部产物（城市级 + 港口级可选 + 政策情景 delta 可选）。"""
    os.makedirs(out_dir, exist_ok=True)
    # ---- 周度港口输出(绿色韧性研究): 逐次挂靠明细 + 港口×ISO周×工况排放 ----
    if WEEKLY_CFG["enabled"] and not policy_only:
        op_raw = op_m if op_m is not None else pd.DataFrame()
        op_raw.to_csv(os.path.join(out_dir, f"port_visits_{ship_type}_{ym}.csv"),
                      index=False, encoding="utf-8-sig")
        wk = weekly_m if weekly_m is not None else pd.DataFrame()
        if len(wk) > 0:
            wk_agg = (wk.groupby(["Port_Name", "YearMonth", "ISO_Week",
                                  "Mode_Class", "Attr"], as_index=False)
                      [["SO2", "PM", "NOx", "CO2"]].sum())
            wk_agg.insert(1, "Ship_Type", ship_type)
        else:
            wk_agg = pd.DataFrame(columns=["Port_Name", "Ship_Type", "YearMonth",
                                           "ISO_Week", "Mode_Class", "Attr",
                                           "SO2", "PM", "NOx", "CO2"])
        wk_agg.to_csv(
            os.path.join(out_dir, f"port_emissions_weekly_{ship_type}_{ym}.csv"),
            index=False, encoding="utf-8-sig")
    # ---- 逐点守恒真实总量(绝对量级核对用; port+edge/集水区口径均为其子集) ----
    if true_tot_row is not None and not policy_only:
        pd.DataFrame([{"YearMonth": ym, "Ship_Type": ship_type,
                       **{p: true_tot_row.get(p, 0.0) for p in pollutants}}]).to_csv(
            os.path.join(out_dir, f"true_totals_{ship_type}_{ym}.csv"),
            index=False, encoding="utf-8-sig")
    # ---- 城市节点 ----
    if not policy_only:
        if len(city_m) > 0:
            _aggregate_city_nodes(city_m, pollutants, ship_type).to_csv(
                os.path.join(out_dir, f"city_emissions_{ship_type}_{ym}.csv"),
                index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["node_id", "YearMonth", "Ship_Type"]).to_csv(
                os.path.join(out_dir, f"city_emissions_{ship_type}_{ym}.csv"),
                index=False, encoding="utf-8-sig")
    # ---- 政策情景 delta（研究 4；仅政策区启用时产出，组合情景=SECA列+NECA列）----
    if policy_active:
        _base_cols = ["B_SO2", "B_PM", "B_NOx"]
        pc = pol_city_m if pol_city_m is not None else pd.DataFrame()
        if len(pc) > 0:
            pc_agg = (pc.groupby(["node_id", "YearMonth", "Mode_Class", "Zone_ID"],
                                 as_index=False)[POLICY_DELTA_COLS].sum())
            pc_agg.insert(2, "Ship_Type", ship_type)
        else:
            pc_agg = pd.DataFrame(columns=["node_id", "YearMonth", "Ship_Type",
                                           "Mode_Class", "Zone_ID"] + POLICY_DELTA_COLS)
        pc_agg.to_csv(os.path.join(out_dir, f"city_policy_delta_{ship_type}_{ym}.csv"),
                      index=False, encoding="utf-8-sig")
        pt = pol_tot_m if pol_tot_m is not None else pd.DataFrame()
        if len(pt) > 0:
            pt_agg = (pt.groupby(["YearMonth", "Mode_Class", "Zone_ID"], as_index=False)
                      [POLICY_DELTA_COLS + POLICY_NO_MARGIN_COLS + _base_cols].sum())
            pt_agg.insert(1, "Ship_Type", ship_type)
        else:
            pt_agg = pd.DataFrame(columns=["YearMonth", "Ship_Type", "Mode_Class",
                                           "Zone_ID"] + POLICY_DELTA_COLS
                                          + POLICY_NO_MARGIN_COLS + _base_cols)
        pt_agg.to_csv(os.path.join(out_dir, f"policy_delta_totals_{ship_type}_{ym}.csv"),
                      index=False, encoding="utf-8-sig")
        if _norm_ship_type(ship_type) == "container":
            # 挂靠级中间表较大，Parquet 可无损保留类型且显著节省磁盘。
            ops = ops_calls_m if ops_calls_m is not None else pd.DataFrame()
            if len(ops) > 0:
                ops = ops.drop_duplicates("Call_ID").sort_values(
                    ["YearMonth", "Port_Name", "Arrive_UTC"]
                )
            ops.to_parquet(
                os.path.join(out_dir, f"ops_calls_{ship_type}_{ym}.parquet"),
                index=False,
            )
            opc = ops_call_city_m if ops_call_city_m is not None else pd.DataFrame()
            if len(opc) > 0:
                val_cols = [c for c in (
                    "OPS_AE_kWh", "OPS_BO_kWh", "Berth_SO2_g", "Berth_PM_g",
                    "Berth_NOx_g", *POLICY_DELTA_COLS,
                ) if c in opc.columns]
                opc = (opc.groupby(
                    ["Call_ID", "Port_Name", "node_id", "YearMonth"], as_index=False
                )[val_cols].sum())
            opc.to_parquet(
                os.path.join(out_dir, f"ops_call_city_{ship_type}_{ym}.parquet"),
                index=False,
            )
    if policy_only:
        with open(os.path.join(out_dir, f"statistics_{ship_type}_{ym}.txt"), "w",
                  encoding="utf-8") as f:
            f.write("V12 policy-only output\n")
            f.write(f"Ship_Type:{ship_type} YearMonth:{ym} Generated:{datetime.now()}\n")
            f.write(f"Success:{counts['ok']} NoData:{counts['nodata']} "
                    f"Failed:{counts['fail']} FailRate:{counts['fail_rate']:.2f}%\n")
        return
    if KEEP_PORT_LEVEL_OUTPUTS:
        _write_monthly_port_operational_stats(
            [op_m] if len(op_m) > 0 else [], ship_type, ym, out_dir)
        _write_edge_voyage_stats_excel(
            [voy_m] if len(voy_m) > 0 else [], ship_type, ym, out_dir)
    for pol in pollutants:
        pe, ee = port_m, edge_m
        # ---- 城市 OD 边（港口边经映射归并；逐点已分月，这里只是按月切片后聚合）----
        if port_city_mapping:
            city_edges, _ = map_edges_to_cities(ee, port_city_mapping, [pol])
            if len(city_edges) > 0:
                city_edges = city_edges.rename(columns={pol: "Total_Emission"}).sort_values(
                    "Total_Emission", ascending=False)
            else:
                city_edges = pd.DataFrame(columns=["From_City", "To_City", "Total_Emission"])
            city_edges.to_csv(
                os.path.join(out_dir, f"city_edge_list_{pol}_{ship_type}_{ym}.csv"),
                index=False, encoding="utf-8-sig")
        if not KEEP_PORT_LEVEL_OUTPUTS:
            continue
        # ---- 港口节点排名 ----
        if len(pe) > 0:
            agg = {pol: "sum"}
            for c, op in [("Turnaround_Hours", "mean"), ("Berth_Hours", "mean"),
                          ("DWT", "mean"), ("Visit_Count", "sum")]:
                if c in pe.columns:
                    agg[c] = op
            ps = pe.groupby("Port_Name").agg(agg).reset_index().rename(columns={
                pol: "Total_Emission", "Turnaround_Hours": "Avg_Turnaround_Hours",
                "Berth_Hours": "Avg_Berth_Hours", "DWT": "Avg_Ship_DWT",
                "Visit_Count": "Total_Visits"})
            for c in ["Avg_Turnaround_Hours", "Avg_Berth_Hours", "Avg_Ship_DWT", "Total_Visits"]:
                if c not in ps.columns:
                    ps[c] = 0 if c == "Total_Visits" else np.nan
            ps = ps[["Port_Name", "Total_Emission", "Avg_Turnaround_Hours",
                     "Avg_Berth_Hours", "Avg_Ship_DWT", "Total_Visits"]].sort_values(
                "Total_Emission", ascending=False)
        else:
            ps = pd.DataFrame(columns=["Port_Name", "Total_Emission", "Avg_Turnaround_Hours",
                                       "Avg_Berth_Hours", "Avg_Ship_DWT", "Total_Visits"])
        if len(ee) > 0:
            es = ee.groupby(["From_Port", "To_Port"])[pol].sum().reset_index()
            es.columns = ["From_Port", "To_Port", "Total_Emission"]
            es = es.sort_values("Total_Emission", ascending=False)
        else:
            es = pd.DataFrame(columns=["From_Port", "To_Port", "Total_Emission"])
        G = nx.DiGraph()
        for _, row in ps.iterrows():
            G.add_node(row["Port_Name"], emission=row["Total_Emission"],
                       avg_duration=row["Avg_Turnaround_Hours"], avg_dwt=row["Avg_Ship_DWT"],
                       visits=row["Total_Visits"])
        for _, row in es.iterrows():
            if row["From_Port"] in G and row["To_Port"] in G:
                G.add_edge(row["From_Port"], row["To_Port"], weight=row["Total_Emission"])
        base = f"network_{pol}_{ship_type}_{ym}"
        nx.write_graphml(G, os.path.join(out_dir, f"{base}.graphml"))
        es.to_csv(os.path.join(out_dir, f"{base}_edge_list_simple.csv"),
                  index=False, encoding="utf-8-sig")
        ps.to_csv(os.path.join(out_dir, f"{base}_port_emissions_ranked.csv"),
                  index=False, encoding="utf-8-sig")
    if KEEP_PORT_LEVEL_OUTPUTS and (len(port_m) > 0 or len(edge_m) > 0):
        try:
            with pd.ExcelWriter(os.path.join(out_dir, f"network_data_{ship_type}_{ym}.xlsx"),
                                engine="openpyxl") as w:
                if len(port_m) > 0:
                    port_m.to_excel(w, sheet_name="ports", index=False)
                if len(edge_m) > 0:
                    edge_m.to_excel(w, sheet_name="edges", index=False)
        except Exception:
            pass
    with open(os.path.join(out_dir, f"statistics_{ship_type}_{ym}.txt"), "w",
              encoding="utf-8") as f:
        f.write("船舶排放网络统计(月度/城市节点; 年±1 边界月处理, 跨月航段已分月)\n")
        f.write("=" * 60 + "\n")
        f.write(f"船型:{ship_type}  月:{ym}  生成:{datetime.now()}\n")
        f.write(f"(年内)成功{counts['ok']} 无数据{counts['nodata']} 失败{counts['fail']} "
                f"有效失败率{counts['fail_rate']:.2f}%\n")


def run_emission_for_year_ship_type(
    current_ship_type: str,
    ships_current_type: pd.DataFrame,
    target_months: List[str],
    window_db_paths: List[str],
    year_label: str,
    port_coor: pd.DataFrame,
    canal_coor: pd.DataFrame,
    LAF: pd.DataFrame,
    ECA_list: dict,
    results_root: str,
    pollutants: List[str],
    n_cores: int,
    berth_distance_threshold: float = None,
    canal_km: float = DEFAULT_CANAL_KM,
    city_catchments=None,
    port_city_mapping: dict = None,
    max_ships: int = 0,
    progress_prefix: str = "",
    policy_zones=None,
    policy_cfg: dict = None,
    policy_only: bool = False,
    ship_start_index: int = 0,
    prefetch_batch_size: int = 0,
    prefetch_sample_interval_sec: int = 0,
    trajectory_cache_dir: str = "",
    strict_db_read: bool = False,
):
    """对一个自然年的单船型：每船整段(含两端边界月)算一次，按月分箱写 12 个月。"""
    policy_active = policy_zones is not None and len(policy_zones) > 0
    if policy_only and not policy_active:
        raise ValueError("--policy-only 必须同时启用 --policy-zones")
    if berth_distance_threshold is None:
        berth_distance_threshold = BERTH_DISTANCE_THRESHOLD_KM
    mmsi_list = sorted(ships_current_type["MMSI"].tolist())
    total_type_ships = len(mmsi_list)
    print(f"该船型船舶数量: {total_type_ships:,}")
    start_idx = max(0, int(ship_start_index))
    if start_idx:
        mmsi_list = mmsi_list[start_idx:]
        print(f"  分片起点: MMSI升序索引 {start_idx:,}（剩余 {len(mmsi_list):,} 艘）")
    if max_ships and max_ships > 0 and len(mmsi_list) > max_ships:
        mmsi_list = mmsi_list[:max_ships]
        print(f"  分片长度: {max_ships:,} 艘（--max-ships-per-type）")
    if not mmsi_list:
        print("  无船舶, 跳过")
        return

    sub = _safe_folder_name(current_ship_type)
    log_dir = os.path.join(results_root, "_logs")
    os.makedirs(log_dir, exist_ok=True)
    error_log_file = os.path.join(log_dir, f"error_log_{sub}_{year_label}.txt")
    with open(error_log_file, "w", encoding="utf-8") as f:
        f.write(f"错误 - {year_label} 船型: {current_ship_type}\n"
                f"读取库(含边界月): {', '.join(os.path.basename(p) for p in window_db_paths)}\n\n")

    ships_by_mmsi = ships_current_type.drop_duplicates(subset=["MMSI"], keep="first").set_index(
        "MMSI", drop=False)
    args_list = []
    for mmsi in mmsi_list:
        row = ships_by_mmsi.loc[mmsi]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        args_list.append(
            (mmsi, row, window_db_paths, year_label, berth_distance_threshold, pollutants))
    ncity = 0 if city_catchments is None else len(city_catchments)
    prefetch_n = max(0, int(prefetch_batch_size))
    sample_sec = max(0, int(prefetch_sample_interval_sec))
    cache_dir = str(trajectory_cache_dir or "").strip()
    if cache_dir and not Path(cache_dir).is_dir():
        raise FileNotFoundError(f"逐船轨迹缓存目录不存在: {cache_dir}")
    if cache_dir:
        if not prefetch_n:
            prefetch_n = 100
        io_mode = f"主进程每 {prefetch_n} 艘读取逐船 Parquet 缓存"
    else:
        io_mode = (
            f"主进程每 {prefetch_n} 艘批量预取轨迹"
            if prefetch_n else "工作进程逐船读取轨迹"
        )
    print(f"开始 {len(mmsi_list):,} 艘船  并行: {n_cores} 核  "
          f"(读 {len(window_db_paths)} 个月库=目标月及边界月; {io_mode})", flush=True)
    if prefetch_n and sample_sec:
        print(f"  AIS 时间桶抽样: 每船每 {sample_sec}s 保留首个真实点", flush=True)
    print(f"  城市集水区(30km) {ncity} 个已加载; 仅输出 {target_months}, "
          "边界月只用于夹住跨月航段",
          flush=True)
    t1 = time.time()
    ok_live = skip_live = fail_live = 0
    success_count = failed_count = nodata_count = 0
    err_lines = []
    nodata_lines = []
    city_buf, port_buf, edge_buf, op_buf, voy_buf = [], [], [], [], []
    pol_city_buf, pol_tot_buf, weekly_buf = [], [], []
    ops_calls_buf, ops_call_city_buf = [], []
    # 逐点守恒真实总量(不经港口/航段/集水区归属): ym -> {pol: 克}
    true_tot: dict = {}
    if policy_active:
        print(f"  政策情景启用: {len(policy_zones)} 个实施单元 "
              f"(SECA {policy_cfg.get('seca_sulfur', 0.001):.3%} / "
              f"NECA Build_Year>={policy_cfg.get('neca_build_year', 2016)})", flush=True)
    need_visit_buffers = (
        not policy_only or _norm_ship_type(current_ship_type) == "container"
    )
    port_coor_worker = port_coor
    if policy_only and _norm_ship_type(current_ship_type) == "container":
        if not policy_cfg or "ops_port_names" not in policy_cfg:
            raise RuntimeError(
                "政策专用 Container 运行缺少入选国 ops_port_names；"
                "禁止退回全球港口缓冲。"
            )
        if not policy_cfg["ops_port_names"]:
            raise RuntimeError("入选政策责任国未映射到任何研究港口")
        wanted_ports = set(policy_cfg["ops_port_names"])
        port_coor_worker = port_coor[port_coor["Port.Name"].isin(wanted_ports)].copy()
        print(
            f"  岸电挂靠识别仅保留入选政策责任国研究港口: "
            f"{port_coor_worker['Port.Name'].nunique():,} 港 / {len(port_coor_worker):,} 泊位",
            flush=True,
        )
    weekly_cfg_run = dict(WEEKLY_CFG)
    if policy_only:
        weekly_cfg_run["enabled"] = False
    with Pool(processes=n_cores, maxtasksperchild=1000,
              initializer=_init_emission_worker,
              initargs=(port_coor_worker, canal_coor, LAF, ECA_list, canal_km, city_catchments,
                        policy_zones, policy_cfg, weekly_cfg_run,
                        need_visit_buffers, policy_only)) as pool:
        tot = len(args_list)
        def _result_stream():
            if not prefetch_n:
                yield from pool.imap_unordered(
                    process_single_ship_multi_month, args_list, chunksize=1)
                return
            nb = (len(mmsi_list) + prefetch_n - 1) // prefetch_n
            for b0 in range(0, len(mmsi_list), prefetch_n):
                batch_ids = mmsi_list[b0:b0 + prefetch_n]
                bt = time.time()
                if cache_dir:
                    tracks = extract_ship_trajectories_cache(batch_ids, cache_dir)
                else:
                    tracks = extract_ship_trajectories_batch(
                        batch_ids, window_db_paths, sample_interval_sec=sample_sec,
                        strict=strict_db_read)
                nrows = sum(len(x) for x in tracks.values())
                print(
                    f"\n  [轨迹预取 {b0 // prefetch_n + 1}/{nb}] "
                    f"{len(batch_ids)} 艘 / {nrows:,} 点 / {time.time()-bt:.1f}s",
                    flush=True,
                )
                batch_args = []
                for mmsi in batch_ids:
                    row = ships_by_mmsi.loc[mmsi]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    batch_args.append((
                        mmsi, row, tracks[mmsi], year_label,
                        berth_distance_threshold, pollutants, True,
                    ))
                yield from pool.imap_unordered(
                    process_single_ship_multi_month, batch_args, chunksize=1)

        bar = tqdm(_result_stream(),
                   total=tot, desc=f"{progress_prefix}{current_ship_type[:16]:<16} {year_label}",
                   unit="船", dynamic_ncols=True, smoothing=0.03, mininterval=0.5)
        for result in bar:
            if result and result.get("success"):
                ok_live += 1
                success_count += 1
                for key, buf in (("city_emissions", city_buf),
                                 ("port_emissions", port_buf),
                                 ("edge_emissions", edge_buf),
                                 ("port_operational_visits", op_buf),
                                 ("edge_voyage_stats", voy_buf),
                                 ("policy_delta_city", pol_city_buf),
                                 ("policy_delta_total", pol_tot_buf),
                                 ("ops_calls", ops_calls_buf),
                                 ("ops_call_city", ops_call_city_buf),
                                 ("port_weekly", weekly_buf)):
                    d = result.get(key)
                    if d is not None and len(d) > 0:
                        buf.append(d)
                tbm = result.get("total_emissions_by_month")
                if tbm:
                    for _ym, dd in tbm.items():
                        acc = true_tot.setdefault(
                            _ym, {p: 0.0 for p in pollutants}
                        )
                        for p in pollutants:
                            acc[p] += float(dd.get(p, 0.0))
            elif result and result.get("no_data"):
                skip_live += 1
                nodata_count += 1
                nodata_lines.append(
                    f"{datetime.now()} | MMSI: {result.get('mmsi', 'NA')} | "
                    f"{result.get('error_message', '无可计算AIS航段')}\n"
                )
            else:
                fail_live += 1
                failed_count += 1
                rr = result or {}
                err_lines.append(
                    f"{datetime.now()} | MMSI: {rr.get('mmsi','NA')} | "
                    f"{rr.get('error_message','Unknown')}\n"
                )
            done_live = ok_live + skip_live + fail_live
            if done_live % 20 == 0 or done_live == tot:
                bar.set_postfix_str(f"成功 {ok_live} 无数据 {skip_live} 失败 {fail_live}", refresh=False)
        bar.close()
    print(f"并行完成, {(time.time()-t1)/60:.2f} 分钟  "
          f"(成功 {ok_live} 无数据 {skip_live} 失败 {fail_live})", flush=True)

    if err_lines:
        with open(error_log_file, "a", encoding="utf-8") as f:
            f.writelines(err_lines)
    nodata_log_file = os.path.join(
        log_dir, f"nodata_log_{sub}_{year_label}.txt"
    )
    with open(nodata_log_file, "w", encoding="utf-8") as f:
        f.write(f"无可计算航段 - {year_label} 船型: {current_ship_type}\n")
        f.write(
            f"读取库(含边界月): "
            f"{', '.join(os.path.basename(p) for p in window_db_paths)}\n\n"
        )
        f.writelines(nodata_lines)
    active = success_count + failed_count
    fail_rate = (100.0 * failed_count / active) if active else 0.0
    print(f"成功 {success_count}  无数据(未活动) {nodata_count}  失败 {failed_count}  "
          f"有效失败率 {failed_count}/{active}={fail_rate:.2f}%")
    if strict_db_read and failed_count:
        raise RuntimeError(
            f"严格模式发现 {failed_count} 艘船计算失败；详见 {error_log_file}"
        )

    print("聚合结果中(目标月份, 待按月分箱)…", flush=True)
    def _cat(b):
        return pd.concat(b, ignore_index=True) if b else pd.DataFrame()
    city_all, port_all = _cat(city_buf), _cat(port_buf)
    edge_all, op_all, voy_all = _cat(edge_buf), _cat(op_buf), _cat(voy_buf)
    pol_city_all, pol_tot_all = _cat(pol_city_buf), _cat(pol_tot_buf)
    ops_calls_all, ops_call_city_all = _cat(ops_calls_buf), _cat(ops_call_city_buf)
    weekly_all = _cat(weekly_buf)
    counts = {"ok": success_count, "nodata": nodata_count, "fail": failed_count,
              "fail_rate": fail_rate}

    def _slice(df, ym):
        return df[df["YearMonth"] == ym] if len(df) and "YearMonth" in df.columns else df.iloc[0:0]
    for ym in target_months:
        out_dir = os.path.join(results_root, ym, sub)
        _write_month_outputs(ym, out_dir, current_ship_type,
                             _slice(city_all, ym), _slice(port_all, ym), _slice(edge_all, ym),
                             _slice(op_all, ym), _slice(voy_all, ym),
                             pollutants, port_city_mapping, counts,
                             pol_city_m=_slice(pol_city_all, ym),
                             pol_tot_m=_slice(pol_tot_all, ym),
                             policy_active=policy_active,
                             true_tot_row=true_tot.get(ym),
                             weekly_m=_slice(weekly_all, ym),
                             ops_calls_m=_slice(ops_calls_all, ym),
                             ops_call_city_m=_slice(ops_call_city_all, ym),
                             policy_only=policy_only)
        print(f"  [写出] {ym}/{sub}", flush=True)
    print(f"  完成船型: {current_ship_type} ({year_label})\n")


# ============================================================================
# 主程序: --year 按季; 每季从多库 ship_info 取 MMSI(按型), 再拼 AIS 后算网
# ============================================================================
def main():
    args = _parse_main_args()
    work_dir = DEFAULT_WORK_DIR
    os.chdir(work_dir)
    year = int(args.year)
    if year < 1990 or year > 2100:
        raise SystemExit(f"年份不合理: {year}")
    db_dir = os.path.abspath(args.db_dir)
    results_root = os.path.abspath(
        args.results_root.strip() or DEFAULT_RESULTS_ROOT
    )
    n_cores = max(1, int(args.n_cores))
    skip_done = not args.no_skip_done
    target_all, full_window = _year_window(year)
    want = _parse_only_months(args.only_months)
    target_months = [ym for ym in target_all if int(ym[-2:]) in want]
    if not target_months:
        raise SystemExit("无待处理(输出)月份")
    # 全年任务仍读取 14 个月。独立月度任务只读取目标月及其相邻月份，
    # 既保留跨月轨迹边界，也避免每个月都重复查询整年数据库。
    window = (
        full_window
        if len(target_months) == len(target_all)
        else _target_month_window(target_months)
    )
    polls = list(POLLUTANTS)
    canal_km = float(args.canal_km)

    print("=" * 80)
    print("船舶排放网络: 自然年连续处理(年±1 边界月) + 按月分箱; 城市节点(30km) + 港口级(可选)")
    print("=" * 80)
    print(f"工作目录: {work_dir}\n年: {year}\n库目录: {db_dir}\n输出根: {results_root}")
    print(f"输出月份: {target_months}")
    print(f"读取窗口(含上年12月+下年1月边界): {window}")
    print(f"并行: {n_cores}  运河半径: {canal_km} km  断点: {skip_done}")
    if not args.all_ship_types:
        print(f"船型白名单: {SHIP_TYPE_WHITELIST}")

    t0 = time.time()
    port_coor, canal_coor, _, LAF, ECA_list = load_port_laf_eca_data(
        port_data_path=os.path.abspath(args.port_xlsx),
        laf_path=os.path.abspath(args.laf_xlsx),
        eca_base=os.path.abspath(args.eca_base),
    )

    print("\n" + "=" * 80)
    print("步骤: 加载城市集水区(缓冲区) 与 港口→城市映射表 (Port2City 输出)")
    print("=" * 80)
    print(f"  集水区 gpkg: {os.path.abspath(args.city_catchments)}")
    city_catchments, port_city_mapping = load_city_data({
        "city_catchments_gpkg": os.path.abspath(args.city_catchments),
        "port_city_mapping_csv": os.path.abspath(args.port_city_mapping),
    })
    # 城市缓冲区(30km 集水区)在 Port2City 阶段(build_catchments.py)预生成；本流程加载后
    # 用作排放空间归属的目标面。这里把它的规模/坐标系/面积分布打印出来，便于核对。
    try:
        _cc = city_catchments
        _crs = _cc.crs
        _area = None
        if "area_km2" in _cc.columns:
            _area = pd.to_numeric(_cc["area_km2"], errors="coerce")
        else:  # 没有现成面积列则等积投影现算
            _area = _cc.to_crs("EPSG:6933").geometry.area / 1e6
        print(
            f"  城市集水区(缓冲区): {len(_cc)} 个节点  CRS={_crs}  "
            f"几何={_cc.geom_type.iloc[0] if len(_cc) else 'NA'}"
        )
        print(
            f"  集水区面积 km^2: 中位数 {float(_area.median()):.0f}  "
            f"最小 {float(_area.min()):.0f}  最大 {float(_area.max()):.0f}"
        )
    except Exception as _e:
        print(f"  集水区: {len(city_catchments)} 城市 (面积统计跳过: {_e})")
    print(f"  港口→城市映射: {len(port_city_mapping)} 港口")
    print(
        "  归属方式: 每艘船每个 AIS 点的排放按 within 落入 30km 集水区, "
        "重叠区重复计入(空间溢出语义), 按停泊/航行拆分 → 城市节点排放; "
        "港口 OD 边经映射表归并为城市 OD 边(同城剔除)。"
    )

    # ---- 政策情景区（研究 4；--policy-zones 或 config.yaml policy.enabled）----
    policy_zones = None
    policy_cfg = None
    _pol_cfg_all = _CFG.get("policy") or {}
    pz_path = args.policy_zones.strip()
    if not pz_path and _pol_cfg_all.get("enabled"):
        pz_path = str(_pol_cfg_all.get("zones_gpkg", "")).strip()
    if args.no_policy:
        pz_path = ""
    if pz_path:
        if not os.path.isfile(pz_path):
            raise SystemExit(f"政策情景区文件不存在: {pz_path}")
        pz_layer = args.policy_zone_layer.strip() or None
        policy_zones = gpd.read_file(pz_path, layer=pz_layer)
        if "zone_id" not in policy_zones.columns:
            raise SystemExit(f"政策情景区缺 zone_id 列: {pz_path}")
        if policy_zones.crs is None:
            policy_zones = policy_zones.set_crs("EPSG:4326")
        elif str(policy_zones.crs).upper() not in ("EPSG:4326", "OGC:CRS84"):
            policy_zones = policy_zones.to_crs("EPSG:4326")
        policy_cfg = {
            "seca_sulfur": float(_pol_cfg_all.get("seca_sulfur", 0.001)),
            "neca_build_year": int(_pol_cfg_all.get("neca_build_year", 2016)),
        }
        if any(c in policy_zones.columns for c in (
            "policy_iso3", "policy_iso3s", "geo_iso3", "geo_iso3s"
        )):
            policy_iso3 = {
                code.strip()
                for col in ("policy_iso3", "policy_iso3s")
                if col in policy_zones.columns
                for cell in policy_zones[col].dropna().astype(str)
                for code in cell.split(",")
                if code.strip()
            }
            # 港口表按实际地理单元编码；海外属地不能只用母国 policy_iso3
            # 筛选，否则会漏掉 GUF 等实施海岸。没有 geo 列时才退回责任国。
            port_geo_iso3 = {
                code.strip()
                for col in ("geo_iso3", "geo_iso3s")
                if col in policy_zones.columns
                for cell in policy_zones[col].dropna().astype(str)
                for code in cell.replace(";", ",").split(",")
                if code.strip()
            }
            if not port_geo_iso3:
                port_geo_iso3 = set(policy_iso3)
            port_meta = pd.read_csv(
                os.path.abspath(args.port_city_mapping), encoding="utf-8-sig"
            )
            if not {"Port_Name", "iso3"}.issubset(port_meta.columns):
                raise SystemExit("港口→城市映射缺少 Port_Name/iso3，无法限定岸电实施港口")
            policy_cfg["ops_port_names"] = sorted(set(
                port_meta.loc[
                    port_meta["iso3"].astype(str).isin(port_geo_iso3), "Port_Name"
                ].dropna().astype(str)
            ))
            policy_cfg["policy_iso3"] = sorted(policy_iso3)
            policy_cfg["ops_geo_iso3"] = sorted(port_geo_iso3)
            if not policy_cfg["ops_port_names"]:
                raise SystemExit("正式政策地理单元未映射到任何岸电研究港口")
            print(
                f"  岸电政策责任国: {len(policy_iso3)}  "
                f"实际地理单元: {len(port_geo_iso3)}  "
                f"可映射研究港口: {len(policy_cfg['ops_port_names'])}",
                flush=True,
            )
        print("\n" + "=" * 80)
        print("政策情景(研究 4): 单遍多情景 delta 输出已启用")
        print("=" * 80)
        print(f"  实施单元: {len(policy_zones)} 个  来源: {pz_path}"
              f"  图层: {pz_layer or '(default)'}")
        print(f"  SECA 硫率上限: {policy_cfg['seca_sulfur']:.3%}  "
              f"NECA Tier III 建造年下限: {policy_cfg['neca_build_year']}")

    print("\n[a] 检查数据库(窗口内文件名匹配, 含两端边界月):")
    window_db_paths, window_found = _find_dbs_in_dir(window, db_dir)
    if not window_db_paths:
        raise SystemExit(f"窗口内无任何库: {window}")
    missing = [m for m in window if m not in window_found]
    if missing:
        print(f"  ⚠ 窗口内缺月库: {missing}（缺边界月→该端跨月航段留极小残余；缺目标月→该月空输出）")
    # 船宇宙取自“目标年”的库（不含边界月，避免引入只在边界月出现的船）
    tset = set(target_months)
    target_db_paths = [p for p in window_db_paths if os.path.basename(p)[4:10] in tset]
    if not target_db_paths:
        raise SystemExit("目标年的月库均缺失")
    print(f"\n[b] 从 {len(target_db_paths)} 个目标月库 ship_info 取 MMSI(去重) -> extract_all_mmsi")
    all_ships = extract_all_mmsi(
        target_db_paths, None, strict=bool(args.strict_db_read)
    )
    all_ships = enrich_ship_attributes(all_ships, args.ship_attributes_csv.strip())
    if args.mmsi_list_csv.strip():
        p_mmsi = os.path.abspath(args.mmsi_list_csv.strip())
        audit_ids = pd.read_csv(p_mmsi)
        if "MMSI" not in audit_ids.columns:
            raise SystemExit(f"--mmsi-list-csv 缺少 MMSI 列: {p_mmsi}")
        wanted_ids = set(pd.to_numeric(
            audit_ids["MMSI"], errors="coerce").dropna().astype(int))
        before_ids = len(all_ships)
        all_ships = all_ships[
            pd.to_numeric(all_ships["MMSI"], errors="coerce")
            .astype("Int64").isin(wanted_ids)
        ].copy()
        print(
            f"  MMSI 审计白名单: {len(all_ships):,}/{before_ids:,} 艘  来源: {p_mmsi}",
            flush=True,
        )
    st_counts = all_ships["Ship_Type"].value_counts()
    print(f"  船型数: {len(st_counts)}  船舶: {len(all_ships):,}")
    st_list = st_counts.index.tolist()
    if not args.all_ship_types:
        skipped = [st for st in st_list if not _ship_type_allowed(st)]
        st_list = [st for st in st_list if _ship_type_allowed(st)]
        print(f"  白名单过滤: 保留 {st_list}  跳过 {len(skipped)} 类: {skipped}")
    if args.only_ship_types.strip():
        norm_want = {_norm_ship_type(s) for s in args.only_ship_types.split(",") if s.strip()}
        st_list = [st for st in st_list if _norm_ship_type(st) in norm_want]
        print(f"  --only-ship-types 过滤 -> 保留: {st_list}")
        if not st_list:
            raise SystemExit(f"--only-ship-types {args.only_ship_types!r} 未匹配任何船型")

    grand = []
    nt = len(st_list)
    for ti, st in enumerate(st_list):
        print("\n" + "#" * 80)
        print(f"  {year} / {st}   (型 {ti+1}/{nt})")
        print("#" * 80)
        sub = _safe_folder_name(st)
        if skip_done and all(
            _month_outputs_done(os.path.join(results_root, ym, sub), st, ym,
                                policy_active=policy_zones is not None,
                                policy_only=bool(args.policy_only))
            for ym in target_months
        ):
            print(f"  [跳过-已完成] {year} / {st}")
            grand.append((str(year), st, "skip"))
            continue
        ships_st = all_ships[all_ships["Ship_Type"] == st].copy()
        prefix = f"[型{ti+1}/{nt}] "
        try:
            run_emission_for_year_ship_type(
                st, ships_st, target_months, window_db_paths, str(year),
                port_coor, canal_coor, LAF, ECA_list, results_root, polls, n_cores,
                canal_km=canal_km, city_catchments=city_catchments,
                port_city_mapping=port_city_mapping,
                max_ships=int(args.max_ships_per_type), progress_prefix=prefix,
                policy_zones=policy_zones, policy_cfg=policy_cfg,
                policy_only=bool(args.policy_only),
                ship_start_index=int(args.ship_start_index),
                prefetch_batch_size=int(args.prefetch_batch_size),
                prefetch_sample_interval_sec=int(args.prefetch_sample_interval_sec),
                trajectory_cache_dir=args.trajectory_cache_dir.strip(),
                strict_db_read=bool(args.strict_db_read),
            )
            grand.append((str(year), st, "ok"))
        except Exception as e:
            import traceback
            print(f"  失败 {year}/{st}: {e}")
            print(traceback.format_exc())
            grand.append((str(year), st, f"err:{e}"))
            if args.strict_db_read:
                raise

    print("\n" + "=" * 80)
    print(f"全部结束, {(time.time()-t0)/60:.1f} 分钟, 子任务 {len(grand)}")
    print("=" * 80)

if __name__ == '__main__':
    main()
