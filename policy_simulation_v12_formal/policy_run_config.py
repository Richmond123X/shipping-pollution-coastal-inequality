# -*- coding: utf-8 -*-
"""V12正式政策模拟的唯一运行年份与路径配置。"""
from pathlib import Path


ROOT = Path(r"E:\transfer justice\policy_simulation_v12_formal")
WORK = Path(r"E:\transfer justice")
BASELINE_MONTHLY_ROOT = (
    WORK / "results_v10_reconstructed_2019_2027" / "reconstructed_monthly"
)

# 用户于2026-08-23固定：政策效果计算使用2024年AIS；政策范围、X=50%、
# S1--S7技术参数、2025锚点SPRI区划和2022电网强度均保持不变。
SIMULATION_YEAR = 2024
AIS_DIR = Path(r"F:\SQL_AIS_SingleShip")
SELECTION_RISK_YEAR = 2025
TRADE_YEAR = 2024
GRID_INTENSITY_YEAR = 2022
# 0 means full-frequency AIS: preserve every valid observation used by the
# established emission-network model. Batching below changes I/O only.
AIS_SAMPLE_INTERVAL_SEC = 0
AIS_PREFETCH_BATCH_SIZE = 0
AIS_SHARD_SIZE = 500

YEAR_MONTHS = [f"{SIMULATION_YEAR}{month:02d}" for month in range(1, 13)]
GT_CROSSWALK = ROOT / "inputs" / f"container_gt_crosswalk_{SIMULATION_YEAR}.csv"
DIRECT_ROOT = ROOT / "intermediate" / f"ais_policy_{SIMULATION_YEAR}_direct"
DIRECT_LOG_ROOT = ROOT / "logs" / f"ais_policy_{SIMULATION_YEAR}_direct"
DIRECT_MANIFEST = ROOT / "tables" / f"ais_policy_{SIMULATION_YEAR}_direct_manifest.csv"
CONSOLIDATED_ROOT = ROOT / "intermediate" / f"ais_policy_{SIMULATION_YEAR}_consolidated"
SCENARIO_ROOT = ROOT / "intermediate" / f"scenarios_{SIMULATION_YEAR}"
