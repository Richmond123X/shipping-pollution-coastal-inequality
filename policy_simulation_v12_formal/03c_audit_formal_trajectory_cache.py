# -*- coding: utf-8 -*-
"""审计正式 300 秒轨迹缓存相对同一 250 km 候选范围内全频 AIS 的误差。

脚本只在 03b 完整结束后运行。它从顺序扫描保留的 raw_margin 中抽取四船型
各若干艘全频轨迹，分别运行全频与正式缓存排放程序，并比较政策 delta、区内
基线、城市归集和集装箱挂靠指标。不会再次随机读取 TB 级 SQLite 轨迹表。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from shapely import contains_xy, union_all


ROOT = Path(r"E:\transfer justice\policy_simulation_v12_formal")
WORK = Path(r"E:\transfer justice")
CACHE = ROOT / "intermediate" / "trajectory_cache_300s_250km"
AUDIT = ROOT / "intermediate" / "trajectory_cache_audit_v12_geodesic75"
ZONES = ROOT / "inputs" / "formal_policy_zones_50pct.gpkg"
GT = ROOT / "inputs" / "container_gt_crosswalk_2025.csv"
PORT_MAP = WORK / "port2city" / "outputs" / "port_city_mapping.csv"
SELECTED_GEO = ROOT / "tables" / "selected_geo_units_50pct.csv"
EMISSION = WORK / "emission_network" / "emission_network_main.py"
AIS_DIR = Path(r"D:\SQL_AIS_SingleShip")
SHIP_TYPES = ["Bulk carrier", "General cargo", "Oil tanker", "Container"]
DELTA = ["D_SO2_SECA", "D_PM_SECA", "D_NOx_NECA"]
NO_MARGIN = ["NME_SO2_CURRENT", "NME_PM_CURRENT", "NME_NOx_CURRENT"]
BASE = ["B_SO2", "B_PM", "B_NOx"]


def require_cache() -> dict:
    path = CACHE / "cache_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"正式缓存未完成: {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("status") != "complete":
        raise RuntimeError(f"缓存状态异常: {state}")
    if state.get("sample_interval_sec") != 300:
        raise RuntimeError(f"正式抽样间隔不是 300 秒: {state.get('sample_interval_sec')}")
    return state


def parquet_rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def select_mmsi(n_per_type: int) -> pd.DataFrame:
    universe = pd.read_csv(CACHE / "formal_ship_universe.csv")
    files = list((CACHE / "by_mmsi").glob("*.parquet"))
    counts = pd.DataFrame({
        "MMSI": [int(p.stem) for p in files],
        "sampled_rows": [parquet_rows(p) for p in files],
        "path": files,
    }).merge(universe, on="MMSI", how="left", validate="1:1")
    zones = gpd.read_file(ZONES, layer="policy_regions").to_crs(4326)
    union = union_all(zones.geometry.values)
    selected_geo = set(pd.read_csv(SELECTED_GEO)["geo_iso3"].astype(str))
    ports = pd.read_csv(PORT_MAP, encoding="utf-8-sig")
    ports = ports[ports["iso3"].astype(str).isin(selected_geo)].copy()
    port_points = gpd.GeoDataFrame(
        ports,
        geometry=gpd.points_from_xy(ports["Lon_port"], ports["Lat_port"]),
        crs=4326,
    ).to_crs(6933)
    # 10 km 仅用于提高审计样本中“真实挂靠”的命中率，不参与正式挂靠识别。
    port_buffer = gpd.GeoSeries(
        [union_all(port_points.geometry.buffer(10_000).values)], crs=6933
    ).to_crs(4326).iloc[0]
    chosen = []
    for ship_type in SHIP_TYPES:
        c = counts[counts["Ship_Type"].astype(str).str.casefold().eq(ship_type.casefold())]
        # 先看轨迹量适中且 MMSI 分布分散的船，避免只审计相邻编号或超大轨迹。
        c = c[c["sampled_rows"].between(200, 12000)].sort_values("MMSI")
        if len(c) > 200:
            idx = np.unique(np.linspace(0, len(c) - 1, 200).round().astype(int))
            c = c.iloc[idx]
        for row in c.itertuples(index=False):
            d = pd.read_parquet(
                row.path, columns=["Longitude", "Latitude", "Speed"]
            )
            inside = contains_xy(
                union, d["Longitude"].to_numpy(), d["Latitude"].to_numpy()
            )
            n_inside = int(np.count_nonzero(inside))
            if n_inside < 20:
                continue
            slow_near_port = 0
            if ship_type == "Container":
                near = contains_xy(
                    port_buffer, d["Longitude"].to_numpy(), d["Latitude"].to_numpy()
                )
                speed = pd.to_numeric(d["Speed"], errors="coerce").to_numpy()
                slow_near_port = int(np.count_nonzero(near & np.isfinite(speed) & (speed <= 1.0)))
                if slow_near_port < 12:
                    continue
            chosen.append({
                "MMSI": int(row.MMSI), "Ship_Type": ship_type,
                "sampled_rows": int(row.sampled_rows),
                "sampled_points_inside_policy": n_inside,
                "sampled_slow_points_near_policy_port": slow_near_port,
            })
            if sum(x["Ship_Type"] == ship_type for x in chosen) >= n_per_type:
                break
        if sum(x["Ship_Type"] == ship_type for x in chosen) < n_per_type:
            raise RuntimeError(f"{ship_type} 未找到足够的政策区活动审计船")
    out = pd.DataFrame(chosen).sort_values(["Ship_Type", "MMSI"])
    AUDIT.mkdir(parents=True, exist_ok=True)
    out.to_csv(AUDIT / "audit_mmsi.csv", index=False, encoding="utf-8-sig")
    return out


def extract_full_margin(sample: pd.DataFrame, n_partitions: int) -> Path:
    out = AUDIT / "full_margin_by_mmsi"
    out.mkdir(parents=True, exist_ok=True)
    ids = set(sample["MMSI"].astype(int))
    parts: dict[int, list[int]] = {}
    for mmsi in ids:
        parts.setdefault(mmsi % n_partitions, []).append(mmsi)
    gathered = {m: [] for m in ids}
    for month in [f"2025{x:02d}" for x in range(1, 13)]:
        manifest = CACHE / "raw_margin" / month / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        for part, mmsis in parts.items():
            src = CACHE / "raw_margin" / month / f"part_{part:03d}.parquet"
            if not src.is_file():
                continue
            d = pd.read_parquet(src, filters=[("MMSI", "in", mmsis)])
            if len(d):
                for mmsi, g in d.groupby("MMSI", sort=False):
                    gathered[int(mmsi)].append(g)
    records = []
    for mmsi in sorted(ids):
        if not gathered[mmsi]:
            raise RuntimeError(f"全频 margin 无轨迹: {mmsi}")
        d = (pd.concat(gathered[mmsi], ignore_index=True)
             .drop_duplicates(["MMSI", "UTCtime"], keep="first")
             .sort_values("UTCtime"))
        d.to_parquet(out / f"{mmsi:09d}.parquet", index=False, compression="zstd")
        records.append({"MMSI": mmsi, "full_margin_rows": len(d)})
    pd.DataFrame(records).to_csv(
        AUDIT / "audit_full_margin_rows.csv", index=False, encoding="utf-8-sig"
    )
    return out


def run_emission(cache_dir: Path, output: Path, mmsi_csv: Path) -> None:
    cmd = [
        sys.executable, str(EMISSION), "--year", "2025",
        "--db-dir", str(AIS_DIR), "--results-root", str(output),
        "--mmsi-list-csv", str(mmsi_csv), "--n-cores", "4",
        "--policy-zones", str(ZONES), "--policy-zone-layer", "policy_units",
        "--ship-attributes-csv", str(GT), "--policy-only",
        "--trajectory-cache-dir", str(cache_dir), "--no-skip-done",
    ]
    log = AUDIT / f"{output.name}.log"
    with log.open("w", encoding="utf-8") as fh:
        subprocess.run(cmd, cwd=WORK, check=True, stdout=fh, stderr=subprocess.STDOUT)


def read_monthly(root: Path, base: str, suffix: str) -> pd.DataFrame:
    files = list(root.rglob(f"{base}_*.{suffix}"))
    parts = []
    for path in files:
        d = pd.read_parquet(path) if suffix == "parquet" else pd.read_csv(path)
        if len(d):
            parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def compare(full_root: Path, sampled_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for base, cols in (
        ("policy_delta_totals", DELTA + NO_MARGIN + BASE),
        ("city_policy_delta", DELTA),
    ):
        f = read_monthly(full_root, base, "csv")
        s = read_monthly(sampled_root, base, "csv")
        for ship_type in ["ALL"] + SHIP_TYPES:
            ff = f if ship_type == "ALL" else f[f["Ship_Type"].eq(ship_type)]
            ss = s if ship_type == "ALL" else s[s["Ship_Type"].eq(ship_type)]
            for col in cols:
                fv = float(pd.to_numeric(ff[col], errors="coerce").sum()) if col in ff else 0.0
                sv = float(pd.to_numeric(ss[col], errors="coerce").sum()) if col in ss else 0.0
                rows.append({
                    "output": base, "ship_type": ship_type, "metric": col,
                    "full_value": fv, "sampled_value": sv,
                    "difference": sv - fv,
                    "relative_error": (sv - fv) / fv if fv else (0.0 if sv == 0 else np.nan),
                })
    metrics = pd.DataFrame(rows)

    ops_rows = []
    for base in ("ops_calls", "ops_call_city"):
        f = read_monthly(full_root, base, "parquet")
        s = read_monthly(sampled_root, base, "parquet")
        # 正式程序只为 Container 写出这两类岸电表；城市挂靠表为节省空间
        # 不重复存 Ship_Type，因此在审计阶段显式恢复这一已知常量。
        if len(f) and "Ship_Type" not in f:
            f["Ship_Type"] = "Container"
        if len(s) and "Ship_Type" not in s:
            s["Ship_Type"] = "Container"
        for ship_type in ["ALL"] + SHIP_TYPES:
            # 岸电正式样本只有 Container。仍把其余船型逐组写入审计表，
            # 但质量闸门必须明确检查 Container，不能用四船型合计的挂靠
            # 冒充集装箱船挂靠已经被审计。
            if ship_type == "ALL":
                ff, ss = f, s
            else:
                ff = f[f["Ship_Type"].eq(ship_type)] if "Ship_Type" in f else f.iloc[0:0]
                ss = s[s["Ship_Type"].eq(ship_type)] if "Ship_Type" in s else s.iloc[0:0]
            for metric, col, func in (
                ("rows", None, lambda d: len(d)),
                ("unique_calls", "Call_ID", lambda d: d["Call_ID"].nunique() if len(d) else 0),
                ("ops_energy_kwh", "OPS_Energy_kWh", lambda d: float(d["OPS_Energy_kWh"].sum()) if len(d) and "OPS_Energy_kWh" in d else 0.0),
                ("berth_so2_g", "Berth_SO2_g", lambda d: float(d["Berth_SO2_g"].sum()) if len(d) and "Berth_SO2_g" in d else 0.0),
                ("berth_pm_g", "Berth_PM_g", lambda d: float(d["Berth_PM_g"].sum()) if len(d) and "Berth_PM_g" in d else 0.0),
                ("berth_nox_g", "Berth_NOx_g", lambda d: float(d["Berth_NOx_g"].sum()) if len(d) and "Berth_NOx_g" in d else 0.0),
            ):
                fv, sv = float(func(ff)), float(func(ss))
                ops_rows.append({
                    "output": base, "ship_type": ship_type, "metric": metric,
                    "full_value": fv, "sampled_value": sv,
                    "difference": sv - fv,
                    "relative_error": (
                        (sv - fv) / fv if fv else (0.0 if sv == 0 else np.nan)
                    ),
                })
    return metrics, pd.DataFrame(ops_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-type", type=int, default=3)
    args = ap.parse_args()
    state = require_cache()
    sample = select_mmsi(args.n_per_type)
    full_cache = extract_full_margin(sample, int(state["n_partitions"]))
    full_out = AUDIT / "full_margin_emission"
    sampled_out = AUDIT / "sampled_300s_emission"
    mmsi_csv = AUDIT / "audit_mmsi.csv"
    run_emission(full_cache, full_out, mmsi_csv)
    run_emission(CACHE / "by_mmsi", sampled_out, mmsi_csv)
    metrics, ops = compare(full_out, sampled_out)
    metrics.to_csv(AUDIT / "sampling_error_emission_metrics.csv", index=False, encoding="utf-8-sig")
    ops.to_csv(AUDIT / "sampling_error_ops_metrics.csv", index=False, encoding="utf-8-sig")

    key_em = metrics[
        metrics["output"].eq("policy_delta_totals")
        & metrics["ship_type"].eq("ALL")
        & metrics["metric"].isin(DELTA + BASE)
        & metrics["relative_error"].notna()
    ]
    key_ops = ops[
        ops["output"].eq("ops_calls")
        & ops["ship_type"].eq("Container")
        & ops["metric"].isin(["unique_calls", "ops_energy_kwh", "berth_so2_g", "berth_pm_g", "berth_nox_g"])
        & ops["relative_error"].notna()
    ]
    max_em = float(key_em["relative_error"].abs().max()) if len(key_em) else np.nan
    max_ops = float(key_ops["relative_error"].abs().max()) if len(key_ops) else np.nan
    full_calls = float(ops.loc[
        ops["output"].eq("ops_calls")
        & ops["ship_type"].eq("Container")
        & ops["metric"].eq("unique_calls"),
        "full_value",
    ].sum())
    passed = bool(
        np.isfinite(max_em) and max_em <= 0.01
        and np.isfinite(max_ops) and max_ops <= 0.02
        and full_calls > 0
    )
    qc = {
        "status": "pass" if passed else "review",
        "n_mmsi": int(len(sample)), "n_per_type": int(args.n_per_type),
        "max_abs_emission_relative_error": max_em,
        "max_abs_ops_relative_error": max_ops,
        "full_frequency_unique_container_calls": full_calls,
        "emission_threshold": 0.01, "ops_threshold": 0.02,
        "scope": "temporal sampling conditional on the same 250-km candidate margin",
    }
    (AUDIT / "sampling_audit_summary.json").write_text(
        json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(qc, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
